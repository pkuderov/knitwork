from dataclasses import dataclass

import torch


@dataclass
class TrainBatch:
    # (B, obs_size)
    o_t: torch.Tensor
    # (B, act_size)
    a_t: torch.Tensor
    # (B, 1)
    r_t: torch.Tensor
    # (B, obs_size)
    o_tn: torch.Tensor
    # reset flags
    # (B, 1)
    fl_t: torch.Tensor


class FlatReplayBuffer:
    """
    Stores transitions from different parallel environments without preserving the episode
    rollouts contiguity. That is, each "i+1"-th transition in RB almost surely does ot relate
    to the same episode as i-th one. So, it's the most simple and performant option for
    async experience collection.
    """
    def __init__(
            self, *, 
            max_size: int, enough_size: int,
            obs_size: int, act_size: int, device, dtype,
            use_multi_batch_sampling: bool = False,
            seed: int | None = None
    ):
        self.max_size = int(max_size)
        self.enough_size = int(enough_size)

        self.ptr, self.size = 0, 0
        # total number of transitions added
        self.n_total_added = 0
        # number of times RB was sampled
        self.n_sampled = 0

        self.device = device
        self.rng = torch.Generator(device)
        if seed is not None:
            self.rng.manual_seed(seed)

        # (N, [obs_size | act_size | 1])
        self.obs = torch.empty((self.max_size, obs_size), dtype=dtype, device=device)
        self.act = torch.empty((self.max_size, act_size), dtype=dtype, device=device)
        self.rew = torch.empty((self.max_size, 1), dtype=dtype, device=device)
        self.flags = torch.empty((self.max_size, 1), dtype=torch.uint8, device=device)
        self.obs_next = torch.empty((self.max_size, obs_size), dtype=dtype, device=device)

        self._last_obs: torch.Tensor = None
        self._last_act: torch.Tensor = None
        self._last_done: torch.Tensor = None

        # sample k batches at a time: so it's infrequent costly large sample (aka mega batch)
        # and the other k-1 times it's just slicing
        self._use_mega_sampling = use_multi_batch_sampling
        self._mega_batch = None
        self._mega_ptr = 0
        self._mega_k = 16

    @property
    def is_initialized(self):
        return self._last_obs is not None
    @property
    def fill_rate(self):
        return self.size / self.max_size

    def put(self, ixs, *, rew, flags, obs_next, act_next):
        # if prev step was done, then now it's reset
        obs, act, reset = self._last_obs[ixs].clone(), self._last_act[ixs].clone(), self._last_done[ixs].clone()
        self._last_obs[ixs] = obs_next
        self._last_act[ixs] = act_next
        self._last_done[ixs] = torch.bitwise_and(flags, RlFlags.DONE)

        # filter out invalid cross-episode transitions, i.e. when episodes are reset
        # NB: since only non-reset transition are added, there's no need to add reset to flags
        m = ~reset.bool().squeeze(-1)
        obs, act, rew, flags, obs_next = obs[m], act[m], rew[m], flags[m], obs_next[m]

        n_samples = len(rew)
        if n_samples == 0:
            return

        n_free = self.max_size - self.ptr
        if n_samples > n_free:
            self._commit(obs[:n_free], act[:n_free], rew[:n_free], flags[:n_free], obs_next[:n_free])
            self._commit(obs[n_free:], act[n_free:], rew[n_free:], flags[n_free:], obs_next[n_free:])
        else:
            self._commit(obs, act, rew, flags, obs_next)

        # only replay buffer can "connect" transitions, so someone might need the reset flag as well
        # NB: I reset bool mask for simplicity and clarity
        return ~m

    def _commit(self, obs, act, rew, flags, obs_next):
        """Adds a batch of transitions to the replay buffer."""
        # overflow is not checked there intentionally, do that outside
        i, bsz = self.ptr, act.shape[0]
        self.obs[i:i+bsz] = obs
        self.act[i:i+bsz] = act
        self.rew[i:i+bsz] = rew
        self.flags[i:i+bsz] = flags
        self.obs_next[i:i+bsz] = obs_next

        self.ptr = (self.ptr + bsz) % self.max_size
        self.size = min(self.size + bsz, self.max_size)
        self.n_total_added += bsz

    def sample(self, batch_size: int):
        assert self.is_initialized
        if self._use_mega_sampling:
            return self.sample_from_mega(batch_size)

        ixs = torch.randint(0, self.size, size=(batch_size,), device=self.device, generator=self.rng)
        self.n_sampled += 1
        return TrainBatch(
            o_t=self.obs[ixs], 
            a_t=self.act[ixs], 
            r_t=self.rew[ixs],
            fl_t=self.flags[ixs],
            o_tn=self.obs_next[ixs]
        )

    def sample_from_mega(self, batch_size: int):
        assert self.is_initialized
        need_resample = (
            self._mega_batch is None
            or (self._mega_ptr + batch_size) > self._mega_batch.o_t.shape[0]
        )

        if need_resample:
            mega_bsz = self._mega_k * batch_size
            ixs = torch.randint(0, self.size, size=(mega_bsz,), device=self.device, generator=self.rng)
            self._mega_batch = TrainBatch(
                o_t=self.obs[ixs], 
                a_t=self.act[ixs], 
                r_t=self.rew[ixs],
                fl_t=self.flags[ixs],
                o_tn=self.obs_next[ixs]
            )
            self._mega_ptr = 0

        b = self._mega_batch
        i = self._mega_ptr
        self._mega_ptr += batch_size
        self.n_sampled += 1
        return TrainBatch(
            o_t=b.o_t[i:i+batch_size],
            a_t=b.a_t[i:i+batch_size],
            r_t=b.r_t[i:i+batch_size],
            fl_t=b.fl_t[i:i+batch_size],
            o_tn=b.o_tn[i:i+batch_size]
        )

    def init_state(self, n_envs, env_bsz):
        obs_size, act_size = self.obs.shape[-1], self.act.shape[-1]

        self._last_obs = self.act.new_empty((n_envs, obs_size))
        self._last_act = self.act.new_empty((n_envs, act_size))
        # init with true (=Terminated) for the first ever semi-transition to be correctly dropped as a "reset" one
        self._last_done = self.flags.new_ones((n_envs, 1))

        self.print_stats(env_bsz)
    
    def get_total_size(self):
        assert self.is_initialized
        total_bytes = 0
        for t in [self.obs, self.act, self.rew, self.flags, self.obs_next]:
            total_bytes += t.numel() * t.element_size()
        return total_bytes

    def print_stats(self, bsz=None):
        max_vec_env_steps = self.max_size // bsz if bsz is not None else None
        sz, sfx = to_readable_size(self.get_total_size())

        max_vec_env_steps_msg = ''
        if max_vec_env_steps is not None:
            max_vec_env_steps_msg = f'  ({format_readable_num(max_vec_env_steps)})'
        print(
            f'Replay buffer: Size = {sz:.2f}{sfx} {max_vec_env_steps_msg}'
        )


class CompressedReplayBuffer:
    """Stub for future implementations."""
    def __init__(self, max_size: int, rng: torch.Generator = None):
        assert False, "Not tested"
        self.max_size = int(max_size)
        self.ptr, self.size = 0, 0
        # total number of transitions added
        self.n_total_added = 0
        # number of times RB was sampled
        self.n_sampled = 0
        self.rng = rng

        # ==> will be initialized after the first episode is finished
        # (N, [obs_size | act_size | 1])
        self.obs, self.act, self.rew, self.flags = None, None, None, None
        self.n_envs = None

    @property
    def is_initialized(self):
        return self.obs is not None
    @property
    def fill_rate(self):
        return self.size / self.max_size

    def put(self, obs, act, rew, flags):
        if not self.is_initialized:
            self.init_buffer(obs, act)

        assert act.shape[0] == self.n_envs

        def _put(i, obs, act, rew, flags):
            bsz = act.shape[0]
            self.obs[i:i+bsz] = obs
            self.act[i:i+bsz] = act
            self.rew[i:i+bsz] = rew
            self.flags[i:i+bsz] = flags
            return i + bsz

        n_added = self.n_envs
        self.ptr = _put(self.ptr, obs, act, rew, flags) % self.max_size
        self.size = min(self.size + n_added, self.max_size)
        self.n_total_added += n_added

    def sample(self, batch_size: int):
        assert self.is_initialized

        # size - n_envs to exclude last step, which doesn't have corresponding obs_next
        n_envs = self.n_envs
        ixs = maybe_with_generator(torch.randint, self.rng)(0, self.size - n_envs, (batch_size,))
        ixs += self.ptr

        o_t = self.obs[ixs]
        o_tn = self.obs[(ixs + n_envs) % self.max_size]
        a_t = self.act[ixs]
        r_t = self.rew[ixs]
        self.n_sampled += 1
        return TrainBatch(o_t=o_t, a_t=a_t, r_t=r_t, o_tn=o_tn)

    def init_buffer(self, obs, act):
        n_envs, obs_size = obs.shape
        act_size = act.shape[-1]
        # round max_size to the nearest divisible to num envs
        self.max_size = n = (self.max_size // n_envs) * n_envs

        self.obs = act.new_empty((n, obs_size))
        self.act = act.new_empty((n, act_size))
        self.rew = act.new_empty((n, 1))
        self.flags = act.new_empty((n, 1))
        self.n_envs = n_envs

        self.print_stats(n_envs)
    
    def get_total_size(self):
        assert self.is_initialized
        total_bytes = 0
        for t in [self.obs, self.act, self.rew]:
            total_bytes += t.numel() * t.element_size()
        return total_bytes

    def print_stats(self, bsz=None):
        max_vec_env_steps = self.max_size // bsz if bsz is not None else None
        sz, sfx = to_readable_size(self.get_total_size())

        max_vec_env_steps_msg = ''
        if max_vec_env_steps is not None:
            max_vec_env_steps_msg = f'  ({format_readable_num(max_vec_env_steps)})'
        print(
            f'Replay buffer: Size = {sz:.2f}{sfx} {max_vec_env_steps_msg}'
        )
