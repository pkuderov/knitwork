from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from gymnasium import spaces

# side-effect import: registers all popgym environment IDs
try:
    import popgym  # noqa: F401
except ImportError:
    pass


def _obs_type(obs_space) -> str:
    """'discrete' if Discrete, else 'continuous'."""
    return 'discrete' if isinstance(obs_space, spaces.Discrete) else 'continuous'


def _obs_dim(obs_space) -> int:
    """Flattened observation dimension for continuous mode."""
    if isinstance(obs_space, spaces.Discrete):
        return 1
    if isinstance(obs_space, spaces.MultiDiscrete):
        return int(obs_space.shape[0]) if obs_space.shape else len(obs_space.nvec)
    if isinstance(obs_space, spaces.Tuple):
        return len(obs_space.spaces)
    if isinstance(obs_space, spaces.Box):
        return int(np.prod(obs_space.shape))
    raise ValueError(f'Unsupported obs space: {obs_space}')


def _n_actions(act_space) -> int:
    if isinstance(act_space, spaces.Discrete):
        return int(act_space.n)
    raise ValueError(f'Only Discrete action spaces supported; got {act_space}')


def _flatten_obs(obs, obs_space) -> np.ndarray:
    """Convert raw vectorised obs to [B, obs_dim] float32."""
    if isinstance(obs_space, spaces.Discrete):
        # obs: [B] int → [B, 1] int64
        return obs.reshape(-1, 1).astype(np.int64)
    if isinstance(obs_space, spaces.MultiDiscrete):
        return obs.reshape(obs.shape[0], -1).astype(np.float32)
    if isinstance(obs_space, spaces.Tuple):
        # obs is a tuple of [B] arrays
        return np.stack(obs, axis=-1).astype(np.float32)
    if isinstance(obs_space, spaces.Box):
        return obs.reshape(obs.shape[0], -1).astype(np.float32)
    raise ValueError(f'Unsupported obs space: {obs_space}')


class MikasakWrapper:
    """Adapts a POPGym/gymnasium env to knitwork's RL interface.

    Compatible with both discrete-obs (token) and continuous-obs (float) environments.
    Exposes observe() / step() / get_stats() matching TreasureHuntEnv API.
    """

    def __init__(self, env_id: str, n_envs: int, seed: int):
        self.env_id  = env_id
        self.n_envs  = n_envs

        # create vectorised env
        def _make():
            return gym.make(env_id)

        self._env = SyncVectorEnv([_make for _ in range(n_envs)])

        single_obs_space = self._env.single_observation_space
        single_act_space = self._env.single_action_space

        self.obs_space  = single_obs_space
        self.act_space  = single_act_space
        self.obs_type   = _obs_type(single_obs_space)
        self.obs_dim    = _obs_dim(single_obs_space)
        self.n_tokens   = int(single_obs_space.n) if self.obs_type == 'discrete' else 0
        self.n_actions  = _n_actions(single_act_space)

        # reset env
        obs_arr, _ = self._env.reset(seed=seed)
        self._current_obs = _flatten_obs(obs_arr, single_obs_space)
        # which envs need hidden-state reset before the NEXT observe()
        self._pending_reset = np.zeros(n_envs, dtype=bool)

        # episode tracking
        self._ep_returns = np.zeros(n_envs, dtype=np.float64)
        self._ep_lengths = np.zeros(n_envs, dtype=np.int64)
        self._completed_returns: list[float] = []
        self._completed_lengths: list[int]   = []

    # ------------------------------------------------------------------

    def observe(self) -> dict:
        """Return current obs and reset mask for just-finished episodes."""
        return {
            'obs':        self._current_obs.copy(),       # [B, obs_dim] or [B, 1] int
            'reset_mask': self._pending_reset.copy(),     # [B] bool
        }

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply actions; return (rewards [B], dones [B])."""
        obs_arr, rewards, terminated, truncated, _ = self._env.step(actions)

        dones = terminated | truncated

        # update episode stats
        self._ep_returns += rewards
        self._ep_lengths += 1
        for i in np.where(dones)[0]:
            self._completed_returns.append(float(self._ep_returns[i]))
            self._completed_lengths.append(int(self._ep_lengths[i]))
            self._ep_returns[i] = 0.0
            self._ep_lengths[i] = 0

        self._current_obs   = _flatten_obs(obs_arr, self.obs_space)
        self._pending_reset = dones.copy()

        return rewards.astype(np.float32), dones.astype(np.float32)

    def get_stats(self) -> dict:
        if not self._completed_returns:
            return {}
        stats = {
            'ep_return': float(np.mean(self._completed_returns[-self.n_envs:])),
            'ep_length': float(np.mean(self._completed_lengths[-self.n_envs:])),
        }
        return stats

    def close(self):
        self._env.close()
