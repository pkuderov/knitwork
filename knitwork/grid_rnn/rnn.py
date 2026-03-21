from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from aim import Run
import numpy as np
import torch
from torch import nn

from knitwork.common.curriculum import CurriculumScheduler
from knitwork.common.dynamic_param import DynamicParameter
from knitwork.common.metrics_collector import MetricsCollector
from knitwork.common.scheduler import Scheduler
from knitwork.common.utils import FpsCounter, convert_hidden_size, flatten_dict, format_readable_num, to_numpy, to_torch
from knitwork.grid_rnn.env_text import TextGenerator, load_dataset, tokenize


@dataclass
class RnnBaselineConfig:
    vocab_size: int
    num_classes: int

    hidden_size: int | None = None
    # Reference hidden size for a 1-layer model (used for auto conversion)
    base_hidden: int | None = 128

    n_layers: int = 1

    embed_dim: int = 64
    dropout: float = 0.0

    use_bias: bool = True


class RnnBaseline(nn.Module):
    def __init__(self, cfg: RnnBaselineConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)

        self.cell = 'gru'
        if cfg.hidden_size is not None:
            self.hidden = cfg.hidden_size
        else:
            assert cfg.base_hidden is not None
            self.hidden = convert_hidden_size(
                base_hid_dim=cfg.base_hidden, in_dim=cfg.embed_dim, out_dim=cfg.num_classes,
                n_layers=cfg.n_layers
            )
        print(f'GRU {self.cfg.n_layers}-layers w/ {self.hidden} hidden units')

        self.rnn: nn.GRU = nn.GRU(
            input_size=cfg.embed_dim,
            hidden_size=self.hidden,
            num_layers=cfg.n_layers,
            batch_first=False,
            dropout=cfg.dropout if cfg.n_layers > 1 else 0.0,
        ) # type: ignore

        self.head = nn.Linear(self.hidden, cfg.num_classes)

        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Param count: {format_readable_num(param_count)}')

    def forward(self, tokens: torch.Tensor, h0=None):
        tokens = to_torch(tokens)
        assert tokens.ndim == 3

        seq_sz, bsz, n_features = tokens.shape
        assert n_features == 1
        x = self.embed(tokens.view(-1))
        x = x.view(seq_sz, bsz, -1)

        y, hN = self.rnn(x, h0)
        logits = self.head(y)
        return logits, hN

    def reset_state(self, state, reset_mask):
        if not torch.any(reset_mask) or state is None:
            return state
        keep_mask = torch.logical_not(reset_mask).float()
 
        def _reset(h):
            # seq, batch, features
            return h * keep_mask[None, :, None]

        return _reset(state) if self.cell == "gru" else (_reset(state[0]), _reset(state[1]))

    def detach_state(self, state):
        if state is None:
            return state
        return state.detach() if self.cell == "gru" else (state[0].detach(), state[1].detach())


def run_sdq():
    setting = 'easy'
    setting = 'hard'

    if setting == 'easy':
        from knitwork.grid_rnn.sdq_gen import SdqConfig, StoreDistractQueryGenerator
        gen_cfg = SdqConfig(
            K=5, V=10, T_range=(10.0, 1_000.0)
        )
        gen = StoreDistractQueryGenerator(
            gen_cfg, n_envs=64, seed=42,
            p_store=0.35, p_query=0.35
        )
    else:
        from knitwork.grid_rnn.sdq_gen_hard import SdqConfig, StoreDistractQueryGenerator
        gen_cfg = SdqConfig(
            K=5, V=10, T_range=(10.0, 1_000.0)
        )
        gen = StoreDistractQueryGenerator(
            gen_cfg, n_envs=64, seed=42,
            p_store=0.35, p_query=0.35,
            odd='store'
        )

    rnn_cfg = RnnBaselineConfig(
        vocab_size=gen_cfg.n_tokens,
        num_classes=gen_cfg.V,
        base_hidden=128,
        n_layers=1,
        embed_dim=64,
    )
    rnn = RnnBaseline(rnn_cfg)
    lr=1e-3
    print(f"Base LR: {lr}")
    wm_lr = DynamicParameter(val=1e-5*lr, tar=lr, n_linear_steps=50, scheduler=Scheduler(100))
    dc_lr = DynamicParameter(val=lr, rel=0.25, lr=0.01, scheduler=Scheduler(5_000))
    def get_lr():
        return wm_lr.val if not wm_lr.scheduler.is_infinite else dc_lr.val
    def step_lr():
        return wm_lr.step() if not wm_lr.scheduler.is_infinite else dc_lr.step()

    optim = torch.optim.RMSprop(rnn.parameters(), lr=get_lr())
    loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=gen_cfg.ignore_index)

    rollout_len = 8
    batch_size = gen.n_envs * rollout_len

    n_steps = 1_000_000_000
    step = 0
    log_stats_schedule = Scheduler(1_000_000)
    print_stats_schedule = Scheduler(5_000_000)
    curriculum_step_schedule = CurriculumScheduler(scheduler=Scheduler(200_000), key='Loss')

    run_config = {
        "log": {
            "wandb": True,
            # "wandb": False,
            "project": "grid-rnn-sdq",
            "tags": [],
        },
        "name": "RNN",
        "gen": asdict(gen_cfg),
        "rnn": asdict(rnn_cfg),
        "rollout_len": rollout_len,
        "batch_size": batch_size,
        "n_steps": n_steps,
        "n_envs": gen.n_envs,
        "p_store_init": gen.p_store,
        "p_query_init": gen.p_query,
        "step": step,
        "lr": lr,
        "print_stats_schedule": print_stats_schedule.schedule,
        "curriculum_step_schedule": curriculum_step_schedule.scheduler.schedule,
        "setting": setting
    }
    logger = start_logger(run_config)

    stats = MetricsCollector()
    fps_counter = FpsCounter()

    rnn_state = None
    batch_y = []
    batch_y_gt = []
    batch_sq_gaps = []

    while step < n_steps:
        obs = gen.next()
        obs = {k: to_torch(v) for k, v in obs.items()}

        rnn_state = rnn.reset_state(rnn_state, obs['reset_mask'])
        x = obs['tokens'].view(1, -1, 1)
        y, rnn_state = rnn(x, rnn_state)
        # eliminate "sequence" dim
        y = y.squeeze(0)

        batch_y.append(y)
        batch_y_gt.append(obs['targets'])
        batch_sq_gaps.append(obs['sq_gaps'])

        step += gen.n_envs

        if step % batch_size == 0:
            y = torch.cat(batch_y, dim=0)
            y_gt = torch.cat(batch_y_gt, dim=0)
            sq_gaps = torch.cat(batch_sq_gaps, dim=0).float()
            m_active = y_gt != gen_cfg.ignore_index

            loss = loss_fn(y, y_gt)
            with torch.no_grad():
                acc = (y[m_active].argmax(dim=-1) == y_gt[m_active]).float()

            mask_misses = sq_gaps < 0.0
            acc_miss = acc[mask_misses].mean()
            acc_non_miss = acc[~mask_misses].mean()
            acc_up_half = acc[sq_gaps > sq_gaps[~mask_misses].mean()].mean()
            acc = acc.mean()

            optim.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)
            if torch.isfinite(grad_norm):
                optim.step()
            else:
                print('Nan loss')

            if step_lr():
                optim.param_groups[0]['lr'] = get_lr()

            stats.put({
                "Loss": to_numpy(loss, copy=False), 
                "Acc": to_numpy(acc, copy=False),
                "Acc-": to_numpy(acc_miss, copy=False),
                "Acc+": to_numpy(acc_non_miss, copy=False),
                "Acc++": to_numpy(acc_up_half, copy=False),
                "|Grad|": to_numpy(grad_norm, copy=False),
                "LR": get_lr(),
            })

            rnn_state = rnn.detach_state(rnn_state)
            batch_y.clear()
            batch_y_gt.clear()
            batch_sq_gaps.clear()

        if curriculum_step_schedule.tick(metrics=stats, n_steps=gen.n_envs):
            K = 10
            dT, dp_store, dp_query = 1.0, -0.0014, -0.0005
            dT, dp_store, dp_query = dT/K, dp_store/K, dp_query/K

            gen.set_metaparams(
                T=gen.T + dT,
                p_store=max(gen.p_store + dp_store, 0.10),
                p_query=max(gen.p_query + dp_query, 0.25)
            )

        if print_stats_schedule.tick(gen.n_envs):
            metrics = {"global_step": step} | stats.get()
            fps = fps_counter.fps(n_iters=step, start=True)
            print(
                f'[{format_readable_num(step)} / {format_readable_num(n_steps, frac=0)}]'
                f' {format_readable_num(fps, frac=0)} fps |'
                f' LR: {int(100*metrics["LR"]/lr)}% | '
                f' L: {metrics["Loss"]:.3f}, A: {metrics["Acc"]:.3f}'
                f' A-: {metrics["Acc-"]:.3f}, A+: {metrics["Acc+"]:.3f},'
                f' A++: {metrics["Acc++"]:.3f}'
            )
            # from pprint import pprint
            # pprint(gen.get_stats(), sort_dicts=False, indent=4)

        if log_stats_schedule.tick(gen.n_envs) and logger is not None:
            fps = fps_counter.fps(n_iters=step, start=True)
            metrics = {
                "global_step": step, "fps": fps, 
                "curr_step": curriculum_step_schedule.cnt_accepted,
                "curr_schedule": curriculum_step_schedule.scheduler.schedule,
            } | stats.get()
            gen_stats = gen.get_stats()
            metrics['gen'] = gen_stats
            logger.track(flatten_dict(metrics))
    
    fps = fps_counter.fps(n_iters=step)
    print(format_readable_num(fps))


def run_text():
    data_path = Path("~/data/text/text8.txt").expanduser()
    data, ds_charset = tokenize(load_dataset(data_path))
    n_chars = ds_charset.size

    gen = TextGenerator(data, n_envs=64, seed=42)

    rnn_cfg = RnnBaselineConfig(
        vocab_size=n_chars,
        num_classes=n_chars,
        base_hidden=512,
        n_layers=1,
        # n_layers=2,
        embed_dim=64,
    )
    rnn = RnnBaseline(rnn_cfg)
    lr=8e-4
    print(f"Base LR: {lr}")
    wm_lr = DynamicParameter(val=1e-5*lr, tar=lr, n_linear_steps=50, scheduler=Scheduler(100))
    dc_lr = DynamicParameter(val=lr, rel=0.02, lr=0.005, scheduler=Scheduler(2_000))
    def get_lr():
        return wm_lr.val if not wm_lr.scheduler.is_infinite else dc_lr.val
    def step_lr():
        return wm_lr.step() if not wm_lr.scheduler.is_infinite else dc_lr.step()

    optim = torch.optim.RMSprop(rnn.parameters(), lr=get_lr())
    loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=gen.cfg.ignore_index)

    rollout_len = 8
    batch_size = gen.n_envs * rollout_len

    n_steps = 500_000_000
    step = 0
    log_stats_schedule = Scheduler(1_000_000)
    print_stats_schedule = Scheduler(5_000_000)

    rng = np.random.default_rng(None)
    p_reset = DynamicParameter(
        val=min(0.1/rollout_len, 0.01), tar=1/10_000, lr=0.007, scheduler=Scheduler(1_000)
    )

    run_config = {
        "log": {
            "wandb": True,
            # "wandb": False,
            "project": "grid-rnn-text",
            "tags": [],
        },
        "name": "RNN",
        "rnn": asdict(rnn_cfg),
        "rollout_len": rollout_len,
        "batch_size": batch_size,
        "n_steps": n_steps,
        "n_envs": gen.n_envs,
        "step": step,
        "lr": lr,
    }
    logger = start_logger(run_config)

    stats = MetricsCollector()
    fps_counter = FpsCounter()

    rnn_state = None
    batch_y = []
    batch_y_gt = []
    ln_2 = np.log(2.0)

    while step < n_steps:
        obs = gen.next()
        obs = {k: to_torch(v) for k, v in obs.items()}

        rnd_reset = torch.from_numpy(rng.random(gen.n_envs) < p_reset.val)
        reset_mask = torch.logical_or(obs['reset_mask'], rnd_reset)
        rnn_state = rnn.reset_state(rnn_state, reset_mask)
        x = obs['tokens'].view(1, -1, 1)
        y, rnn_state = rnn(x, rnn_state)
        # eliminate "sequence" dim
        y = y.squeeze(0)


        batch_y.append(y)
        batch_y_gt.append(obs['targets'])

        step += gen.n_envs

        if step % batch_size == 0:
            y = torch.cat(batch_y, dim=0)
            y_gt = torch.cat(batch_y_gt, dim=0)
            m_active = y_gt != gen.cfg.ignore_index

            loss = loss_fn(y, y_gt)
            with torch.no_grad():
                acc = (y[m_active].argmax(dim=-1) == y_gt[m_active]).float()
                bpc = loss / ln_2
                perplexity = torch.exp(loss)
            acc = acc.mean()

            optim.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)
            if torch.isfinite(grad_norm):
                optim.step()
            else:
                print('Nan loss')

            stats.put({
                "Loss": to_numpy(loss, copy=False),
                "BPC": to_numpy(bpc, copy=False),
                "Perplexity": to_numpy(perplexity, copy=False),
                "Acc": to_numpy(acc, copy=False),
                "|Grad|": to_numpy(grad_norm, copy=False),
                "LR": get_lr(),
                "T": 1 / p_reset.val,
            })

            p_reset.step()
            if step_lr():
                optim.param_groups[0]['lr'] = get_lr()

            rnn_state = rnn.detach_state(rnn_state)
            batch_y.clear()
            batch_y_gt.clear()

        if print_stats_schedule.tick(gen.n_envs):
            metrics = {"global_step": step} | stats.get()
            fps = fps_counter.fps(n_iters=step, start=True)
            print(
                f'[{format_readable_num(step)} / {format_readable_num(n_steps, frac=0)}]'
                f' {format_readable_num(fps, frac=0)} fps |'
                f' LR: {int(100*metrics["LR"]/lr)}%  '
                f' T: {int(metrics["T"])} | '
                f' L: {metrics["Loss"]:.3f}, A: {metrics["Acc"]:.3f}'
            )
            # from pprint import pprint
            # pprint(gen.get_stats(), sort_dicts=False, indent=4)

        if log_stats_schedule.tick(gen.n_envs) and logger is not None:
            fps = fps_counter.fps(n_iters=step, start=True)
            metrics = {
                "global_step": step, "fps": fps, 
            } | stats.get()
            gen_stats = gen.get_stats()
            metrics['gen'] = gen_stats
            logger.track(flatten_dict(metrics))
    
    fps = fps_counter.fps(n_iters=step)
    print(format_readable_num(fps))


def main():
    # run_sdq()
    run_text()


def start_logger(config):
    cfg_log = config['log']
    if not cfg_log['wandb']:
        return None

    run = Run(
        experiment=cfg_log['project']
    )
    print(f'Logging to Aim: {run.hash} {cfg_log["project"]}')
    run['hparams'] = dict(
        project=cfg_log['project'],
    ) | config
    return run

if __name__ == "__main__":
    main()
