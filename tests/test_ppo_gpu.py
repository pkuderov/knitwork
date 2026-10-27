import pytest
import torch

from knitwork.env.popgym_torch import CardMemoryEnv
from knitwork.models.utils import build_model
from knitwork.rl.ppo_gpu import GraphedPolicy, collect, compute_gae, index_state, replay

CFG = {
    'grnn_lru': dict(hidden_size=16, n_layers=2, n_columns=3, horizon=[1.0, 100.0], fb_norm=True),
    'grnn': dict(hidden_size=16, n_layers=2, n_columns=3, n_attn_heads=2, ln_msg=True, bank=2, mha=2, noise_std=0.0),
    'lru': dict(hidden_size=16, n_layers=2, horizon=[1.0, 100.0]),
    'grnn_lru:topk': dict(hidden_size=16, n_layers=2, n_columns=3, horizon=[1.0, 100.0], fb_norm=True, x_add=True, write_topk=1),
    'grnn_lru:dynamic': dict(hidden_size=16, n_layers=2, n_columns=3, horizon=[1.0, 100.0], fb_norm=True, x_add=True, routing='dynamic'),
    'grnn_lru:selective': dict(hidden_size=16, n_layers=2, n_columns=3, horizon=[1.0, 100.0], fb_norm=True, x_add=True, selective_rank=4),
    'grnn_lru:table': dict(hidden_size=16, n_layers=2, n_columns=3, horizon=[1.0, 100.0], fb_norm=True, x_add=True, routing='table'),
    'grnn_lru:rounds2': dict(hidden_size=16, n_layers=2, n_columns=3, horizon=[1.0, 100.0], fb_norm=True, x_add=True, routing='dynamic', comm_rounds=2),
    'grnn_lru:read': dict(hidden_size=16, n_layers=2, n_columns=3, horizon=[1.0, 100.0], fb_norm=True, x_add=True, routing='dynamic', read_attn=True),
    'grnn_lru:statekey': dict(hidden_size=16, n_layers=2, n_columns=3, horizon=[1.0, 100.0], fb_norm=True, x_add=True, write_topk=1, write_mode='state', write_free=1.0),
    'grnn_lru:gate': dict(hidden_size=16, n_layers=2, n_columns=3, horizon=[1.0, 100.0], fb_norm=True, x_add=True, input_gate_rank=4),
    'grnn_lru:gru': dict(hidden_size=16, n_layers=2, n_columns=3, horizon=[1.0, 100.0], fb_norm=True, x_add=True, gru_columns=1),
    'grnn_lru:route_topk': dict(hidden_size=16, n_layers=2, n_columns=3, horizon=[1.0, 100.0], fb_norm=True, x_add=True, routing='dynamic', route_topk=2),
    'grnn_lru:rounds3': dict(hidden_size=16, n_layers=2, n_columns=3, horizon=[1.0, 100.0], fb_norm=True, x_add=True, routing='dynamic', comm_rounds=3),
    'grnn_lru:inall': dict(hidden_size=16, n_layers=2, n_columns=3, horizon=[1.0, 100.0], fb_norm=True, x_add=True, routing='dynamic', input_all_layers=True),
    'grnn_lru:colhz': dict(hidden_size=16, n_layers=2, n_columns=3, horizon=[1.0, 100.0], fb_norm=True, x_add=True, column_horizons=True),
    'grnn_lru:alloc_sq': dict(hidden_size=16, n_layers=2, n_columns=3, horizon=[1.0, 100.0], fb_norm=True, x_add=True, routing='dynamic', write_alloc=True, route_query='state'),
    'grnn_lru:clock': dict(hidden_size=16, n_layers=2, n_columns=3, horizon=[1.0, 100.0], fb_norm=True, x_add=True, routing='dynamic', layer_strides=[1, 3]),
    'grnn_lru:mem': dict(hidden_size=16, n_layers=2, n_columns=3, horizon=[1.0, 100.0], fb_norm=True, x_add=True, routing='dynamic', mem_slots=5, mem_stride=2),
    'grnn_lru:hybrid_share': dict(hidden_size=16, n_layers=2, n_columns=3, horizon=[1.0, 100.0], fb_norm=True, x_add=True, routing='dynamic', backbone=True, share_columns=True),
    'grnn_lru:raw_lin': dict(hidden_size=16, n_layers=1, n_columns=3, horizon=[1.0, 100.0], fb_norm=True, x_add=True, routing='dynamic', raw_columns=1, cell_act='linear'),
    'grnn_many:full': dict(hidden_size=16, n_columns=6, horizon=[1.0, 100.0]),
    'grnn_many:active': dict(hidden_size=16, n_columns=6, horizon=[1.0, 100.0], active_k=2, active_balance=0.0),
    'grnn_many:workspace': dict(hidden_size=16, n_columns=6, horizon=[1.0, 100.0], comm='workspace', n_slots=3, slot_topk=2),
    'grnn_many:groups': dict(hidden_size=16, n_columns=6, horizon=[1.0, 100.0], comm='groups', group_size=3),
    'grnn_lru:ff': dict(hidden_size=16, n_layers=2, n_columns=3, horizon=[1.0, 100.0], fb_norm=True, x_add=True, routing='dynamic', feedback=False, route_query='current'),
    'grnn_bank:read': dict(hidden_size=16, n_layers=2, n_columns=3, horizon=[1.0, 100.0]),
    'grnn_bank:write': dict(hidden_size=16, n_layers=2, n_columns=3, horizon=[1.0, 100.0], write='attn'),
    'grnn_bank:write_fb_h2': dict(hidden_size=16, n_layers=2, n_columns=3, horizon=[1.0, 100.0], write='attn', write_heads=2, write_feedback=True),
}


def test_gae_resets_on_done():
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.zeros(3, 1)
    dones = torch.tensor([[False], [True], [False]])
    advs, _ = compute_gae(rewards, values, dones, torch.tensor([10.0]), gamma=0.5, lambda_=1.0)
    assert torch.allclose(advs[:, 0], torch.tensor([1 + 0.5 * 2, 2.0, 3 + 0.5 * 10]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graphs need a GPU')
@pytest.mark.parametrize('rnn', sorted(CFG))
def test_graphed_policy_matches_eager(rnn):
    torch.manual_seed(0)
    dev = torch.device('cuda')
    env = CardMemoryEnv('RepeatPreviousEasy', 8, device=dev)
    model = build_model('rl_vector', dict(input_size=env.obs_dim, output_size=env.n_actions, dtype=torch.float32, device=dev),
                        rnn.split(':')[0], CFG[rnn]).to(dev)
    obs = env.reset()
    state = model.rnn.reset_state(None, bsz=8)
    graphed = GraphedPolicy(model, state, obs, rollout_len=6, minibatch=4)
    batch, _, _ = collect(env, graphed, state, obs, 6)
    idx = torch.arange(4, device=dev)
    s0 = index_state(batch['state_init'], idx, 8)
    args = batch['obs'][:, idx], batch['dones'][:, idx], s0
    with torch.no_grad():
        eager_logits = replay(model, *args)[0]
    lp = torch.distributions.Categorical(logits=eager_logits).log_prob(batch['actions'][:, idx])
    assert torch.allclose(lp, batch['log_probs'][:, idx], atol=1e-5)
    grads = []
    for fn in (lambda: replay(model, *args), lambda: graphed.replay_minibatch(*args)):
        model.zero_grad()
        logits, values, feats, route_ent = fn()
        (logits.square().mean() + values.mean() + feats.square().mean() + route_ent.mean()).backward()
        grads.append(torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]))
    assert torch.allclose(grads[0], grads[1], atol=1e-5)


def test_topk_write_freezes_unselected_columns():
    torch.manual_seed(0)
    cfg = dict(CFG['grnn_lru:topk'], n_columns=4)
    model = build_model('rl_vector', dict(input_size=5, output_size=3, dtype=torch.float32, device='cpu'), 'grnn_lru', cfg)
    state = model.rnn.reset_state(None, bsz=7)
    _, new, _ = model.rnn(torch.randn(7, 1, 16), state)
    # h: [L, C, B, 2H]; exactly one column per (layer, env) changes
    changed = (new['h'] != state['h']).any(-1)
    assert (changed.sum(1) == 1).all()


def test_selective_decay_starts_as_plain_lru():
    torch.manual_seed(0)
    cfg = CFG['grnn_lru:selective']
    model = build_model('rl_vector', dict(input_size=5, output_size=3, dtype=torch.float32, device='cpu'), 'grnn_lru', cfg)
    plain = build_model('rl_vector', dict(input_size=5, output_size=3, dtype=torch.float32, device='cpu'), 'grnn_lru',
                        dict(cfg, selective_rank=0))
    plain.load_state_dict(model.state_dict(), strict=False)
    state = model.rnn.reset_state(None, bsz=7)
    x = torch.randn(7, 1, 16)
    assert torch.allclose(model.rnn(x, state)[1]['h'], plain.rnn(x, state)[1]['h'], atol=1e-5)


def test_write_balance_pushes_bias_toward_even_load():
    torch.manual_seed(0)
    cfg = dict(CFG['grnn_lru:topk'], n_columns=4, write_balance=0.1)
    model = build_model('rl_vector', dict(input_size=5, output_size=3, dtype=torch.float32, device='cpu'), 'grnn_lru', cfg)
    rnn = model.rnn
    # make column 0 always win: its selection bias must go down, the others' up
    with torch.no_grad():
        rnn.write_b[:, 0] += 100.0
    state = rnn.reset_state(None, bsz=7)
    rnn(torch.randn(7, 1, 16), state)
    assert (rnn.write_bias[:, 0] < 0).all() and (rnn.write_bias[:, 1:] > 0).all()


def test_input_gate_and_gru_columns_shapes():
    torch.manual_seed(0)
    for extra in (dict(input_gate_rank=4), dict(gru_columns=2)):
        cfg = dict(CFG['grnn_lru'], n_columns=4, x_add=True, **extra)
        model = build_model('rl_vector', dict(input_size=5, output_size=3, dtype=torch.float32, device='cpu'), 'grnn_lru', cfg)
        state = model.rnn.reset_state(None, bsz=7)
        _, new, _ = model.rnn(torch.randn(7, 1, 16), state)
        assert new['h'].shape == state['h'].shape and torch.isfinite(new['h']).all()
    # GRU columns keep the imaginary half at zero
    assert (new['h'][:, :2, :, 16:] == 0).all()


def test_ordinal_policy_starts_near_uniform():
    torch.manual_seed(0)
    model = build_model('rl_vector', dict(input_size=5, output_size=26, dtype=torch.float32, device='cpu', policy='ordinal',
                                          policy_gain=0.01), 'lru', CFG['lru'])
    state = model.rnn.reset_state(None, bsz=7)
    logits, _, _, _ = model(torch.randn(7, 5), state)
    assert logits.shape == (7, 26)
    p = logits.softmax(-1)
    # mid actions are most likely, far ends still have non-trivial mass
    assert p.argmax(-1).sub(12.5).abs().max() <= 1 and p.min() > 0.005


def test_column_horizons_are_disjoint_bands():
    torch.manual_seed(0)
    cfg = dict(CFG['grnn_lru:colhz'], n_columns=4, horizon=[1.0, 1000.0])
    model = build_model('rl_vector', dict(input_size=5, output_size=3, dtype=torch.float32, device='cpu'), 'grnn_lru', cfg)
    # horizon = exp(-log_r); column c spans [1000^(c/4), 1000^((c+1)/4))
    hz = torch.exp(-model.rnn.cells.log_r)
    for c in range(4):
        assert hz[:, c].min() >= 1000 ** (c / 4) - 1e-3 and hz[:, c].max() <= 1000 ** ((c + 1) / 4) + 1e-3


def test_aux_memory_loss_is_zero_for_perfect_recall_and_ignores_episode_starts():
    from knitwork.rl.ppo_gpu import aux_memory_loss
    T, b, D = 6, 2, 3
    obs = torch.randn(T, b, D)
    dones = torch.zeros(T, b, dtype=torch.bool)
    dones[2, 0] = True  # env 0: new episode from t = 3
    # pred[t] holds obs[t - 1] (garbage across the episode boundary)
    pred = torch.cat([torch.zeros(1, b, D), obs[:-1]])
    pred[3, 0] = 100.0
    loss = aux_memory_loss(pred, (1,), obs, dones)
    assert loss.item() < 1e-6


def test_reverse_column_horizons_give_readout_the_longest_band():
    torch.manual_seed(0)
    cfg = dict(CFG['grnn_lru:colhz'], n_columns=4, horizon=[1.0, 1000.0], column_horizons='reverse')
    model = build_model('rl_vector', dict(input_size=5, output_size=3, dtype=torch.float32, device='cpu'), 'grnn_lru', cfg)
    hz = torch.exp(-model.rnn.cells.log_r)
    assert hz[:, 0].min() >= 1000 ** 0.75 - 1e-3 and hz[:, 3].max() <= 1000 ** 0.25 + 1e-3


def test_autoencode_min_cards_mixes_episode_lengths():
    env = CardMemoryEnv('AutoencodeMedium', 64, device='cpu', min_cards=10)
    env.reset(desync=False)
    assert env.lens.min() >= 10 and env.lens.max() <= 104 and env.lens.unique().numel() > 1
    # env 0 runs its full episode: N watch steps, then N - 1 rewarded recalls, done after 2N - 1 steps
    N, total, n_rewarded = int(env.lens[0]), 0.0, 0
    for t in range(2 * N - 1):
        target = env.target()[0].item()
        actions = env.target().clamp(min=0)
        _, reward, done = env.step(actions)
        total += reward[0].item()
        n_rewarded += target >= 0
        assert bool(done[0]) == (t == 2 * N - 2)
    assert n_rewarded == N and abs(total - 1.0) < 1e-5


def test_autoencode_grow_cards_raises_upper_length_on_high_accuracy():
    env = CardMemoryEnv('AutoencodeMedium', 32, device='cpu', min_cards=10, grow_cards=8, grow_acc=0.8)
    env.reset()
    assert env.lens.max() <= 10
    env.grow(0.5)
    assert env.max_cards == 10
    for _ in range(20):
        env.grow(0.9)
    assert env.max_cards == 104
    env.load_state_dict(CardMemoryEnv('AutoencodeMedium', 32, device='cpu', min_cards=10, grow_cards=8).state_dict())
    assert env.max_cards == 10


def test_memory_slots_store_snapshots_in_a_ring_and_clockwork_holds_layers():
    torch.manual_seed(0)
    cfg = dict(CFG['grnn_lru:mem'], layer_strides=[1, 3])
    model = build_model('rl_vector', dict(input_size=5, output_size=3, dtype=torch.float32, device='cpu'), 'grnn_lru', cfg)
    core = model.rnn
    state = core.reset_state(None, bsz=7)
    outs1 = []
    for step in range(7):
        y, new, _ = core(torch.randn(7, 1, 16), state)
        # layer 1 updates only on steps 0, 3, 6
        assert torch.equal(new['outs'][1], state['outs'][1]) == (step % 3 != 0)
        if step % 2 == 0:
            # slot (step // 2) % 5 now holds this step's core output
            assert torch.equal(new['mem'][(step // 2) % 5], y)
        state = new
    assert (state['t'] == 7).all()
    state = core.reset_state(state, torch.tensor([True] + [False] * 6))
    assert state['t'][0] == 0 and state['mem'][:, 0].abs().sum() == 0


def test_many_active_columns_hold_state_and_balance_load():
    torch.manual_seed(0)
    cfg = dict(CFG['grnn_many:active'], active_balance=0.01)
    model = build_model('rl_vector', dict(input_size=5, output_size=3, dtype=torch.float32, device='cpu'), 'grnn_many', cfg)
    core = model.rnn
    state = core.reset_state(None, bsz=7)
    state = core(torch.randn(7, 1, 16), state)[1]
    _, new, _ = core(torch.randn(7, 1, 16), state)
    # h: (1, C, B, 2H); column 0 plus exactly active_k others change per env, the rest hold without decay
    changed = (new['h'][0] != state['h'][0]).any(-1)  # (C, B)
    assert changed[0].all() and (changed.sum(0) == 3).all()
    before = core.route_bias.clone()
    for _ in range(20):
        state = core(torch.randn(7, 1, 16), state)[1]
    assert not torch.equal(before, core.route_bias)
