import pytest
import torch

from knitwork.env.popgym_torch import CardMemoryEnv
from knitwork.models.utils import build_model
from knitwork.rl.ppo_gpu import GraphedPolicy, collect, compute_gae, index_state, replay

CFG = {
    'grnn_lru': dict(hidden_size=16, n_layers=2, n_columns=3, horizon=[1.0, 100.0], fb_norm=True),
    'grnn': dict(hidden_size=16, n_layers=2, n_columns=3, n_attn_heads=2, ln_msg=True, bank=2, mha=2, noise_std=0.0),
    'lru': dict(hidden_size=16, n_layers=2, horizon=[1.0, 100.0]),
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
                        rnn, CFG[rnn]).to(dev)
    obs = env.reset()
    state = model.rnn.reset_state(None, bsz=8)
    graphed = GraphedPolicy(model, state, obs, rollout_len=6, minibatch=4)
    batch, _, _ = collect(env, graphed, state, obs, 6)
    idx = torch.arange(4, device=dev)
    s0 = index_state(batch['state_init'], idx, 8)
    args = batch['obs'][:, idx], batch['dones'][:, idx], s0
    with torch.no_grad():
        eager_logits, _ = replay(model, *args)
    lp = torch.distributions.Categorical(logits=eager_logits).log_prob(batch['actions'][:, idx])
    assert torch.allclose(lp, batch['log_probs'][:, idx], atol=1e-5)
    grads = []
    for fn in (lambda: replay(model, *args), lambda: graphed.replay_minibatch(*args)):
        model.zero_grad()
        logits, values = fn()
        (logits.square().mean() + values.mean()).backward()
        grads.append(torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]))
    assert torch.allclose(grads[0], grads[1], atol=1e-5)
