"""Observation/action adapter for existing recurrent cores; no privileged inputs."""

import torch
from torch import nn

from knitwork.models.utils import REGISTRY, resolve_model


class ActionObservationModel(nn.Module):
    def __init__(self, *, n_observations, n_actions, rnn_type, rnn_cfg, dtype, device):
        super().__init__()
        self.rnn = resolve_model(rnn_type, REGISTRY)(**rnn_cfg, dtype=dtype, device=device)
        hidden = self.rnn.hidden_size
        self.observation_embedding = nn.Embedding(n_observations, hidden)
        self.action_embedding = nn.Embedding(n_actions, hidden)
        self.head = nn.Linear(hidden, n_observations)

    def forward(self, observation, action, state):
        x = self.observation_embedding(observation) + self.action_embedding(action)
        x = x.unsqueeze(1) if getattr(self.rnn, 'batch_first', False) else x.unsqueeze(0)
        output, state, _ = self.rnn(x, state)
        return self.head(output), state


def reactive_probabilities(train_walks, n_observations, n_actions):
    """Training-only smoothed P(next observation | current observation, action)."""
    observations, actions = train_walks['observations'], train_walks['actions']
    counts = torch.ones(n_observations, n_actions, n_observations, dtype=torch.float64)
    flat_indices = (
        (torch.as_tensor(observations[:, :-1]) * n_actions + torch.as_tensor(actions))
        * n_observations + torch.as_tensor(observations[:, 1:])
    )
    counts.view(-1).scatter_add_(0, flat_indices.flatten(), torch.ones(flat_indices.numel(), dtype=counts.dtype))
    return counts / counts.sum(-1, keepdim=True)
