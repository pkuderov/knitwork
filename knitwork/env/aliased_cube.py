"""Paper-inspired aliased cube surface, not an official ICML-2024 implementation.

Dedieu et al.: https://proceedings.mlr.press/v235/dedieu24a.html.
The local topology and seam action semantics are explicit implementation choices.
An NPZ transition/emission table can replace them without changing the experiment.
"""

import hashlib
from pathlib import Path

import gymnasium as gym
import numpy as np


def cube_surface_transitions(edge_size):
    if not isinstance(edge_size, int) or edge_size < 2:
        raise ValueError('edge_size must be an integer >= 2')
    # Each face has an outward normal and fixed local horizontal/vertical axes.
    frames = np.array(
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
            [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            [[0, 0, -1], [1, 0, 0], [0, -1, 0]],
        ], dtype=np.int64,
    )
    normals = {tuple(frame[0]): face for face, frame in enumerate(frames)}
    transitions = np.empty((6 * edge_size ** 2, 4), dtype=np.int64)
    for face, (normal, u, v) in enumerate(frames):
        for row in range(edge_size):
            for column in range(edge_size):
                state = (face * edge_size + row) * edge_size + column
                for action, (dr, dc) in enumerate(((0, 1), (0, -1), (1, 0), (-1, 0))):
                    next_row, next_column = row + dr, column + dc
                    next_face = face
                    if not (0 <= next_row < edge_size and 0 <= next_column < edge_size):
                        direction = dc * u + dr * v
                        next_face = normals[tuple(direction)]
                        _, next_u, next_v = frames[next_face]
                        # Fold the cell center around the crossed cube edge.
                        tangent = v * (2 * row - edge_size + 1) if dc else u * (2 * column - edge_size + 1)
                        point = direction * edge_size + normal * (edge_size - 1) + tangent
                        next_column = int((point @ next_u + edge_size - 1) // 2)
                        next_row = int((point @ next_v + edge_size - 1) // 2)
                    transitions[state, action] = (next_face * edge_size + next_row) * edge_size + next_column
    return transitions


class AliasedCube(gym.Env):
    """Infinite deterministic navigation with categorical, many-to-one observations.

    Actions are face-local east/west/north/south; crossing a seam changes the frame.
    Rewards are always zero. Episode lengths are chosen by the random-walk sampler.
    """

    metadata = {'render_modes': []}

    def __init__(self, edge_size=6, n_observations=12, map_seed=0, graph_path=None):
        super().__init__()
        if graph_path is None:
            transitions = cube_surface_transitions(edge_size)
            if not isinstance(n_observations, int) or not 1 < n_observations < len(transitions):
                raise ValueError('n_observations must be between 2 and n_states - 1')
            emissions = np.arange(len(transitions), dtype=np.int64) % n_observations
            np.random.default_rng(map_seed).shuffle(emissions)
            self.source = 'paper_inspired_cube_surface'
        else:
            with np.load(Path(graph_path).expanduser(), allow_pickle=False) as graph:
                transitions = graph['transitions'].copy()
                emissions = graph['observations'].copy()
            self.source = 'external_transition_table'
        if transitions.ndim != 2 or min(transitions.shape) < 1 or emissions.shape != (len(transitions),):
            raise ValueError('Expected transitions[n_states, n_actions] and observations[n_states]')
        if not np.issubdtype(transitions.dtype, np.integer) or not np.issubdtype(emissions.dtype, np.integer):
            raise ValueError('Transition and observation IDs must be integers')
        if transitions.min() < 0 or transitions.max() >= len(transitions) or emissions.min() < 0:
            raise ValueError('Invalid transition or observation ID')
        self.transitions = np.asarray(transitions, dtype=np.int64)
        self.observations = np.asarray(emissions, dtype=np.int64)
        self.n_states, self.n_actions = self.transitions.shape
        self.n_observations = int(self.observations.max()) + 1
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.observation_space = gym.spaces.Discrete(self.n_observations)
        self.state = None
        self.fingerprint = hashlib.sha256(self.transitions.tobytes() + self.observations.tobytes()).hexdigest()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and 'state' in options:
            self.state = int(options['state'])
            if not 0 <= self.state < self.n_states:
                raise ValueError('Invalid start state')
        else:
            self.state = int(self.np_random.integers(self.n_states))
        return int(self.observations[self.state]), {}

    def step(self, action):
        if self.state is None:
            raise RuntimeError('Call reset before step')
        if not self.action_space.contains(action):
            raise ValueError('Invalid action')
        self.state = int(self.transitions[self.state, action])
        return int(self.observations[self.state]), 0.0, False, False, {}

    def sample_walks(self, n_sequences, sequence_length, seed):
        if n_sequences < 1 or sequence_length < 2:
            raise ValueError('Need positive n_sequences and sequence_length >= 2')
        rng = np.random.default_rng(seed)
        actions = rng.integers(self.n_actions, size=(n_sequences, sequence_length - 1))
        states = np.empty((n_sequences, sequence_length), dtype=np.int64)
        states[:, 0] = rng.integers(self.n_states, size=n_sequences)
        for step in range(sequence_length - 1):
            states[:, step + 1] = self.transitions[states[:, step], actions[:, step]]
        # Latent states are deliberately absent from the learner-facing dataset.
        return {'observations': self.observations[states], 'actions': actions}
