import torch
from torch import nn

from knitwork.common.utils import convert_hidden_size, count_learnable_params, to_torch


class GruBaseline(nn.Module):
    cell_type: str = 'gru'

    def __init__(
            self, *,
            input_size, embedding_size, output_size,
            hidden_size = None, base_hidden_size = None,
            n_layers, use_bias = True, dropout = 0.0
    ):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.embedding = nn.Embedding(input_size, self.embedding_size)

        self.n_layers = n_layers
        self.base_hidden_size = base_hidden_size

        if hidden_size is not None:
            self.hidden_size = hidden_size
        else:
            self.hidden_size = convert_hidden_size(
                base_hid_dim=self.base_hidden_size, 
                in_dim=self.embedding_size, out_dim=self.output_size,
                n_layers=self.n_layers
            )
        print(
            f'RNN ({self.cell_type.upper()})'
            f' {self.n_layers}-layers w/ {self.hidden_size} hidden units'
        )

        if self.n_layers == 1:
            dropout = 0
        self.rnn = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            batch_first=False,
            bias=use_bias,
            dropout=dropout,
        )

        self.head = nn.Linear(self.hidden_size, self.output_size)

        print(f'Param count: {count_learnable_params(self, as_str=True)}')

    def forward(self, tokens: torch.Tensor, h0=None):
        tokens = to_torch(tokens)
        # bsz, n_features
        assert tokens.ndim == 2

        # since it isn't built on GRUCell, but on GRU, add sequence dim
        tokens = tokens.unsqueeze(0)
        seq_sz, bsz, n_features = tokens.shape
        assert n_features == 1
        x = self.embedding(tokens.view(-1))
        x = x.view(seq_sz, bsz, -1)

        y, hN = self.rnn(x, h0)
        logits = self.head(y)

        # remove sequence dim
        logits = logits.squeeze(0)
        return logits, hN

    def reset_state(self, state, reset_mask):
        if not torch.any(reset_mask) or state is None:
            return state
        keep_mask = torch.logical_not(reset_mask)
        # to float of the current model precision
        keep_mask = keep_mask.to(self.head.weight.dtype)
 
        def _reset(h):
            # layer, batch, features
            return h * keep_mask[None, :, None]

        return _reset(state)

    def detach_state(self, state):
        if state is None:
            return state
        return state.detach()


class GruCore(nn.Module):
    cell_type: str = 'gru'
    has_attn = False

    def __init__(
            self, *,
            hidden_size, n_layers, 
            use_bias = True, dropout = 0.0,
            dtype, device,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.dtype = dtype
        self.device = device

        print(
            f'RNN ({self.cell_type.upper()})'
            f' {self.n_layers}-layers w/ {self.hidden_size} hidden units'
        )

        if self.n_layers == 1:
            dropout = 0.0
        self.rnn = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            batch_first=False,
            bias=use_bias,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, state: dict, **_):
        # x shape: (n_inputs, batch, hidden_size)
        # keep the same shape, using the first dim as sequence dim
        # but make sure it's always 1
        assert x.shape[0] == 1
        y, h_new = self.rnn(x, state['h'])

        # remove sequence dim
        y = y.squeeze(0)
        state = {'h': h_new}
        info = {}

        return y, state, info

    def reset_state(self, state=None, reset_mask=None, *, bsz=None):
        if state is None:
            bsz = reset_mask.shape[0] if reset_mask is not None else bsz
            return self.init_state(bsz)

        keep = (~reset_mask.flatten())[None, :, None]
        h = state['h'] * keep
        return {'h': h}

    def detach_state(self, state):
        if state is None:
            return state
        return {'h': state['h'].detach()}

    def init_state(self, bsz):
        h = torch.zeros(
            self.n_layers, bsz, self.hidden_size,
            device=self.device, dtype=self.dtype
        )
        return {'h': h}
