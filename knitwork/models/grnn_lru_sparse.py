"""Sparse LRU ablations; prediction always reads column zero.

Defaults preserve grnn_lru_core, including its parameter names and initialization.
Clockwork skips cell projections; routing masks still use dense message products.
"""

import math
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F

from knitwork.models.grnn_lru_core import GridRnn as DenseGridRnn, LruBank


class _Entmax15(torch.autograd.Function):
    """Exact alpha=1.5 simplex mapping with its analytic Jacobian.

    Threshold algorithm: Peters et al., https://arxiv.org/abs/1905.05702.
    The custom backward avoids differentiating inactive square-root thresholds.
    """

    @staticmethod
    def forward(ctx, logits):
        x = (logits - logits.amax(dim=-1, keepdim=True)) / 2
        ordered = x.sort(dim=-1, descending=True).values
        rank = torch.arange(1, x.shape[-1] + 1, device=x.device, dtype=x.dtype)
        mean = ordered.cumsum(-1) / rank
        variance_sum = ordered.square().cumsum(-1) - rank * mean.square()
        threshold = mean - ((1 - variance_sum) / rank).clamp_min(0).sqrt()
        support = (threshold <= ordered).sum(-1, keepdim=True)
        tau = threshold.gather(-1, support - 1)
        probabilities = (x - tau).clamp_min(0).square()
        ctx.save_for_backward(probabilities)
        return probabilities

    @staticmethod
    def backward(ctx, grad_output):
        probabilities, = ctx.saved_tensors
        root = probabilities.sqrt()
        grad = grad_output * root
        return grad - root * grad.sum(-1, keepdim=True) / root.sum(-1, keepdim=True)


def entmax15(logits):
    return _Entmax15.apply(logits)


class SparseCommunication(nn.Module):
    def __init__(
            self, original, *, topology='dense', routing_init='diagonal',
            routing_activation='softmax', hub_query_size=0,
    ):
        super().__init__()
        if topology not in ('dense', 'star', 'star_ring'):
            raise ValueError(f'Unknown topology: {topology}')
        if routing_init not in ('diagonal', 'self_heavy', 'soft'):
            raise ValueError(f'Unknown routing initialization: {routing_init}')
        if routing_activation not in ('softmax', 'entmax15'):
            raise ValueError(f'Unknown routing activation: {routing_activation}')
        if routing_activation == 'entmax15' and topology != 'dense':
            raise ValueError('entmax15 ablation requires dense candidate connectivity')
        if not isinstance(hub_query_size, int) or hub_query_size < 0:
            raise ValueError('hub_query_size must be a nonnegative integer')
        self.dim = original.dim
        self.pi_route_logits = original.pi_route_logits
        self.routing_activation = routing_activation
        self.hub_query_size = hub_query_size
        self.n_columns, n_kv = self.pi_route_logits.shape
        allowed = torch.ones(self.n_columns, n_kv, dtype=torch.bool)
        if topology != 'dense':
            allowed[1:, :self.n_columns] = False
            for column in range(1, self.n_columns):
                allowed[column, 0] = allowed[column, column] = True
                if topology == 'star_ring':
                    neighbor = 1 + column % (self.n_columns - 1)
                    allowed[column, neighbor] = True
        self.register_buffer('allowed', allowed, persistent=False)
        self._initialize(routing_init, topology)
        if hub_query_size:
            self.query = nn.Linear(self.dim, hub_query_size, bias=False)
            self.key = nn.Linear(self.dim, hub_query_size, bias=False)

    @torch.no_grad()
    def _initialize(self, routing_init, topology):
        logits = self.pi_route_logits
        if routing_init == 'soft':
            indices = torch.arange(1, self.n_columns)
            logits[indices, indices] = 0.1
        elif routing_init == 'self_heavy':
            # Keep the hub's original initialization and keep inputs readable.
            n_external = logits.shape[1] - self.n_columns
            for column in range(1, self.n_columns):
                probabilities = torch.empty_like(logits[column])
                peer_mass = 0.15 if n_external else 0.30
                probabilities[:self.n_columns] = peer_mass / (self.n_columns - 1)
                probabilities[column] = 0.70
                if n_external:
                    probabilities[self.n_columns:] = 0.15 / n_external
                logits[column].copy_(probabilities.log())
        if topology != 'dense':
            # Preserve self/input mass; redirect removed peer mass to allowed peers.
            probabilities = logits.softmax(-1)
            for column in range(1, self.n_columns):
                peers = self.allowed[column, :self.n_columns].clone()
                peers[column] = False
                peer_mass = probabilities[column, :self.n_columns].sum() - probabilities[column, column]
                retained_mass = probabilities[column, :self.n_columns][peers].sum()
                probabilities[column, :self.n_columns][peers] *= peer_mass / retained_mass
            logits.copy_(probabilities.log())

    def forward(self, q, k, v, return_weights=False):
        logits = self.pi_route_logits.masked_fill(~self.allowed, float('-inf'))
        if self.routing_activation == 'entmax15':
            weights = torch.cat((logits[:1].softmax(-1), entmax15(logits[1:])), dim=0)
        else:
            weights = logits.softmax(-1)
        weights = weights.unsqueeze(0).expand(v.shape[0], -1, -1)
        if self.hub_query_size:
            # Read using the previous hub and the fresh input/lower-layer hub.
            if k.shape[1] > self.n_columns:
                fresh = k[:, self.n_columns:].mean(dim=1)
            else:
                fresh = k[:, 0]
            query = self.query(q[:, 0] + fresh)
            keys = self.key(k)
            scores = (keys * query.unsqueeze(1)).sum(-1) / math.sqrt(self.hub_query_size)
            hub = (logits[0].unsqueeze(0) + scores).softmax(-1)
            weights = torch.cat((hub.unsqueeze(1), weights[:, 1:]), dim=1)
        message = torch.bmm(weights, v)
        info = {'attn_weights': weights.detach().mean(0)} if return_weights else {}
        return torch.transpose_copy(message, 0, 1), info


class GroupedLruBank(LruBank):
    """Block diagonal input/output projections; complex state packing stays unchanged."""

    def __init__(self, original, groups):
        # Reuse the already initialized recurrence and replace only the projections.
        nn.Module.__init__(self)
        for attribute in ('n_layers', 'n_columns', 'hidden_size', 'horizon', 'hz_min', 'hz_max'):
            setattr(self, attribute, getattr(original, attribute))
        self.log_r, self.theta = original.log_r, original.theta
        self.groups = groups
        width = self.hidden_size // groups
        self.weight_in = nn.Parameter(torch.empty(self.n_layers, self.n_columns, groups, width, 2 * width))
        self.weight_out = nn.Parameter(torch.empty(self.n_layers, self.n_columns, groups, 2 * width, width))
        for weight in (self.weight_in, self.weight_out):
            nn.init.uniform_(weight, -1 / math.sqrt(width), 1 / math.sqrt(width))

    def forward(self, layer, x, h, columns=None):
        c, b, width = x.shape
        g, d = self.groups, width // self.groups
        log_r, theta = self.log_r[layer], self.theta[layer]
        weight_in, weight_out = self.weight_in[layer], self.weight_out[layer]
        if columns is not None:
            log_r, theta = log_r[columns], theta[columns]
            weight_in, weight_out = weight_in[columns], weight_out[columns]
        decay = log_r.exp()
        radius = (-decay).exp()
        lam_re, lam_im = radius * theta.cos(), radius * theta.sin()
        gamma = (-torch.expm1(-2 * decay)).sqrt()
        grouped_x = x.reshape(c, b, g, d).permute(0, 2, 1, 3).reshape(c * g, b, d)
        u = torch.bmm(grouped_x, weight_in.reshape(c * g, d, 2 * d))
        u_re, u_im = u.reshape(c, g, b, 2 * d).chunk(2, -1)
        u_re = u_re.permute(0, 2, 1, 3).reshape(c, b, width)
        u_im = u_im.permute(0, 2, 1, 3).reshape(c, b, width)
        h_re, h_im = h.chunk(2, -1)
        new_re = lam_re * h_re - lam_im * h_im + gamma * u_re
        new_im = lam_re * h_im + lam_im * h_re + gamma * u_im
        h_n = torch.cat((new_re, new_im), dim=-1)
        grouped_h = torch.cat(
            (new_re.reshape(c, b, g, d), new_im.reshape(c, b, g, d)), dim=-1,
        ).permute(0, 2, 1, 3).reshape(c * g, b, 2 * d)
        y = torch.bmm(grouped_h, weight_out.reshape(c * g, 2 * d, d))
        y = y.reshape(c, g, b, d).permute(0, 2, 1, 3).reshape(c, b, width)
        return torch.transpose_copy(F.silu(y) + x, 0, 1), h_n


def shuffle_channels(x, groups):
    return x.reshape(*x.shape[:-1], groups, x.shape[-1] // groups).transpose(-1, -2).flatten(-2)


class GridRnn(DenseGridRnn):
    def __init__(
            self, *, topology='dense', routing_init='diagonal',
            routing_activation='softmax', hub_query_size=0, projection_groups=1,
            shuffle_between_layers=False, update_periods=None, update_offsets=None, **kwargs,
    ):
        super().__init__(**kwargs)
        if not isinstance(projection_groups, int) or projection_groups < 1 or self.hidden_size % projection_groups:
            raise ValueError('projection_groups must divide hidden_size')
        self.projection_groups = projection_groups
        self.shuffle_between_layers = shuffle_between_layers
        if projection_groups > 1:
            self.cells = GroupedLruBank(self.cells, projection_groups)
        if (topology, routing_init, routing_activation, hub_query_size) != ('dense', 'diagonal', 'softmax', 0):
            self.comm = self.attn = nn.ModuleList([
                SparseCommunication(
                    comm, topology=topology, routing_init=routing_init,
                    routing_activation=routing_activation, hub_query_size=hub_query_size,
                )
                for comm in self.comm
            ])
        periods = [1] * self.n_columns if update_periods is None else list(update_periods)
        offsets = [0] * self.n_columns if update_offsets is None else list(update_offsets)
        if len(periods) != self.n_columns or len(offsets) != self.n_columns:
            raise ValueError('update periods/offsets must have one entry per column')
        if any(not isinstance(p, int) or p < 1 for p in periods):
            raise ValueError('update periods must be positive integers')
        if any(not isinstance(o, int) or not 0 <= o < p for p, o in zip(periods, offsets)):
            raise ValueError('update offsets must be integers in [0, period)')
        if periods[0] != 1:
            raise ValueError('The output hub must update every step')
        self.has_clockwork = any(p > 1 for p in periods)
        self.clock_cycle = math.lcm(*periods)
        if self.clock_cycle > 64:
            raise ValueError('Clockwork cycle exceeds 64 compiled phases')
        for phase in range(self.clock_cycle):
            active = [c for c, (p, o) in enumerate(zip(periods, offsets)) if phase % p == o]
            self.register_buffer(f'active_{phase}', torch.tensor(active), persistent=False)

    def forward(self, x, state, *, capture=False, **kwargs):
        if not self.has_clockwork and not self.shuffle_between_layers:
            return super().forward(x, state, capture=capture, **kwargs)
        assert x.shape[1] == self.n_inputs
        phase = state.get('clock_phase', -1)
        columns = None
        if self.has_clockwork and phase >= 0:
            # Branches specialize a finite clock; formatting a symbolic int breaks Dynamo.
            for candidate in range(self.clock_cycle):
                if phase == candidate:
                    columns = getattr(self, f'active_{candidate}')
                    break
        out = torch.cat((state['out'], x), dim=1)
        hidden, outputs, info = [], [], defaultdict(list)
        for layer in range(self.n_layers):
            message, comm_info = self.comm[layer](state['outs'][layer], out, out, return_weights=capture)
            previous_h = state['h'][layer]
            if columns is None:
                cell_out, next_h = self.cells(layer, message, previous_h)
            else:
                selected_h = previous_h.index_select(0, columns)
                selected_message = message.index_select(0, columns)
                if self.projection_groups > 1:
                    selected_out, selected_h = self.cells(layer, selected_message, selected_h, columns=columns)
                else:
                    selected_out, selected_h = self._selected_cells(layer, selected_message, selected_h, columns)
                next_h = previous_h.index_copy(0, columns, selected_h)
                cell_out = state['outs'][layer].index_copy(1, columns, selected_out)
            hidden.append(next_h)
            outputs.append(cell_out)
            for key, value in comm_info.items():
                info[key].append(value)
            out = cell_out
            if self.shuffle_between_layers and layer + 1 < self.n_layers:
                out = shuffle_channels(out, self.projection_groups)
        hidden, outputs = torch.stack(hidden), torch.stack(outputs)
        feedback = F.rms_norm(outputs[-1], (self.hidden_size,)) if self.fb_norm else outputs[-1]
        new_state = {'h': hidden, 'outs': outputs, 'out': feedback}
        if self.has_clockwork:
            new_state['clock_phase'] = (phase + 1) % self.clock_cycle
        return outputs[-1][:, 0], new_state, info

    def _selected_cells(self, layer, x, h, columns):
        # Index parameters before bmm so inactive columns perform no cell projection.
        theta = self.cells.theta[layer][columns]
        decay = self.cells.log_r[layer][columns].exp()
        radius = (-decay).exp()
        lam_re, lam_im = radius * theta.cos(), radius * theta.sin()
        gamma = (-torch.expm1(-2 * decay)).sqrt()
        h_re, h_im = h.chunk(2, -1)
        u_re, u_im = torch.bmm(x, self.cells.weight_in[layer][columns]).chunk(2, -1)
        new_re = lam_re * h_re - lam_im * h_im + gamma * u_re
        new_im = lam_re * h_im + lam_im * h_re + gamma * u_im
        next_h = torch.cat((new_re, new_im), dim=-1)
        y = torch.bmm(next_h, self.cells.weight_out[layer][columns])
        return torch.transpose_copy(F.silu(y) + x, 0, 1), next_h

    def init_state(self, bsz):
        state = super().init_state(bsz)
        if self.has_clockwork:
            # Initialize every column on the first step; subsequent clocks are global.
            state['clock_phase'] = -1
        return state

    def reset_state(self, state=None, reset_mask=None, *, bsz=None):
        new_state = super().reset_state(state, reset_mask, bsz=bsz)
        if state is not None and self.has_clockwork:
            new_state['clock_phase'] = state['clock_phase']
        return new_state

    def detach_state(self, state):
        if state is None:
            return None
        return {key: value.detach() if torch.is_tensor(value) else value for key, value in state.items()}

    def carried_state_floats(self):
        # Count information needed for the next step, excluding redundant output views.
        h_size = 2 * self.n_layers * self.n_columns * self.hidden_size
        if self.has_clockwork:
            return h_size + self.n_layers * self.n_columns * self.hidden_size
        hub_layers = self.n_layers if self.fb_norm else self.n_layers - 1
        hub_cache = hub_layers * self.hidden_size if self.hub_uses_queries else 0
        return h_size + self.n_columns * self.hidden_size + hub_cache

    @property
    def hub_uses_queries(self):
        return any(getattr(comm, 'hub_query_size', 0) > 0 for comm in self.comm)
