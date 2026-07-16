from typing import Sequence

from torch import nn
import torch

from knitwork.common.torch import chain, fw, make_layers, symlog


class FiLM(nn.Module):
    """
    FiLM modulator, which scales and shifts input
    based on the modulating conditioner.
    """
    def __init__(self, modulator_dim: int, base_dim: int, rng: torch.Generator = None):
        super().__init__()
        _, self.layer = make_layers(
            name='FiLM', input_size=modulator_dim, layers=[2*base_dim],
            out_logits=True, is_output=True, rng=rng, print_module=False
        )
        torch.nn.init.ones_(self.layer.bias[base_dim:])

    def forward(self, x, modulator):
        m = self.layer(modulator)
        scale, shift = torch.chunk(m, 2, dim=-1)
        return x * scale + shift

    def __repr__(self):
        return f'FiLM({self.layer})'


class MlpEncoder(nn.Module):
    def __init__(
            self, *,
            obs_size: int, body: Sequence[int] = (),
            use_symlog=False, fn_act=nn.SiLU, 
            rng: torch.Generator = None,
    ):
        super().__init__()
        self.use_symlog = use_symlog

        self.enc_size, self.encoder = make_layers(
            name='Encoder', input_size=obs_size, layers=body, rng=rng,
            activation=fn_act
        )

    def forward(self, obs):
        x = obs
        x = symlog(x) if self.use_symlog else x
        e = fw(self.encoder, x)
        return e


class AdaptiveTemperature(nn.Module):
    def __init__(
            self, *,
            stateless: bool = True,
            input_size: int = None,
            body: Sequence[int] = (),
            init_value: float = None,

            fn_act=nn.SiLU,
            rng: torch.Generator = None,
    ):
        super().__init__()
        self.stateless = stateless
        if self.stateless:
            assert init_value is not None
            self.log_temp = nn.Parameter(torch.tensor([init_value]).log())
            print(f'AdaTemp: stateless, single parameter')
            return

        _, self.log_temp = make_layers(
            name='AdaTemp', input_size=input_size, layers=chain(body, 1),
            out_logits=True, is_output=True, activation=fn_act, rng=rng
        )

    def forward(self, x=None):
        t = fw(self.log_temp, x) if not self.stateless else self.log_temp
        return torch.clamp(t, -30.0, 3.0)
