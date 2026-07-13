import importlib


# Model registry
REGISTRY: dict[str, tuple[str, str] | None] = {
    'rnn':            ('knitwork.models.gru',           'GruBaseline'),
    'grnn':           ('knitwork.models.grnn',           'GridRnn'),
    'grnn_err':       ('knitwork.models.grnn_err',       'GridRnn'),
    'grnn2':          ('knitwork.models.grnn2',          'GridRnn2'),
    'grnn_lru':       ('knitwork.models.grnn_lru',       'GridLRU'),
    'grnn_lru_wide':  ('knitwork.models.grnn_lru',       'GridLRU'),
    'hgrnn':          ('knitwork.models.hgrnn',          'HopfieldGridRnn'),
    'hgrnn_lru':      ('knitwork.models.hgrnn_lru',      'HopfieldGridLRU'),
    'hgrn_grnn':      ('knitwork.models.hgrn_grnn',      'HGRN_GridRnn'),
    'grnn_fw':        ('knitwork.models.grnn_fw',        'GridRnnFW'),
    'grnn_reservoir': ('knitwork.models.grnn_reservoir', 'GridRnnReservoir'),
    'grnn_loss':      ('knitwork.models.grnn_loss',      'GridRnnLoss'),
    'grnn_engram':    ('knitwork.models.engram_grnn',    'EngramGridRnn'),
    'grnn_prec_delta': ('knitwork.models.grnn_prec_delta', 'GridRnnPrecDelta'),
    'grnn_ema_mem':   ('knitwork.models.grnn_ema_mem',   'GridRnnEmaMem'),
    'grnn_fix':       ('knitwork.models.grnn_fix',       'GridRnnFix'),
    'grnn_fix_v3':    ('knitwork.models.grnn_fix_v3',    'GridRnnFixV3'),
    'grnn_fix_v4':    ('knitwork.models.grnn_fix_v4',    'GridRnnFixV4'),
    'hgrnn_fix_v4':   ('knitwork.models.hgrnn_fix_v4',   'HopfieldGridRnnFixV4'),
    'grnn_fix_v5':    ('knitwork.models.grnn_fix_v5',    'GridRnnFixV5'),
    'hgrnn_fix':      ('knitwork.models.hgrnn_fix',      'HopfieldGridRnnFix'),
    'grnn_fusion':    None,  # factory
    # config aliases
    'grnn_res':       ('knitwork.models.grnn_reservoir', 'GridRnnReservoir'),
    'grnn_delta':     ('knitwork.models.grnn_delta',    'GridDelta'),
    'grnn_delta_wide':('knitwork.models.grnn_delta',    'GridDelta'),
    'grnn_harmonic':  ('knitwork.models.grnn_harmonic', 'HarmonicGridRNN'),
    'grnn_base':      ('knitwork.models.grnn_base',           'GridRnnBase'),
    'transformer':    ('knitwork.models.baseline.transformer', 'Transformer'),
    # external baselines
    'delta_net':      ('knitwork.models.baseline.delta_net', 'DeltaNet'),
    'hgrn2':          ('knitwork.models.baseline.hgrn2',     'HGRN2'),
    'mlstm':          ('knitwork.models.baseline.mlstm',     'mLSTM'),
    'mamba':          ('knitwork.models.baseline.mamba',     'Mamba'),
}


def build_model(rnn_type: str, rnn_cfg: dict, n_chars: int):
    if rnn_type == 'grnn_fusion':
        from knitwork.models.grnn_fusion import build_fusion_from_config
        return build_fusion_from_config(rnn_cfg, n_chars, n_chars)
    entry = REGISTRY.get(rnn_type)
    if entry is None:
        raise ValueError(f'Unknown model type: {rnn_type!r}')
    mod_path, cls_name = entry
    cls = getattr(importlib.import_module(mod_path), cls_name)
    return cls(**rnn_cfg, input_size=n_chars, output_size=n_chars)


def model_forward(rnn, x, state, *, capture: bool):
    capture = capture and _supports_attn(rnn)
    result = rnn(x, state, return_attn=True) if capture else rnn(x, state)
    y, state = result[0], result[1]
    if len(result) == 2:
        return y, state, {}, None
    if len(result) == 3:
        third = result[2]
        if isinstance(third, dict):
            return y, state, third, None
        return y, state, {}, third
    if len(result) == 4:
        return y, state, result[2], result[3]
    return y, state, {}, None


def _supports_attn(rnn) -> bool:
    flag = getattr(rnn, '_supports_return_attn', None)
    if flag is None:
        import inspect
        flag = 'return_attn' in inspect.signature(rnn.forward).parameters
        rnn._supports_return_attn = flag
    return flag
