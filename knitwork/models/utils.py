import importlib


# Model registry
MODELS_ROOT = 'knitwork.models'

REGISTRY = {
    'rnn': 'gru.GruCore',
    'grnn': 'grnn_core.GridRnn',
    'delta_net': 'baseline.delta_net.DeltaNetCore',
    'hgrn2': 'baseline.hgrn2.HGRN2Core',
    'mlstm': 'baseline.mlstm.mLSTMCore',
    'transformer': 'baseline.transformer.TransformerCore',
    # 'grnn_err':       ('knitwork.models.grnn_err',       'GridRnn'),
    # 'grnn2':          ('knitwork.models.grnn2',          'GridRnn2'),
    # 'grnn_lru':       ('knitwork.models.grnn_lru',       'GridLRU'),
    # 'grnn_lru_wide':  ('knitwork.models.grnn_lru',       'GridLRU'),
    # 'hgrnn':          ('knitwork.models.hgrnn',          'HopfieldGridRnn'),
    # 'hgrnn_lru':      ('knitwork.models.hgrnn_lru',      'HopfieldGridLRU'),
    # 'hgrn_grnn':      ('knitwork.models.hgrn_grnn',      'HGRN_GridRnn'),
    # 'grnn_fw':        ('knitwork.models.grnn_fw',        'GridRnnFW'),
    # 'grnn_reservoir': ('knitwork.models.grnn_reservoir', 'GridRnnReservoir'),
    # 'grnn_loss':      ('knitwork.models.grnn_loss',      'GridRnnLoss'),
    # 'grnn_engram':    ('knitwork.models.engram_grnn',    'EngramGridRnn'),
    # 'grnn_prec_delta': ('knitwork.models.grnn_prec_delta', 'GridRnnPrecDelta'),
    # 'grnn_ema_mem':   ('knitwork.models.grnn_ema_mem',   'GridRnnEmaMem'),
    # 'grnn_fix':       ('knitwork.models.grnn_fix',       'GridRnnFix'),
    # 'grnn_fix_v3':    ('knitwork.models.grnn_fix_v3',    'GridRnnFixV3'),
    # 'grnn_fix_v4':    ('knitwork.models.grnn_fix_v4',    'GridRnnFixV4'),
    # 'grnn_fix_v4_L2C8':    ('knitwork.models.grnn_fix_v4',    'GridRnnFixV4'),
    # 'hgrnn_fix_v4':   ('knitwork.models.hgrnn_fix_v4',   'HopfieldGridRnnFixV4'),
    # 'grnn_fix_v5':    ('knitwork.models.grnn_fix_v5',    'GridRnnFixV5'),
    # 'hgrnn_fix':      ('knitwork.models.hgrnn_fix',      'HopfieldGridRnnFix'),
    # 'grnn_feedback':  ('knitwork.models.grnn_feedback',  'GridRnnFeedback'),
    # 'grnn_fusion':    None,  # factory
    # # config aliases
    # 'grnn_res':       ('knitwork.models.grnn_reservoir', 'GridRnnReservoir'),
    # 'grnn_delta':     ('knitwork.models.grnn_delta',    'GridDelta'),
    # 'grnn_delta_wide':('knitwork.models.grnn_delta',    'GridDelta'),
    # 'grnn_harmonic':  ('knitwork.models.grnn_harmonic', 'HarmonicGridRNN'),
    # 'grnn_base':      ('knitwork.models.grnn_base',           'GridRnnBase'),
    # # external baselines
    # 'mamba':          ('knitwork.models.baseline.mamba',     'Mamba'),
}

WRAPPER_REGISTRY = {
    'token': 'wrappers.TokenModel',
}


def build_model(wrapper_type: str, wrapper_cfg: dict, rnn_type: str, rnn_cfg: dict):
    model_fn = resolve_model(wrapper_type, WRAPPER_REGISTRY)
    rnn_fn = resolve_model(rnn_type, REGISTRY)

    return model_fn(
        **wrapper_cfg,
        rnn=rnn_cfg, rnn_fn=rnn_fn,
    )


def resolve_model(rnn_type: str, registry: dict):
    cls_path = registry[rnn_type]
    import_path, cls_name = cls_path.rsplit('.', 1)
    import_path = f'{MODELS_ROOT}.{import_path}'
    return getattr(importlib.import_module(import_path), cls_name)
