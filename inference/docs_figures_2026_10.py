"""Figures and RL run summary for the October 2026 documentation update (docs/experiments/figures_2026-10)."""

import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

OUT = 'docs/experiments/figures_2026-10'
RL_JSON = 'docs/experiments/rl_2026-10_runs.json'

# categorical slots in fixed order (identity follows the model family, never the rank)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')
GRAY, INK, INK2 = '#9a9893', '#0b0b0b', '#52514e'
FAMILY = {'Grid-GRU': BLUE, 'GRU': ORANGE, 'Grid-LRU': AQUA, 'Transformer': VIOLET, 'Other': GRAY, 'RIMs': GRAY}
BETTER, WORSE, NEUTRAL = BLUE, ORANGE, GRAY

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'axes.edgecolor': '#333333', 'axes.linewidth': 0.65,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e1e5e8',
    'grid.linewidth': 0.55, 'axes.axisbelow': True, 'figure.dpi': 150, 'savefig.bbox': 'tight',
    'text.color': INK, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'legend.frameon': False,
})


def save(fig, name):
    fig.savefig(f'{OUT}/{name}.png')
    fig.savefig(f'{OUT}/{name}.svg')
    plt.close(fig)


def family(name):
    for key in ('Grid-GRU', 'Grid-LRU', 'Transformer', 'GRU', 'RIMs'):
        if name.startswith(key) or (key == 'RIMs' and 'RIMs' in name):
            return key
    return 'Other'


def hbars(ax, labels, values, errs=None, colors=None, fmt='{:.4f}', xlim=None):
    y = range(len(labels))[::-1]
    ax.barh(list(y), values, xerr=errs, color=colors, height=0.62, error_kw=dict(lw=0.8, capsize=2, ecolor=INK2))
    ax.set_yticks(list(y), labels)
    ax.grid(axis='y', visible=False)
    if xlim:
        ax.set_xlim(*xlim)
    lo = ax.get_xlim()[0]
    for yi, v, e in zip(y, values, errs or [0] * len(values)):
        ax.text(v + (e or 0) + (ax.get_xlim()[1] - lo) * 0.01, yi, fmt.format(v), va='center', fontsize=7.5, color=INK2)


# ---------------------------------------------------------------- AISTATS: same-scale comparisons

def fig_10m():
    # article/latex/aistats_historical_main.tex: Text8 val BPC and SDQ Acc++ at ~10M params, 1B tokens
    rows = [
        ('Grid-GRU L3C4', 1.4345, .0017, .9204, .0128), ('Grid-GRU L2C4', 1.4397, .0050, .8433, .0055),
        ('Grid-GRU L2C8', 1.4452, .0019, .8664, .0099), ('Grid-GRU L2C16', 1.4548, .0023, .8911, .0049),
        ('Grid-GRU L1C8', 1.4687, .0046, .7051, .0163), ('Transformer-256', 1.4492, .0120, None, None),
        ('Transformer-64', 1.4816, .0049, None, None), ('GRU L3', 1.4830, .0061, .3241, .0862),
        ('GRU L2', 1.5004, .0119, .5444, .0743), ('GRU L1', 1.5632, .0025, .6177, .0052),
    ]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.6), sharey=False)
    r = sorted(rows, key=lambda t: t[1])
    hbars(a, [t[0] for t in r], [t[1] for t in r], [t[2] for t in r], [FAMILY[family(t[0])] for t in r], xlim=(1.40, 1.59))
    a.set_title('Text8 validation BPC (lower is better)')
    s = sorted([t for t in rows if t[3] is not None], key=lambda t: -t[3])
    hbars(b, [t[0] for t in s], [t[3] for t in s], [t[4] for t in s], [FAMILY[family(t[0])] for t in s], fmt='{:.3f}', xlim=(0, 1.08))
    b.set_title('SDQ Acc++ (higher is better)')
    fig.suptitle('~10M parameters, 1B training tokens, 2-3 seeds (mean +- SD)', fontsize=10, color=INK2)
    save(fig, 'aistats_10m_same_scale')


MID = [
    # article/latex/aistats_mid_main.tex + aistats_mid_state.tex: (model, BPC, SD, runs, state KiB)
    ('Grid-GRU L3C4', 1.5918, .0009, 3, 5.25), ('Grid-GRU L2C4', 1.5960, .0058, 3, 4.25),
    ('Grid-GRU L2C8', 1.6084, .0017, 2, 6.00), ('GRU L3', 1.6205, .0080, 2, 2.72),
    ('GRU L2', 1.6359, .0007, 3, 2.25), ('Grid-GRU L1C8', 1.6367, .0013, 2, 4.38),
    ('Transformer-256', 1.6396, .0098, 2, 1408.0), ('HGRN2', 1.6695, .0005, 2, 1097.3),
    ('Grid-LRU L3C4', 1.6699, .0042, 3, 16.19), ('Grid-LRU L2C4', 1.6720, .0006, 3, 14.06),
    ('GRU L1', 1.6807, .0003, 2, 1.58), ('Grid-LRU L2C8', 1.7156, .0047, 2, 20.00),
    ('mLSTM', 1.7163, .0039, 2, 549.29), ('Grid-LRU L1C8', 1.7471, .0002, 2, 16.50),
    ('DeltaNet', 1.7844, .0012, 2, 551.82), ('RIMs', 1.8741, .0965, 2, 5.44), ('BRIMs', 1.8853, .0179, 2, 7.50),
]


def fig_1m():
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 4.6), gridspec_kw=dict(width_ratios=[1.35, 1]))
    labels = [t[0] for t in MID]
    colors = [FAMILY[family(t[0])] for t in MID]
    hbars(a, labels, [t[1] for t in MID], [t[2] for t in MID], colors, xlim=(1.55, 2.0))
    a.set_title('Text8 validation BPC at ~1M params (lower is better)')
    y = list(range(len(MID)))[::-1]
    b.scatter([t[4] for t in MID], y, c=colors, s=36, edgecolors='white', linewidths=1.5, zorder=3)
    b.set_xscale('log')
    b.set_yticks(y, [''] * len(y))
    b.set_title('Carried state per sequence (KiB, log)')
    for yi, t in zip(y, MID):
        b.text(t[4] * 1.25, yi, f'{t[4]:g}', va='center', fontsize=7.5, color=INK2)
    b.set_xlim(1, 8000)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, AQUA, VIOLET, GRAY)]
    b.legend(handles, ['Grid-GRU (MoSAIC)', 'GRU', 'Grid-LRU', 'Transformer', 'other baselines'], loc='lower right', fontsize=7.5)
    save(fig, 'aistats_1m_same_scale')


def fig_scale():
    # best configuration per family at ~1M and ~10M (Text8 val BPC)
    pairs = [('Grid-GRU L3C4', 1.5918, 1.4345, BLUE), ('Transformer-256', 1.6396, 1.4492, VIOLET),
             ('GRU L3', 1.6205, 1.4830, ORANGE), ('GRU L1', 1.6807, 1.5632, YELLOW)]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for name, m1, m10, c in pairs:
        ax.plot([0, 1], [m1, m10], color=c, lw=2, marker='o', ms=7, mec='white', mew=1.5)
        ax.text(1.04, m10, f'{name}  {m10:.4f}', va='center', fontsize=8, color=INK)
        ax.text(-0.04, m1, f'{m1:.4f}', va='center', ha='right', fontsize=8, color=INK2)
    ax.set_xticks([0, 1], ['~1M params', '~10M params'])
    ax.set_xlim(-0.35, 1.9)
    ax.set_ylabel('Text8 validation BPC')
    ax.grid(axis='x', visible=False)
    ax.set_title('Scaling 1M -> 10M at a fixed 1B-token budget')
    save(fig, 'aistats_scaling_1m_10m')


def fig_ablations():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.4))
    fig.subplots_adjust(wspace=0.6)
    # aistats_ablation_main.tex: MoSAIC L2C4 regularizers, 67M-token pilot, delta vs full
    reg = [('no comm. cost', -.0040), ('no entropy bonus', -.0036), ('no logit noise', -.0016), ('all three off', +.0022)]
    # aistats_lru_variants.tex: Grid-LRU L2C4 mechanisms, 1B tokens, delta vs control
    lru = [('two reads', -.0185), ('write gate', -.0023), ('rotations', -.0023), ('query read', +.0199), ('associative', +.0265)]
    # aistats_sparse_lru.tex: sparse Grid-LRU, 1B tokens, delta vs dense control
    sp = [('dynamic hub', -.0102), ('self heavy', -.0082), ('star', +.0055), ('star ring', +.0067),
          ('entmax', +.0099), ('clockwork', +.0323), ('block (0.53M core)', +.0877)]
    for ax, data, title in ((axes[0], reg, 'Grid-GRU L2C4 regularizers\n(67M-token pilot)'),
                            (axes[1], lru, 'Grid-LRU L2C4 memory mechanisms\n(1B tokens)'),
                            (axes[2], sp, 'Sparse Grid-LRU routing\n(1B tokens)')):
        labels, vals = [d[0] for d in data], [d[1] for d in data]
        y = list(range(len(data)))[::-1]
        ax.barh(y, vals, color=[BETTER if v < 0 else WORSE for v in vals], height=0.6)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_yticks(y, labels)
        ax.grid(axis='y', visible=False)
        span = max(abs(v) for v in vals)
        ax.set_xlim(-span * 1.7, span * 1.7)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        for yi, v in zip(y, vals):
            ax.text(v + (span * 0.05 if v >= 0 else -span * 0.05), yi, f'{v:+.4f}', va='center',
                    ha='left' if v >= 0 else 'right', fontsize=7.5, color=INK2)
        ax.set_title(title)
        ax.set_xlabel('delta BPC vs control (negative = better)')
    save(fig, 'aistats_ablations_delta')


def fig_communication():
    # communication_2026-10-13/report.md: L2C4 H136, clean, seed 42, ~1B tokens
    conds = ['dynamic\ncolumn0', 'static\ncolumn0', 'none\ncolumn0', 'dynamic\nmean', 'static\nmean', 'none\nmean']
    text8 = [1.6006, 1.6004, 1.7235, 1.6204, 1.6144, 1.6452]
    sdq = [45.71, 85.54, 23.32, 79.79, 82.26, None]
    gaps = [52.68, 72.89, 44.80, 70.88, 70.27, None]
    colors = [BLUE, AQUA, GRAY, BLUE, AQUA, GRAY]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4))
    x = range(len(conds))
    for ax, vals, title, ylim, fmt in ((axes[0], text8, 'Text8 validation BPC (lower is better)', (1.55, 1.75), '{:.4f}'),
                                      (axes[1], sdq, 'SDQ held-out accuracy, % (higher is better)', (0, 100), '{:.1f}'),
                                      (axes[2], gaps, 'SDQ fixed-gap macro accuracy, %', (0, 100), '{:.1f}')):
        for xi, v, c in zip(x, vals, colors):
            if v is None:
                ax.text(xi, ylim[0] + (ylim[1] - ylim[0]) * 0.04, 'not\nfinished', ha='center', fontsize=7, color=INK2)
                continue
            ax.bar(xi, v - ylim[0], bottom=ylim[0], color=c, width=0.62, hatch='//' if 'mean' in conds[xi] else None,
                   edgecolor='white', linewidth=0)
            ax.text(xi, v + (ylim[1] - ylim[0]) * 0.015, fmt.format(v), ha='center', fontsize=7.5, color=INK2)
        ax.set_xticks(list(x), conds, fontsize=7)
        ax.set_ylim(*ylim)
        ax.grid(axis='x', visible=False)
        ax.set_title(title)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, AQUA, GRAY)]
    fig.legend(handles + [plt.Rectangle((0, 0), 1, 1, facecolor=GRAY, hatch='//', edgecolor='white')],
               ['dynamic routing', 'static routing', 'no communication', 'hatched: readout = mean of columns'],
               fontsize=7.5, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.08))
    save(fig, 'communication_ablation')


# ---------------------------------------------------------------- RL (logs/run_fast/*.log)

LINE = re.compile(r'^\[([\d.]+)M\] (\d+) sps (.*)$')


def parse_rl():
    runs = {}
    for f in sorted(glob.glob('logs/run_fast/*.log')):
        key = os.path.basename(f)[:-4]
        pts, params, arch = [], None, None
        for line in open(f, errors='ignore'):
            line = line.strip()
            if line.startswith('Param count:'):
                params = line.split(':')[1].strip()
            elif ' of ' in line and 'hidden units' in line:
                arch = line
            m = LINE.match(line)
            if m:
                tok = m.group(3).split()
                d = {k: float(v) for k, v in zip(tok[::2], tok[1::2]) if re.match(r'^-?[\d.]+$', v)}
                pts.append((float(m.group(1)), d.get('Eval', d.get('EpRet')), d.get('Acc')))
        if pts:
            runs[key] = {'params': params, 'arch': arch, 'steps_M': [p[0] for p in pts],
                         'eval': [p[1] for p in pts], 'acc': [p[2] for p in pts]}
    return runs


def last5(run, key, upto=None):
    vals = [v for m, v in zip(run['steps_M'], run[key]) if upto is None or m <= upto + 1e-6]
    vals = vals[-5:]
    return sum(vals) / len(vals)


def fig_rl(runs):
    rl = lambda k: last5(runs[k], 'eval')
    # AutoencodeEasy, PPO, 20M steps, seed 42, ~1M params: (label, run, verdict)
    ae = [
        ('L1C4 colhz H224 (best)', 'p26_AE_l1', 'win'), ('L1C5 colhz', 'p30_AE_c5', 'win'),
        ('L2C4 dyn+colhz', 'p13_AE_colhz', 'win'), ('L2C4 dyn+colhz+aux .1', 'p16_AE_aux01', 'same'),
        ('colhz reverse order', 'p17_AE_colhz_rev', 'same'), ('L2C2 H208', 'p19_AE_c2', 'same'),
        ('readout=columns', 'p18_AE_cols', 'same'), ('L1 no colhz', 'p27_AE_l1nohz', 'same'),
        ('L2C4 dyn, shared columns', 'p26_AE_share', 'same'), ('L1 cell_act=linear', 'p28_AE_lin', 'same'),
        ('L1 raw_columns=1', 'p28_AE_raw', 'same'), ('L2C4 dynamic + x_add', 'p10_AE_dyn', 'same'),
        ('route_query=state', 'p24_AE_stateq', 'same'), ('route_topk=2', 'p13_AE_route_topk2', 'same'),
        ('L1C6', 'p27_AE_l1c6', 'lose'), ('input_all_layers', 'p13_AE_inall', 'same'), ('read_attn', 'p11_AE_read', 'same'),
        ('LRU backbone hybrid', 'p26_AE_hybrid', 'lose'), ('route entropy +.01', 'p16_AE_rent_p', 'lose'),
        ('L1 readout=columns', 'p27_AE_l1cols', 'lose'), ('bank, attn write + fb', 'p23_AE_bank_fb', 'lose'),
        ('bank, attn write', 'p22_AE_bank_write', 'lose'), ('L2C8', 'p13_AE_c8', 'lose'),
        ('route entropy -.01', 'p16_AE_rent_n', 'lose'), ('L2C8 colhz', 'p16_AE_c8_colhz', 'lose'),
        ('bank, 4 heads', 'p23_AE_bank_h4', 'lose'), ('route_query=current, no fb', 'p22_AE_grid_ff', 'lose'),
        ('comm_rounds=2', 'p12_AE_rounds2', 'lose'), ('mem_slots=16', 'p25_AE_mem', 'lose'),
        ('ManyRnn C16 full (0.31M)', 'p31_AE_m1', 'lose'), ('bank, attention read', 'p22_AE_bank_read', 'lose'),
        ('write_alloc', 'p24_AE_alloc', 'lose'), ('raw_columns=2', 'p29_AE_raw2', 'lose'),
        ('static routing + x_add', 'p17_AE_static', 'lose'), ('routing=table', 'p11_AE_table', 'lose'),
        ('selective decay', 'p10_AE_sel', 'lose'), ('dynamic, no x_add', 'p11_AE_dyn_noxadd', 'lose'),
        ('comm_rounds=3', 'p13_AE_rounds3', 'lose'), ('clockwork strides [1,4]', 'p25_AE_clock', 'lose'),
        ('ManyRnn active_k=4 (0.32M)', 'p31_AE_m2', 'lose'), ('L1 feedback=false', 'p28_AE_nofb', 'lose'),
    ]
    ae = sorted(ae, key=lambda t: -rl(t[1]))
    fig, ax = plt.subplots(figsize=(7.2, 9.6))
    y = list(range(len(ae)))[::-1]
    vals = [rl(t[1]) for t in ae]
    col = {'win': BLUE, 'same': GRAY, 'lose': ORANGE}
    ax.barh(y, vals, color=[col[t[2]] for t in ae], height=0.66)
    ax.axvline(0, color=INK, lw=0.6)
    lru = (0.369 + rl('p11_AE_lru_s1') + rl('p11_AE_lru_s2')) / 3
    ax.axvline(lru, color=INK, lw=1, ls='--')
    ax.text(lru + 0.01, -1.1, f'plain LRU L2, 3 seeds: {lru:.2f}', fontsize=7.5, ha='left', color=INK)
    ax.set_yticks(y, [t[0] for t in ae], fontsize=7.5)
    for yi, v in zip(y, vals):
        ax.text(v + (0.015 if v >= 0 else -0.015), yi, f'{v:+.3f}', va='center', ha='left' if v >= 0 else 'right',
                fontsize=7, color=INK2)
    ax.set_xlim(-0.45, 1.12)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('greedy eval return, mean of last 5 evaluations (max 1.0)')
    ax.set_title('AutoencodeEasy, PPO 20M steps, seed 42, ~1M params\n(Grid-LRU variants unless stated)')
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, GRAY, ORANGE)]
    ax.legend(handles, ['adopted / best', 'no clear change', 'rejected'], loc='lower right', fontsize=7.5)
    save(fig, 'rl_ae_easy_variants')

    # three-protocol comparison of the main candidates
    cands = [('LRU L2\n(seed 1)', 'p11_AE_lru_s1', 'p17_AEM_curr_lru', 'p12_sup_AEH_lru_s1', ORANGE),
             ('Grid L2C4\nstatic+x_add', 'p17_AE_static', None, None, YELLOW),
             ('Grid L2C4\ndynamic', 'p10_AE_dyn', 'p17_AEM_curr_dyn', 'p12_sup_AEH_dyn_s1', AQUA),
             ('Grid L2C4\ndyn+colhz', 'p13_AE_colhz', 'p17_AEM_curr', 'p13_sup_AEH_colhz', GREEN),
             ('Grid L1C4\ndyn+colhz', 'p26_AE_l1', 'p26_AEM_curr_l1', 'p26_sup_AEH_l1', BLUE),
             ('Bank\nattn write', 'p22_AE_bank_write', None, 'p22_sup_AEH_bank_write', VIOLET)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.4))
    for ax, idx, title, key, upto in ((axes[0], 1, 'RL AutoencodeEasy (eval return)', 'eval', None),
                                      (axes[1], 2, 'RL AutoencodeMedium, init from Easy', 'eval', None),
                                      (axes[2], 3, 'Oracle AutoencodeHard @4.6M (accuracy)', 'acc', 4.62)):
        for xi, c in enumerate(cands):
            k = c[idx]
            if k is None or k not in runs:
                ax.text(xi, 0.02, 'n/a', ha='center', fontsize=7, color=INK2)
                continue
            v = last5(runs[k], key, upto)
            ax.bar(xi, v, color=c[4], width=0.62)
            ax.text(xi, v + (0.02 if v >= 0 else -0.06), f'{v:.2f}', ha='center', fontsize=7.5, color=INK2)
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_xticks(range(len(cands)), [c[0] for c in cands], fontsize=6.8)
        ax.grid(axis='x', visible=False)
        ax.set_title(title)
    axes[0].set_ylim(-0.1, 1.05)
    axes[1].set_ylim(-0.1, 1.05)
    axes[2].set_ylim(0, 1.0)
    save(fig, 'rl_three_protocols')

    # learning curves
    curves = [('LRU L2 (seed 1)', 'p11_AE_lru_s1', ORANGE), ('Grid static + x_add', 'p17_AE_static', YELLOW),
              ('Grid dynamic, no x_add', 'p11_AE_dyn_noxadd', MAGENTA), ('Grid dynamic + x_add', 'p10_AE_dyn', AQUA),
              ('Grid dyn + colhz', 'p13_AE_colhz', GREEN), ('Grid L1C4 dyn + colhz', 'p26_AE_l1', BLUE),
              ('Bank attn write', 'p22_AE_bank_write', VIOLET), ('Grid feedback=false', 'p28_AE_nofb', RED)]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for label, k, c in curves:
        r = runs[k]
        ax.plot(r['steps_M'], r['eval'], color=c, lw=1.6, label=label)
    ax.set_xlabel('environment steps (M)')
    ax.set_ylabel('eval return')
    ax.set_title('AutoencodeEasy learning curves (seed 42)')
    ax.legend(fontsize=7.5, loc='upper left', ncol=2)
    ax.set_ylim(-0.6, 1.05)
    save(fig, 'rl_ae_easy_curves')

    # other POPGym tasks, GRU vs LRU vs Grid-LRU static+x_add (p7/p8)
    tasks = [('RepeatFirstMedium', 'p7_RepeatFirstMedium'), ('MineSweeperEasy', 'p8_MineSweeperEasy'),
             ('ConcentrationEasy', 'p8_ConcentrationEasy'), ('CountRecallMedium', 'p7_CountRecallMedium')]
    models = [('GRU', 'rnn', ORANGE), ('LRU', 'lru', YELLOW), ('Grid-LRU static + x_add', 'glru_xadd', AQUA)]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    w = 0.26
    for j, (label, suf, c) in enumerate(models):
        xs, vs = [], []
        for i, (_, pre) in enumerate(tasks):
            k = f'{pre}_{suf}'
            if k in runs:
                xs.append(i + (j - 1) * w)
                vs.append(last5(runs[k], 'eval'))
        ax.bar(xs, vs, width=w - 0.02, color=c, label=label)
        for xx, v in zip(xs, vs):
            ax.text(xx, v + (0.02 if v >= 0 else -0.07), f'{v:.2f}', ha='center', fontsize=7, color=INK2)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(range(len(tasks)), [t[0] for t in tasks])
    ax.grid(axis='x', visible=False)
    ax.set_ylabel('eval return (last 5)')
    ax.set_title('Other POPGym tasks, ~1M params, seed 42')
    ax.legend(fontsize=7.5, loc='upper right')
    save(fig, 'rl_popgym_tasks')

    # curriculum / transfer to AutoencodeMedium
    tr = [('LRU, init from Easy', 'p17_AEM_curr_lru', ORANGE), ('Grid L1C4 raw_columns=1', 'p28_AEM_curr_raw', BLUE),
          ('Grid L2C4 readout=columns', 'p18_AEM_curr_cols', AQUA), ('Grid L1C4 colhz', 'p26_AEM_curr_l1', BLUE),
          ('Grid L2C4 dyn', 'p17_AEM_curr_dyn', AQUA), ('Grid L2C4 colhz', 'p17_AEM_curr', AQUA),
          ('Grid L1C5', 'p30_AEM_curr_c5', BLUE), ('LRU adaptive grow', 'p21_AEM_grow_lru', ORANGE),
          ('Grid adaptive grow', 'p21_AEM_grow', AQUA), ('LRU length mix', 'p20_AEM_mix_lru', ORANGE),
          ('Grid length mix', 'p20_AEM_mix', AQUA), ('LRU from scratch', 'p12_AEM_lru', ORANGE),
          ('Grid from scratch, 60M', 'p17_AEM_long', AQUA), ('Grid from scratch', 'p12_AEM_dyn', AQUA)]
    tr = sorted(tr, key=lambda t: -last5(runs[t[1]], 'eval'))
    fig, ax = plt.subplots(figsize=(7, 4.6))
    y = list(range(len(tr)))[::-1]
    vals = [last5(runs[t[1]], 'eval') for t in tr]
    ax.barh(y, vals, color=[t[2] for t in tr], height=0.64)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_yticks(y, [t[0] for t in tr], fontsize=7.5)
    for yi, v in zip(y, vals):
        ax.text(v + (0.015 if v >= 0 else -0.015), yi, f'{v:+.2f}', va='center', ha='left' if v >= 0 else 'right',
                fontsize=7, color=INK2)
    ax.set_xlim(-0.65, 0.8)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('eval return on AutoencodeMedium, last 5 (20M steps)')
    ax.set_title('AutoencodeMedium: curriculum and transfer')
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (ORANGE, AQUA, BLUE)]
    ax.legend(handles, ['LRU', 'Grid-LRU L2C4', 'Grid-LRU L1'], fontsize=7.5, loc='lower right')
    save(fig, 'rl_aem_transfer')


def main():
    os.makedirs(OUT, exist_ok=True)
    fig_10m()
    fig_1m()
    fig_scale()
    fig_ablations()
    fig_communication()
    if glob.glob('logs/run_fast/*.log'):
        runs = parse_rl()
        # compact summary: curves subsampled to at most 40 points
        summary = {}
        for k, r in runs.items():
            step = max(1, len(r['steps_M']) // 40)
            summary[k] = {'params': r['params'], 'arch': r['arch'], 'final_M': r['steps_M'][-1],
                          'eval_last5': round(last5(r, 'eval'), 4),
                          'acc_last5': round(last5(r, 'acc'), 4) if r['acc'][-1] is not None else None,
                          'curve': [[m, e] for m, e in list(zip(r['steps_M'], r['eval']))[::step]]}
        with open(RL_JSON, 'w') as f:
            json.dump(summary, f, indent=0)
    else:
        with open(RL_JSON) as f:
            runs = {k: {'steps_M': [p[0] for p in v['curve']], 'eval': [p[1] for p in v['curve']], 'acc': [None]}
                    for k, v in json.load(f).items()}
        print('logs/run_fast missing: RL figures need the raw logs', file=sys.stderr)
        return
    fig_rl(runs)


if __name__ == '__main__':
    main()
