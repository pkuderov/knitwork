"""Read-only analysis of the frozen October 4 Comet snapshot; no tracker access."""

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--snapshot', default='docs/experiments/frontier_2026-10-04_snapshot.json')
parser.add_argument('--output', default='docs/experiments/frontier_2026-10-04_analysis.json')
parser.add_argument('--figure-prefix', default='frontier_2026-10-04')
args = parser.parse_args()
SOURCE = ROOT / args.snapshot
OUT = ROOT / args.output
FIG = ROOT / 'docs/experiments/figures'
snapshot = json.loads(SOURCE.read_text())
runs = {r['metadata']['experimentKey'][:8]: r for r in snapshot['runs'] if r['project'] == 'knitwork-aistat'}


def curve(run_id, metric, scale=1.0):
    result = {}
    points = runs[run_id]['metrics'].get(metric, [])
    for p in sorted(points, key=lambda p: p.get('timestamp') or 0):
        try:
            step, value = int(p['step']), float(p['metricValue']) * scale
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            result[step] = value
    return dict(sorted(result.items()))


def common(ids, metric='val/Loss'):
    return sorted(set.intersection(*(set(curve(i, metric)) for i in ids)))


def row(run_id):
    r = runs[run_id]
    return dict(id=r['metadata']['experimentKey'], name=r['metadata']['experimentName'])


def text_comparison(named, control=None):
    control = control or next(iter(named.values()))
    steps = common(list(named.values()))
    step = steps[-1]
    ref = curve(control, 'val/Loss', 1 / math.log(2))
    rows = []
    for name, i in named.items():
        loss = curve(i, 'val/Loss', 1 / math.log(2))
        late = [s for s in steps if s >= 800_000_000]
        window = curve(i, 'val_w1024/Loss', 1 / math.log(2))
        rows.append(row(i) | dict(variant=name, bpc=loss[step], delta=loss[step]-ref[step],
                                 window1024_bpc=window.get(step),
                                 late_800M_mean_delta=statistics.mean(loss[s]-ref[s] for s in late) if late else None))
    return dict(common_step=step, common_steps=steps, rows=rows)


ARCH = dict(control='24a0f721', write_gate='54b43903', query_read='a6b467d1', associative='8621e147', rotations='f3a8110e', two_reads='c47fb505')
ABL = dict(full='ba83b1d6', none='40eb9dbd', no_noise='8f98d51b', no_comm='5525cd85', no_entropy='3efba9c4')
MQAR = dict(control='a7eb48b2', write_gate='3e509bfb', query_read='4b364702', associative='bcc7b8a1', rotations='a5157b08', two_reads='668690f2')
RP = dict(GRU='d7027e85', LRU='96ec3b8a', Grid_GRU='b0c36ba9', Grid_LRU='ebbc8da6', dynamic_hub='4589138c', two_reads='775dd695')
AE = dict(GRU='ef2c91cc', LRU='39dbcf25', dynamic_hub='cc147c27', two_reads='63eebaf4')
BINS = ['T64_K4', 'T128_K8', 'T256_K16', 'T256_K32', 'T256_K64']
TRAIN_WEIGHTS = [100_000 / 180_000] + [20_000 / 180_000] * 4
result = dict(source=str(SOURCE.relative_to(ROOT)), retrieval_started_utc=snapshot['retrieval_started_utc'],
              method='Last timestamp per duplicate step; exact common steps, no interpolation; Text8 CE divided by ln(2); MQAR CE in nats; train and RL diagnostic metrics may be EMAs.',
              project_counts=dict(Counter(r['project'] for r in snapshot['runs'])),
              architecture=text_comparison(ARCH), ablation=text_comparison(ABL))
aux_step = common(list(ABL.values()), 'L_comm')[-1]
for a in result['ablation']['rows']:
    i = a['id'][:8]
    a['trajectory_delta'] = {str(s): curve(i, 'val/Loss', 1/math.log(2))[s] - curve('ba83b1d6', 'val/Loss', 1/math.log(2))[s] for s in result['ablation']['common_steps']}
    a['aux_step'] = aux_step
    a['L_comm'] = curve(i, 'L_comm')[aux_step]
    a['H_comm'] = curve(i, 'H_comm')[aux_step]

sp_control = '6dbc5ce5'
sparse = dict(self_heavy='191b7533', star='7f45618a', star_ring='c428fe4c', clockwork='9b65fb68', block='d5ee1404', entmax='798342c6', dynamic_hub='010b7096')
result['sparse_pairs'] = []
for name, i in sparse.items():
    steps = common([i, sp_control])
    s = steps[-1]
    result['sparse_pairs'].append(row(i) | dict(variant=name, step=s, bpc=curve(i,'val/Loss',1/math.log(2))[s], delta=(curve(i,'val/Loss')[s]-curve(sp_control,'val/Loss')[s])/math.log(2)))

result['rollout'] = []
for name, r64, r512 in [('GRU','f7539af5','c55b14c3'), ('Grid_GRU','ba83b1d6','0376834a'), ('Grid_LRU','136031bb','cbddaab7'), ('LRU','f3a8995e','814aa951')]:
    s = common([r64,r512])[-1]
    a,b = (curve(i,'val/Loss',1/math.log(2))[s] for i in [r64,r512])
    result['rollout'].append(dict(model=name, step=s, r64_bpc=a,r512_bpc=b,delta=b-a,ids=[runs[i]['metadata']['experimentKey'] for i in [r64,r512]]))

baselines = dict(Mamba='428bbf5e', Grid_GRU='9a7f6783', GRU='7803e007', Grid_LRU='6c724575', two_reads='c47fb505', DeltaNet='8667f731', mLSTM='a2f65cc2', HGRN2='6f9ad9de')
result['baselines_seed42'] = text_comparison(baselines, control='428bbf5e')
seed_ids = dict(Mamba=['428bbf5e','671835b4'], Grid_GRU=['9a7f6783','5196277e'])
s = common(sum(seed_ids.values(), []))[-1]
result['mamba_two_seed'] = dict(step=s, groups={})
for name, ids in seed_ids.items():
    values = [curve(i,'val/Loss',1/math.log(2))[s] for i in ids]
    result['mamba_two_seed']['groups'][name] = dict(ids=[runs[i]['metadata']['experimentKey'] for i in ids], values=values, mean=statistics.mean(values), sample_sd=statistics.stdev(values))
for name, ids in [('Grid_LRU_L2', ['6c724575','55de969d','5c0724b5']), ('Grid_LRU_L3', ['08b65912','5423e339','7bf4b64d'])]:
    s = common(ids)[-1]
    values = [curve(i,'val/Loss',1/math.log(2))[s] for i in ids]
    result[name] = dict(step=s, values=values, mean=statistics.mean(values), sample_sd=statistics.stdev(values), ids=[runs[i]['metadata']['experimentKey'] for i in ids])
depth_ids = ['6c724575','55de969d','5c0724b5','08b65912','5423e339','7bf4b64d']
s = common(depth_ids)[-1]
result['depth_matched'] = dict(step=s, groups={})
for name, ids in [('L2', depth_ids[:3]), ('L3', depth_ids[3:])]:
    values = [curve(i,'val/Loss',1/math.log(2))[s] for i in ids]
    result['depth_matched']['groups'][name] = dict(values=values, mean=statistics.mean(values), sample_sd=statistics.stdev(values))

mqar_steps = common(list(MQAR.values()), 'val/Acc')
s = mqar_steps[-1]
result['mqar'] = dict(common_step=s, common_steps=mqar_steps, bins=BINS, train_bin_weights=TRAIN_WEIGHTS, val_query_weights=[k/124 for k in [4,8,16,32,64]], rows=[])
for name, i in MQAR.items():
    acc = curve(i,'val/Acc')
    loss = curve(i,'val/Loss')
    grad = curve(i,'train/Grad_norm')
    skips = curve(i,'train/skipped_steps')
    train_acc = curve(i,'train/Acc')
    bins = [curve(i,f'val/{b}/Acc')[s] for b in BINS]
    spikes = [st for st,v in grad.items() if v>1e6]
    last = max(acc)
    best = max(acc, key=acc.get)
    result['mqar']['rows'].append(row(i) | dict(variant=name, val_acc=acc[s],val_loss=loss[s], bin_acc=bins,
        macro_bin_acc=statistics.mean(bins), val_acc_reweighted_train_bins=sum(w*a for w,a in zip(TRAIN_WEIGHTS,bins)),
        train_ema_acc=train_acc.get(s), exact_match=curve(i,'val/Exact_match').get(s),
        latest_val_step=last, latest_val_acc=acc[last], latest_val_loss=loss[last],
        best_val_acc_step=best,best_val_acc=acc[best],
        max_grad_norm_ema=max(grad.values()),max_grad_step=max(grad,key=grad.get),first_grad_above_1e6=spikes[0] if spikes else None,
        last_skipped_steps_ema=skips[max(skips)], max_val_loss=max(loss.values()),max_val_loss_step=max(loss,key=loss.get),
        last_logged_global_step=max(curve(i,'global_step')), metadata_running=runs[i]['metadata']['running'],
        test_presence_audit=runs[i].get('test_presence_audit'), stdout_audit=runs[i].get('stdout_audit')))

result['rl'] = {}
for env, named in [('RepeatPreviousHard',RP), ('AutoencodeEasy',AE)]:
    steps = common(list(named.values()), 'eval/EpRet')
    s = steps[-1]
    late = steps[-4:]
    rows = []
    for name,i in named.items():
        returns = curve(i,'eval/EpRet')
        last4 = [returns[st] for st in late]
        metrics = {}
        for metric in ['H','Skipped','Upd','ApproxKL','KLStop','ClipFrac','state/AbsMax','state/RMS','state/PoleRadiusMax']:
            c = curve(i,metric)
            if c:
                metrics[metric] = dict(last=c[max(c)], max=max(c.values()), late_mean=statistics.mean(list(c.values())[-4:]))
        peak = max(returns,key=returns.get)
        rows.append(row(i)|dict(model=name,last_common_return=returns[s],last4_common_mean=statistics.mean(last4),last4_common_values=last4,
            best_logged_return=returns[peak],best_logged_return_step=peak,last_eval_step=max(returns),last_eval_return=returns[max(returns)],
            metrics=metrics))
    result['rl'][env] = dict(common_step=s, last4_common_steps=late, rows=rows)

result['coverage'] = dict(stdout_available=sum(r.get('stdout_audit',{}).get('available',False) for r in snapshot['runs']),
                          stdout_audited=sum('stdout_audit' in r for r in snapshot['runs']),
                          runs_with_retrieval_errors=[r['metadata']['experimentKey'] for r in snapshot['runs'] if r['errors']],
                          slot_runs=[r['metadata']['experimentKey'] for r in snapshot['runs'] if 'slot_' in r['metadata']['experimentName']])
cohorts = {
    'Mamba': ['428bbf5e', '671835b4'],
    'Grid_GRU_L2C4': ['9a7f6783', '5196277e', '3d9e9cec'],
    'Grid_GRU_L3C4': ['44063b70', '4a3faa47', '971aace3'],
    'Grid_GRU_L2C8': ['f75e22df', 'd0f5cc50'],
    'GRU_L2': ['7803e007', '5792d3c7', '3735558c'],
    'Grid_LRU_L2C4': ['6c724575', '55de969d', '5c0724b5'],
    'Grid_LRU_L3C4': ['08b65912', '5423e339', '7bf4b64d'],
    'Grid_LRU_L2C8': ['09fb7a42', '0b24e705'],
    'HGRN2': ['6f9ad9de', '14cc943b'],
    'mLSTM': ['a2f65cc2', '7aa3cf4a'],
    'DeltaNet': ['8667f731', '68ed71d0'],
    'RIMs': ['e7095a9b', '0a817e21'],
    'BRIMs': ['c306b9b2', '3b24fb37'],
}
cohorts = {name: ids for name, ids in cohorts.items() if all(i in runs for i in ids)}
s = common([i for ids in cohorts.values() for i in ids])[-1]
result['baseline_cohorts'] = {'common_step': s, 'rows': []}
for name, ids in cohorts.items():
    values = [curve(i, 'val/Loss', 1 / math.log(2))[s] for i in ids]
    result['baseline_cohorts']['rows'].append({
        'model': name, 'n': len(ids), 'values': values,
        'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values),
        'ids': [runs[i]['metadata']['experimentKey'] for i in ids],
    })
ids = ['ad72c9a0', '37180ab2']
s = common(ids)[-1]
values = [curve(i, 'val/Loss', 1 / math.log(2))[s] for i in ids]
result['transformer_separate_horizon'] = {
    'step': s, 'values': values, 'mean': statistics.mean(values),
    'sample_sd': statistics.stdev(values),
}
OUT.write_text(json.dumps(result,indent=2)+'\n')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.25,'figure.dpi':150})
FIG.mkdir(parents=True,exist_ok=True)
colors = dict(zip(ARCH, ['#555555','#dd8a00','#d62728','#9467bd','#2ca02c','#1f77b4']))
def draw(ax,i,metric,label,scale=1,color=None):
    c = curve(i,metric,scale)
    ax.plot([s/1e6 for s in c],list(c.values()),label=label,color=color)

fig,axs=plt.subplots(1,2,figsize=(13,4.2),layout='constrained')
ref=curve(ARCH['control'],'val/Loss',1/math.log(2))
for name,i in ARCH.items():
    if name=='control':continue
    c=curve(i,'val/Loss',1/math.log(2));steps=sorted(set(c)&set(ref))
    axs[0].plot([s/1e6 for s in steps],[c[s]-ref[s] for s in steps],label=name,color=colors[name])
for name,i in list(baselines.items())[:5]:draw(axs[1],i,'val/Loss',name,1/math.log(2))
axs[0].axhline(0,color='black',lw=.8);axs[0].set(title='Text8: changes relative to dynamic control',ylabel='Validation BPC difference',xlabel='Training tokens (million)');axs[0].legend(fontsize=8)
axs[1].set(title='Text8: selected local baselines, seed 42',ylabel='Validation BPC',xlabel='Training tokens (million)',ylim=(1.5,2.4));axs[1].legend(fontsize=8)
fig.savefig(FIG/f'{args.figure_prefix}_text8.png');plt.close(fig)

fig,axs=plt.subplots(1,3,figsize=(16,4.1),layout='constrained')
for name,i in MQAR.items():
    draw(axs[0],i,'val/Acc',name,100,colors[name]);draw(axs[1],i,'val/Loss',name,color=colors[name]);draw(axs[2],i,'train/Grad_norm',name,color=colors[name])
axs[0].set(title='MQAR: query-weighted validation accuracy',ylabel='Correct queries (%)');axs[0].legend(fontsize=8)
axs[1].set(title='Validation CE',ylabel='Nats',yscale='log');axs[1].axhline(math.log(4096),color='black',ls=':',label='ln(4096)')
axs[2].set(title='Pre-clip gradient norm (EMA)',ylabel='Norm, log scale',yscale='log')
for ax in axs:ax.set_xlabel('Training tokens (million)')
fig.savefig(FIG/f'{args.figure_prefix}_mqar.png');plt.close(fig)

fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for name,i in RP.items():draw(axs[0,0],i,'eval/EpRet',name)
for name,i in AE.items():
    draw(axs[0,1],i,'eval/EpRet',name);draw(axs[1,0],i,'H',name);draw(axs[1,1],i,'state/RMS',name)
for ax,title in zip(axs[0],['RepeatPreviousHard','AutoencodeEasy']):ax.axhline(-.5,color='black',ls=':');ax.set(title=title,ylabel='Evaluation episode return');ax.legend(fontsize=8)
axs[1,0].axhline(math.log(4),color='black',ls=':');axs[1,0].set(title='AutoencodeEasy: policy entropy (EMA)',ylabel='Nats');axs[1,0].legend(fontsize=8)
axs[1,1].set(title='AutoencodeEasy: raw LRU/GRU hidden RMS',ylabel='RMS (EMA)',yscale='log');axs[1,1].legend(fontsize=8)
for ax in axs.flat:ax.set_xlabel('Vector slots (million)')
fig.savefig(FIG/f'{args.figure_prefix}_rl.png');plt.close(fig)
print(OUT.relative_to(ROOT))
print(json.dumps({k:result[k] for k in ['mamba_two_seed','Grid_LRU_L2','Grid_LRU_L3','coverage']},indent=2))
for r in result['mqar']['rows']:print('MQAR',r['variant'],r['latest_val_step'],r['latest_val_acc'],r['best_val_acc_step'],r['best_val_acc'])
for env,g in result['rl'].items():
    for r in g['rows']:print('RL',env,r['model'],r['last4_common_mean'],r['best_logged_return_step'],r['best_logged_return'])
