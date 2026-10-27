"""Build a separate H100 report from frozen server logs and Comet GET snapshots."""

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.report_plot_style import prepare_figure, text_overlaps

ARCH = {'control':'24a0f721','write_gate':'54b43903','query_read':'a6b467d1','associative':'8621e147','rotations':'f3a8110e','two_reads':'c47fb505'}
SPARSE = {'control':'6dbc5ce5','self_heavy':'191b7533','star':'7f45618a','star_ring':'c428fe4c','clockwork':'9b65fb68','block':'d5ee1404','entmax':'798342c6','dynamic_hub':'010b7096'}
ABL = {'full':'ba83b1d6','no_aux':'40eb9dbd','no_noise':'8f98d51b','no_comm_penalty':'5525cd85','no_entropy_bonus':'3efba9c4'}
MQAR = {'control':'a7eb48b2','write_gate':'3e509bfb','query_read':'4b364702','associative':'bcc7b8a1','rotations':'a5157b08','two_reads':'668690f2'}
RL = {
    'RepeatPreviousHard':{'GRU':'d7027e85','LRU':'96ec3b8a','Grid-GRU':'b0c36ba9','Grid-LRU':'ebbc8da6','dynamic_hub':'4589138c','two_reads':'775dd695'},
    'AutoencodeEasy':{'GRU':'ef2c91cc','LRU':'39dbcf25','dynamic_hub':'cc147c27','two_reads':'63eebaf4'},
}
ROLLOUT = {'Grid-GRU':('ba83b1d6','0376834a'),'Grid-LRU':('136031bb','cbddaab7'),'GRU':('f7539af5','c55b14c3'),'LRU':('f3a8995e','814aa951')}
BINS=['T64_K4','T128_K8','T256_K16','T256_K32','T256_K64']
PALETTE=['#465a70','#e08028','#ad4959','#8860ae','#27997f','#2e77b8','#9c983e','#bc617f']


def display_model(name):
    head, separator, tail = name.partition('.')
    head = {'rnn':'GRU','grnn':'Grid-GRU','grnn_lru':'Grid-LRU',
        'mamba':'Mamba','hgrn2':'HGRN2','mlstm':'mLSTM','delta_net':'DeltaNet',
        'rims':'RIMs','brims':'BRIMs','transformer':'Transformer'}.get(head, head)
    return head + (' '+tail if separator else '')


def curve(run, metric):
    values={}
    dropped=0
    for point in sorted(run['metrics'].get(metric,[]), key=lambda p:p.get('timestamp') or 0):
        try:
            step,value=int(point['step']),float(point['metricValue'])
        except (TypeError,ValueError):
            dropped+=1
            continue
        if not math.isfinite(value):
            dropped+=1
            continue
        values[step]=value
    return dict(sorted(values.items()))


def common(runs, metric='val/Loss'):
    steps=set.intersection(*(set(curve(r,metric)) for r in runs))
    if not steps:
        raise ValueError(f'No exact common steps for {metric}')
    return sorted(steps)


def stats(values):
    return {'values':values,'n':len(values),'mean':statistics.mean(values),'sd':statistics.stdev(values) if len(values)>1 else None}


def load(directory):
    snapshot=json.loads((directory/'comet_snapshot.json').read_text())
    server=json.loads((directory/'server_snapshot.json').read_text())
    runs={r['metadata']['experimentKey']:r for r in snapshot['runs']}
    def get(prefix):
        selected=[r for key,r in runs.items() if key.startswith(prefix)]
        if len(selected)!=1:
            raise ValueError(f'Missing/ambiguous experiment: {prefix}')
        return selected[0]
    groups={}
    inventory=[]
    for job in server['records']:
        run=runs.get(job['comet_ids'][-1]) if job['comet_ids'] else None
        if job['queue_directory']=='pending' and not job['log']:
            outcome='not_started'
        elif job['rc']==0 or re.search(r'\[1\.00B/1000M EVAL\]|STATUS complete|Best update',job['log']):
            outcome='completed'
        else:
            outcome='incomplete_or_failed'
        inventory.append({'job_file':job['job_file'],'name':job['name'],'queue_directory':job['queue_directory'],
            'outcome':outcome,'rc':job['rc'],'process_alive':job['training_process_verified_alive'],
            'id':job['comet_ids'][-1] if job['comet_ids'] else None,
            'metrics_available':bool(run and run['metrics']),
            'start':job['start'],'end':job['end']})
        if not run or not job['name'].startswith('mid_') or 'topk' in job['name']:
            continue
        key=re.sub(r'_seed\d+$','',job['name'][4:])
        groups.setdefault(key,[]).append(run)
    primary={k:v for k,v in groups.items() if k!='transformer'}
    common_steps=common([r for group in primary.values() for r in group])
    baseline_step=common_steps[-1]
    baseline_rows=[]
    for model,group in primary.items():
        if len({r['parameters'].get('seed') for r in group})!=len(group):
            raise ValueError(f'Duplicate seed: {model}')
        values=[curve(r,'val/Loss')[baseline_step]/math.log(2) for r in group]
        baseline_rows.append({'model':model,'step':baseline_step,**stats(values),
            'seeds':[r['parameters'].get('seed') for r in group],
            'ids':[r['metadata']['experimentKey'] for r in group]})
    baseline_rows.sort(key=lambda r:r['mean'])
    tr_step=common(groups['transformer'])[-1]
    transformer={'step':tr_step,**stats([curve(r,'val/Loss')[tr_step]/math.log(2) for r in groups['transformer']])}
    # A late snapshot must not silently drop a seed with missing measurements.
    coverage = {}
    candidate_steps = {step for group in primary.values() for r in group for step in curve(r, 'val/Loss') if step >= 900_000_000}
    for step in candidate_steps:
        coverage[step] = [name for name, group in primary.items() if all(step in curve(r, 'val/Loss') for r in group)]
    late_step = max(coverage, key=lambda step: (len(coverage[step]), step))
    late_rows = []
    separate_rows = []
    for name, group in primary.items():
        step = late_step if name in coverage[late_step] else common(group)[-1]
        row = {'model':name, 'step':step,
            **stats([curve(r,'val/Loss')[step]/math.log(2) for r in group]),
            'seeds':[r['parameters'].get('seed') for r in group],
            'ids':[r['metadata']['experimentKey'] for r in group]}
        (late_rows if name in coverage[late_step] else separate_rows).append(row)
    all_models_rows = baseline_rows
    baseline_rows = sorted(late_rows, key=lambda row:row['mean'])
    all_models_step = baseline_step
    baseline_step = late_step
    topology_groups = {}
    for family_name in ('grnn','grnn_lru'):
        subset = {name:group for name,group in primary.items() if name.startswith(family_name+'.')}
        step = common([r for group in subset.values() for r in group])[-1]
        topology_groups[family_name] = {'step':step, 'rows':[
            {'model':name, **stats([curve(r,'val/Loss')[step]/math.log(2) for r in group])}
            for name,group in subset.items()]}

    def comparison(named, metric='val/Loss', scale=1/math.log(2)):
        selected={name:get(prefix) for name,prefix in named.items()}
        shared=common(list(selected.values()),metric)
        step=shared[-1]
        return {'step':step,'common_steps':shared,'rows':[{'variant':name,'value':curve(r,metric)[step]*scale,'id':r['metadata']['experimentKey']} for name,r in selected.items()]}
    analysis={'baseline_common_step':baseline_step,'baseline':baseline_rows,'transformer':transformer,
        'baseline_all_models':all_models_rows, 'baseline_all_models_step':all_models_step,
        'baseline_separate':separate_rows, 'topologies':topology_groups,
        'inventory':inventory,'outcome_counts':dict(Counter(r['outcome'] for r in inventory)),
        'queue_counts':dict(Counter(r['queue_directory'] for r in inventory)),
        'metrics_resolved':len(runs),'unresolved_ids':snapshot['unresolved_ids'],
        'architecture':comparison(ARCH),'sparse':comparison(SPARSE),'ablation':comparison(ABL),
        'topk':None,
        'rollout':[],'mqar':comparison(MQAR,'val/Acc',100),'rl':{}}
    # IDs for top-k are taken from the actual queue jobs, rather than guessed.
    topk={'dense':'9a7f6783'}
    for job in server['records']:
        if job['name'] in ('mid_grnn.L2C4_topk1','mid_grnn.L2C4_topk2') and job['comet_ids']:
            topk['top1' if 'topk1' in job['name'] else 'top2']=job['comet_ids'][-1]
    analysis['topk']=comparison(topk)
    for name,prefixes in ROLLOUT.items():
        selected=[get(p) for p in prefixes]
        step=common(selected)[-1]
        analysis['rollout'].append({'model':name,'step':step,'r64_bpc':curve(selected[0],'val/Loss')[step]/math.log(2),'r512_bpc':curve(selected[1],'val/Loss')[step]/math.log(2)})
    for row in analysis['mqar']['rows']:
        r=get(row['id'])
        step=analysis['mqar']['step']
        row['bins']=[100*curve(r,f'val/{b}/Acc')[step] for b in BINS]
        row['macro_bin_acc']=statistics.mean(row['bins'])
        c=curve(r,'val/Acc')
        row['last_step']=max(c)
        row['last_acc']=100*c[max(c)]
        row['best_acc']=100*max(c.values())
        row['max_grad_norm_ema']=max(curve(r,'train/Grad_norm').values())
    for env,named in RL.items():
        selected={name:get(prefix) for name,prefix in named.items()}
        shared=common(list(selected.values()),'eval/EpRet')
        late=shared[-4:]
        analysis['rl'][env]={'late_steps':late,'rows':[]}
        for name,r in selected.items():
            c=curve(r,'eval/EpRet')
            analysis['rl'][env]['rows'].append({'model':name,'late_mean':statistics.mean(c[s] for s in late),'last_common':c[late[-1]],'peak':max(c.values()),'id':r['metadata']['experimentKey']})
    sources={filename:hashlib.sha256((directory/filename).read_bytes()).hexdigest() for filename in ('server_snapshot.json','comet_snapshot.json')}
    if (directory/'mqar_artifacts.json').exists():
        sources['mqar_artifacts.json']=hashlib.sha256((directory/'mqar_artifacts.json').read_bytes()).hexdigest()
    stdout_final = {}
    for job in server['records']:
        if not job['comet_ids']:
            continue
        matches = re.findall(r'\[67\.11M/67M EVAL\].*?BPC:([0-9.]+)', job['log'])
        if matches:
            stdout_final[job['comet_ids'][-1]] = float(matches[-1])
    analysis['pilot_stdout_final_bpc_rounded'] = stdout_final
    analysis['sources_sha256']=sources
    analysis['method']='Comet GET snapshot restricted to server-log IDs. Per-step duplicate values: last timestamp wins; nonfinite points removed. Comparisons use exact common token/slot steps, with no interpolation. Text8 CE/ln(2). Different training budgets and tasks remain separate.'
    return snapshot,server,get,groups,analysis


def style(ax,xlabel='Tokens (M)',ylabel='BPC'):
    ax.grid(True,alpha=0.18)
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_curve(ax,run,metric,label,scale=1,color=None):
    c=curve(run,metric)
    ax.plot(np.asarray(list(c))/1e6,np.asarray(list(c.values()))*scale,label=label,color=color)


def figures(directory,get,groups,analysis):
    gallery=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.axisbelow':True,'axes.titleweight':'bold'})
    def save(fig,name,title,note,pdf):
        prepare_figure(fig)
        collisions = text_overlaps(fig)
        if collisions:
            print('LAYOUT', name, collisions)
        for extension in ('png','svg','pdf'):
            fig.savefig(directory/f'{name}.{extension}',dpi=170,bbox_inches='tight',facecolor='white')
        pdf.savefig(fig,bbox_inches='tight')
        gallery.append({'name':name,'title':title,'note':note,'text_overlaps':collisions})
        plt.close(fig)
    def family(named,name,title,note,pdf):
        fig,axs=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
        table=analysis[name]
        ref=table['rows'][0]['value']
        for i,(label,prefix) in enumerate(named.items()):
            plot_curve(axs[0],get(prefix),'val/Loss',label,1/math.log(2),PALETTE[i%len(PALETTE)])
        labels=[r['variant'] for r in table['rows']]
        delta=[r['value']-ref for r in table['rows']]
        axs[1].barh(range(len(labels)),delta,color=[PALETTE[i%len(PALETTE)] for i in range(len(labels))])
        axs[1].set_yticks(range(len(labels)),labels)
        axs[1].invert_yaxis()
        axs[1].axvline(0,color='#222222',lw=0.8)
        for i,v in enumerate(delta):
            axs[1].annotate(f'{v:+.4f}',(v,i),xytext=(4 if v>=0 else -4,0),textcoords='offset points',ha='left' if v>=0 else 'right',va='center',fontsize=9)
        style(axs[0])
        style(axs[1],'ΔBPC','')
        axs[0].legend(fontsize=8)
        axs[0].set_title('Validation')
        axs[1].set_title(f"Δ vs {labels[0]} @ {table['step']/1e6:.2f}M")
        margin=max(max(abs(v) for v in delta)*0.35,0.004)
        axs[1].set_xlim(min(delta)-margin,max(delta)+margin)
        save(fig,name,title,note,pdf)
    with PdfPages(directory/'all_figures.pdf') as pdf:
        fig,ax=plt.subplots(figsize=(11,8),constrained_layout=True)
        rows=analysis['baseline']
        for i,row in enumerate(rows):
            ax.errorbar(row['mean'],i,xerr=row['sd'] or 0,fmt='o',color='#2563a6',capsize=3)
            jitter=np.linspace(-0.08,0.08,row['n'])
            ax.scatter(row['values'],i+jitter,s=20,color='#555555',alpha=0.7)
            ax.text(1.025,i,f"{row['mean']:.4f} ± {row['sd']:.4f}",transform=ax.get_yaxis_transform(),va='center',fontsize=8,clip_on=False)
        ax.set_yticks(range(len(rows)),[display_model(r['model'])  + f" (n={r['n']})" for r in rows])
        ax.invert_yaxis()
        ax.set_xlim(min(r['mean'] for r in rows)-0.035,max(max(r['values']) for r in rows)+0.06)
        style(ax,'BPC','')
        save(fig,'01_baselines',f"Text8: сравнение моделей на {analysis['baseline_common_step']/1e6:.2f}M токенов",
            'Точки — отдельные seed; центр и whiskers — среднее ± SD между seed, не CI. Все эти модели около 1M полных весов. Transformer приведён отдельно из-за другого набора точек оценки.',pdf)

        fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
        early = analysis['baseline_all_models']
        for i, row in enumerate(early):
            ax.errorbar(row['mean'], i, xerr=row['sd'] or 0, fmt='o', color='#355c7d', capsize=3)
            ax.scatter(row['values'], i+np.linspace(-0.06,0.06,row['n']), s=16, color='#777777')
            ax.text(1.025, i, f"{row['mean']:.4f}", transform=ax.get_yaxis_transform(), va='center', fontsize=8, clip_on=False)
        ax.set_yticks(range(len(early)), [display_model(r['model'])+f" (n={r['n']})" for r in early])
        ax.invert_yaxis()
        style(ax, 'BPC', '')
        save(fig, '14_all_models_common', 'Дополнительный общий срез всех моделей',
            f"Все 17 non-Transformer групп, без удаления seed: {analysis['baseline_all_models_step']:,} токенов. Это ранний срез. Основной поздний рейтинг приведён отдельно.", pdf)

        fig,axs=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
        selected=['grnn.L2C4','rnn.L3','mamba','delta_net','hgrn2','mlstm','rims','brims']
        for i,key in enumerate(selected):
            group=groups[key]
            r=next((r for r in group if r['parameters'].get('seed')=='42'),group[0])
            plot_curve(axs[0],r,'val/Loss',display_model(key),1/math.log(2),PALETTE[i])
        for i,r in enumerate(groups['transformer']):
            plot_curve(axs[1],r,'val/Loss','Transformer seed '+r['parameters'].get('seed','?'),1/math.log(2),PALETTE[i])
        for ax in axs:
            style(ax)
            ax.set_ylim(1.45,2.4)
            ax.legend(fontsize=8)
        axs[0].set_title('Seed 42')
        axs[1].set_title('Transformer · separate horizon')
        save(fig,'02_baseline_curves','Text8: как обучались основные сравниваемые модели',
            'Кривые continuous validation. Область BPC выше 2.4 в начале выходит за ось; полные значения сохранены в исходном snapshot. Transformer нельзя подставлять в общую точку без отдельного правила согласования.',pdf)

        fig,axs=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
        for col,family_name in enumerate(('grnn','grnn_lru')):
            subset=analysis['topologies'][family_name]['rows']
            for i,r in enumerate(subset):
                axs[col].errorbar(i,r['mean'],yerr=r['sd'] or 0,fmt='o',capsize=4,color='#2563a6')
                axs[col].scatter([i]*r['n'],r['values'],s=22,color='#666666')
            axs[col].set_xticks(range(len(subset)),[r['model'].split('.',1)[1] for r in subset])
            axs[col].set_title('Grid-GRU' if col==0 else 'Grid-LRU')
            style(axs[col],'Topology','BPC')
        save(fig,'03_topologies','Text8: глубина и число колонок',
            f"Общий шаг {analysis['baseline_common_step']}; каждая точка seed видна. Размер state и вычисления различаются между формами, несмотря на близкое число весов.",pdf)

        top={r['variant']:r['id'] for r in analysis['topk']['rows']}
        family(top,'topk','Text8: dense, top-1 и top-2 обмен',
            'Один seed 42. Top-k меняет допустимые источники сообщения; эти запуски не измеряют выигрыш sparse execution или полной стоимости FLOPs.',pdf)
        family(SPARSE,'sparse','Grid-LRU: разные схемы обмена между колонками',
            'Сравнение с отдельным LRU control, не с текущей локальной clean-серией. Разные схемы распределяют сообщения и частоту обновлений; скорость экономии вычислений отдельно не доказана.',pdf)
        family(ARCH,'architecture','Grid-LRU: пять изменений записи и чтения памяти',
            'Ширина сохранена, но параметры, state и вычисления не строго равны. Two-reads добавляет вычисления; associative заменяет LRU-колонку матрицей памяти.',pdf)

        fig,axs=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
        rr=analysis['rollout']
        for i,row in enumerate(rr):
            axs[0].plot([i-0.15,i+0.15],[row['r64_bpc'],row['r512_bpc']],color=PALETTE[i],alpha=0.6)
            axs[0].scatter(i-0.15,row['r64_bpc'],color=PALETTE[i],marker='o',s=45)
            axs[0].scatter(i+0.15,row['r512_bpc'],color=PALETTE[i],marker='D',s=40)
            for j,prefix in enumerate(ROLLOUT[row['model']]):
                plot_curve(axs[1],get(prefix),'val/Loss',row['model']+f' r{64 if j==0 else 512}',1/math.log(2),PALETTE[i])
                axs[1].lines[-1].set_linestyle('-' if j==0 else '--')
        for marker, label in [('o','r64'),('D','r512')]:
            axs[0].scatter([],[],marker=marker,color='#333333',label=label)
        axs[0].legend()
        axs[0].set_xticks(range(len(rr)),[r['model'] for r in rr])
        style(axs[0],'Model','BPC')
        style(axs[1])
        axs[1].legend(fontsize=7,ncol=2)
        save(fig,'07_rollout','Короткий пилот: обучение на отрезках 64 или 512 шагов',
            'Каждому pilot дан бюджет 67 108 864 токенов, 2048 updates; точки слева сравниваются на общей Comet-точке 50 331 648. Tokens/update одинаковы; streams меняются 512→64, поэтому это не изолированный эффект длины backprop. Линии — один seed.',pdf)
        family(ABL,'ablation','Короткая ablation: шум и вспомогательные штрафы',
            'no_comm_penalty отключает штраф, а no_aux — все три добавки. Сам обмен между колонками остаётся во всех этих H100 вариантах. Это не локальный routing=none.',pdf)

        fig,axs=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
        for i,(label,prefix) in enumerate(MQAR.items()):
            plot_curve(axs[0],get(prefix),'val/Acc',label,100,PALETTE[i])
            plot_curve(axs[1],get(prefix),'val/Loss',label,1,PALETTE[i])
        style(axs[0],ylabel='Acc (%)')
        style(axs[1],ylabel='CE')
        axs[0].legend(fontsize=8)
        axs[1].set_yscale('log')
        save(fig,'09_mqar_curves','MQAR: качество поиска запомненных пар',
            'MQAR — отдельная задача, V8192 и около 4M полных весов, из них около 1M core. CE на логарифмической шкале. Разные конец/полнота Comet-логов показаны в таблице отдельно; stdout может содержать более поздние округлённые оценки.',pdf)

        fig,axs=plt.subplots(1,2,figsize=(15,5),constrained_layout=True)
        mrows=analysis['mqar']['rows']
        values=np.array([r['bins'] for r in mrows])
        image=axs[0].imshow(values,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
        axs[0].set_xticks(range(5),BINS,rotation=25,ha='right')
        axs[0].set_yticks(range(len(mrows)),[r['variant'] for r in mrows])
        axs[0].set_xlabel('Bin')
        for i in range(len(mrows)):
            for j in range(5):
                axs[0].text(j,i,(f'{values[i,j]:.2f}' if values[i,j]<1 else f'{values[i,j]:.1f}'),ha='center',va='center',color='white' if values[i,j]>55 else '#222222',fontsize=9)
        fig.colorbar(image,ax=axs[0],label='Acc (%)')
        for i,r in enumerate(mrows):
            axs[1].scatter(r['value'],i,color=PALETTE[i],marker='o',s=55)
            axs[1].scatter(r['macro_bin_acc'],i,color=PALETTE[i],marker='s',s=55)
        axs[1].set_yticks(range(len(mrows)),[r['variant'] for r in mrows])
        axs[1].invert_yaxis()
        for marker,label in [('o','Query-weighted'),('s','Macro over bins')]:
            axs[1].scatter([],[],marker=marker,color='#333333',label=label)
        style(axs[1],'Acc (%)','')
        axs[1].legend(fontsize=8)
        axs[1].set_xlim(0,max(values.max(),25)*1.2)
        save(fig,'10_mqar_bins',f"MQAR: разные уровни сложности, {analysis['mqar']['step']/1e6:.2f}M токенов",
            'T — длина последовательности, K — число пар. В общей accuracy длинные примеры имеют больше queries. Macro сначала одинаково взвешивает пять bins. Это разные критерии.',pdf)

        fig,axs=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
        for i,(label,prefix) in enumerate(MQAR.items()):
            plot_curve(axs[0],get(prefix),'train/Grad_norm',label,1,PALETTE[i])
            plot_curve(axs[1],get(prefix),'train/skipped_steps',label,1,PALETTE[i])
        axs[0].set_yscale('log')
        style(axs[0],ylabel='Grad norm (EMA)')
        style(axs[1],ylabel='Skipped steps (EMA)')
        axs[0].legend(fontsize=8)
        save(fig,'11_mqar_stability','MQAR: устойчивость обучения',
            'Это сглаженные training diagnostics, а не точный cumulative counter или повторные обучения. Большие private-state/gradient колебания — наблюдение, не доказанный диагноз причины.',pdf)

        fig,axs=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
        for col,(env,named) in enumerate(RL.items()):
            for i,(label,prefix) in enumerate(named.items()):
                plot_curve(axs[col],get(prefix),'eval/EpRet',label,1,PALETTE[i])
            axs[col].set_title(env)
            style(axs[col],'Slots (M)','Return')
            axs[col].legend(fontsize=8)
        save(fig,'12_rl_curves','RL: оценка поведения по мере обучения',
            'Return — награда за эпизод, больше лучше. X считает vector slots, не только valid environment transitions. Среда и протокол отличаются от Text8/MQAR.',pdf)

        fig,axs=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
        for col,(env,info) in enumerate(analysis['rl'].items()):
            for i,r in enumerate(info['rows']):
                axs[col].plot([r['late_mean'],r['peak']],[i,i],color=PALETTE[i],alpha=0.5)
                axs[col].scatter(r['late_mean'],i,color=PALETTE[i],marker='o',s=60)
                axs[col].scatter(r['peak'],i,color=PALETTE[i],marker='D',s=50)
            axs[col].set_yticks(range(len(info['rows'])),[r['model'] for r in info['rows']])
            axs[col].invert_yaxis()
            axs[col].set_title(env)
            style(axs[col],'Return','')
            for marker,label in [('o','Last 4 common evals'),('D','Best logged eval')]:
                axs[col].scatter([],[],marker=marker,color='#222222',label=label)
            axs[col].legend(fontsize=8)
        save(fig,'13_rl_endpoints','RL: лучший момент и результат к концу обучения',
            'Последние четыре оценки — четыре точки одного обучения, не четыре seed. Best logged evaluation приведена как диагностика, не как выбранный основной результат.',pdf)
    return gallery


def write_report(directory,snapshot,server,get,groups,analysis,gallery):
    analysis['figures'] = gallery
    if any(item['text_overlaps'] for item in gallery):
        raise ValueError('Figure text overlaps remain')
    (directory/'analysis.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (directory/'inventory.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(analysis['inventory'][0]))
        writer.writeheader()
        writer.writerows(analysis['inventory'])
    with (directory/'baselines.csv').open('w',newline='') as handle:
        columns=['model','step','n','mean','sd','seeds','ids']
        writer=csv.DictWriter(handle,fieldnames=columns,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(analysis['baseline'])
    counts=analysis['outcome_counts']
    lines=['# H100: что проверили и что получилось', '',
        'Это **отдельный отчёт о серверной серии**. Данные заново прочитаны 12 октября 2026 с `aicenter3` и из Comet; сами найденные запуски проводились 1–4 октября. Локальная серия `dynamic/static/none` на RTX 5060 сюда не включена. Обучения на сервере при подготовке отчёта не запускались и не останавливались.', '',
        '## Что найдено и чему можно доверять', '',
        f"В очереди зарегистрировано **{len(server['records'])} заданий**. По логам/return codes: {counts.get('completed',0)} имеют признак нормального завершения, {counts.get('incomplete_or_failed',0)} не завершены нормально или завершились с ошибкой, {counts.get('not_started',0)} не запускались. **В папке running осталось 8 старых записей, но соответствующие наши training processes не живы.** По одному имени папки нельзя судить о текущем обучении.", '',
        f"Свежий Comet snapshot содержит **{len(snapshot['runs'])} experiment IDs** этой серверной серии, с кривыми/параметрами; ошибок чтения у них нет. Ещё {len(snapshot['unresolved_ids'])} старых IDs ранних остановленных MQAR-попыток не найдены в доступных списках Comet и не используются для сравнения качества. Retries и зарегистрированные задания — не одно и то же число.", '',
        'На ноде сейчас все семь GPU — H100 80GB HBM3. Для включения эксперимента использована связь job → stdout Comet URL → experiment id. Источники и их hashes сохранены. Никакие метрики чужих текущих процессов не использовались.', '',
        '## Простая постановка', '',
        '**Text8:** модель читает текст по символам и угадывает следующий символ. BPC — ошибка, меньше лучше. Основные новые модели имеют примерно 1M полных параметров, train/val/test split 90M/5M/5M, бюджет около 1B токенов, 512 потоков и отрезки обучения длиной 64. В основной mid-серии есть seed 42 и 1337, у ряда GRU/Grid форм ещё 31337. Размер памяти и вычисления между разными семействами не уравнены.', '',
        '**MQAR:** модель получает несколько пар ключ—значение и затем должна вспомнить правильное значение по ключу. Есть пять уровней сложности с разной длиной и числом пар. В этой серии около 1M параметров recurrent core, но **около 4M полных параметров** из-за большого vocabulary/embedding. Бюджет новых запусков — 8 эпох, batch 32, полный градиент через последовательность. Старые прерванные попытки и slot-crashes не являются результатами качества.', '',
        '**RL:** модель выбирает действия в средах, где нужно помнить прошлые наблюдения. В `RepeatPreviousHard` вспоминается прошлый элемент; в `AutoencodeEasy` надо воспроизвести показанную последовательность. Return — награда за эпизод, больше лучше. Бюджет каждого запуска около 8,39M vector slots, один seed. Эти slots нельзя автоматически приравнивать к числу полезных переходов среды.', '',
        '## Что означает каждый набор вариантов', '',
        '| Проверка | Простое объяснение | Что остаётся ограничением |',
        '|---|---|---|',
        '| Основные baselines | GRU, Mamba, HGRN2, DeltaNet, mLSTM, RIMs, BRIMs, Grid-GRU, Grid-LRU и Transformer получают сопоставимый бюджет данных | Наши реализации и настройки; это не лучшие опубликованные версии каждого семейства |',
        '| L1C8 / L2C4 / L2C8 / L3C4 | Меняем число слоёв и колонок памяти | Вместе меняются state и вычисления |',
        '| Top-1 / Top-2 | Колонке разрешён один или два выбранных источника вместо всех | Качество не доказывает реальную экономию FLOPs |',
        '| Self-heavy | Больше опоры на свою память | Отдельная Grid-LRU серия |',
        '| Star | Остальные колонки обмениваются через центральную | Фиксированная геометрия маршрутов |',
        '| Star-ring | К центральным связям добавлено кольцо соседей | Меняется доступность источников |',
        '| Clockwork | Некоторые колонки обновляются реже | Меняется частота recurrent updates |',
        '| Block | Входные и выходные проекции LRU разбиты на две группы каналов, между слоями каналы переставляются | Связи между колонками остаются плотными; меняются проекции и число параметров |',
        '| Entmax | Обучаемое более разреженное распределение весов связи | Нет отдельного speedup benchmark |',
        '| Dynamic hub | Центральная колонка выбирает периферию по текущему контексту | Это старый LRU-router, не локальный GRU communication control |',
        '| Write gate | Центр решает, насколько обновлять каждую периферийную память | Добавляются параметры; сохранение памяти не равно sparse execution |',
        '| Query read | Центр читает из памяти только узкую часть, нужную для текущего запроса | Дополнительные read projections |',
        '| Associative | Одна LRU-колонка заменена матрицей, которая записывает и извлекает пары | Весов меньше, state больше; параметры не строго matched |',
        '| Rotations | Небольшие обучаемые повороты напрямую перемешивают состояния | Сохранение нормы поворота не гарантирует стабильность всей модели |',
        '| Two reads | После обычного шага центр ещё раз читает свежие сообщения | Те же веса, но больше вычислений |',
        '| Rollout 64 / 512 | Градиент проходит через короткий или длинный кусок истории | При одинаковом tokens/update streams меняются 512→64 |',
        '| No noise | Убираем шум в маршрутах | Короткий пилот 67M, один seed |',
        '| No communication penalty | Убираем дополнительный штраф за обмен | **Сам обмен остаётся** |',
        '| No entropy bonus | Убираем дополнительную награду за разнообразие маршрутов | Сам dynamic-router остаётся |',
        '| No aux | Убираем все три добавки к loss | **Это не режим none без межколоночного обмена** |', '',
        'Названия control/static/dynamic/none между разными сериями нельзя приравнивать без проверки кода. Реализация локальных `dynamic`, `static`, `none` подробно описана в [отдельном отчёте](../communication_2026-10-12/report.md). В серверной абляции регуляризаторов `none` означает только отсутствие auxiliary noise/loss/bonus.', '',
        '## Результаты Text8', '',
        f"Основная таблица сравнивает **точно одинаковый шаг {analysis['baseline_common_step']:,} токенов** по continuous validation. Среднее ± SD считается по отдельным seed. Это не доверительный интервал и не формальный тест значимости.", '',
        '| Модель | Seed | BPC ↓ |', '|---|---:|---:|']
    for r in analysis['baseline']:
        spread=f" ± {r['sd']:.4f}" if r['sd'] is not None else ''
        lines.append(f"| {display_model(r['model'])} | {r['n']} | {r['mean']:.4f}{spread} |")
    for row in analysis['baseline_separate']:
        lines += ['', f"{display_model(row['model'])} оставлен отдельно со всеми {row['n']} seed: {row['mean']:.4f} ± {row['sd']:.4f} BPC в собственной общей точке {row['step']:,}. У одного seed отсутствует метрика основной поздней точки; seed не удалён ради совпадения budget."]
    lines += ['', f"Общая точка абсолютно всех recurrent/baseline групп в свежем snapshot только {analysis['baseline_all_models_step']:,}: после неё пропуски у GRU-L2 и Grid-LRU-L2C4 чередуются. Поэтому основной поздний срез выбирается только по наличию точек у всех seed группы, без просмотра их качества. Ранний общий срез сохранён отдельно в analysis.json и на дополнительном графике."]
    tr=analysis['transformer']
    lines += ['',f"Transformer показан отдельно: **{tr['mean']:.4f} ± {tr['sd']:.4f} BPC**, {tr['n']} seed, собственная общая точка {tr['step']:,}. Эта точка не совпадает с основной таблицей; интерполяция не применялась.", '',
        'В основном позднем срезе Mamba даёт самый низкий средний BPC; далее идут Grid-GRU L3C4 и L2C4. Grid-LRU в целом хуже соответствующего Grid-GRU. Большая глубина сама по себе не улучшает результат одинаково у всех семейств. Точный порядок виден в таблице и на графиках; он относится к этим настройкам и seed.', '',
        '### Изменения памяти и маршрутов', '', '| Серия | Вариант | Общие токены | BPC ↓ |', '|---|---|---:|---:|']
    for name in ('topk','sparse','architecture','ablation'):
        info=analysis[name]
        for r in info['rows']:
            lines.append(f"| {name} | {r['variant']} | {info['step']:,} | {r['value']:.4f} |")
    lines += ['', 'Варианты с дополнительной памятью не дают общего улучшения на Text8. Результат отдельного варианта надо сравнивать с его собственным control, а не с другим LRU baseline. Короткие регуляризаторные ablations отвечают только на ранний участок обучения.', '',
        '### Пилот длины обучения', '', '| Модель | r64 BPC | r512 BPC | Разница r512−r64 |', '|---|---:|---:|---:|']
    for r in analysis['rollout']:
        lines.append(f"| {r['model']} | {r['r64_bpc']:.4f} | {r['r512_bpc']:.4f} | {r['r512_bpc']-r['r64_bpc']:+.4f} |")
    lines += ['', 'Каждому pilot дан budget 67 108 864 токена, 2048 updates. В свежих Comet-кривых финальные оценки отсутствуют: последняя точно общая точка сравнительной таблицы — 50 331 648 токенов. Финальные округлённые BPC из stdout приведены отдельно ниже. Более длинный backprop помогает не всем моделям. Это пилот для выбора следующих проверок, а не финальная оценка после 1B.', '',
        '## Финальные округлённые pilot-результаты из stdout', '',
        '| Pilot | Вариант | BPC на 67.11M (округлено в логе) |', '|---|---|---:|']
    for name, prefixes in ROLLOUT.items():
        for rollout, prefix in zip((64,512), prefixes):
            run_id = get(prefix)['metadata']['experimentKey']
            value = analysis['pilot_stdout_final_bpc_rounded'].get(run_id)
            lines.append(f'| rollout | {name} r{rollout} | {value:.3f} |' if value is not None else f'| rollout | {name} r{rollout} | — |')
    for name, prefix in ABL.items():
        run_id = get(prefix)['metadata']['experimentKey']
        value = analysis['pilot_stdout_final_bpc_rounded'].get(run_id)
        lines.append(f'| auxiliary ablation | {name} | {value:.3f} |' if value is not None else f'| auxiliary ablation | {name} | — |')
    lines += ['', 'Эти цифры округлены до трёх знаков исходным stdout. Они не подменяют Comet-точки большей точности; малые различия ниже разрешения округления не интерпретируются.', '',
        '**Что показывает rollout pilot.** На финальных 67.11M токенов r512 хуже r64 у Grid-GRU примерно на 0.021 BPC, у Grid-LRU — на 0.017, у GRU — на 0.016. У LRU разница около −0.001 BPC, практически на уровне округления. Поэтому здесь не видно общего выигрыша от перехода к длинному backpropagation. Это совместное изменение длины rollout 64→512 и числа потоков 512→64 при одинаковых 32,768 tokens/update, а не изолированная проверка длины градиента. Частота случайных resets по токенам в конфигурациях компенсирована.', '',
        '**Что показывает auxiliary pilot.** Все пять вариантов заканчивают в узком диапазоне 1.861–1.868 BPC. Относительно full удаление entropy bonus даёт примерно −0.001, communication penalty — +0.001, всех добавок — +0.004, routing noise — +0.006 BPC. Значения ±0.001 нельзя использовать для уверенного ранжирования по округлённому логу. На точной Comet-точке 50.33M no_noise и no_comm_penalty ещё немного лучше full; к финалу знак их различий меняется. Значит, порядок чувствителен к моменту оценки.', '',
        '**Граница вывода.** Каждый вариант представлен одним seed и коротким бюджетом 67M вместо 1B. Эти результаты не доказывают ненужность регуляризаторов и не объясняют сами по себе низкую SDQ accuracy локального dynamic_column0: pilot относится к Text8. Приоритетное наблюдение здесь — отсутствие общего раннего выигрыша r512; влияние отдельных auxiliary additions требует полного бюджета и повторных seed.', '',
        '## Результаты MQAR', '',
        f"Все шесть вариантов сравниваются в точке **{analysis['mqar']['step']:,} токенов**, доступной у каждого. Отдельно показана последняя доступная точка каждого запуска: её нельзя использовать как одинаковый budget для ранжирования.", '',
        '| Вариант | Accuracy на общей точке | Macro по пяти bins | Последняя доступная Acc | Токены последней точки |',
        '|---|---:|---:|---:|---:|']
    for r in analysis['mqar']['rows']:
        lines.append(f"| {r['variant']} | {r['value']:.2f}% | {r['macro_bin_acc']:.2f}% | {r['last_acc']:.2f}% | {r['last_step']:,} |")
    lines += ['', 'У всех моделей общая accuracy остаётся низкой. Короткие и лёгкие примеры заметно проще длинных и нагруженных памятью. Большая training accuracy не означает такую же held-out accuracy; query-weighting усиливает вес сложных bins.', '',
        'В query_read и associative видны большие всплески gradients, а query_read пропускает часть updates. Лучший ранний момент и поздний результат могут сильно различаться. Поэтому здесь показаны полные траектории, а не только лучший score. Comet не содержит все финальные оценки, которые есть в stdout. Серверные status.json/metrics.json для этих старых MQAR-запусков в проверенном каталоге runs не найдены. Свежие Comet кривые и stdout доступны; полного подтверждения финальной оценки каждого варианта из run artifacts здесь нет.', '',
        '## Результаты RL', '',
        'Для основного сравнения используем среднее последних четырёх **общих** eval steps. Это четыре точки одного обучения, а не четыре независимых запуска. Отдельно показываем лучший записанный момент, чтобы видеть возможное ухудшение.', '',
        '| Среда | Модель | Последние 4 общие оценки | Лучший записанный Return |', '|---|---|---:|---:|']
    for env,info in analysis['rl'].items():
        for r in info['rows']:
            lines.append(f"| {env} | {r['model']} | {r['late_mean']:.4f} | {r['peak']:.4f} |")
    lines += ['', 'Эта короткая RL-серия не показывает устойчивого общего преимущества модульных моделей. Отдельный лучший момент не означает, что улучшение сохраняется к концу обучения. По одному seed нельзя делать вывод о надёжности RL обучения.', '',
        '## Графики', '']
    for item in gallery:
        lines += [f"### {item['title']}", '',f"![{item['title']}]({item['name']}.png)",'',item['note'],'']
    lines += ['## Как получены числа', '',
        'Каждая кривая прочитана из свежего Comet snapshot по ID из H100 stdout. Если у одного шага несколько значений, используется запись с последним timestamp. Нечисловые/нефинитные точки не рисуются. Для сравнения вариантов берётся последний точно общий шаг; значения между точками не интерполируются. BPC вычисляется как cross-entropy / ln(2).', '',
        'Text8 основной серии использует continuous validation для quality curves; checkpoint selector исторически использовал reset-1024 validation. В этом отчёте нет выбора вариантов по test. Pilot и full-budget curves не объединяются в один рейтинг; MQAR и RL не приравниваются к Text8.', '',
        'Много seed есть только у части Text8 baselines. Остальные механизмы/пилоты/RL в основном представлены одним seed. Error bars в baseline-графиках — SD между seed. Соседние checkpoints не считаются повторными обучениями.', '',
        'Скорость H100 из этих совместных запусков не является изолированным hardware benchmark: одновременно работали другие процессы, а SDK мог задерживать upload. Поэтому в основные графики качества не включены шумные Comet perf/fps и durationMillis как «чистое время модели».', '',
        '## Источники и воспроизведение', '',
        '[Снимок серверной очереди и stdout](server_snapshot.json), [свежие метрики Comet](comet_snapshot.json), [Результат поиска MQAR run artifacts: пустой](mqar_artifacts.json), [числовой анализ](analysis.json), [все задания CSV](inventory.csv), [baselines CSV](baselines.csv), [PDF всех графиков](all_figures.pdf), [галерея](index.html). SHA256 исходных снимков сохранены в analysis.json.', '',
        '```bash', '# Повторить графики без сети и без обучения:', '.venv/bin/python inference/h100_report.py', '', '# Только при необходимости обновить read-only Comet snapshot:', '.venv/bin/python inference/fetch_h100_report.py', '```', '',
        'Незавершённые/упавшие slot-проверки и ранние остановленные MQAR-запуски остаются в inventory, но не получают придуманных значений качества. Нет claims о state of the art или полном закрытии замечаний статьи: это отчёт о доступных проведённых проверках.']
    (directory/'report.md').write_text('\n'.join(lines)+'\n')
    from markdown_it import MarkdownIt
    rendered = MarkdownIt('commonmark').enable('table').render('\n'.join(lines))
    (directory/'report.html').write_text(
        '<!doctype html><html lang="ru"><meta charset="utf-8"><title>H100 report</title>'
        '<style>body{font:16px system-ui;max-width:1250px;margin:32px auto;padding:0 20px;line-height:1.55;color:#222}'
        'img{max-width:100%}table{border-collapse:collapse;margin:20px 0;display:block;overflow-x:auto}'
        'th,td{border:1px solid #d8dde2;padding:7px 12px}th{background:#f4f6f8}a{color:#2563a6}'
        'pre{overflow-x:auto;background:#f6f7f8;padding:12px}</style>' + rendered + '</html>'
    )
    html=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>H100 experiments</title>',
        '<style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 20px;background:#f7f9fc;color:#233044}img{width:100%;background:white}section{margin:36px 0}a{color:#2563a6}</style>',
        '<h1>H100 · серверная серия · данные прочитаны 12 октября 2026</h1><p><a href="report.md">Markdown</a> · <a href="report.html">Полный HTML-отчёт</a> · <a href="all_figures.pdf">PDF</a> · <a href="baselines.csv">CSV</a></p>']
    for item in gallery:
        html += [f"<section><h2>{item['title']}</h2><img src='{item['name']}.png'><p>{item['note']}</p><a href='{item['name']}.svg'>SVG</a> · <a href='{item['name']}.pdf'>PDF</a></section>"]
    html.append('</html>')
    (directory/'index.html').write_text('\n'.join(html))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',default='docs/experiments/h100_2026-10-12')
    args=parser.parse_args()
    directory=Path(args.directory)
    snapshot,server,get,groups,analysis=load(directory)
    gallery=figures(directory,get,groups,analysis)
    write_report(directory,snapshot,server,get,groups,analysis,gallery)
    print('Report:',directory/'report.md',len(gallery),'figures')
    print('Inventory:',analysis['outcome_counts'])
    print('Baseline common step:',analysis['baseline_common_step'])
    print('Baseline best:',[(r['model'],round(r['mean'],4)) for r in analysis['baseline'][:5]])
    print('MQAR common step:',analysis['mqar']['step'])


if __name__=='__main__':
    main()
