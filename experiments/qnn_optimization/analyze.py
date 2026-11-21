"""Verify completed Phase-1 records and generate the research figures."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from challenge.data import load_data
from challenge.metrics import score
from challenge.coverage import metric_vector
from experiments.qnn_optimization.model import NativeQiskitClassifier

LABELS=['alfalfa','maize','wheat']
NAMES={'A':'A: 1 block / COB100','B':'B: 1 block / L-BFGS','C':'C: 2 blocks / COB100',
       'D':'D: 2 blocks / L-BFGS','E':'E: 1 block / COB1700'}
COLORS={'A':'#607b94','B':'#ca8625','C':'#9576a8','D':'#238462','E':'#8e713f'}


def analyze(out):
    out=Path(out);figs=out/'figures';figs.mkdir(exist_ok=True)
    meta=json.loads((out/'metadata.json').read_text());assert meta['status']=='complete'
    records=json.loads((out/'runs.json').read_text());assert len(records)==15
    selection=json.loads((out/'training_only_selection.json').read_text());selected=selection['selected_configuration']
    tables,_,manifest,_=load_data();v=tables['validation'];y=v.target.to_numpy()
    ref=json.loads((out/'svm.json').read_text())
    original=json.loads((ROOT/'outputs/full_baseline_914_2026-11-20/runs.json').read_text())
    rows=[];predictions={};agreement=[]
    for r in records:
        name,seed=r['configuration'],r['seed']
        assert r['selection_sha256']==meta['selection_sha256']
        p=pd.read_csv(out/f'{name}_seed_{seed}_predictions.csv')
        assert len(p)==150 and p.sample_id.is_unique
        np.testing.assert_array_equal(p.sample_id,v.sample_id)
        np.testing.assert_array_equal(p.target,y)
        assert score(y,p.prediction,LABELS)==r['metrics']
        predictions[(name,seed)]=p.prediction.to_numpy()
        row={'configuration':name,'seed':seed,'macro_f1':r['metrics']['macro_f1'],
             'accuracy':r['metrics']['accuracy'],'training_macro_f1':r['training_metrics']['macro_f1'],
             'fit_seconds':r['fit_seconds'],'total_seconds':sum(r[k] for k in ('fit_seconds','predict_seconds','preprocessing_seconds')),
             'evaluations':r['resources']['objective_evaluations'],
             'gradient_evaluations':r['resources']['gradient_evaluations'],
             'probability_passes':r['resources']['native_operator_builds'],
             'logical_field_evaluations':r['resources']['logical_field_probability_evaluations']}
        for j,crop in enumerate(LABELS):
            for metric in ('precision','recall'):
                row[f'{metric}_{crop}']=r['metrics'][metric][j]
        rows.append(row)
        if name=='A':
            old=next(o for o in original if o['model']=='qnn' and o['seed']==seed)
            oldp=pd.read_csv(ROOT/f'outputs/full_baseline_914_2026-11-20/qnn_{seed}_predictions.csv')
            same=bool(np.array_equal(oldp.prediction,p.prediction))
            assert same and old['metrics']==r['metrics']
            agreement.append({'seed':seed,'all_labels_and_metrics_match_prior_local_original':same})
    df=pd.DataFrame(rows);df.to_csv(out/'per_seed_metrics.csv',index=False)
    summary={'selected_from_training_only':selected,'selection_cv_seed':0,'selection_cv_is_spatial':False,
             'local_qiskit_runs':15,'original_baseline_agreement':agreement,'metrics':[],
             'original_svm':ref,'historical_independent_prototype_equivalence':'not established; source unavailable',
             'test_inference':False,'seed_sd':'Population SD, ddof=0; not confidence over new fields'}
    for name in NAMES:
        group=df[df.configuration==name];item={'configuration':name}
        for c in df.columns[2:]:
            item[f'{c}_mean']=float(group[c].mean());item[f'{c}_sd']=float(group[c].std(ddof=0))
        item['cv_oof_macro_f1']=next(c['oof_metrics']['macro_f1'] for c in selection['candidates'] if c['configuration']==name)
        item['confusion_sum_three_seeds']=np.asarray([r['metrics']['confusion'] for r in records if r['configuration']==name]).sum(axis=0).tolist()
        item['resources']=next(r['resources'] for r in records if r['configuration']==name and r['seed']==0)
        summary['metrics'].append(item)

    field_review=v[['sample_id','target']].copy();changes=[]
    for name in NAMES:
        for seed in (0,1,2): field_review[f'{name}_seed_{seed}']=predictions[(name,seed)]
    for seed in (0,1,2):
        a,d=predictions[('A',seed)],predictions[(selected,seed)]
        repaired=(a!=y)&(d==y);lost=(a==y)&(d!=y)
        for mask,kind in ((repaired,'repaired'),(lost,'newly wrong')):
            for i in np.flatnonzero(mask):
                changes.append({'seed':seed,'sample_id':v.sample_id.iloc[i],'target':y[i],
                                'baseline_prediction':a[i],'selected_prediction':d[i],'change':kind})
    field_review.to_csv(out/'validation_field_review.csv',index=False)
    pd.DataFrame(changes).to_csv(out/'selected_changed_predictions.csv',index=False)
    summary['selected_prediction_changes']={kind:sum(r['change']==kind for r in changes) for kind in ('repaired','newly wrong')}

    # Paired validation spatial-block resampling, conditional on fitted seeds.
    encode={crop:i for i,crop in enumerate(LABELS)};yi=np.array([encode[c] for c in y])
    mapping=json.loads((ROOT/'data/validation_groups.json').read_text())
    groups=np.array([mapping[s] for s in v.sample_id]);unique,index=np.unique(groups,return_inverse=True)
    rng=np.random.default_rng(2026);weights=[]
    while len(weights)<2000:
        w=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))[index]
        if all(w[yi==j].sum()>0 for j in (0,1,2)): weights.append(w)
    psel=np.array([[encode[c] for c in predictions[(selected,seed)]] for seed in (0,1,2)])
    pa=np.array([[encode[c] for c in predictions[('A',seed)]] for seed in (0,1,2)])
    svmp=pd.read_csv(out/'svm_predictions.csv');np.testing.assert_array_equal(svmp.sample_id,v.sample_id)
    psvm=np.array([[encode[c] for c in svmp.prediction]]*3)
    intervals={}
    for name,base in [('original_qnn',pa),('original_svm',psvm)]:
        delta=metric_vector(yi,psel,np.ones(150))-metric_vector(yi,base,np.ones(150))
        boot=np.array([metric_vector(yi,psel,w)-metric_vector(yi,base,w) for w in weights])
        lo,hi=np.quantile(boot,[.025,.975],axis=0)
        intervals[name]={k:{'difference':float(delta[j]),'lower_95':float(lo[j]),'upper_95':float(hi[j])}
                         for j,k in enumerate(('macro_f1','accuracy','recall_alfalfa','recall_maize','recall_wheat'))}
    summary['paired_spatial_bootstrap']={'method':'2000 paired resamples of 18 spatial validation blocks, RNG 2026; two-sided percentile intervals; conditional on fitted models, not test performance',
                                         'comparisons':intervals}

    plt.rcParams.update({'font.size':10,'figure.dpi':140,'axes.spines.top':False,'axes.spines.right':False})
    for reps in (1,2):
        m=NativeQiskitClassifier(reps=reps)
        fig=m.circuit.draw(output='mpl',fold=-1)
        fig.savefig(figs/f'01_circuit_{reps}_blocks.png',dpi=150,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    for x,item in enumerate(summary['metrics']):
        name=item['configuration'];group=df[df.configuration==name].sort_values('seed')
        axes[0].bar(x,item['macro_f1_mean'],yerr=item['macro_f1_sd'],capsize=4,color=COLORS[name],alpha=.5)
        for seed,marker in zip((0,1,2),('o','s','^')):
            axes[0].scatter(x+(seed-1)*.14,group.iloc[seed].macro_f1,marker=marker,s=30,color=COLORS[name])
        axes[0].text(x,max(group.macro_f1)+.025,f'{item["macro_f1_mean"]:.4f}',ha='center',fontsize=9)
        axes[1].bar(x,item['total_seconds_mean'],color=COLORS[name],alpha=.8)
        axes[1].text(x,item['total_seconds_mean']+.12,f'{item["total_seconds_mean"]:.2f}s',ha='center',fontsize=9)
    axes[0].axhline(ref['metrics']['macro_f1'],color='#626262',linestyle='--',linewidth=1,label='Original SVM')
    axes[0].set(ylabel='Validation macro-F1',ylim=(0,1),title=f'All seeds shown; training-CV choice = {selected}')
    axes[0].legend(loc='lower left')
    axes[1].set(ylabel='Preprocess + fit + validation prediction (seconds)',title='Same native Qiskit caching engine for A-E')
    axes[1].set_ylim(0,max(r['total_seconds_mean'] for r in summary['metrics'])*1.25)
    for ax in axes:
        ax.set_xticks(range(5),list(NAMES));ax.set_xlabel('Predeclared configuration');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.savefig(figs/'02_macro_f1_runtime.png',bbox_inches='tight');plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,3.7),layout='constrained')
    bars=ax.bar(list(NAMES),[r['cv_oof_macro_f1'] for r in summary['metrics']],color=[COLORS[n] for n in NAMES])
    ax.bar_label(bars,fmt='%.4f',padding=3)
    ax.set(ylim=(0,1),ylabel='Training-only out-of-fold macro-F1',title=f'Configuration selection BEFORE validation: {selected}')
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.savefig(figs/'03_training_only_selection.png',bbox_inches='tight');plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(12,7.6),layout='constrained')
    for ax,name in zip(axes.flat,list(NAMES)+['SVM']):
        cm=np.asarray(next(r['confusion_sum_three_seeds'] for r in summary['metrics'] if r['configuration']==name)) if name!='SVM' else np.asarray(ref['metrics']['confusion'])*3
        ax.imshow(cm/150,vmin=0,vmax=1,cmap='Blues')
        for i in range(3):
            for j in range(3): ax.text(j,i,f'{cm[i,j]}\n{cm[i,j]/150:.0%}',ha='center',va='center',color='white' if cm[i,j]>82 else '#18202a',fontsize=9)
        ax.set(xticks=range(3),yticks=range(3),xticklabels=LABELS,yticklabels=LABELS,
               xlabel='Predicted crop',ylabel='Actual crop',title=NAMES.get(name,'Original SVM'))
    fig.suptitle('Same 150 fields repeated over 3 seeds: 150 decisions per crop, not new independent fields',fontsize=11)
    fig.savefig(figs/'04_confusion_all_configurations.png',bbox_inches='tight');plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(11,4.2),layout='constrained')
    for ax,name in zip(axes,('A',selected,'SVM')):
        cm=np.asarray(next(r['confusion_sum_three_seeds'] for r in summary['metrics'] if r['configuration']==name)) if name!='SVM' else np.asarray(ref['metrics']['confusion'])*3
        ax.imshow(cm/150,vmin=0,vmax=1,cmap='Blues')
        for i in range(3):
            for j in range(3): ax.text(j,i,f'{cm[i,j]} / 150\n{cm[i,j]/150:.0%}',ha='center',va='center',color='white' if cm[i,j]>82 else '#18202a',fontsize=9)
        ax.set(xticks=range(3),yticks=range(3),xticklabels=LABELS,yticklabels=LABELS,
               xlabel='Predicted crop',ylabel='Actual crop',title={'A':'Original QNN',selected:'Selected QNN '+selected,'SVM':'Original SVM'}[name])
    fig.suptitle('Same 50 fields per crop, evaluated three times: 150 decisions per row',fontsize=11)
    fig.savefig(figs/'07_selected_confusion.png',bbox_inches='tight');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,key in zip(axes,('recall','precision')):
        for offset,name in zip((-.24,0,.24),('A',selected,'SVM')):
            means=[next(r[f'{key}_{c}_mean'] for r in summary['metrics'] if r['configuration']==name)*100 for c in LABELS] if name!='SVM' else np.array(ref['metrics'][key])*100
            bars=ax.bar(np.arange(3)+offset,means,width=.22,color=COLORS.get(name,'#999999'),label=name)
            ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
        ax.set(xticks=range(3),xticklabels=LABELS,ylim=(0,110),ylabel='Percent (mean over seeds)',title='Per-crop '+key)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncols=3)
    fig.savefig(figs/'05_selected_crop_tradeoffs.png',bbox_inches='tight');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for name in NAMES:
        h=pd.DataFrame(json.loads((out/f'{name}_seed_0_history.json').read_text()))
        loss=np.minimum.accumulate(h.objective_bits)*np.log(2)
        axes[0].plot(h.evaluation,loss,color=COLORS[name],label=name)
        axes[1].plot(h.probability_passes,loss,color=COLORS[name],label=name)
    axes[0].set(xlabel='Objective evaluations (log scale)',xscale='log',title='Seed 0: optimizer attempts')
    axes[1].set(xlabel='Logical probability passes (log scale)',xscale='log',title='Seed 0: account for gradient work')
    for ax in axes:
        ax.set_ylabel('Best training cross-entropy (nats)');ax.legend();ax.grid(alpha=.15)
    fig.savefig(figs/'06_loss_and_work.png',bbox_inches='tight');plt.close(fig)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    pd.DataFrame(summary['metrics']).drop(columns=['confusion_sum_three_seeds','resources']).to_csv(out/'summary.csv',index=False)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('metrics',)},indent=2))
    print(pd.DataFrame(summary['metrics'])[['configuration','cv_oof_macro_f1','macro_f1_mean','macro_f1_sd','accuracy_mean','recall_alfalfa_mean','recall_maize_mean','recall_wheat_mean','total_seconds_mean','probability_passes_mean']].to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('results',type=Path)
    analyze(parser.parse_args().results)
