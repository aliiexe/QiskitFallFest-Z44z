"""Replay saved models, analyze paired errors, and plot the topology study."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from challenge.data import load_data
from challenge.metrics import score
from challenge.coverage import metric_vector
from experiments.qnn_optimization.run import sha
from experiments.entanglement_topology.model import TopologyClassifier,EDGES

LABELS=['alfalfa','maize','wheat']
TOPOLOGIES=list(EDGES)
COLORS={'linear':'#176f9e','ring':'#cb8625','full':'#32825a'}


def analyze(out):
    out=Path(out);figs=out/'figures';figs.mkdir(exist_ok=True)
    meta=json.loads((out/'metadata.json').read_text());assert meta['status']=='complete'
    selection=json.loads((out/'training_only_selection.json').read_text());selected=selection['selected_topology']
    assert sha(out/'training_only_selection.json')==meta['selection_sha256']
    records=json.loads((out/'runs.json').read_text());assert len(records)==9
    tables,_,manifest,_=load_data();v=tables['validation'];y=v.target.to_numpy()
    prep=joblib.load(out/'preprocessing.joblib');B=prep.transform(v[manifest['features']].to_numpy())
    rows=[];predictions={};replay=[]
    for name in TOPOLOGIES:
        rs=sorted([r for r in records if r['topology']==name],key=lambda r:r['seed'])
        assert [r['seed'] for r in rs]==[0,1,2]
        for r in rs:
            seed=r['seed'];tag=f'{name}_seed_{seed}'
            p=pd.read_csv(out/f'{tag}_predictions.csv')
            assert len(p)==150 and p.sample_id.is_unique
            np.testing.assert_array_equal(p.sample_id,v.sample_id);np.testing.assert_array_equal(p.target,y)
            assert score(y,p.prediction,LABELS)==r['metrics']
            predictions[(name,seed)]=p.prediction.to_numpy()
            m=TopologyClassifier(topology=name)
            with np.load(out/f'{tag}_weights.npz',allow_pickle=False) as weights:
                np.testing.assert_array_equal(weights['initial'],np.random.default_rng(seed).uniform(-np.pi,np.pi,12))
                m.model=SimpleNamespace(weights=weights['final'].copy())
            restored=m.predict_proba(B)
            np.testing.assert_allclose(restored,p[[f'p_{crop}' for crop in LABELS]].to_numpy(),atol=1e-10,rtol=0)
            np.testing.assert_array_equal(m.labels[restored.argmax(axis=1)],p.prediction)
            history=json.loads((out/f'{tag}_history.json').read_text())
            assert len(history)==r['resources']['gradient_evaluations']==r['resources']['objective_evaluations']
            assert r['resources']['native_operator_builds']==25*len(history)
            replay.append({'topology':name,'seed':seed,'status':'passed','validation_rows':150,
                           'weights_sha256':sha(out/f'{tag}_weights.npz')})
            row={'topology':name,'seed':seed,'macro_f1':r['metrics']['macro_f1'],'accuracy':r['metrics']['accuracy'],
                 'training_macro_f1':r['training_metrics']['macro_f1'],'fit_seconds':r['fit_seconds'],
                 'total_seconds':sum(r[k] for k in ('fit_seconds','predict_seconds','preprocessing_seconds')),
                 'objective_evaluations':len(history),'gradient_evaluations':r['resources']['gradient_evaluations'],
                 'probability_passes':r['resources']['native_operator_builds'],
                 'logical_field_probability_evaluations':r['resources']['logical_field_probability_evaluations'],
                 'cx_gates':r['resources']['two_qubit_gates'],'depth':r['resources']['depth'],
                 'converged':r['termination']['success'],'final_training_loss_nats':history[-1]['objective_bits']*np.log(2)}
            for j,crop in enumerate(LABELS):
                for key in ('precision','recall'):row[f'{key}_{crop}']=r['metrics'][key][j]
            rows.append(row)
    df=pd.DataFrame(rows);df.to_csv(out/'per_seed_metrics.csv',index=False)
    summary={'selected_from_training_only':selected,'cv_seeds':[0,1,2],'cv_is_spatial':False,
             'all_full_training_runs':'9 fresh local Qiskit fits; linear reproduces previous Model D',
             'seed_sd':'Population SD across initializations; not uncertainty over new fields',
             'test_inference':False,'eight_qubit_or_reuploading_experiments':False,'metrics':[]}
    for name in TOPOLOGIES:
        group=df[df.topology==name];item={'topology':name}
        for c in df.columns[2:]:
            item[f'{c}_mean']=float(group[c].mean());item[f'{c}_sd']=float(group[c].std(ddof=0))
        cv=next(r for r in selection['candidates'] if r['topology']==name)
        item['cv_mean_oof_macro_f1']=cv['mean_oof_macro_f1'];item['cv_seed_sd_oof_macro_f1']=cv['seed_sd_oof_macro_f1']
        item['confusion_sum_three_seeds']=np.array([r['metrics']['confusion'] for r in records if r['topology']==name]).sum(axis=0).tolist()
        item['resources']=next(r['resources'] for r in records if r['topology']==name and r['seed']==0)
        summary['metrics'].append(item)
    field_review=v[['sample_id','target','NDVI']].copy();changes=[];paired={}
    for name in TOPOLOGIES:
        for seed in (0,1,2):field_review[f'{name}_seed_{seed}']=predictions[(name,seed)]
    for name in ('ring','full'):
        paired[name]=[]
        for seed in (0,1,2):
            a,b=predictions[('linear',seed)],predictions[(name,seed)]
            base=df[(df.topology=='linear')&(df.seed==seed)].iloc[0]
            candidate=df[(df.topology==name)&(df.seed==seed)].iloc[0]
            repaired=(a!=y)&(b==y);introduced=(a==y)&(b!=y)
            paired[name].append({'seed':seed,'macro_f1_delta':float(candidate.macro_f1-base.macro_f1),
                                 'accuracy_delta':float(candidate.accuracy-base.accuracy),
                                 'repaired':int(repaired.sum()),'newly_wrong':int(introduced.sum()),
                                 'recall_delta':{c:float(candidate[f'recall_{c}']-base[f'recall_{c}']) for c in LABELS}})
            for mask,kind in ((repaired,'repaired'),(introduced,'newly wrong'),((a!=y)&(b!=y)&(a!=b),'changed but still wrong')):
                for i in np.flatnonzero(mask):
                    changes.append({'topology':name,'seed':seed,'sample_id':v.sample_id.iloc[i],
                                    'target':y[i],'NDVI':float(v.NDVI.iloc[i]),'linear_prediction':a[i],
                                    'candidate_prediction':b[i],'change':kind})
    field_review.to_csv(out/'validation_field_review.csv',index=False)
    pd.DataFrame(changes).to_csv(out/'changed_predictions.csv',index=False)
    summary['paired_seed_changes']=paired
    summary['consistency']={name:{'positive_f1_seed_count':sum(r['macro_f1_delta']>0 for r in paired[name]),
                                  'negative_f1_seed_count':sum(r['macro_f1_delta']<0 for r in paired[name]),
                                  'repaired_decisions':sum(r['repaired'] for r in paired[name]),
                                  'newly_wrong_decisions':sum(r['newly_wrong'] for r in paired[name])} for name in paired}
    summary['validation_labels_identical_across_seeds']={name:all(np.array_equal(predictions[(name,0)],predictions[(name,seed)]) for seed in (1,2)) for name in TOPOLOGIES}
    for name in paired:
        summary['consistency'][name]['unique_fields_repaired_in_at_least_one_seed']=len({r['sample_id'] for r in changes if r['topology']==name and r['change']=='repaired'})
        summary['consistency'][name]['unique_fields_newly_wrong_in_at_least_one_seed']=len({r['sample_id'] for r in changes if r['topology']==name and r['change']=='newly wrong'})
    # Preselected motivating examples, plus the most consistently repaired maize/wheat fields.
    example_ids=['r2_614f27a98e1657c5','r2_0f0d5893227ca21d','r2_1f036cbcc9c718d1','r2_19678996b95020df','r2_47a4274b239df0f5']
    if changes:
        changed=pd.DataFrame(changes)
        for name in ('ring','full'):
            for crop in ('maize','wheat'):
                for kind in ('repaired','newly wrong'):
                    counts=changed[(changed.topology==name)&(changed.change==kind)&(changed.target==crop)].groupby('sample_id').size().sort_values(ascending=False)
                    example_ids.extend(counts.head(2).index.tolist())
    field_review[field_review.sample_id.isin(example_ids)].to_json(out/'field_examples.json',orient='records',indent=2)

    encode={c:i for i,c in enumerate(LABELS)};yi=np.array([encode[c] for c in y])
    mapping=json.loads((ROOT/'data/validation_groups.json').read_text())
    groups=np.array([mapping[s] for s in v.sample_id]);unique,index=np.unique(groups,return_inverse=True)
    rng=np.random.default_rng(2026);weights=[]
    while len(weights)<2000:
        w=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))[index]
        if all(w[yi==j].sum()>0 for j in (0,1,2)): weights.append(w)
    ip={name:np.array([[encode[c] for c in predictions[(name,seed)]] for seed in (0,1,2)]) for name in TOPOLOGIES}
    bootstrap={}
    for name in ('ring','full'):
        point=metric_vector(yi,ip[name],np.ones(150))-metric_vector(yi,ip['linear'],np.ones(150))
        boot=np.array([metric_vector(yi,ip[name],w)-metric_vector(yi,ip['linear'],w) for w in weights])
        lo,hi=np.quantile(boot,[.025,.975],axis=0)
        bootstrap[name]={key:{'difference':float(point[j]),'lower_95':float(lo[j]),'upper_95':float(hi[j])}
                         for j,key in enumerate(('macro_f1','accuracy','recall_alfalfa','recall_maize','recall_wheat'))}
    summary['paired_spatial_bootstrap']={'method':'2000 paired resamples of 18 validation spatial blocks, RNG 2026; two-sided percentile intervals, conditional on fitted seeds; no multiple-comparison adjustment',
                                         'candidate_minus_linear':bootstrap}
    assert all(sha(ROOT/p)==h for p,h in meta['protected_sha256'].items())
    verification={'status':'passed','models_replayed':9,'validation_rows_reproduced':1350,
                  'identical_initializations':True,'probabilities_labels_metrics_reproduced':True,
                  'gradient_pass_counts_verified':True,'previous_files_unchanged':True,'models':replay}
    (out/'artifact_verification.json').write_text(json.dumps(verification,indent=2))

    plt.rcParams.update({'font.size':11,'figure.dpi':140,'axes.spines.top':False,'axes.spines.right':False})
    # Simple directed connectivity sketches; native circuit drawings retain exact order.
    fig,axes=plt.subplots(1,3,figsize=(10,3.8),layout='constrained')
    positions=np.array([[-1,1],[1,1],[1,-1],[-1,-1]])
    for ax,name in zip(axes,TOPOLOGIES):
        for a,b in EDGES[name]:
            ax.annotate('',xy=positions[b],xytext=positions[a],
                        arrowprops={'arrowstyle':'->','color':COLORS[name],'lw':1.8,'shrinkA':17,'shrinkB':17},zorder=1)
        ax.scatter(positions[:,0],positions[:,1],s=650,color=COLORS[name],alpha=.18,zorder=2)
        for q,(x,z) in enumerate(positions):ax.text(x,z,f'q{q}',ha='center',va='center',zorder=3)
        resource=next(r['resources'] for r in summary['metrics'] if r['topology']==name)
        ax.set_title(f'{name.title()}: {len(EDGES[name])} CX per block')
        ax.text(0,-1.65,f'{resource["two_qubit_gates"]} total CX; depth {resource["depth"]}',ha='center')
        ax.set(xlim=(-1.7,1.7),ylim=(-1.9,1.6),aspect='equal');ax.axis('off')
    fig.suptitle('Same four qubits and 12 angles; change only the directed CX connections',fontsize=12)
    fig.savefig(figs/'01_connectivity.png',bbox_inches='tight');plt.close(fig)
    for name in TOPOLOGIES:
        fig=TopologyClassifier(topology=name).circuit.draw(output='mpl',fold=-1)
        fig.savefig(figs/f'circuit_{name}.png',dpi=150,bbox_inches='tight');plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(13,4.1),layout='constrained')
    for x,item in enumerate(summary['metrics']):
        name=item['topology'];g=df[df.topology==name].sort_values('seed')
        axes[0].bar(x,item['macro_f1_mean'],yerr=item['macro_f1_sd'],capsize=4,color=COLORS[name],alpha=.4)
        for seed,marker in zip((0,1,2),('o','s','^')):
            axes[0].scatter(x+(seed-1)*.13,g.iloc[seed].macro_f1,marker=marker,s=40,color=COLORS[name],label=f'seed {seed}' if x==0 else None,zorder=4)
        axes[0].text(x,max(g.macro_f1)+.035,f'{item["macro_f1_mean"]:.4f}',ha='center')
        axes[1].bar(x,item['total_seconds_mean'],color=COLORS[name],alpha=.8)
        axes[1].text(x,item['total_seconds_mean']+.12,f'{item["total_seconds_mean"]:.2f}s',ha='center')
        axes[2].bar(x,item['gradient_evaluations_mean'],color=COLORS[name],alpha=.8)
        axes[2].text(x,item['gradient_evaluations_mean']+1.2,f'{item["gradient_evaluations_mean"]:.1f}',ha='center')
    axes[0].set(ylabel='Validation macro-F1',ylim=(0,1.06),title='All seeds, mean +/- seed SD');axes[0].legend(loc='lower left')
    axes[1].set(ylabel='Preprocess + fit + validation prediction (s)',title='Mean runtime; same Qiskit engine')
    axes[1].set_ylim(0,max(r['total_seconds_mean'] for r in summary['metrics'])*1.25)
    axes[2].set(ylabel='Mean loss/gradient evaluations',title='25 probability passes per evaluation')
    axes[2].set_ylim(0,max(r['gradient_evaluations_mean'] for r in summary['metrics'])*1.3)
    for ax in axes:
        ax.set_xticks(range(3),TOPOLOGIES);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle(f'Training-only CV selected {selected}; every predefined validation result is shown',fontsize=12)
    fig.savefig(figs/'02_scores_runtime_gradient_work.png',bbox_inches='tight');plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,3.8),layout='constrained')
    for x,r in enumerate(selection['candidates']):
        name=r['topology'];ax.bar(x,r['mean_oof_macro_f1'],yerr=r['seed_sd_oof_macro_f1'],capsize=4,color=COLORS[name],alpha=.5)
        for seed,marker in zip((0,1,2),('o','s','^')):
            ax.scatter(x+(seed-1)*.13,r['seeds'][seed]['metrics']['macro_f1'],marker=marker,color=COLORS[name],s=35)
        ax.text(x,max(s['metrics']['macro_f1'] for s in r['seeds'])+.035,f'{r["mean_oof_macro_f1"]:.4f}',ha='center')
    ax.set(xticks=range(3),xticklabels=TOPOLOGIES,ylim=(0,1.05),ylabel='Training-only out-of-fold macro-F1',
           title=f'27 CV fits; selection locked before validation: {selected}')
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.savefig(figs/'03_training_only_selection.png',bbox_inches='tight');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.1),layout='constrained')
    for ax,key in zip(axes,('recall','precision')):
        for offset,name in zip((-.24,0,.24),TOPOLOGIES):
            item=next(r for r in summary['metrics'] if r['topology']==name)
            means=[item[f'{key}_{crop}_mean']*100 for crop in LABELS]
            bars=ax.bar(np.arange(3)+offset,means,width=.22,color=COLORS[name],alpha=.65,label=name)
            ax.bar_label(bars,fmt='%.1f',padding=4,fontsize=9)
            g=df[df.topology==name].sort_values('seed')
            for seed,marker in zip((0,1,2),('o','s','^')):
                vals=[g.iloc[seed][f'{key}_{crop}']*100 for crop in LABELS]
                ax.scatter(np.arange(3)+offset+(seed-1)*.045,vals,marker=marker,s=22,color=COLORS[name],edgecolors='#18202a',linewidths=.35,zorder=4)
        ax.set(xticks=range(3),xticklabels=LABELS,ylim=(0,112),ylabel='Percent; bars = means, dots = seeds',title='Per-crop '+key)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncols=3)
    fig.savefig(figs/'04_per_crop_metrics.png',bbox_inches='tight');plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(12,4.4),layout='constrained')
    for ax,name in zip(axes,TOPOLOGIES):
        cm=np.asarray(next(r['confusion_sum_three_seeds'] for r in summary['metrics'] if r['topology']==name))
        ax.imshow(cm/150,vmin=0,vmax=1,cmap='Blues')
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{cm[i,j]} / 150\n{cm[i,j]/150:.0%}',ha='center',va='center',color='white' if cm[i,j]>82 else '#18202a')
        ax.set(xticks=range(3),yticks=range(3),xticklabels=LABELS,yticklabels=LABELS,xlabel='Predicted crop',ylabel='Actual crop',title=name.title())
    fig.suptitle('Same 50 fields per crop evaluated with 3 seeds: 150 decisions per row',fontsize=12)
    fig.savefig(figs/'05_confusion_matrices.png',bbox_inches='tight');plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(12,6.7),layout='constrained',sharey=True)
    seed_colors=('#176f9e','#cb8625','#32825a');loss_stats={}
    for col,name in enumerate(TOPOLOGIES):
        loss_stats[name]=[]
        for seed,color in zip((0,1,2),seed_colors):
            h=pd.DataFrame(json.loads((out/f'{name}_seed_{seed}_history.json').read_text()))
            losses=h.objective_bits.to_numpy()*np.log(2);best=np.minimum.accumulate(losses)
            axes[0,col].plot(h.evaluation,best,color=color,label=f'seed {seed}')
            axes[1,col].plot(h.seconds,best,color=color,label=f'seed {seed}')
            loss_stats[name].append({'seed':seed,'best_training_loss_nats':float(losses.min()),
                                     'best_evaluation':int(h.evaluation.iloc[losses.argmin()])})
        axes[0,col].set(title=name.title(),xlabel='Loss/gradient evaluations (25 passes each)')
        axes[1,col].set_xlabel('Elapsed fitting time (seconds)')
        for row in (0,1):axes[row,col].legend();axes[row,col].grid(alpha=.15)
    axes[0,0].set_ylabel('Best training loss (nats; lower is better)');axes[1,0].set_ylabel('Best training loss (nats; lower is better)')
    fig.savefig(figs/'06_training_curves.png',bbox_inches='tight');plt.close(fig)
    summary['loss_diagnostics']=loss_stats

    fig,ax=plt.subplots(figsize=(9,3.8),layout='constrained')
    x=np.arange(6);paired_rows=[(name,r) for name in ('ring','full') for r in paired[name]]
    ax.bar(x-.18,[r['repaired'] for _,r in paired_rows],width=.34,color='#32825a',label='Repaired linear mistakes')
    ax.bar(x+.18,[r['newly_wrong'] for _,r in paired_rows],width=.34,color='#b35645',label='Introduced new mistakes')
    ax.set(xticks=x,xticklabels=[f'{name}\nseed {r["seed"]}' for name,r in paired_rows],ylabel='Validation fields',title='Paired error trade-offs against linear Model D')
    ax.legend();ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.savefig(figs/'07_error_changes.png',bbox_inches='tight');plt.close(fig)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    pd.DataFrame(summary['metrics']).drop(columns=['resources','confusion_sum_three_seeds']).to_csv(out/'summary.csv',index=False)
    print(pd.DataFrame(summary['metrics'])[['topology','cv_mean_oof_macro_f1','macro_f1_mean','macro_f1_sd','accuracy_mean',
                                          'recall_alfalfa_mean','recall_maize_mean','recall_wheat_mean','total_seconds_mean','gradient_evaluations_mean','probability_passes_mean','converged_mean']].to_string(index=False))
    print('Training-only selection:',selected)
    print('Paired seed changes:',json.dumps(paired,indent=2))
    print('Bootstrap:',json.dumps(summary['paired_spatial_bootstrap'],indent=2))
    print('PASSED: nine saved native Qiskit models reproduce all 1350 validation rows. Prior files unchanged.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('results',type=Path)
    analyze(parser.parse_args().results)
