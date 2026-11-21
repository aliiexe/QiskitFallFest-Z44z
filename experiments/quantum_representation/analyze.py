"""Rescore saved OOF predictions and explain the standalone quantum comparison."""
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
from experiments.qnn_optimization.run import sha

LABELS=['alfalfa','maize','wheat']
NAMES=['first4_q4','selected4_q4','first6_q6']
DISPLAY=['First 4 PCs / 4q','Select 4 of 8 PCs / 4q','First 6 PCs / 6q']
COLORS=['#23779d','#c38a2c','#32825a']


def analyze(out):
    out=Path(out);figs=out/'figures';figs.mkdir(exist_ok=True)
    meta=json.loads((out/'metadata.json').read_text());assert meta['status']=='complete'
    tables,_,_,_=load_data();train=tables['train'];y=train.target.to_numpy()
    runs=json.loads((out/'cv_runs.json').read_text());seed_runs=json.loads((out/'seed_runs.json').read_text())
    selection=json.loads((out/'training_only_selection.json').read_text())
    assert len(runs)==27 and len(seed_runs)==9
    predictions={};rows=[]
    for r in seed_runs:
        name,seed=r['configuration'],r['seed'];p=pd.read_csv(out/f'{name}_seed_{seed}_oof.csv')
        np.testing.assert_array_equal(p.sample_id,train.sample_id);np.testing.assert_array_equal(p.target,y)
        assert score(y,p.prediction,LABELS)==r['metrics']
        probs=p[[f'p_{c}' for c in LABELS]].to_numpy()
        np.testing.assert_allclose(probs.sum(axis=1),1.,atol=1e-10,rtol=0)
        assert np.all(probs>=0)
        np.testing.assert_array_equal(np.asarray(LABELS)[probs.argmax(axis=1)],p.prediction)
        predictions[name,seed]=p
        subset=[f for f in runs if f['configuration']==name and f['seed']==seed]
        for f in subset:
            resources=f['resources'];calls=resources['objective_evaluations']
            assert calls==resources['gradient_evaluations']<=100
            passes=calls*(1+2*resources['trainable'])
            assert resources['native_operator_builds']==passes
            assert resources['logical_field_probability_evaluations']==passes*f['training_rows']
            baseline=next(a for a in runs if a['configuration']=='first4_q4' and a['seed']==seed and a['fold']==f['fold'])
            np.testing.assert_array_equal(f['initial_weights'][:12],baseline['initial_weights'])
        row={'configuration':name,'seed':seed,'macro_f1':r['metrics']['macro_f1'],'accuracy':r['metrics']['accuracy'],
             'cv_fit_seconds':sum(a['fit_seconds'] for a in subset),
             'gradient_calls':sum(a['resources']['gradient_evaluations'] for a in subset),
             'probability_passes':sum(a['resources']['native_operator_builds'] for a in subset),
             'logical_field_evaluations':sum(a['resources']['logical_field_probability_evaluations'] for a in subset)}
        for j,c in enumerate(LABELS):
            for key in ('precision','recall'):row[f'{key}_{c}']=r['metrics'][key][j]
        rows.append(row)
    df=pd.DataFrame(rows);df.to_csv(out/'per_seed_metrics.csv',index=False)
    summary={'scope':'Training-only model-selection evidence; no validation or test scoring',
             'seed_sd':'Initialization variation on the same folds, not a confidence interval',
             'selection':selection['selected_configuration'],'metrics':[],'paired_changes':{},
             'limitations':['Adaptive follow-up study on reused training data and folds',
                            'Stratified, not geographic cross-validation',
                            'Ideal exact simulation, not hardware performance',
                            'Width changes parameter count, gates and modulo-3 readout group proportions']}
    changed=[]
    for name in NAMES:
        group=df[df.configuration==name];item={'configuration':name}
        for c in df.columns[2:]:item[c+'_mean']=float(group[c].mean());item[c+'_sd']=float(group[c].std(ddof=0))
        item['confusion_sum_three_seeds']=np.asarray([r['metrics']['confusion'] for r in seed_runs if r['configuration']==name]).sum(axis=0).tolist()
        first=next(r for r in runs if r['configuration']==name)
        item['resources']={k:first['resources'][k] for k in ('qubits','trainable','depth','two_qubit_gates','state_dimension','readout_group_sizes','probability_passes_per_loss_and_gradient')}
        item['budget_caps_reached']=sum(r['resources']['objective_evaluations']==100 for r in runs if r['configuration']==name)
        item['fold_selected_components']=[r['selected_components'] for r in runs if r['configuration']==name and r['seed']==0]
        histories=[json.loads((out/f'{name}_seed_{r["seed"]}_fold_{r["fold"]}_history.json').read_text()) for r in runs if r['configuration']==name]
        item['final_training_loss_bits_mean']=float(np.mean([h[-1]['objective_bits'] for h in histories]))
        item['termination_messages']=sorted(set(r['termination']['message'] for r in runs if r['configuration']==name))
        summary['metrics'].append(item)
        if name!='first4_q4':
            details=[]
            for seed in (0,1,2):
                a=predictions['first4_q4',seed].prediction.to_numpy();b=predictions[name,seed].prediction.to_numpy()
                repaired=(a!=y)&(b==y);new=(a==y)&(b!=y)
                baseline=df[(df.configuration=='first4_q4')&(df.seed==seed)].iloc[0]
                cand=df[(df.configuration==name)&(df.seed==seed)].iloc[0]
                details.append({'seed':seed,'macro_f1_delta':float(cand.macro_f1-baseline.macro_f1),
                                'repaired':int(repaired.sum()),'newly_wrong':int(new.sum())})
                for mask,kind in ((repaired,'repaired'),(new,'newly wrong')):
                    for i in np.flatnonzero(mask):
                        changed.append({'configuration':name,'seed':seed,'sample_id':train.sample_id.iloc[i],
                                        'target':y[i],'NDVI':float(train.NDVI.iloc[i]),
                                        'baseline_prediction':a[i],'candidate_prediction':b[i],'change':kind,
                                        **{f'candidate_p_{c}':float(predictions[name,seed][f'p_{c}'].iloc[i]) for c in LABELS}})
            summary['paired_changes'][name]=details
    changes=pd.DataFrame(changed);changes.to_csv(out/'changed_field_decisions.csv',index=False)
    examples=[]
    for name in NAMES[1:]:
        for kind in ('repaired','newly wrong'):
            for crop in ('maize','wheat'):
                subset=changes[(changes.configuration==name)&(changes.change==kind)&(changes.target==crop)]
                counts=subset.groupby('sample_id').size().sort_values(ascending=False)
                for field in counts.head(2).index:
                    rows=subset[subset.sample_id==field]
                    examples.append({'configuration':name,'change':kind,'sample_id':field,'target':crop,
                                     'seed_count':int(counts[field]),'NDVI':float(rows.NDVI.iloc[0]),
                                     'baseline_predictions':rows.baseline_prediction.tolist(),
                                     'candidate_predictions':rows.candidate_prediction.tolist()})
    (out/'field_examples.json').write_text(json.dumps(examples,indent=2))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    pd.DataFrame([{k:v for k,v in r.items() if k not in ('resources','confusion_sum_three_seeds','fold_selected_components')} for r in summary['metrics']]).to_csv(out/'summary.csv',index=False)

    plt.rcParams.update({'font.size':10,'figure.dpi':140,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for j,(name,color) in enumerate(zip(NAMES,COLORS)):
        g=df[df.configuration==name].sort_values('seed');axes[0].bar(j,g.macro_f1.mean(),color=color,alpha=.4)
        for seed,marker in zip((0,1,2),('o','s','^')):axes[0].scatter(j+(seed-1)*.12,g.iloc[seed].macro_f1,marker=marker,color=color,s=45,label=f'Seed {seed}' if j==0 else None)
        axes[0].text(j,g.macro_f1.max()+.012,f'{g.macro_f1.mean():.4f}',ha='center')
        bar=axes[1].bar(j,g.cv_fit_seconds.mean(),color=color,alpha=.8)
        axes[1].bar_label(bar,fmt='%.1fs',padding=3)
    axes[0].set(ylabel='Pooled training-only OOF macro-F1',ylim=(.6,1),title='Quantum representation: each marker is one seed')
    axes[0].legend(loc='lower right')
    axes[1].set(ylabel='Mean total fit seconds across three folds',title='Cost of one complete CV pass')
    for ax in axes:ax.set(xticks=range(3),xticklabels=['First4 / 4q','Selected4 / 4q','First6 / 6q']);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.savefig(figs/'01_quantum_performance_runtime.png',bbox_inches='tight');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    for ax,key in zip(axes,('recall','precision')):
        for offset,r,color,label in zip((-.24,0,.24),summary['metrics'],COLORS,DISPLAY):
            bars=ax.bar(np.arange(3)+offset,[100*r[f'{key}_{c}_mean'] for c in LABELS],width=.22,color=color,label=label)
            ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=8)
        ax.set(xticks=range(3),xticklabels=LABELS,ylim=(0,105),ylabel='Percent, mean over seeds',title='Training-only crop '+key)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncols=3,fontsize=9)
    fig.savefig(figs/'02_quantum_crop_metrics.png',bbox_inches='tight');plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(12,4.3),layout='constrained')
    support=train.target.value_counts().reindex(LABELS).to_numpy()*3
    for ax,r,label in zip(axes,summary['metrics'],DISPLAY):
        cm=np.asarray(r['confusion_sum_three_seeds']);fraction=cm/support[:,None]
        ax.imshow(fraction,vmin=0,vmax=1,cmap='Blues')
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{cm[i,j]}\n{fraction[i,j]:.1%}',ha='center',va='center',color='white' if fraction[i,j]>.55 else '#18202a',fontsize=9)
        ax.set(xticks=range(3),yticks=range(3),xticklabels=LABELS,yticklabels=LABELS,xlabel='Predicted crop',ylabel='Actual crop',title=label)
    fig.suptitle('Same 914 training fields predicted out of fold once per seed; not validation',fontsize=11)
    fig.savefig(figs/'03_quantum_confusion.png',bbox_inches='tight');plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained',sharey=True)
    for ax,name,label,color in zip(axes,NAMES,DISPLAY,COLORS):
        for seed,style in zip((0,1,2),('-','--',':')):
            for fold in (0,1,2):
                h=json.loads((out/f'{name}_seed_{seed}_fold_{fold}_history.json').read_text())
                ax.plot([a['evaluation'] for a in h],[a['objective_bits'] for a in h],color=color,linestyle=style,alpha=.6,label=f'Seed {seed}, folds 0–2' if fold==0 else None)
        ax.set(xlabel='Loss-and-gradient evaluation',title=label);ax.grid(alpha=.15);ax.legend(fontsize=8)
    axes[0].set_ylabel('Training cross-entropy loss (bits; lower is better)')
    fig.suptitle('Training loss measures fit, not held-out classification quality',fontsize=11)
    fig.savefig(figs/'04_quantum_training_curves.png',bbox_inches='tight');plt.close(fig)

    # A small resource table is more readable than six complete circuit diagrams.
    fig,ax=plt.subplots(figsize=(10,2.8),layout='constrained');ax.axis('off')
    table_rows=[]
    for r,label in zip(summary['metrics'],DISPLAY):
        a=r['resources'];table_rows.append([label,a['qubits'],a['trainable'],a['two_qubit_gates'],a['depth'],a['probability_passes_per_loss_and_gradient'],f"{r['gradient_calls_mean']/3:.1f}"])
    table=ax.table(cellText=table_rows,colLabels=['Candidate','Qubits','Angles','CNOTs','Depth','Passes / gradient','Mean calls / fit'],cellLoc='center',loc='center',colWidths=[.27,.08,.08,.08,.08,.18,.18])
    table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.8)
    ax.set_title('Logical circuit resources and measured optimization work\nExact ideal simulation; physical routing and hardware noise excluded',pad=6)
    fig.savefig(figs/'05_quantum_resources.png',bbox_inches='tight');plt.close(fig)

    # Cases from the classical diagnostic show the gain and its tradeoff, with
    # actual PCA values plotted only descriptively (not used to preprocess CV).
    context=ROOT/'outputs/feature_representation_2026-11-21'
    fields=['r2_0786546ce39aeece','r2_1da35aa4b575dc80']
    pcs=pd.read_csv(context/'training_pca_scores.csv').set_index('sample_id')
    fig,axes=plt.subplots(1,2,figsize=(11,3.7),layout='constrained',sharey=True)
    for ax,field,title in zip(axes,fields,['Wheat: 6-PC SVM repairs an error\nin all three repeats','Wheat: 6-PC SVM introduces an error\nin all three repeats']):
        row=pcs.loc[field];vals=row[[f'PC{i}' for i in range(1,9)]].to_numpy(dtype=float)
        ax.bar(range(1,9),vals,color=['#23779d']*4+['#c38a2c']*4);ax.axhline(0,color='#777777',linewidth=.7)
        ndvi=float(train.loc[train.sample_id==field,'NDVI'].iloc[0])
        ax.set(xticks=range(1,9),xlabel='PCA component (full-training descriptive fit)',title=f'{title}\n{field}; NDVI={ndvi:.4f}');ax.tick_params(labelsize=9)
    axes[0].set_ylabel('Standardized PC value')
    fig.savefig(figs/'06_real_field_examples.png',bbox_inches='tight');plt.close(fig)
    assert all(sha(ROOT/p)==h for p,h in meta['protected_sha256'].items())
    full_reference_checks=[]
    if selection['selected_configuration']=='first4_q4':
        for seed in (0,1,2):
            with np.load(out/f'selected_seed_{seed}_weights.npz',allow_pickle=False) as fresh:
                with np.load(ROOT/f'outputs/entanglement_topology_2026-11-21/full_seed_{seed}_weights.npz',allow_pickle=False) as old:
                    np.testing.assert_allclose(fresh['final'],old['final'],atol=1e-10,rtol=0)
            full_reference_checks.append({'seed':seed,'prior_full_training_weights_match':True})
    result={'status':'passed','oof_files_rescored':9,'oof_rows_rescored':9*914,'cv_fits_checked':27,
            'reference_fold_checks':9,'same_initial_points_verified':True,'cost_counters_verified':True,
            'prior_files_unchanged':True,'full_training_reference_checks':full_reference_checks,
            'analysis_source_sha256':sha(Path(__file__)),
            'validation_or_test_scoring':False}
    (out/'artifact_verification.json').write_text(json.dumps(result,indent=2))
    print(df.to_string(index=False));print('Selected:',summary['selection']);print('Verification:',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);analyze(p.parse_args().results)
