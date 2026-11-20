"""Rescore the nine paired QNN runs, plot comparisons, and quantify trade-offs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from challenge.coverage import metric_vector
from challenge.data import load_data
from challenge.metrics import score

LABELS = ['alfalfa','maize','wheat']
SCALES = [1.0,0.5,0.25]
COLORS = {1.0:'#176f9e',0.5:'#ca8625',0.25:'#32825a'}


def analyze(out,baseline):
    out, baseline = Path(out),Path(baseline)
    figs = out/'figures'
    figs.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'figure.dpi':140,'axes.spines.top':False,'axes.spines.right':False})
    tables,_,manifest,_=load_data()
    val=tables['validation']
    records=json.loads((out/'runs.json').read_text())
    assert len(records)==9
    predictions={}
    rows=[]
    for scale in SCALES:
        selected=sorted([r for r in records if r['scale']==scale],key=lambda r:r['seed'])
        assert [r['seed'] for r in selected]==[0,1,2]
        for r in selected:
            seed=r['seed']
            p=pd.read_csv(out/f'scale_{scale:g}_seed_{seed}_predictions.csv')
            assert len(p)==150 and p.sample_id.is_unique and set(p.sample_id)==set(val.sample_id)
            p=p.set_index('sample_id').loc[val.sample_id].reset_index()
            np.testing.assert_array_equal(p.target,val.target)
            assert score(p.target,p.prediction,LABELS)==r['metrics']
            predictions[(scale,seed)]=p.prediction.to_numpy()
            assert r['training_rows']==914 and r['resources']['objective_evaluations']==100
            assert r['resources']['qubits']==4 and r['resources']['trainable']==8
            row={'scale':scale,'seed':seed,'origin':r['origin'],
                 'macro_f1':r['metrics']['macro_f1'],'accuracy':r['metrics']['accuracy'],
                 'correct_fields':int(np.trace(np.asarray(r['metrics']['confusion']))),
                 'fit_seconds':r['fit_seconds'],'predict_seconds':r['predict_seconds'],
                 'total_seconds':sum(r[k] for k in ('fit_seconds','predict_seconds','preprocessing_seconds'))}
            for i,label in enumerate(LABELS):
                for metric in ('recall','precision'):
                    row[f'{metric}_{label}']=r['metrics'][metric][i]
            rows.append(row)
    df=pd.DataFrame(rows)
    df.to_csv(out/'per_seed_metrics.csv',index=False)
    aggregate=[]
    numeric=[c for c in df if c not in ('scale','seed','origin')]
    for scale in SCALES:
        group=df[df.scale==scale]
        item={'scale':scale}
        for metric in numeric:
            item[f'{metric}_mean']=float(group[metric].mean())
            item[f'{metric}_sd']=float(group[metric].std(ddof=0))
        item['confusion_sum_three_seeds']=np.asarray([r['metrics']['confusion'] for r in records if r['scale']==scale]).sum(axis=0).tolist()
        aggregate.append(item)
    summary={'provenance':'All nine QNN results are LOCAL. Scale=1 reused the separately verified full baseline.',
             'class_order':LABELS,'metrics':aggregate,
             'sd_definition':'Population SD across 3 initialization seeds; not an interval over new fields.',
             'test_inference':False}
    summary['per_seed_deltas']={}
    y=val.target.to_numpy()
    changes=[]
    field_table=val[['sample_id','target']].copy()
    for scale in SCALES:
        for seed in (0,1,2):
            field_table[f'scale_{scale:g}_seed_{seed}']=predictions[(scale,seed)]
    for scale in (0.5,0.25):
        delta=[]
        for seed in (0,1,2):
            old=df[(df.scale==1)&(df.seed==seed)].iloc[0]
            new=df[(df.scale==scale)&(df.seed==seed)].iloc[0]
            basepred=predictions[(1.,seed)]
            newpred=predictions[(scale,seed)]
            repaired=(basepred!=y)&(newpred==y)
            introduced=(basepred==y)&(newpred!=y)
            delta.append({'seed':seed,'macro_f1':float(new.macro_f1-old.macro_f1),
                          'accuracy':float(new.accuracy-old.accuracy),
                          'repaired_fields':int(repaired.sum()),'newly_wrong_fields':int(introduced.sum()),
                          'recall_delta':{label:float(new[f'recall_{label}']-old[f'recall_{label}']) for label in LABELS}})
            for mask,kind in [(repaired,'repaired'),(introduced,'newly wrong')]:
                for i in np.flatnonzero(mask):
                    changes.append({'scale':scale,'seed':seed,'sample_id':val.sample_id.iloc[i],
                                    'target':y[i],'original_prediction':basepred[i],
                                    'candidate_prediction':newpred[i],'change':kind})
        summary['per_seed_deltas'][str(scale)]=delta
    field_table.to_csv(out/'validation_field_review.csv',index=False)
    pd.DataFrame(changes).to_csv(out/'changed_predictions.csv',index=False)

    # Paired spatial block resampling: keep each block and all seeds paired.
    # This is conditional on the three fitted models and is validation analysis.
    encoding={label:i for i,label in enumerate(LABELS)}
    yi=val.target.map(encoding).to_numpy()
    spatial=json.loads((ROOT/'data/validation_groups.json').read_text())
    groups=np.array([spatial[s] for s in val.sample_id])
    unique,inverse=np.unique(groups,return_inverse=True)
    rng=np.random.default_rng(2026)
    weights=[]
    while len(weights)<2000:
        w=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))[inverse]
        if all(w[yi==k].sum()>0 for k in range(3)):
            weights.append(w)
    integer_preds={scale:np.array([[encoding[v] for v in predictions[(scale,seed)]] for seed in (0,1,2)]) for scale in SCALES}
    bootstrap={}
    metric_names=['macro_f1','accuracy','recall_alfalfa','recall_maize','recall_wheat']
    for scale in (0.5,0.25):
        point=metric_vector(yi,integer_preds[scale],np.ones(150))-metric_vector(yi,integer_preds[1.],np.ones(150))
        boot=np.array([metric_vector(yi,integer_preds[scale],w)-metric_vector(yi,integer_preds[1.],w) for w in weights])
        lower,upper=np.quantile(boot,[.025,.975],axis=0)
        bootstrap[str(scale)]={name:{'difference':float(point[i]),'lower_95':float(lower[i]),'upper_95':float(upper[i])}
                               for i,name in enumerate(metric_names)}
    summary['paired_spatial_bootstrap']={'method':'2000 paired resamples of 18 validation spatial blocks, RNG seed 2026; two-sided percentile intervals; conditional on these 3 fitted seeds; no multiplicity correction',
                                          'spatial_blocks':len(unique),'candidate_minus_original':bootstrap}

    # Mathematical rule and actual field encodings, with no model alteration.
    with np.load(out/'prepared_inputs.npz',allow_pickle=False) as prepared:
        B=prepared['X_validation']
        np.testing.assert_array_equal(prepared['ids_validation'],val.sample_id.to_numpy())
    ids=['r2_614f27a98e1657c5','r2_0f0d5893227ca21d','r2_1f036cbcc9c718d1','r2_19678996b95020df']
    positions=[int(np.flatnonzero(val.sample_id.eq(s))[0]) for s in ids]
    examples=[]
    for idx in positions:
        for scale in SCALES:
            theta=np.pi/2*(1+np.tanh(scale*B[idx]))
            item={'sample_id':val.sample_id.iloc[idx],'target':y[idx],'scale':scale,'NDVI':float(val.NDVI.iloc[idx]),
                  **{f'PC{j+1}':float(B[idx,j]) for j in range(4)},
                  **{f'angle{j+1}':float(theta[j]) for j in range(4)},
                  **{f'prediction_seed_{seed}':predictions[(scale,seed)][idx] for seed in (0,1,2)}}
            examples.append(item)
    pd.DataFrame(examples).to_csv(out/'actual_field_encoding_examples.csv',index=False)
    summary['actual_field_examples']=examples
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    xx=np.linspace(-10,10,600)
    for scale in SCALES:
        axes[0].plot(xx,np.pi/2*(1+np.tanh(scale*xx)),color=COLORS[scale],label=f'scale {scale:g}')
        vals=np.pi/2*(1+np.tanh(scale*B[positions,0]))
        axes[1].plot(B[positions,0],vals,marker='o',linestyle='none',color=COLORS[scale],label=f'scale {scale:g}')
    axes[0].set(xlabel='Prepared PCA component z',ylabel='Rotation angle (radians)',title='Same formula; change only s inside tanh(s*z)')
    axes[1].set(xlabel='Actual PC1: three maize fields and one wheat field',ylabel='Rotation angle (radians; zoom near pi)',title='Actual extreme fields: zoom near the upper limit',
                ylim=(3.045,np.pi+.004),yticks=[3.06,3.08,3.10,3.12,np.pi],
                yticklabels=['3.06','3.08','3.10','3.12','pi'])
    axes[0].set_yticks([0,np.pi/2,np.pi],['0','pi / 2','pi'])
    for ax in axes:
        ax.grid(alpha=.15)
        ax.legend()
    fig.savefig(figs/'01_encoding_rule_and_actual_fields.png',bbox_inches='tight')
    plt.close(fig)

    baseline_records=json.loads((baseline/'runs.json').read_text())
    svm_f1=next(r['metrics']['macro_f1'] for r in baseline_records if r['model']=='reference' and r['seed']==0)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    for j,scale in enumerate(SCALES):
        a=next(r for r in aggregate if r['scale']==scale)
        group=df[df.scale==scale].sort_values('seed')
        axes[0].bar(j,a['macro_f1_mean'],yerr=a['macro_f1_sd'],capsize=5,width=.65,color=COLORS[scale],alpha=.35)
        for seed,marker in zip((0,1,2),('o','s','^')):
            axes[0].scatter(j+(seed-1)*.13,group.iloc[seed].macro_f1,marker=marker,s=48,
                            color=COLORS[scale],label=f'seed {seed}' if j==0 else None,zorder=4)
        axes[0].text(j,max(group.macro_f1)+.05,f'{a["macro_f1_mean"]:.4f}',ha='center')
        axes[1].bar(j,a['total_seconds_mean'],color=COLORS[scale],alpha=.8)
        axes[1].text(j,a['total_seconds_mean']+.8,f'{a["total_seconds_mean"]:.1f} s',ha='center')
    axes[0].axhline(svm_f1,color='#64748b',linestyle='--',linewidth=1,label=f'Local SVM: {svm_f1:.4f}')
    axes[0].set(ylabel='Validation macro-F1',ylim=(0,1.07),title='Mean +/- seed SD; dots show every run')
    axes[0].legend(loc='lower left',ncols=2)
    axes[1].set(ylabel='Preprocess + fit + validation prediction (seconds)',title='Mean runtime on this computer')
    axes[1].set_ylim(0,max(a['total_seconds_mean'] for a in aggregate)*1.2)
    for ax in axes:
        ax.set_xticks(range(3),['1.0 (original)','0.5','0.25'])
        ax.set_xlabel('Fixed input scale s')
        ax.grid(axis='y',alpha=.15)
        ax.set_axisbelow(True)
    fig.savefig(figs/'02_macro_f1_and_runtime.png',bbox_inches='tight')
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    for ax,key in zip(axes,('recall','precision')):
        for offset,scale in zip((-.24,0,.24),SCALES):
            a=next(r for r in aggregate if r['scale']==scale)
            means=[a[f'{key}_{crop}_mean']*100 for crop in LABELS]
            bars=ax.bar(np.arange(3)+offset,means,width=.22,color=COLORS[scale],alpha=.65,label=f'scale {scale:g}')
            ax.bar_label(bars,fmt='%.0f',padding=4,fontsize=9)
            group=df[df.scale==scale].sort_values('seed')
            for seed,marker in zip((0,1,2),('o','s','^')):
                vals=[group.iloc[seed][f'{key}_{crop}']*100 for crop in LABELS]
                ax.scatter(np.arange(3)+offset+(seed-1)*.045,vals,color=COLORS[scale],
                           marker=marker,s=23,edgecolors='#18202a',linewidths=.35,zorder=4)
        ax.set(xticks=range(3),xticklabels=LABELS,yticks=[0,25,50,75,100],ylim=(0,112),
               ylabel='Percent; bars = means, dots = individual seeds',title=f'Per-crop {key}')
        ax.grid(axis='y',alpha=.15)
        ax.set_axisbelow(True)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncols=3)
    fig.savefig(figs/'03_per_crop_recall_precision.png',bbox_inches='tight')
    plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(12,4.5),layout='constrained')
    for ax,scale in zip(axes,SCALES):
        a=next(r for r in aggregate if r['scale']==scale)
        cm=np.asarray(a['confusion_sum_three_seeds'])
        ax.imshow(cm/150,vmin=0,vmax=1,cmap='Blues')
        for i in range(3):
            for j in range(3):
                ax.text(j,i,f'{cm[i,j]} / 150\n{cm[i,j]/150:.0%}',ha='center',va='center',
                        color='white' if cm[i,j]/150>.55 else '#18202a')
        ax.set(xticks=range(3),yticks=range(3),xticklabels=LABELS,yticklabels=LABELS,
               xlabel='Predicted crop',ylabel='Actual crop',title=f'Input scale {scale:g}')
    fig.suptitle('Three runs of the SAME 150 validation fields: counts are 450 decisions, not new fields',fontsize=12)
    fig.savefig(figs/'04_confusion_comparison.png',bbox_inches='tight')
    plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained',sharey=True)
    loss_diagnostics={}
    for ax,scale in zip(axes,SCALES):
        stats=[]
        for seed,color in zip((0,1,2),('#176f9e','#ca8625','#32825a')):
            h=pd.DataFrame(json.loads((out/f'scale_{scale:g}_seed_{seed}_history.json').read_text()))
            loss=h.objective.to_numpy()*np.log(2)
            ax.plot(h.evaluation,np.minimum.accumulate(loss),label=f'seed {seed}',color=color)
            stats.append({'seed':seed,'best_loss_nats':float(loss.min()),
                          'best_at_80':float(loss[:80].min()),'best_evaluation':int(h.evaluation.iloc[loss.argmin()])})
        loss_diagnostics[str(scale)]=stats
        ax.set(xlabel='Optimizer evaluations',title=f'Input scale {scale:g}')
        ax.legend()
        ax.grid(alpha=.15)
    axes[0].set_ylabel('Best training cross-entropy (nats; lower is better)')
    fig.savefig(figs/'05_training_loss.png',bbox_inches='tight')
    plt.close(fig)
    summary['loss_diagnostics']=loss_diagnostics

    fig,ax=plt.subplots(figsize=(8,3.2),layout='constrained')
    for j,scale in enumerate((0.5,0.25)):
        b=bootstrap[str(scale)]['macro_f1']
        ax.plot([b['lower_95'],b['upper_95']],[j,j],color=COLORS[scale],linewidth=2)
        ax.scatter(b['difference'],j,color=COLORS[scale],s=65)
    ax.axvline(0,color='#64748b',linestyle='--')
    ax.set(yticks=[0,1],yticklabels=['0.5 versus original','0.25 versus original'],ylim=(-.7,1.7),
           xlabel='Change in mean validation macro-F1 (candidate minus original)',
           title='Paired spatial-block resampling: 95% percentile intervals')
    ax.grid(axis='x',alpha=.15)
    fig.savefig(figs/'06_paired_spatial_uncertainty.png',bbox_inches='tight')
    plt.close(fig)

    summary['selected_by_mean_validation_macro_f1']=float(max(aggregate,key=lambda a:a['macro_f1_mean'])['scale'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    pd.DataFrame(aggregate).drop(columns=['confusion_sum_three_seeds']).to_csv(out/'summary.csv',index=False)
    print(df[['scale','seed','macro_f1','accuracy','recall_alfalfa','recall_maize','recall_wheat','total_seconds']].to_string(index=False),flush=True)
    print('All 9 prediction sets rescored. Paired spatial analysis complete. No test predictions made.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    parser.add_argument('--baseline',type=Path,required=True)
    args=parser.parse_args()
    analyze(args.results,args.baseline)
