"""Analyze training-only representation evidence; never score validation."""
import argparse
from collections import Counter
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
from sklearn.dummy import DummyClassifier
from challenge.data import load_data
from challenge.metrics import score
from experiments.qnn_optimization.run import sha
from experiments.feature_representation.models import VIEWS

LABELS=['alfalfa','maize','wheat']
LEARNERS=['logistic','svm_fixed','svm_nested']
COLORS={'alfalfa':'#23865d','maize':'#c98919','wheat':'#596fb0'}


def analyze(out):
    out=Path(out);figs=out/'figures';figs.mkdir(exist_ok=True)
    meta=json.loads((out/'metadata.json').read_text());assert meta['status']=='complete'
    tables,_,manifest,_=load_data();t=tables['train'];F=manifest['features'];y=t.target.to_numpy()
    records=json.loads((out/'repeat_runs.json').read_text());folds=json.loads((out/'fold_runs.json').read_text())
    assert len(records)==63 and len(folds)==189
    rows=[];preds={}
    for r in records:
        name,view,repeat=r['learner'],r['view'],r['repeat']
        p=pd.read_csv(out/f'{name}_{view}_repeat_{repeat}_oof.csv')
        assert len(p)==914 and p.sample_id.is_unique
        np.testing.assert_array_equal(p.sample_id,t.sample_id);np.testing.assert_array_equal(p.target,y)
        assert score(y,p.prediction,LABELS)==r['metrics']
        preds[(name,view,repeat)]=p.prediction.to_numpy()
        row={'learner':name,'view':view,'repeat':repeat,'macro_f1':r['metrics']['macro_f1'],
             'accuracy':r['metrics']['accuracy'],'fit_and_search_seconds':r['fit_and_search_seconds']}
        for j,crop in enumerate(LABELS):
            for key in ('precision','recall'):row[f'{key}_{crop}']=r['metrics'][key][j]
        rows.append(row)
    df=pd.DataFrame(rows);df.to_csv(out/'per_repeat_metrics.csv',index=False)
    summary={'scope':'training-only repeated nested CV; NO validation/test evaluation',
             'repeat_sd':'Variation across overlapping CV partitions; not an independent-sample standard error',
             'metrics':[],'paired_differences':{},'cv_is_spatial':False}
    for learner in LEARNERS:
        for view in VIEWS:
            group=df[(df.learner==learner)&(df.view==view)];item={'learner':learner,'view':view}
            for c in df.columns[3:]:
                item[f'{c}_mean']=float(group[c].mean());item[f'{c}_sd']=float(group[c].std(ddof=0))
            item['confusion_sum_three_repeats']=np.array([r['metrics']['confusion'] for r in records if r['learner']==learner and r['view']==view]).sum(axis=0).tolist()
            summary['metrics'].append(item)
        base=df[(df.learner==learner)&(df.view=='pca4')].sort_values('repeat')
        summary['paired_differences'][learner]={}
        for view in ('pca6','pca8','selected4of8'):
            candidate=df[(df.learner==learner)&(df.view==view)].sort_values('repeat')
            summary['paired_differences'][learner][view]=[
                {'repeat':int(a['repeat']),'macro_f1_delta':float(b.macro_f1-a.macro_f1),
                 'recall_delta':{c:float(b[f'recall_{c}']-a[f'recall_{c}']) for c in LABELS}}
                for (_,a),(_,b) in zip(base.iterrows(),candidate.iterrows())]
    selected_records=[r for r in folds if r['view']=='selected4of8']
    summary['component_selection_frequency']={learner:dict(Counter(pc for r in selected_records if r['learner']==learner for pc in r['selected_components'])) for learner in LEARNERS}
    # Dummy references computed only on the exact outer training folds.
    assignments=pd.read_csv(out/'outer_fold_assignments.csv');dummy=[]
    for repeat in (0,1,2):
        a=assignments[assignments.repeat==repeat]
        for strategy in ('most_frequent','stratified'):
            p=np.empty(len(y),dtype=object)
            for fold in (0,1,2):
                held=np.flatnonzero(a.fold.to_numpy()==fold);fit=np.flatnonzero(a.fold.to_numpy()!=fold)
                d=DummyClassifier(strategy=strategy,random_state=2026+10*repeat+fold).fit(np.zeros((len(fit),1)),y[fit])
                p[held]=d.predict(np.zeros((len(held),1)))
            dummy.append({'repeat':repeat,'strategy':strategy,'metrics':score(y,p,LABELS)})
    summary['dummy_training_only_references']=dummy
    changes=[];examples=[]
    for view in ('pca6','pca8','selected4of8'):
        for repeat in (0,1,2):
            a=preds[('svm_nested','pca4',repeat)];b=preds[('svm_nested',view,repeat)]
            for mask,kind in (((a!=y)&(b==y),'repaired'),((a==y)&(b!=y),'newly wrong')):
                for i in np.flatnonzero(mask):
                    changes.append({'view':view,'repeat':repeat,'sample_id':t.sample_id.iloc[i],'target':y[i],
                                    'NDVI':float(t.NDVI.iloc[i]),'pca4_prediction':a[i],'candidate_prediction':b[i],'change':kind})
    changed=pd.DataFrame(changes);changed.to_csv(out/'training_oof_changed_predictions.csv',index=False)
    for view in ('pca6','pca8'):
        for kind in ('repaired','newly wrong'):
            for crop in ('maize','wheat'):
                group=changed[(changed.view==view)&(changed.change==kind)&(changed.target==crop)]
                counts=group.groupby('sample_id').size().sort_values(ascending=False)
                for field in counts.head(2).index:
                    examples.append({'view':view,'change':kind,'repeat_count':int(counts[field]),
                                     'sample_id':field,'target':crop,'NDVI':float(t.loc[t.sample_id==field,'NDVI'].iloc[0])})
    (out/'field_examples.json').write_text(json.dumps(examples,indent=2))
    summary['changed_decisions']={view:{kind:int(((changed.view==view)&(changed.change==kind)).sum()) for kind in ('repaired','newly wrong')} for view in ('pca6','pca8','selected4of8')}
    diag=json.loads((out/'pca_diagnostics.json').read_text());summary['pca_variance']=[{k:r[k] for k in ('component','variance_ratio','cumulative_variance','eta2_all_crops','eta2_maize_wheat')} for r in diag['components']]
    summary['selected_classical_representation']=json.loads((out/'training_only_selection.json').read_text())['selected_representation']
    assert all(sha(ROOT/p)==h for p,h in meta['protected_sha256'].items())
    (out/'verification.json').write_text(json.dumps({'status':'passed','oof_files_rescored':63,'oof_rows_rescored':63*914,
                                                    'fold_runs':189,'previous_files_unchanged':True,
                                                    'full_PCA_reconstruction':'passed','validation_or_test_scoring':False},indent=2))

    plt.rcParams.update({'font.size':10,'figure.dpi':140,'axes.spines.top':False,'axes.spines.right':False})
    variance=np.array([r['variance_ratio'] for r in diag['components']]);pc=np.arange(1,14)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].bar(pc,variance*100,color=['#23779d' if j<=4 else '#c38a2c' if j<=8 else '#9ba4ac' for j in pc])
    axes[0].set(yscale='log',xticks=pc,xlabel='PCA component',ylabel='Explained variance (%) -- logarithmic scale',title='Complete decomposition of 13 standardized features')
    axes[1].plot(pc,np.cumsum(variance)*100,marker='o',color='#23779d')
    for k in (4,6,8):
        val=variance[:k].sum()*100;axes[1].scatter(k,val,s=55,color='#c38a2c',zorder=4)
        axes[1].annotate(f'{k} PCs: {val:.2f}%',xy=(k,val),xytext=(k,82-(k-4)*7),arrowprops={'arrowstyle':'->','color':'#666666'},ha='center')
    axes[1].set(xticks=pc,xlabel='Number of retained components',ylabel='Cumulative explained variance (%)',ylim=(45,103),title='High variance retention does not guarantee crop separation')
    for ax in axes:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.savefig(figs/'01_complete_pca_variance.png',bbox_inches='tight');plt.close(fig)

    fig,axes=plt.subplots(4,4,figsize=(12,9),layout='constrained')
    for ax,feature in zip(axes.flat,F):
        values=[t.loc[t.target==crop,feature].to_numpy() for crop in LABELS]
        b=ax.boxplot(values,patch_artist=True,showfliers=True,flierprops={'markersize':2,'alpha':.4})
        for patch,crop in zip(b['boxes'],LABELS):patch.set_facecolor(COLORS[crop]);patch.set_alpha(.5)
        ax.set(xticks=(1,2,3),xticklabels=['Alf','Maize','Wheat'],title=feature)
    for ax in list(axes.flat)[len(F):]:ax.axis('off')
    fig.suptitle('Original training features: separate axes retain each feature\'s numerical scale',fontsize=12)
    fig.savefig(figs/'02_original_feature_distributions.png',bbox_inches='tight');plt.close(fig)

    scores=pd.read_csv(out/'training_pca_scores.csv')
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for ax,(a,b) in zip(axes,((1,2),(5,6),(7,8))):
        for crop in LABELS:
            rows=scores.target==crop;ax.scatter(scores.loc[rows,f'PC{a}'],scores.loc[rows,f'PC{b}'],s=9,alpha=.45,color=COLORS[crop],label=crop)
        ax.set(xlabel=f'Standardized PC{a}',ylabel=f'Standardized PC{b}',title=f'Training components {a} and {b}')
        ax.grid(alpha=.12)
    axes[0].legend(markerscale=2)
    fig.suptitle('Descriptive training views; overlap in two dimensions does not settle predictive value',fontsize=11)
    fig.savefig(figs/'03_class_separation.png',bbox_inches='tight');plt.close(fig)

    load=pd.read_csv(out/'pca_loadings.csv',index_col=0)
    fig,ax=plt.subplots(figsize=(10,4.4),layout='constrained')
    image=ax.imshow(load.iloc[4:8].to_numpy(),cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto')
    for i in range(4):
        for j in range(13):ax.text(j,i,f'{load.iloc[i+4,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if abs(load.iloc[i+4,j])>.65 else '#18202a')
    ax.set(xticks=range(13),xticklabels=F,yticks=range(4),yticklabels=['PC5','PC6','PC7','PC8'],title='Later PCs describe small contrasts among standardized bands and NDVI')
    fig.colorbar(image,ax=ax,label='PCA coefficient (sign arbitrary; not causal feature importance)')
    fig.savefig(figs/'04_late_component_loadings.png',bbox_inches='tight');plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained',sharey=True)
    for ax,learner in zip(axes,LEARNERS):
        for view,k in (('pca4',4),('pca6',6),('pca8',8),('pca13',13)):
            item=next(r for r in summary['metrics'] if r['learner']==learner and r['view']==view)
            group=df[(df.learner==learner)&(df.view==view)].sort_values('repeat')
            ax.errorbar(k,item['macro_f1_mean'],yerr=item['macro_f1_sd'],marker='o',color='#23779d',capsize=4)
            ax.text(k,item['macro_f1_mean']+.015,f'{item["macro_f1_mean"]:.4f}',ha='center',fontsize=9)
        ax.set(xticks=[4,6,8,13],xlabel='Number of PCA components',ylim=(.80,.96),title={'logistic':'Logistic regression (fixed)','svm_fixed':'SVM (fixed original settings)','svm_nested':'SVM (nested training-only tuning)'}[learner])
        ax.grid(alpha=.15)
    axes[0].set_ylabel('Training-only out-of-fold macro-F1; mean +/- split SD')
    fig.savefig(figs/'05_components_vs_cv_performance.png',bbox_inches='tight');plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    for j,view in enumerate(VIEWS):
        g=df[(df.learner=='svm_nested')&(df.view==view)].sort_values('repeat');ax.bar(j,g.macro_f1.mean(),color='#8fb9ce')
        for repeat,marker in zip((0,1,2),('o','s','^')):ax.scatter(j+(repeat-1)*.12,g.iloc[repeat].macro_f1,marker=marker,s=35,color='#17658b')
        ax.text(j,g.macro_f1.max()+.025,f'{g.macro_f1.mean():.4f}',ha='center',fontsize=9)
    ax.set(xticks=range(len(VIEWS)),xticklabels=VIEWS,ylim=(0,1.02),ylabel='Training-only OOF macro-F1',title='Nested SVM controls: later PCs alone, selected inputs, and all original information')
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.savefig(figs/'06_representation_controls.png',bbox_inches='tight');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.1),layout='constrained')
    for ax,key in zip(axes,('recall','precision')):
        for offset,view,color in zip((-.24,0,.24),('pca4','pca6','pca8'),('#23779d','#c48b2d','#32825a')):
            r=next(r for r in summary['metrics'] if r['learner']=='svm_nested' and r['view']==view)
            bars=ax.bar(np.arange(3)+offset,[r[f'{key}_{c}_mean']*100 for c in LABELS],width=.22,color=color,label=view)
            ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
        ax.set(xticks=range(3),xticklabels=LABELS,ylim=(0,105),ylabel='Percent (mean across CV repeats)',title='Nested-SVM crop '+key)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncols=3)
    fig.savefig(figs/'07_crop_metrics.png',bbox_inches='tight');plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(12,4.4),layout='constrained')
    support=t.target.value_counts().reindex(LABELS).to_numpy()*3
    for ax,view in zip(axes,('pca4','pca6','pca8')):
        cm=np.asarray(next(r['confusion_sum_three_repeats'] for r in summary['metrics'] if r['learner']=='svm_nested' and r['view']==view));fractions=cm/support[:,None]
        ax.imshow(fractions,vmin=0,vmax=1,cmap='Blues')
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{cm[i,j]}\n{fractions[i,j]:.1%}',ha='center',va='center',color='white' if fractions[i,j]>.55 else '#18202a',fontsize=9)
        ax.set(xticks=range(3),yticks=range(3),xticklabels=LABELS,yticklabels=LABELS,xlabel='Predicted crop',ylabel='Actual crop',title=view)
    fig.suptitle('OOF decisions across 3 repeats of the SAME 914 training fields; not validation results',fontsize=11)
    fig.savefig(figs/'08_training_oof_confusion.png',bbox_inches='tight');plt.close(fig)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    pd.DataFrame(summary['metrics']).drop(columns=['confusion_sum_three_repeats']).to_csv(out/'summary.csv',index=False)
    print(pd.DataFrame(summary['metrics'])[['learner','view','macro_f1_mean','macro_f1_sd','recall_maize_mean','recall_wheat_mean']].to_string(index=False))
    print('Component selection frequencies:',summary['component_selection_frequency'])
    print('No validation/test predictions or scores were computed.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('results',type=Path)
    analyze(parser.parse_args().results)
