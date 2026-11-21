"""Complete PCA diagnostics and nested training-only classical comparisons."""
import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import traceback
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer,f1_score
from challenge.data import load_data
from challenge.metrics import score
from experiments.qnn_optimization.run import sha,now
from experiments.feature_representation.models import VIEWS,model,selected_components

LABELS=['alfalfa','maize','wheat']
LEARNERS=['logistic','svm_fixed','svm_nested']


def eta_squared(z,y):
    total=np.square(z-z.mean()).sum()
    between=sum((y==c).sum()*(z[y==c].mean()-z.mean())**2 for c in np.unique(y))
    return float(between/total) if total else 0.


def diagnostics(train,F,out):
    raw=train[F].to_numpy();y=train.target.to_numpy();scaler=StandardScaler()
    X=scaler.fit_transform(raw);pca=PCA(svd_solver='full');Z=pca.fit_transform(X)
    S=StandardScaler().fit_transform(Z)
    np.testing.assert_allclose(pca.inverse_transform(Z),X,atol=1e-10,rtol=0)
    mw=np.isin(y,['maize','wheat'])
    rows=[]
    for j in range(13):
        rows.append({'component':j+1,'variance_ratio':float(pca.explained_variance_ratio_[j]),
                     'cumulative_variance':float(pca.explained_variance_ratio_[:j+1].sum()),
                     'eta2_all_crops':eta_squared(S[:,j],y),
                     'eta2_maize_wheat':eta_squared(S[mw,j],y[mw]),
                     'largest_coefficients':[{ 'feature':F[k],'coefficient':float(pca.components_[j,k])}
                                             for k in np.argsort(abs(pca.components_[j]))[-4:][::-1]]})
    pd.DataFrame([{k:v for k,v in r.items() if k!='largest_coefficients'} for r in rows]).to_csv(out/'pca_variance_and_separation.csv',index=False)
    pd.DataFrame(pca.components_,index=[f'PC{j+1}' for j in range(13)],columns=F).to_csv(out/'pca_loadings.csv')
    scores=pd.DataFrame(S,columns=[f'PC{j+1}' for j in range(13)])
    scores.insert(0,'target',y);scores.insert(0,'sample_id',train.sample_id)
    scores.to_csv(out/'training_pca_scores.csv',index=False)
    recon=[]
    for k in (4,6,8,13):
        represented=Z.copy();represented[:,k:]=0
        error=np.sqrt(np.mean((pca.inverse_transform(represented)-X)**2,axis=0))
        recon.extend({'components':k,'feature':feature,'reconstruction_rmse_in_standardized_units':float(e)} for feature,e in zip(F,error))
    pd.DataFrame(recon).to_csv(out/'feature_reconstruction.csv',index=False)
    train.groupby('target')[F].describe().to_csv(out/'feature_distributions.csv')
    diag={'scope':'descriptive full-training fit only; never used as preprocessing in CV',
          'components':rows,'class_counts':train.target.value_counts().to_dict(),
          'full_decomposition_reconstruction_passed':True,
          'extra_variance_PC5_through_PC8':float(pca.explained_variance_ratio_[4:8].sum()),
          'interpretation':'eta-squared measures descriptive mean separation; it does not prove incremental prediction or causality'}
    (out/'pca_diagnostics.json').write_text(json.dumps(diag,indent=2))
    joblib.dump({'input_scaler':scaler,'pca':pca},out/'descriptive_full_pca.joblib')


def run(out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((ROOT/'experiments/feature_representation/protocol.json').read_text())
    paths=[ROOT/p for p in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0') if p]
    paths=[p for p in paths if 'experiments/feature_representation' not in str(p)]
    # Preserve all prior experiment outputs, including ignored local artifacts.
    for child in (ROOT/'outputs').iterdir():
        if child.resolve()!=out:
            paths.extend(p for p in child.rglob('*') if p.is_file() and '__pycache__' not in p.parts) if child.is_dir() else paths.append(child)
    locked={str(p.relative_to(ROOT)):sha(p) for p in set(paths) if p.is_file()}
    sources={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'experiments/feature_representation').iterdir() if p.suffix in ('.py','.json')}
    meta={'status':'running','started_at':now(),'branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
          'starting_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'protocol':protocol,'protected_sha256':locked,'implementation_sha256':sources,'platform':platform.platform(),
          'package_versions':{n:version(n) for n in ('numpy','scipy','scikit-learn','qiskit')},
          'validation_predictions_or_scores':False,'test_inference':False}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2));(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    log=(out/'execution.log').open('w',buffering=1)
    def progress(msg):print(msg,flush=True);log.write(msg+'\n')
    tick_all=perf_counter()
    try:
        tables,_,manifest,_=load_data();train=tables['train'];F=manifest['features']
        raw=train[F].to_numpy();y=train.target.to_numpy();diagnostics(train,F,out)
        scorer=make_scorer(f1_score,labels=LABELS,average='macro',zero_division=0)
        old=pd.read_csv(ROOT/'outputs/qnn_optimization_2026-11-21/cv_fold_assignments.csv')
        np.testing.assert_array_equal(old.sample_id,train.sample_id)
        records=[];repeat_records=[];assignments=[]
        progress('Training-only study: 7 representations, 3 classical recipes, 3 repeats of 3 outer folds; nested SVM has 9 inner-grid candidates.')
        for repeat,split_seed in enumerate((2026,2027,2028)):
            outer=list(StratifiedKFold(3,shuffle=True,random_state=split_seed).split(raw,y))
            fold_ids=np.empty(len(train),dtype=int)
            for fold,(_,held) in enumerate(outer):fold_ids[held]=fold
            if repeat==0:np.testing.assert_array_equal(fold_ids,old.fold)
            assignments.extend({'sample_id':s,'repeat':repeat,'split_seed':split_seed,'fold':int(f)} for s,f in zip(train.sample_id,fold_ids))
            for learner in LEARNERS:
                for view in VIEWS:
                    oof=np.empty(len(y),dtype=object)
                    for fold,(fit,held) in enumerate(outer):
                        estimator=model(view,learner);start=perf_counter()
                        if learner=='svm_nested':
                            inner=StratifiedKFold(3,shuffle=True,random_state=30000+100*repeat+fold)
                            grid=GridSearchCV(estimator,{'classifier__C':[1,10,100],'classifier__gamma':[.01,.1,'scale']},
                                              scoring=scorer,cv=inner,n_jobs=1,refit=True,error_score='raise')
                            grid.fit(raw[fit],y[fit]);fitted=grid.best_estimator_;parameters=grid.best_params_
                            inner_scores=[{'parameters':p,'mean_f1':float(s)} for p,s in zip(grid.cv_results_['params'],grid.cv_results_['mean_test_score'])]
                        else:
                            fitted=estimator.fit(raw[fit],y[fit]);parameters={'C':fitted.named_steps['classifier'].C}
                            if learner=='svm_fixed':parameters['gamma']=fitted.named_steps['classifier'].gamma
                            inner_scores=None
                        fit_seconds=perf_counter()-start;start=perf_counter();pred=fitted.predict(raw[held]);predict_seconds=perf_counter()-start
                        oof[held]=pred
                        selected=selected_components(fitted,view)
                        r={'learner':learner,'view':view,'repeat':repeat,'split_seed':split_seed,'fold':fold,
                           'training_rows':len(fit),'held_out_rows':len(held),'metrics':score(y[held],pred,LABELS),
                           'fit_and_search_seconds':fit_seconds,'predict_seconds':predict_seconds,
                           'selected_components':selected,'best_parameters':parameters,'inner_grid_scores':inner_scores}
                        clf=fitted.named_steps['classifier']
                        if hasattr(clf,'n_support_'):r['support_vectors']=int(clf.n_support_.sum())
                        records.append(r)
                    metrics=score(y,oof,LABELS)
                    repeat_records.append({'learner':learner,'view':view,'repeat':repeat,'split_seed':split_seed,'metrics':metrics,
                                           'fit_and_search_seconds':sum(r['fit_and_search_seconds'] for r in records if r['learner']==learner and r['view']==view and r['repeat']==repeat)})
                    pd.DataFrame({'sample_id':train.sample_id,'target':y,'prediction':oof,'fold':fold_ids}).to_csv(out/f'{learner}_{view}_repeat_{repeat}_oof.csv',index=False)
                    (out/'fold_runs.json').write_text(json.dumps(records,indent=2));(out/'repeat_runs.json').write_text(json.dumps(repeat_records,indent=2))
                    progress(f'{learner} {view} repeat={repeat} F1={metrics["macro_f1"]:.6f} recall={metrics["recall"]}')
        pd.DataFrame(assignments).to_csv(out/'outer_fold_assignments.csv',index=False)
        primary=[{'view':v,'mean_oof_macro_f1':float(np.mean([r['metrics']['macro_f1'] for r in repeat_records if r['learner']=='svm_nested' and r['view']==v]))} for v in VIEWS]
        sizes={'pca4':4,'pca6':6,'pca8':8,'pca13':13,'raw13':13,'late5_8':4,'selected4of8':4}
        selected=max(primary,key=lambda r:(r['mean_oof_macro_f1'],-sizes[r['view']]))['view']
        selection={'selected_representation':selected,'selected_at':now(),'criterion':'mean three-repeat nested-SVM pooled OOF macro-F1; ties favor fewer output features',
                   'candidates':primary,'selection_scope':'training only','validation_or_test_used':False}
        (out/'training_only_selection.json').write_text(json.dumps(selection,indent=2))
        # Store deployable classical diagnostic references using training-only hyperparameter selection.
        for view in set(('pca4',selected)):
            grid=GridSearchCV(model(view,'svm_nested'),{'classifier__C':[1,10,100],'classifier__gamma':[.01,.1,'scale']},
                              scoring=scorer,cv=StratifiedKFold(3,shuffle=True,random_state=2026),n_jobs=1,error_score='raise')
            grid.fit(raw,y);joblib.dump(grid.best_estimator_,out/f'full_training_{view}_svm.joblib')
            (out/f'full_training_{view}_parameters.json').write_text(json.dumps({'view':view,'best_parameters':grid.best_params_,
                                       'selected_components':selected_components(grid.best_estimator_,view),
                                       'used_for_validation_prediction':False},indent=2))
        assert all(sha(ROOT/p)==h for p,h in locked.items())
        assert all(sha(ROOT/p)==h for p,h in sources.items())
        meta.update(status='complete',completed_at=now(),wall_seconds=perf_counter()-tick_all,
                    prior_files_unchanged=True,outer_fits=len(records),repeat_predictions=len(repeat_records),
                    selected_representation=selected,validation_predictions_or_scores=False,test_inference=False)
        progress('Complete. Selected from training only: '+selected+'. No validation or test scoring performed.')
    except BaseException:
        meta.update(status='failed',error=traceback.format_exc(),completed_at=now(),wall_seconds=perf_counter()-tick_all);raise
    finally:
        log.close();(out/'metadata.json').write_text(json.dumps(meta,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results',type=Path,required=True)
    run(parser.parse_args().results)
