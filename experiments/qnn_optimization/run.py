"""Training-only selection followed by predeclared standalone QNN comparisons."""
import argparse
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import traceback
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from challenge.data import load_data
from challenge.metrics import score
from challenge.classical import reference
from challenge.models import MatchedHead
from experiments.qnn_optimization.model import NativeQiskitClassifier

LABELS=['alfalfa','maize','wheat']


def now():
    return datetime.now(ZoneInfo('Africa/Casablanca')).isoformat()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def preprocessing():
    return make_pipeline(StandardScaler(),PCA(4,svd_solver='full'),StandardScaler())


def classifier(config):
    return NativeQiskitClassifier(LABELS,reps=config['entangling_blocks'],
                                  optimizer=config['optimizer'],max_evals=config['max_evals'])


def execute(out):
    assert json.loads((out/'verification.json').read_text())['status']=='passed'
    if (out/'runs.json').exists() or (out/'cv_runs.json').exists():
        raise FileExistsError('Refusing to overwrite an existing experiment.')
    protocol=json.loads((ROOT/'experiments/qnn_optimization/protocol.json').read_text())
    configs=protocol['configurations']
    locked={str(p.relative_to(ROOT)):sha(p) for folder in ('challenge','data','configs','results','notebooks')
            for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    locked['README.md']=sha(ROOT/'README.md')
    metadata={'status':'running','started_at':now(),'branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
              'starting_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'protocol':protocol,'protected_sha256':locked,'platform':platform.platform(),
              'package_versions':{name:version(name) for name in ('numpy','scipy','scikit-learn','qiskit','qiskit-machine-learning')},
              'historical_prototype':'unavailable; no claimed reproduction of its scores',
              'test_inference':False}
    meta_path=out/'metadata.json'
    meta_path.write_text(json.dumps(metadata,indent=2))
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    log=(out/'execution.log').open('w',buffering=1)

    def progress(message):
        print(message,flush=True);log.write(message+'\n')

    tick=perf_counter()
    try:
        tables,_,manifest,_=load_data()
        train=tables['train'];F=manifest['features'];raw=train[F].to_numpy();y=train.target.to_numpy()
        cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=2026)
        folds=list(cv.split(raw,y));assignment=np.empty(len(train),dtype=int)
        for fold,(_,held) in enumerate(folds): assignment[held]=fold
        pd.DataFrame({'sample_id':train.sample_id,'fold':assignment}).to_csv(out/'cv_fold_assignments.csv',index=False)
        cv_records=[];cv_summary=[]
        progress('Training-only selection: 5 configurations x 3 folds, training seed 0. Validation is not scored.')
        for config in configs:
            oof=np.empty(len(y),dtype=object)
            for fold,(fit_ids,held_ids) in enumerate(folds):
                start=perf_counter();prep=preprocessing()
                X=prep.fit_transform(raw[fit_ids]);V=prep.transform(raw[held_ids]);prep_seconds=perf_counter()-start
                model=classifier(config).fit(X,y[fit_ids],seed=0)
                start=perf_counter();p=model.predict_proba(V);pred=model.labels[p.argmax(axis=1)];predict_seconds=perf_counter()-start
                oof[held_ids]=pred
                r={'configuration':config['name'],'fold':fold,'seed':0,'training_rows':len(fit_ids),'held_out_rows':len(held_ids),
                   'metrics':score(y[held_ids],pred,LABELS),'fit_seconds':model.fit_seconds,
                   'predict_seconds':predict_seconds,'preprocessing_seconds':prep_seconds,
                   'resources':model.resources(),'termination':model.termination,
                   'initial_weights':model.initial_point.tolist(),'final_weights':model.model.weights.tolist()}
                cv_records.append(r)
                (out/f'cv_{config["name"]}_fold_{fold}_history.json').write_text(json.dumps(model.history,indent=2))
                (out/'cv_runs.json').write_text(json.dumps(cv_records,indent=2))
                progress(f'CV {config["name"]} fold={fold} F1={r["metrics"]["macro_f1"]:.6f} fit={model.fit_seconds:.2f}s evaluations={len(model.history)}')
            metrics=score(y,oof,LABELS)
            cv_summary.append({'configuration':config['name'],'oof_metrics':metrics})
            pd.DataFrame({'sample_id':train.sample_id,'target':y,'prediction':oof,'fold':assignment}).to_csv(out/f'cv_{config["name"]}_oof_predictions.csv',index=False)
        selected=max(cv_summary,key=lambda r:r['oof_metrics']['macro_f1'])['configuration']
        selection={'selected_configuration':selected,'selected_at':now(),'selection_data':'914 training fields only',
                   'criterion':'pooled out-of-fold macro-F1, all three classes','folds':'3-fold stratified; not spatial',
                   'training_seed_for_selection':0,'candidates':cv_summary}
        selection_path=out/'training_only_selection.json'
        selection_path.write_text(json.dumps(selection,indent=2))
        metadata['selection_sha256']=sha(selection_path)
        metadata['selected_configuration']=selected
        meta_path.write_text(json.dumps(metadata,indent=2))
        progress('SELECTION LOCKED FROM TRAINING ONLY: '+selected+'. Now evaluating predefined full-training runs.')

        validation=tables['validation'];val_raw=validation[F].to_numpy();val_y=validation.target.to_numpy()
        start=perf_counter();prep=preprocessing();A=prep.fit_transform(raw);B=prep.transform(val_raw);prep_seconds=perf_counter()-start
        joblib.dump(prep,out/'preprocessing.joblib')
        records=[]
        for config in configs:
            for seed in (0,1,2):
                model=classifier(config).fit(A,y,seed=seed)
                expected=np.random.default_rng(seed).uniform(-np.pi,np.pi,len(model.weights))
                np.testing.assert_array_equal(model.initial_point,expected)
                assert len(model.history)<=config['max_evals']
                start=perf_counter();p=model.predict_proba(B);pred=model.labels[p.argmax(axis=1)];predict_seconds=perf_counter()-start
                direct=model.statevector_proba(model.prepare_states(B[:6]),model.model.weights,count=False)
                np.testing.assert_allclose(p[:6],direct,atol=1e-10,rtol=0)
                train_p=model.statevector_proba(model.prepare_states(A),model.model.weights,count=False)
                train_pred=model.labels[train_p.argmax(axis=1)]
                tag=f'{config["name"]}_seed_{seed}'
                r={'configuration':config['name'],'seed':seed,'config':config,'training_rows':914,'validation_rows':150,
                   'selection_sha256':metadata['selection_sha256'],'origin':'new local native-Qiskit experiment',
                   'metrics':score(val_y,pred,LABELS),'training_metrics':score(y,train_pred,LABELS),
                   'fit_seconds':model.fit_seconds,'predict_seconds':predict_seconds,'preprocessing_seconds':prep_seconds,
                   'resources':model.resources(),'termination':model.termination,'encoding_state_preparations':914}
                records.append(r)
                pd.DataFrame({'sample_id':validation.sample_id,'target':val_y,'prediction':pred,
                              **{f'p_{label}':p[:,j] for j,label in enumerate(LABELS)}}).to_csv(out/f'{tag}_predictions.csv',index=False)
                np.savez_compressed(out/f'{tag}_weights.npz',initial=model.initial_point,final=model.model.weights)
                (out/f'{tag}_history.json').write_text(json.dumps(model.history,indent=2))
                (out/'runs.json').write_text(json.dumps(records,indent=2))
                progress(f'FULL {config["name"]} seed={seed} F1={r["metrics"]["macro_f1"]:.6f} recall={r["metrics"]["recall"]} fit={model.fit_seconds:.2f}s evaluations={len(model.history)} probability_passes={model.forward_passes}')
        start=perf_counter();svm=reference().fit(raw,y);fit_seconds=perf_counter()-start
        start=perf_counter();pred=svm.predict(val_raw);predict_seconds=perf_counter()-start
        ref={'model':'original SVM reference','metrics':score(val_y,pred,LABELS),'fit_seconds':fit_seconds,
             'predict_seconds':predict_seconds,'support_vectors':int(sum(svm.named_steps['svc'].n_support_)),
             'origin':'locally refitted original challenge.classical.reference'}
        pd.DataFrame({'sample_id':validation.sample_id,'target':val_y,'prediction':pred}).to_csv(out/'svm_predictions.csv',index=False)
        (out/'svm.json').write_text(json.dumps(ref,indent=2))
        controls=[]
        for seed in (0,1,2):
            control=MatchedHead(LABELS,max_evals=100).fit(A,y,seed=seed)
            controls.append({'seed':seed,'metrics':score(val_y,control.predict(B),LABELS),
                             'weights':control.weights.tolist(),'resources':control.resources(),
                             'fit_seconds':control.fit_seconds,'label':'original tiny classical control, not capacity/cost matched to all candidates'})
        (out/'tiny_control.json').write_text(json.dumps(controls,indent=2))
        assert sha(selection_path)==metadata['selection_sha256']
        assert all(sha(ROOT/path)==digest for path,digest in locked.items())
        metadata.update(status='complete',completed_at=now(),wall_seconds=perf_counter()-tick,
                        protected_files_unchanged=True,initializations_verified=True,probabilities_verified=True)
        progress('Training and verification complete. Original code, notebook, data and organizer results unchanged.')
    except BaseException:
        metadata.update(status='failed',error=traceback.format_exc(),completed_at=now(),wall_seconds=perf_counter()-tick)
        raise
    finally:
        log.close();meta_path.write_text(json.dumps(metadata,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    args=parser.parse_args();args.results.mkdir(parents=True,exist_ok=True)
    execute(args.results)
