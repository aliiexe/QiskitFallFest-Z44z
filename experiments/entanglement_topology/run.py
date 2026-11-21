"""Paired training-only topology selection, then nine predefined local fits."""
import argparse
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
from challenge.data import load_data
from challenge.metrics import score
from experiments.qnn_optimization.run import preprocessing,now,sha
from experiments.entanglement_topology.model import TopologyClassifier

LABELS=['alfalfa','maize','wheat']
PREVIOUS=ROOT/'outputs/qnn_optimization_2026-11-21'


def run(out):
    out=Path(out).resolve()
    assert json.loads((out/'verification.json').read_text())['status']=='passed'
    if (out/'cv_runs.json').exists() or (out/'runs.json').exists():
        raise FileExistsError('Use a new result directory; existing runs are never overwritten.')
    protocol=json.loads((ROOT/'experiments/entanglement_topology/protocol.json').read_text())
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    protected={ROOT/p for p in tracked if p and not p.startswith(('experiments/entanglement_topology/','outputs/entanglement_topology_2026-11-21/'))}
    # Include previously generated local experiments even when Git ignores them.
    for folder in (ROOT/'outputs').iterdir():
        if folder.resolve()!=out:
            if folder.is_dir(): protected.update(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
            elif folder.is_file(): protected.add(folder)
    locked={str(p.relative_to(ROOT)):sha(p) for p in protected if p.is_file()}
    source_files=[p for p in (ROOT/'experiments/entanglement_topology').iterdir() if p.suffix in ('.py','.json')]
    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in source_files}
    metadata={'status':'running','started_at':now(),'branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
              'starting_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'protocol':protocol,'implementation_sha256':source_hashes,'protected_sha256':locked,
              'platform':platform.platform(),'package_versions':{n:version(n) for n in ('numpy','scipy','scikit-learn','qiskit','qiskit-machine-learning')},
              'reference':'Verified Model D, prior local Qiskit runs; linear is also freshly refitted here.',
              'test_inference':False,'eight_qubit_or_reuploading_experiments':False}
    meta_path=out/'metadata.json';meta_path.write_text(json.dumps(metadata,indent=2))
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    logfile=(out/'execution.log').open('w',buffering=1)
    def log(message):
        print(message,flush=True);logfile.write(message+'\n')
    start_all=perf_counter()
    try:
        tables,_,manifest,_=load_data();train=tables['train'];F=manifest['features']
        raw=train[F].to_numpy();y=train.target.to_numpy()
        assignments=pd.read_csv(PREVIOUS/'cv_fold_assignments.csv')
        np.testing.assert_array_equal(assignments.sample_id,train.sample_id)
        fold_id=assignments.fold.to_numpy();assert set(fold_id)=={0,1,2}
        assignments.to_csv(out/'cv_fold_assignments.csv',index=False)
        fold_data=[]
        for fold in (0,1,2):
            fit=np.flatnonzero(fold_id!=fold);held=np.flatnonzero(fold_id==fold)
            tick=perf_counter();prep=preprocessing();X=prep.fit_transform(raw[fit]);V=prep.transform(raw[held])
            fold_data.append((fit,held,X,V,perf_counter()-tick))
        log('Training-only selection: 3 topologies x 3 seeds x 3 identical folds. No validation scoring yet.')
        cv_records=[];cv_summary=[]
        for name in protocol['topologies']:
            seed_metrics=[]
            for seed in (0,1,2):
                oof=np.empty(len(train),dtype=object)
                for fold,(fit,held,X,V,prep_seconds) in enumerate(fold_data):
                    model=TopologyClassifier(topology=name).fit(X,y[fit],seed=seed)
                    tick=perf_counter();p=model.predict_proba(V);pred=model.labels[p.argmax(axis=1)];predict_seconds=perf_counter()-tick
                    oof[held]=pred
                    r={'topology':name,'seed':seed,'fold':fold,'training_rows':len(fit),'held_out_rows':len(held),
                       'metrics':score(y[held],pred,LABELS),'fit_seconds':model.fit_seconds,'predict_seconds':predict_seconds,
                       'preprocessing_seconds':prep_seconds,'resources':model.resources(),'termination':model.termination,
                       'initial_weights':model.initial_point.tolist(),'final_weights':model.model.weights.tolist()}
                    cv_records.append(r)
                    (out/f'cv_{name}_seed_{seed}_fold_{fold}_history.json').write_text(json.dumps(model.history,indent=2))
                    (out/'cv_runs.json').write_text(json.dumps(cv_records,indent=2))
                    log(f'CV {name} seed={seed} fold={fold} F1={r["metrics"]["macro_f1"]:.6f} fit={model.fit_seconds:.2f}s evals={len(model.history)}')
                metrics=score(y,oof,LABELS);seed_metrics.append({'seed':seed,'metrics':metrics})
                pd.DataFrame({'sample_id':train.sample_id,'target':y,'prediction':oof,'fold':fold_id}).to_csv(out/f'cv_{name}_seed_{seed}_oof_predictions.csv',index=False)
                if name=='linear' and seed==0:
                    old=pd.read_csv(PREVIOUS/'cv_D_oof_predictions.csv')
                    np.testing.assert_array_equal(old.sample_id,train.sample_id)
                    np.testing.assert_array_equal(old.prediction,oof)
            values=[s['metrics']['macro_f1'] for s in seed_metrics]
            cv_summary.append({'topology':name,'mean_oof_macro_f1':float(np.mean(values)),
                               'seed_sd_oof_macro_f1':float(np.std(values)),'seeds':seed_metrics,
                               'two_qubit_gates':model.resources()['two_qubit_gates']})
        selected=max(cv_summary,key=lambda r:(r['mean_oof_macro_f1'],-r['two_qubit_gates']))['topology']
        selection={'selected_topology':selected,'locked_at':now(),'criterion':'mean seed-specific pooled OOF macro-F1',
                   'selection_data':'914 training fields only','cv_seeds':[0,1,2],
                   'fold_policy':'previous fixed stratified folds; NOT spatial','candidates':cv_summary,
                   'linear_seed0_oof_matches_previous_model_D':True}
        selection_path=out/'training_only_selection.json';selection_path.write_text(json.dumps(selection,indent=2))
        selection_hash=sha(selection_path)
        metadata.update(selected_topology=selected,selection_sha256=selection_hash)
        meta_path.write_text(json.dumps(metadata,indent=2))
        log('SELECTION LOCKED: '+selected+'. Now refitting all nine predefined full-training models.')

        v=tables['validation'];tick=perf_counter();prep=preprocessing()
        A=prep.fit_transform(raw);B=prep.transform(v[F].to_numpy());prep_seconds=perf_counter()-tick
        joblib.dump(prep,out/'preprocessing.joblib')
        prior=json.loads((PREVIOUS/'runs.json').read_text());records=[];linear_checks=[]
        for name in protocol['topologies']:
            for seed in (0,1,2):
                model=TopologyClassifier(topology=name).fit(A,y,seed=seed)
                np.testing.assert_array_equal(model.initial_point,np.random.default_rng(seed).uniform(-np.pi,np.pi,12))
                assert len(model.history)<=100
                tick=perf_counter();p=model.predict_proba(B);pred=model.labels[p.argmax(axis=1)];predict_seconds=perf_counter()-tick
                direct=model.statevector_proba(model.prepare_states(B[:6]),model.model.weights,count=False)
                np.testing.assert_allclose(p[:6],direct,atol=1e-10,rtol=0)
                train_p=model.statevector_proba(model.prepare_states(A),model.model.weights,count=False)
                metrics=score(v.target,pred,LABELS)
                if name=='linear':
                    old=next(r for r in prior if r['configuration']=='D' and r['seed']==seed)
                    oldp=pd.read_csv(PREVIOUS/f'D_seed_{seed}_predictions.csv')
                    np.testing.assert_array_equal(oldp.prediction,pred);assert old['metrics']==metrics
                    with np.load(PREVIOUS/f'D_seed_{seed}_weights.npz',allow_pickle=False) as weights:
                        np.testing.assert_allclose(model.model.weights,weights['final'],atol=1e-10,rtol=0)
                    linear_checks.append({'seed':seed,'prior_D_predictions_metrics_and_weights_match':True})
                r={'topology':name,'seed':seed,'origin':'fresh local Qiskit topology fit','selection_sha256':selection_hash,
                   'training_rows':914,'validation_rows':150,'metrics':metrics,
                   'training_metrics':score(y,model.labels[train_p.argmax(axis=1)],LABELS),
                   'fit_seconds':model.fit_seconds,'predict_seconds':predict_seconds,'preprocessing_seconds':prep_seconds,
                   'resources':model.resources(),'termination':model.termination,'encoding_state_preparations':914}
                records.append(r);tag=f'{name}_seed_{seed}'
                pd.DataFrame({'sample_id':v.sample_id,'target':v.target,'prediction':pred,
                              **{f'p_{crop}':p[:,j] for j,crop in enumerate(LABELS)}}).to_csv(out/f'{tag}_predictions.csv',index=False)
                np.savez_compressed(out/f'{tag}_weights.npz',initial=model.initial_point,final=model.model.weights)
                (out/f'{tag}_history.json').write_text(json.dumps(model.history,indent=2))
                (out/'runs.json').write_text(json.dumps(records,indent=2))
                log(f'FULL {name} seed={seed} F1={metrics["macro_f1"]:.6f} recall={metrics["recall"]} fit={model.fit_seconds:.2f}s evals={len(model.history)} passes={model.forward_passes}')
        assert sha(selection_path)==selection_hash
        assert all(sha(ROOT/p)==digest for p,digest in locked.items())
        assert all(sha(ROOT/p)==digest for p,digest in source_hashes.items())
        metadata.update(status='complete',completed_at=now(),wall_seconds=perf_counter()-start_all,
                        prior_experiments_and_organizer_files_unchanged=True,
                        identical_initializations_verified=True,linear_reference_checks=linear_checks)
        log('Complete: all previous experiments and organizer files unchanged; linear reproduces verified Model D.')
    except BaseException:
        metadata.update(status='failed',completed_at=now(),wall_seconds=perf_counter()-start_all,error=traceback.format_exc());raise
    finally:
        logfile.close();meta_path.write_text(json.dumps(metadata,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results',type=Path,required=True)
    run(parser.parse_args().results)
