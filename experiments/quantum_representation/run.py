"""Predefined, training-only quantum representation comparison."""
import argparse
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
from experiments.feature_representation.models import representation,selected_components
from experiments.quantum_representation.model import RepresentationClassifier
from experiments.qnn_optimization.run import now,sha

LABELS=['alfalfa','maize','wheat']
PREVIOUS=ROOT/'outputs/entanglement_topology_2026-11-21'


def run(out):
    out=Path(out).resolve()
    verification=json.loads((out/'verification.json').read_text())
    assert verification['status']=='passed'
    for p,h in verification['implementation_sha256'].items():assert sha(ROOT/p)==h
    if (out/'metadata.json').exists():raise FileExistsError('Existing runs are never overwritten')
    protocol=json.loads((ROOT/'experiments/quantum_representation/protocol.json').read_text())
    tracked=[ROOT/p for p in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0') if p]
    protected={p for p in tracked if 'experiments/quantum_representation' not in str(p)}
    for folder in (ROOT/'outputs').iterdir():
        if folder.resolve()!=out:
            if folder.is_dir():protected.update(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
            elif folder.is_file():protected.add(folder)
    locked={str(p.relative_to(ROOT)):sha(p) for p in protected if p.is_file()}
    sources={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'experiments/quantum_representation').iterdir() if p.suffix in ('.py','.json')}
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    assert branch=='research/quantum-model-optimization'
    meta={'status':'running','started_at':now(),'branch':branch,
          'starting_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'protocol':protocol,'protected_sha256':locked,'implementation_sha256':sources,
          'package_versions':{n:version(n) for n in ('numpy','scipy','scikit-learn','qiskit','qiskit-machine-learning')},
          'platform':platform.platform(),'validation_or_test_inference':False}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    log=(out/'execution.log').open('w',buffering=1)
    def progress(msg):print(msg,flush=True);log.write(msg+'\n')
    tick_all=perf_counter()
    try:
        tables,_,manifest,_=load_data();train=tables['train'];F=manifest['features']
        raw=train[F].to_numpy();y=train.target.to_numpy()
        assignments=pd.read_csv(PREVIOUS/'cv_fold_assignments.csv')
        np.testing.assert_array_equal(assignments.sample_id,train.sample_id)
        fold_id=assignments.fold.to_numpy();assert set(fold_id)=={0,1,2}
        assignments.to_csv(out/'cv_fold_assignments.csv',index=False)
        old_records=json.loads((PREVIOUS/'cv_runs.json').read_text())
        records=[];seed_records=[];reference_checks=[]
        progress('27 fresh quantum CV fits; training-only, no validation/test inference.')
        for candidate in protocol['candidates']:
            name,view,n=candidate['name'],candidate['representation'],candidate['qubits']
            fold_data=[]
            for fold in (0,1,2):
                fit=np.flatnonzero(fold_id!=fold);held=np.flatnonzero(fold_id==fold)
                tick=perf_counter();prep=representation(view)
                X=prep.fit_transform(raw[fit],y[fit]);V=prep.transform(raw[held])
                fold_data.append((fit,held,X,V,selected_components(prep,view),perf_counter()-tick))
                joblib.dump(prep,out/f'{name}_fold_{fold}_preprocessing.joblib')
            for seed in (0,1,2):
                oof=np.empty(len(y),dtype=object);oof_p=np.empty((len(y),3))
                for fold,(fit,held,X,V,pcs,prep_seconds) in enumerate(fold_data):
                    m=RepresentationClassifier(n).fit(X,y[fit],seed=seed)
                    assert len(m.history)<=100
                    np.testing.assert_array_equal(m.initial_point,np.random.default_rng(seed).uniform(-np.pi,np.pi,3*n))
                    tick=perf_counter();p=m.predict_proba(V);predict_seconds=perf_counter()-tick
                    direct=m.statevector_proba(m.prepare_states(V[:6]),m.model.weights,count=False)
                    np.testing.assert_allclose(p[:6],direct,atol=1e-10,rtol=0)
                    pred=m.labels[p.argmax(axis=1)];oof[held]=pred;oof_p[held]=p
                    metrics=score(y[held],pred,LABELS)
                    if name=='first4_q4':
                        old=next(r for r in old_records if r['topology']=='full' and r['seed']==seed and r['fold']==fold)
                        np.testing.assert_allclose(m.model.weights,old['final_weights'],atol=1e-10,rtol=0)
                        assert metrics==old['metrics']
                        reference_checks.append({'seed':seed,'fold':fold,'prior_weights_and_metrics_match':True})
                    record={'configuration':name,'representation':view,'qubits':n,'seed':seed,'fold':fold,
                            'training_rows':len(fit),'held_out_rows':len(held),'selected_components':pcs,
                            'metrics':metrics,'fit_seconds':m.fit_seconds,'predict_seconds':predict_seconds,
                            'preprocessing_seconds':prep_seconds,'resources':m.resources(),'termination':m.termination,
                            'initial_weights':m.initial_point.tolist(),'final_weights':m.model.weights.tolist()}
                    records.append(record);tag=f'{name}_seed_{seed}_fold_{fold}'
                    (out/f'{tag}_history.json').write_text(json.dumps(m.history,indent=2))
                    (out/'cv_runs.json').write_text(json.dumps(records,indent=2))
                    progress(f'{name} seed={seed} fold={fold} F1={metrics["macro_f1"]:.6f} fit={m.fit_seconds:.2f}s gradient_calls={len(m.history)}')
                if name=='first4_q4':
                    oldp=pd.read_csv(PREVIOUS/f'cv_full_seed_{seed}_oof_predictions.csv')
                    np.testing.assert_array_equal(oldp.prediction,oof)
                metrics=score(y,oof,LABELS)
                seed_records.append({'configuration':name,'seed':seed,'metrics':metrics,
                                     'fit_seconds':sum(r['fit_seconds'] for r in records if r['configuration']==name and r['seed']==seed)})
                pd.DataFrame({'sample_id':train.sample_id,'target':y,'prediction':oof,'fold':fold_id,
                              **{f'p_{c}':oof_p[:,j] for j,c in enumerate(LABELS)}}).to_csv(out/f'{name}_seed_{seed}_oof.csv',index=False)
                (out/'seed_runs.json').write_text(json.dumps(seed_records,indent=2))
        summaries=[]
        for c in protocol['candidates']:
            group=[r for r in seed_records if r['configuration']==c['name']]
            summaries.append({**c,'mean_oof_macro_f1':float(np.mean([r['metrics']['macro_f1'] for r in group])),
                              'seed_sd':float(np.std([r['metrics']['macro_f1'] for r in group]))})
        selected=max(summaries,key=lambda r:(r['mean_oof_macro_f1'],-r['qubits'],r['name']=='first4_q4'))
        selection={'selected_configuration':selected['name'],'locked_at':now(),'selection_scope':'training only',
                   'criterion':protocol['selection'],'candidates':summaries,'prior_full_connectivity_reference_checks':reference_checks,
                   'validation_or_test_inference':False}
        selection_path=out/'training_only_selection.json';selection_path.write_text(json.dumps(selection,indent=2))
        selection_hash=sha(selection_path)
        progress('Training-only selection locked: '+selected['name']+'. Saving three full-training fits without held-out release evaluation.')
        prep=representation(selected['representation']);X=prep.fit_transform(raw,y)
        joblib.dump(prep,out/'selected_full_training_preprocessing.joblib')
        final_records=[]
        for seed in (0,1,2):
            m=RepresentationClassifier(selected['qubits']).fit(X,y,seed=seed)
            np.savez_compressed(out/f'selected_seed_{seed}_weights.npz',initial=m.initial_point,final=m.model.weights)
            (out/f'selected_seed_{seed}_history.json').write_text(json.dumps(m.history,indent=2))
            final_records.append({'configuration':selected['name'],'seed':seed,'training_rows':len(y),
                                  'selected_components':selected_components(prep,selected['representation']),
                                  'selection_sha256':selection_hash,'fit_seconds':m.fit_seconds,
                                  'resources':m.resources(),'termination':m.termination,
                                  'validation_or_test_inference':False})
            (out/'full_training_runs.json').write_text(json.dumps(final_records,indent=2))
            progress(f'Full-training saved seed={seed} fit={m.fit_seconds:.2f}s; no validation/test scoring.')
        assert sha(selection_path)==selection_hash
        assert all(sha(ROOT/p)==h for p,h in locked.items())
        assert all(sha(ROOT/p)==h for p,h in sources.items())
        meta.update(status='complete',completed_at=now(),wall_seconds=perf_counter()-tick_all,
                    prior_files_unchanged=True,selected_configuration=selected['name'],cv_fits=len(records),full_training_fits=3)
    except BaseException:
        meta.update(status='failed',error=traceback.format_exc(),completed_at=now(),wall_seconds=perf_counter()-tick_all);raise
    finally:
        log.close();(out/'metadata.json').write_text(json.dumps(meta,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);run(p.parse_args().results)
