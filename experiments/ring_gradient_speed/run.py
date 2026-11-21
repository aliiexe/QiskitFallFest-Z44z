"""Same ring model, same optimizer: paired training-only gradient-engine selection."""
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
from experiments.qnn_optimization.run import preprocessing,sha,now
from experiments.entanglement_topology.model import TopologyClassifier
from experiments.ring_gradient_speed.model import BatchedRingClassifier

LABELS=['alfalfa','maize','wheat']
METHODS=['parameter_shift','batched_shift','operator_adjoint']
OLD=ROOT/'outputs/entanglement_topology_2026-11-21'


def classifier(method):
    return TopologyClassifier(topology='ring') if method=='parameter_shift' else BatchedRingClassifier(method)


def resources(model,method):
    r=model.resources()
    if method=='parameter_shift':
        r.update(gradient_method=method,complete_forward_probability_passes=model.forward_passes,
                 reverse_sweeps=0,state_preparation_seconds=model.state_preparation_seconds,
                 fixed_gate_cache_seconds=0.,input_batch_conversion_seconds=0.)
    return r


def run(out):
    out=Path(out).resolve()
    verified=json.loads((out/'verification.json').read_text());assert verified['status']=='passed'
    assert all(sha(ROOT/p)==h for p,h in verified['source_sha256'].items())
    if (out/'metadata.json').exists():raise FileExistsError('Existing runs preserved; use a fresh destination')
    protocol=json.loads((ROOT/'experiments/ring_gradient_speed/protocol.json').read_text())
    locked=json.loads((out/'preservation.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in locked.items())
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    assert branch=='research/quantum-model-optimization'
    sources={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'experiments/ring_gradient_speed').iterdir() if p.suffix in ['.py','.json']}
    metadata={'status':'running','started_at':now(),'branch':branch,
              'starting_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'protocol':protocol,'source_sha256':sources,'protected_sha256':locked,
              'platform':platform.platform(),'python':sys.version,
              'packages':{n:version(n) for n in ['qiskit','qiskit-machine-learning','numpy','scipy','scikit-learn']},
              'new_notebook':False,'test_inference':False}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2));(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (out/'cv').mkdir();(out/'full').mkdir()
    logfile=(out/'execution.log').open('w',buffering=1)
    def log(s):print(s,flush=True);logfile.write(s+'\n')
    start_all=perf_counter()
    try:
        tables,release,manifest,_=load_data();t=tables['train'];F=manifest['features']
        raw=t[F].to_numpy();y=t.target.to_numpy()
        assignment=pd.read_csv(OLD/'cv_fold_assignments.csv')
        np.testing.assert_array_equal(assignment.sample_id,t.sample_id)
        fold_id=assignment.fold.to_numpy();assert set(fold_id)=={0,1,2}
        assignment.to_csv(out/'cv_fold_assignments.csv',index=False)
        old_cv=json.loads((OLD/'cv_runs.json').read_text())
        records=[];oof={(m,s):np.empty(len(t),dtype=object) for m in METHODS for s in [0,1,2]}
        oof_p={(m,s):np.empty((len(t),3)) for m in METHODS for s in [0,1,2]}
        log('Training only: 3 unchanged-ring gradient engines x 3 folds x seeds 0/1/2. Inference uses the same standard SamplerQNN.')
        for fold in [0,1,2]:
            fit=np.flatnonzero(fold_id!=fold);held=np.flatnonzero(fold_id==fold)
            for seed in [0,1,2]:
                rotation=(fold+seed)%3;order=METHODS[rotation:]+METHODS[:rotation]
                for method in order:
                    # Each fit pays its own preprocessing, constructor and every cache.
                    tick=perf_counter();prep=preprocessing();X=prep.fit_transform(raw[fit]);V=prep.transform(raw[held]);prep_seconds=perf_counter()-tick
                    tick=perf_counter();m=classifier(method);constructor_seconds=perf_counter()-tick
                    m.fit(X,y[fit],seed=seed)
                    tick=perf_counter();p=m.predict_proba(V);inference_seconds=perf_counter()-tick
                    pred=m.labels[p.argmax(axis=1)];oof[method,seed][held]=pred;oof_p[method,seed][held]=p
                    metrics=score(y[held],pred,LABELS)
                    np.testing.assert_array_equal(m.initial_point,np.random.default_rng(seed).uniform(-np.pi,np.pi,12))
                    if method=='parameter_shift':
                        old=next(r for r in old_cv if r['topology']=='ring' and r['seed']==seed and r['fold']==fold)
                        np.testing.assert_allclose(m.model.weights,old['final_weights'],atol=1e-10,rtol=0)
                        assert metrics==old['metrics']
                    r={'method':method,'seed':seed,'fold':fold,'fit_fields':len(fit),'held_out_fields':len(held),
                       'metrics':metrics,'preprocessing_seconds':prep_seconds,'model_construction_seconds':constructor_seconds,
                       'fit_seconds':m.fit_seconds,'inference_seconds':inference_seconds,
                       'total_seconds':prep_seconds+constructor_seconds+m.fit_seconds+inference_seconds,
                       'resources':resources(m,method),'termination':m.termination,
                       'initial_weights':m.initial_point.tolist(),'final_weights':m.model.weights.tolist(),
                       'final_training_loss_bits':float(m.history[-1]['objective_bits'])}
                    records.append(r)
                    (out/'cv'/f'{method}_seed_{seed}_fold_{fold}_history.json').write_text(json.dumps(m.history,indent=2))
                    (out/'cv_runs.json').write_text(json.dumps(records,indent=2))
                    log(f'CV {method} fold={fold} seed={seed} F1={metrics["macro_f1"]:.6f} fit={m.fit_seconds:.4f}s cache={m.state_preparation_seconds:.4f}s calls={len(m.history)}')
        seed_records=[];summaries=[]
        for method in METHODS:
            for seed in [0,1,2]:
                metrics=score(y,oof[method,seed],LABELS)
                group=[r for r in records if r['method']==method and r['seed']==seed]
                seed_records.append({'method':method,'seed':seed,'metrics':metrics,
                                     'three_fold_total_seconds':sum(r['total_seconds'] for r in group),
                                     'three_fold_fit_seconds':sum(r['fit_seconds'] for r in group),
                                     'same_predictions_as_baseline':bool(np.array_equal(oof[method,seed],oof['parameter_shift',seed]))})
                pd.DataFrame({'sample_id':t.sample_id,'target':y,'prediction':oof[method,seed],'fold':fold_id,
                              **{f'p_{c}':oof_p[method,seed][:,j] for j,c in enumerate(LABELS)}}).to_csv(out/'cv'/f'{method}_seed_{seed}_oof.csv',index=False)
            rows=[r for r in seed_records if r['method']==method]
            losses=[r['final_training_loss_bits'] for r in records if r['method']==method]
            base_losses=[r['final_training_loss_bits'] for r in records if r['method']=='parameter_shift']
            eligible=all(r['same_predictions_as_baseline'] for r in rows) and abs(np.mean(losses)-np.mean(base_losses))<=1e-7
            summaries.append({'method':method,'mean_oof_macro_f1':float(np.mean([r['metrics']['macro_f1'] for r in rows])),
                              'seed_SD':float(np.std([r['metrics']['macro_f1'] for r in rows])),
                              'mean_final_training_loss_bits':float(np.mean(losses)),
                              'all_seed_OOF_predictions_match_baseline':all(r['same_predictions_as_baseline'] for r in rows),
                              'quality_gate_passed':bool(eligible),
                              'median_three_fold_total_seconds':float(np.median([r['three_fold_total_seconds'] for r in rows])),
                              'median_three_fold_fit_seconds':float(np.median([r['three_fold_fit_seconds'] for r in rows]))})
        (out/'cv_seed_runs.json').write_text(json.dumps(seed_records,indent=2))
        selected=min((r for r in summaries if r['quality_gate_passed']),
                     key=lambda r:(r['median_three_fold_total_seconds'],r['method']!='batched_shift'))['method']
        selection={'selected_method':selected,'locked_at':now(),'selection_scope':'914 training fields only, no validation scores',
                   'criterion':protocol['selection'],'candidates':summaries}
        selection_path=out/'training_only_selection.json';selection_path.write_text(json.dumps(selection,indent=2))
        selection_hash=sha(selection_path)
        metadata.update(selected_method=selected,selection_sha256=selection_hash)
        (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
        log('SELECTION LOCKED: '+selected+'. Now one full-training comparison against baseline; no validation retuning.')
        v=tables['validation'];val_raw=v[F].to_numpy();full=[]
        final_methods=['parameter_shift'] if selected=='parameter_shift' else ['parameter_shift',selected]
        for seed in [0,1,2]:
            order=final_methods if seed%2==0 else final_methods[::-1]
            for method in order:
                tick=perf_counter();prep=preprocessing();X=prep.fit_transform(raw);V=prep.transform(val_raw);prep_seconds=perf_counter()-tick
                np.testing.assert_allclose(X,release['X_train'],atol=1e-10,rtol=0)
                tick=perf_counter();m=classifier(method);constructor_seconds=perf_counter()-tick
                m.fit(X,y,seed=seed)
                tick=perf_counter();p=m.predict_proba(V);inference_seconds=perf_counter()-tick
                pred=m.labels[p.argmax(axis=1)];metrics=score(v.target,pred,LABELS)
                old=pd.read_csv(OLD/f'ring_seed_{seed}_predictions.csv')
                if method=='parameter_shift':
                    np.testing.assert_array_equal(pred,old.prediction)
                    with np.load(OLD/f'ring_seed_{seed}_weights.npz',allow_pickle=False) as w:
                        np.testing.assert_allclose(m.model.weights,w['final'],atol=1e-10,rtol=0)
                r={'method':method,'seed':seed,'training_fields':914,'validation_fields':150,'metrics':metrics,
                   'preprocessing_seconds':prep_seconds,'model_construction_seconds':constructor_seconds,
                   'fit_seconds':m.fit_seconds,'inference_seconds':inference_seconds,
                   'total_seconds':prep_seconds+constructor_seconds+m.fit_seconds+inference_seconds,
                   'resources':resources(m,method),'termination':m.termination,
                   'final_training_loss_bits':float(m.history[-1]['objective_bits']),
                   'selection_sha256':selection_hash,'historical_validation_labels_match':bool(np.array_equal(pred,old.prediction))}
                full.append(r)
                np.savez_compressed(out/'full'/f'{method}_seed_{seed}_weights.npz',initial=m.initial_point,final=m.model.weights)
                joblib.dump(prep,out/'full'/f'{method}_seed_{seed}_preprocessing.joblib')
                (out/'full'/f'{method}_seed_{seed}_history.json').write_text(json.dumps(m.history,indent=2))
                pd.DataFrame({'sample_id':v.sample_id,'target':v.target,'prediction':pred,
                              **{f'p_{c}':p[:,j] for j,c in enumerate(LABELS)}}).to_csv(out/'full'/f'{method}_seed_{seed}_predictions.csv',index=False)
                (out/'full_runs.json').write_text(json.dumps(full,indent=2))
                log(f'FINAL {method} seed={seed} F1={metrics["macro_f1"]:.6f} fit={m.fit_seconds:.4f}s cache={m.state_preparation_seconds:.4f}s total={r["total_seconds"]:.4f}s old-labels-match={r["historical_validation_labels_match"]}')
        assert sha(selection_path)==selection_hash
        assert all(sha(ROOT/p)==h for p,h in locked.items())
        assert all(sha(ROOT/p)==h for p,h in sources.items())
        metadata.update(status='complete',completed_at=now(),wall_seconds=perf_counter()-start_all,
                        prior_files_unchanged=True,CV_fits=len(records),full_fits=len(full))
        log('Complete. Demo notebook and all earlier files unchanged; no test inference or validation-driven settings.')
    except BaseException:
        metadata.update(status='failed',error=traceback.format_exc(),completed_at=now(),wall_seconds=perf_counter()-start_all);raise
    finally:
        logfile.close();(out/'metadata.json').write_text(json.dumps(metadata,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);run(p.parse_args().results)
