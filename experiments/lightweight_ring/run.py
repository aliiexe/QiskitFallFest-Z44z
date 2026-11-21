"""Training-only speed/accuracy selection, then a single locked validation comparison."""
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
from challenge.classical import reference
from challenge.models import angles
from challenge.metrics import score
from experiments.qnn_optimization.run import preprocessing,sha,now
from experiments.qnn_optimization.model import NativeQiskitClassifier
from experiments.entanglement_topology.model import TopologyClassifier
from experiments.lightweight_ring.model import LightweightRingClassifier

LABELS=['alfalfa','maize','wheat']
OLD=ROOT/'outputs/entanglement_topology_2026-11-21'


def classifier(config):
    if config['name']=='ring12':return TopologyClassifier(topology='ring',max_evals=100)
    if config['name']=='original8':return NativeQiskitClassifier(reps=1,optimizer='COBYLA',max_evals=100)
    return LightweightRingClassifier(optimizer=config['optimizer'],max_evals=config['max_evals'])


def run(out):
    out=Path(out).resolve()
    check=json.loads((out/'verification.json').read_text());assert check['status']=='passed'
    assert all(sha(ROOT/p)==h for p,h in check['verified_sources'].items())
    if (out/'metadata.json').exists():raise FileExistsError('Existing experiment is preserved; use a new destination.')
    protocol=json.loads((ROOT/'experiments/lightweight_ring/protocol.json').read_text())
    tracked=[ROOT/p for p in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0') if p]
    protected={p for p in tracked if not str(p.relative_to(ROOT)).startswith('experiments/lightweight_ring/')}
    for folder in (ROOT/'outputs').iterdir():
        if folder.resolve()!=out:
            if folder.is_dir():protected.update(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
            elif folder.is_file():protected.add(folder)
    locked={str(p.relative_to(ROOT)):sha(p) for p in protected if p.is_file()}
    sources={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'experiments/lightweight_ring').iterdir() if p.suffix in ('.py','.json')}
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    assert branch=='research/quantum-model-optimization'
    metadata={'status':'running','started_at':now(),'branch':branch,
              'starting_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'protocol':protocol,'protected_sha256':locked,'implementation_sha256':sources,
              'platform':platform.platform(),'python':sys.version,
              'packages':{p:version(p) for p in ['qiskit','qiskit-machine-learning','numpy','scipy','scikit-learn']},
              'test_inference':False}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    (out/'cv').mkdir();(out/'full').mkdir()
    logfile=(out/'execution.log').open('w',buffering=1)
    def log(message):print(message,flush=True);logfile.write(message+'\n')
    start_all=perf_counter()
    try:
        tables,release,manifest,_=load_data();t=tables['train'];F=manifest['features']
        raw=t[F].to_numpy();y=t.target.to_numpy()
        assignment=pd.read_csv(OLD/'cv_fold_assignments.csv')
        np.testing.assert_array_equal(assignment.sample_id,t.sample_id)
        fold_ids=assignment.fold.to_numpy();assert set(fold_ids)=={0,1,2}
        assignment.to_csv(out/'cv_fold_assignments.csv',index=False)
        fold_data=[]
        for fold in (0,1,2):
            fit=np.flatnonzero(fold_ids!=fold);held=np.flatnonzero(fold_ids==fold)
            tick=perf_counter();prep=preprocessing();X=prep.fit_transform(raw[fit]);V=prep.transform(raw[held])
            fold_data.append((fit,held,X,V,perf_counter()-tick))
        old_cv=json.loads((OLD/'cv_runs.json').read_text())
        configurations=protocol['candidates']+[{'name':'ring12','optimizer':'L-BFGS-B','max_evals':100}]
        cv_records=[];seed_records=[];summaries=[]
        log('Training-only: six eight-angle budgets + twelve-angle ring; three folds and seeds 0/1/2. Validation not scored.')
        for config in configurations:
            name=config['name']
            for seed in (0,1,2):
                oof=np.empty(len(t),dtype=object);oof_p=np.empty((len(t),3));seed_time=0.
                for fold,(fit,held,X,V,prep_seconds) in enumerate(fold_data):
                    m=classifier(config).fit(X,y[fit],seed=seed)
                    tick=perf_counter();p=m.predict_proba(V);inference_seconds=perf_counter()-tick
                    pred=m.labels[p.argmax(axis=1)];oof[held]=pred;oof_p[held]=p
                    metrics=score(y[held],pred,LABELS)
                    expected=np.random.default_rng(seed).uniform(-np.pi,np.pi,len(m.weights))
                    np.testing.assert_array_equal(m.initial_point,expected)
                    assert len(m.history)<=config['max_evals']
                    reference_match=None
                    if name=='ring12':
                        old=next(r for r in old_cv if r['topology']=='ring' and r['seed']==seed and r['fold']==fold)
                        np.testing.assert_allclose(m.model.weights,old['final_weights'],atol=1e-10,rtol=0)
                        assert metrics==old['metrics'];reference_match=True
                    resources=m.resources()
                    passes_per_call=1+2*len(m.weights) if config['optimizer']=='L-BFGS-B' else 1
                    assert resources['native_operator_builds']==len(m.history)*passes_per_call
                    assert resources['logical_field_probability_evaluations']==resources['native_operator_builds']*len(fit)
                    total=prep_seconds+m.fit_seconds+inference_seconds;seed_time+=total
                    record={'configuration':name,'seed':seed,'fold':fold,'fit_fields':len(fit),'evaluation_fields':len(held),
                            'metrics':metrics,'preprocessing_seconds':prep_seconds,'fit_seconds':m.fit_seconds,
                            'inference_seconds':inference_seconds,'total_seconds':total,'resources':resources,
                            'termination':m.termination,'initial_weights':m.initial_point.tolist(),
                            'final_weights':m.model.weights.tolist(),'historical_reference_match':reference_match}
                    cv_records.append(record)
                    (out/'cv'/f'{name}_seed_{seed}_fold_{fold}_history.json').write_text(json.dumps(m.history,indent=2))
                    (out/'cv_runs.json').write_text(json.dumps(cv_records,indent=2))
                    log(f'CV {name} seed={seed} fold={fold} F1={metrics["macro_f1"]:.4f} fit={m.fit_seconds:.2f}s calls={len(m.history)} passes={resources["native_operator_builds"]}')
                metrics=score(y,oof,LABELS)
                seed_records.append({'configuration':name,'seed':seed,'metrics':metrics,'three_fold_total_seconds':seed_time})
                pd.DataFrame({'sample_id':t.sample_id,'target':y,'prediction':oof,'fold':fold_ids,
                              **{f'p_{c}':oof_p[:,j] for j,c in enumerate(LABELS)}}).to_csv(out/'cv'/f'{name}_seed_{seed}_oof.csv',index=False)
                (out/'cv_seed_runs.json').write_text(json.dumps(seed_records,indent=2))
            rows=[r for r in seed_records if r['configuration']==name]
            summaries.append({**config,'mean_oof_macro_f1':float(np.mean([r['metrics']['macro_f1'] for r in rows])),
                              'seed_SD':float(np.std([r['metrics']['macro_f1'] for r in rows])),
                              'worst_seed_macro_f1':float(min(r['metrics']['macro_f1'] for r in rows)),
                              'median_three_fold_total_seconds':float(np.median([r['three_fold_total_seconds'] for r in rows])),
                              'trainable':12 if name=='ring12' else 8})
        # Deterministic fixed SVM CV reference: one fit per fold, no fictitious seed replicates.
        svm_oof=np.empty(len(t),dtype=object);svm_cv=[]
        for fold,(fit,held,_,_,_) in enumerate(fold_data):
            svm=reference();tick=perf_counter()
            X=svm[0].fit_transform(raw[fit]);V=svm[0].transform(raw[held]);prep_seconds=perf_counter()-tick
            tick=perf_counter();svm[-1].fit(X,y[fit]);fit_seconds=perf_counter()-tick
            tick=perf_counter();pred=svm[-1].predict(V);infer=perf_counter()-tick
            svm_oof[held]=pred
            svm_cv.append({'fold':fold,'metrics':score(y[held],pred,LABELS),
                           'preprocessing_seconds':prep_seconds,'fit_seconds':fit_seconds,'inference_seconds':infer})
        pd.DataFrame({'sample_id':t.sample_id,'target':y,'prediction':svm_oof,'fold':fold_ids}).to_csv(out/'cv/svm_oof.csv',index=False)
        (out/'svm_cv.json').write_text(json.dumps({'folds':svm_cv,'pooled_metrics':score(y,svm_oof,LABELS)},indent=2))
        best=max(r['mean_oof_macro_f1'] for r in summaries)
        ref=next(r for r in summaries if r['name']=='ring12')
        for r in summaries:
            r['eligible_eight_angle']=r['trainable']==8 and r['mean_oof_macro_f1']>=best-.01 and r['worst_seed_macro_f1']>=ref['worst_seed_macro_f1']-.01
            r['pareto_nondominated']=not any(a['mean_oof_macro_f1']>=r['mean_oof_macro_f1'] and a['median_three_fold_total_seconds']<=r['median_three_fold_total_seconds'] and
                                             (a['mean_oof_macro_f1']>r['mean_oof_macro_f1'] or a['median_three_fold_total_seconds']<r['median_three_fold_total_seconds']) for a in summaries)
        eligible=[r for r in summaries if r['eligible_eight_angle']]
        selected=(min(eligible,key=lambda r:(r['median_three_fold_total_seconds'],-r['mean_oof_macro_f1'],r['max_evals'])) if eligible
                  else max((r for r in summaries if r['trainable']==8),key=lambda r:(r['mean_oof_macro_f1'],-r['median_three_fold_total_seconds'])))
        selection={'selected_configuration':selected['name'],'selected_config':{k:selected[k] for k in ['name','optimizer','max_evals']},
                   'locked_at':now(),'selection_data':'914 training fields only; no validation scores used',
                   'rule':protocol['selection'],'quality_criterion_met':bool(eligible),'candidates':summaries,
                   'best_quantum_mean_oof_macro_f1':best,'eligibility_mean_floor':best-.01,
                   'worst_seed_floor':ref['worst_seed_macro_f1']-.01}
        path=out/'training_only_selection.json';path.write_text(json.dumps(selection,indent=2));selection_hash=sha(path)
        log(f'SELECTION LOCKED: {selected["name"]}; quality criterion met={bool(eligible)}. Now evaluate only selected configuration and references on validation.')
        metadata.update(selected_configuration=selected['name'],selection_sha256=selection_hash)
        (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
        # Fresh, same-machine final comparison. No candidate/seed reselection follows.
        v=tables['validation'];val_raw=v[F].to_numpy();full_records=[];fitted_models={};prepared={}
        old_full=json.loads((OLD/'runs.json').read_text())
        old_original=json.loads((ROOT/'outputs/qnn_optimization_2026-11-21/runs.json').read_text())
        final_configs=[{**selection['selected_config'],'name':'selected8'},
                       {'name':'ring12','optimizer':'L-BFGS-B','max_evals':100},
                       {'name':'original8','optimizer':'COBYLA','max_evals':100},{'name':'svm'}]
        for config in final_configs:
            name=config['name']
            for seed in (0,1,2):
                if name=='svm':
                    m=reference(seed);tick=perf_counter()
                    X=m[0].fit_transform(raw);V=m[0].transform(val_raw);prep_seconds=perf_counter()-tick
                    tick=perf_counter();m[-1].fit(X,y);fit_seconds=perf_counter()-tick
                    tick=perf_counter();pred=m[-1].predict(V);infer_seconds=perf_counter()-tick;p=None
                    resources={'inputs':13,'qubits':0,'trainable_quantum':0,'support_vectors':int(m[-1].n_support_.sum())}
                    joblib.dump(m,out/'full'/f'svm_seed_{seed}.joblib')
                    termination=None
                else:
                    tick=perf_counter();prep=preprocessing();X=prep.fit_transform(raw);V=prep.transform(val_raw);prep_seconds=perf_counter()-tick
                    np.testing.assert_allclose(X,release['X_train'],atol=1e-10,rtol=0)
                    m=classifier(config).fit(X,y,seed=seed);fit_seconds=m.fit_seconds
                    tick=perf_counter();p=m.predict_proba(V);infer_seconds=perf_counter()-tick
                    pred=m.labels[p.argmax(axis=1)]
                    np.testing.assert_allclose(p[:6],m.statevector_proba(m.prepare_states(V[:6]),m.model.weights,count=False),atol=1e-10,rtol=0)
                    resources=m.resources();termination=m.termination
                    np.savez_compressed(out/'full'/f'{name}_seed_{seed}_weights.npz',initial=m.initial_point,final=m.model.weights)
                    (out/'full'/f'{name}_seed_{seed}_history.json').write_text(json.dumps(m.history,indent=2))
                    if name=='selected8' and seed==0:joblib.dump(prep,out/'preprocessing.joblib')
                    fitted_models[name,seed]=m;prepared[name,seed]=X
                metrics=score(v.target,pred,LABELS)
                if name=='ring12':
                    old=next(r for r in old_full if r['topology']=='ring' and r['seed']==seed)
                    assert metrics==old['metrics']
                    with np.load(OLD/f'ring_seed_{seed}_weights.npz',allow_pickle=False) as w:np.testing.assert_allclose(m.model.weights,w['final'],atol=1e-10,rtol=0)
                if name=='original8':
                    old=next(r for r in old_original if r['configuration']=='A' and r['seed']==seed)
                    assert metrics==old['metrics']
                frame=pd.DataFrame({'sample_id':v.sample_id,'target':v.target,'prediction':pred})
                if p is not None:
                    for j,c in enumerate(LABELS):frame[f'p_{c}']=p[:,j]
                frame.to_csv(out/'full'/f'{name}_seed_{seed}_predictions.csv',index=False)
                record={'model':name,'selected_candidate':selected['name'] if name=='selected8' else None,'seed':seed,
                        'origin':'FRESH local fit after training-only selection was locked','training_fields':914,'validation_fields':150,
                        'metrics':metrics,'preprocessing_seconds':prep_seconds,'fit_seconds':fit_seconds,
                        'inference_seconds':infer_seconds,'total_seconds':prep_seconds+fit_seconds+infer_seconds,
                        'resources':resources,'termination':termination,'selection_sha256':selection_hash,
                        'training_engine':'StandardScaler + SVC (classical)' if name=='svm' else 'Qiskit cached fixed-input states; bound ansatz Operator each pass',
                        'inference_engine':'SVC' if name=='svm' else 'standard SamplerQNN / QMLSampler(shots=None)'}
                full_records.append(record);(out/'runs.json').write_text(json.dumps(full_records,indent=2))
                log(f'VALIDATION {name} seed={seed} F1={metrics["macro_f1"]:.4f} accuracy={metrics["accuracy"]:.4f} total={record["total_seconds"]:.3f}s')
        # Isolate evaluator overhead with identical frozen circuits and weights.
        log('Fixed-weight evaluator microbenchmark; not standard optimizer training-time measurement.')
        micro=[]
        for name in ['selected8','ring12','original8']:
            m=fitted_models[name,0];X=prepared[name,0];w=m.model.weights
            tick=perf_counter();states=m.prepare_states(X);state_seconds=perf_counter()-tick
            cached=[];sampler=[]
            for repeat in (0,1,2):
                tick=perf_counter();a=m.statevector_proba(states,w,count=False);cached.append(perf_counter()-tick)
                tick=perf_counter();b=m.qnn.forward(angles(X),w);sampler.append(perf_counter()-tick)
                np.testing.assert_allclose(a,b,atol=1e-10,rtol=0)
            micro.append({'model':name,'fields':914,'weights_seed':0,'input_state_preparation_seconds':state_seconds,
                          'cached_forward_seconds':cached,'standard_sampler_forward_seconds':sampler,
                          'probabilities_match':True,'scope':'one fixed-weight full-training forward pass; not optimizer training'})
            log(f'ENGINE {name}: median cached pass={np.median(cached):.4f}s, standard pass={np.median(sampler):.4f}s')
        (out/'evaluator_microbenchmark.json').write_text(json.dumps(micro,indent=2))
        assert sha(path)==selection_hash
        assert all(sha(ROOT/p)==h for p,h in locked.items())
        assert all(sha(ROOT/p)==h for p,h in sources.items())
        metadata.update(status='complete',completed_at=now(),wall_seconds=perf_counter()-start_all,
                        protected_files_unchanged=True,cv_quantum_fits=len(cv_records),svm_cv_fits=3,full_training_fits=len(full_records))
        log('Complete. All previous files unchanged. No validation-driven reselection or test inference.')
    except BaseException:
        metadata.update(status='failed',error=traceback.format_exc(),completed_at=now(),wall_seconds=perf_counter()-start_all);raise
    finally:
        logfile.close();(out/'metadata.json').write_text(json.dumps(metadata,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);run(p.parse_args().results)
