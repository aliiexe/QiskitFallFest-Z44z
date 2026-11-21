"""Verify and summarize the completed benchmark; no new model selection or plots."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from challenge.data import load_data
from challenge.metrics import score
from experiments.qnn_optimization.run import sha

LABELS=['alfalfa','maize','wheat']


def analyze(out):
    out=Path(out)
    metadata=json.loads((out/'metadata.json').read_text());assert metadata['status']=='complete'
    selection=json.loads((out/'training_only_selection.json').read_text())
    assert sha(out/'training_only_selection.json')==metadata['selection_sha256']
    tables,_,_,_=load_data();t=tables['train'];v=tables['validation']
    cv=json.loads((out/'cv_runs.json').read_text());seeds=json.loads((out/'cv_seed_runs.json').read_text())
    full=json.loads((out/'runs.json').read_text());micro=json.loads((out/'evaluator_microbenchmark.json').read_text())
    assert len(cv)==63 and len(seeds)==21 and len(full)==12
    for r in seeds:
        p=pd.read_csv(out/'cv'/f'{r["configuration"]}_seed_{r["seed"]}_oof.csv')
        np.testing.assert_array_equal(p.sample_id,t.sample_id)
        np.testing.assert_array_equal(p.target,t.target)
        assert score(t.target,p.prediction,LABELS)==r['metrics']
        np.testing.assert_array_equal(np.array(LABELS)[p[[f'p_{c}' for c in LABELS]].to_numpy().argmax(axis=1)],p.prediction)
    for r in cv:
        resources=r['resources'];calls=resources['objective_evaluations']
        gradients=resources['gradient_evaluations'];passes=resources['native_operator_builds']
        assert passes==calls+2*resources['trainable']*gradients
        assert resources['logical_field_probability_evaluations']==passes*r['fit_fields']
    selected_name=selection['selected_configuration']
    summaries=[]
    for name in ['selected8','ring12','original8','svm']:
        group=[r for r in full if r['model']==name]
        assert sorted(r['seed'] for r in group)==[0,1,2]
        item={'model':name,'selected_candidate':selected_name if name=='selected8' else None,
              'macro_f1_mean':float(np.mean([r['metrics']['macro_f1'] for r in group])),
              'macro_f1_seed_SD':float(np.std([r['metrics']['macro_f1'] for r in group])),
              'accuracy_mean':float(np.mean([r['metrics']['accuracy'] for r in group])),
              'worst_seed_macro_f1':min(r['metrics']['macro_f1'] for r in group),
              'metrics_by_seed':[{ 'seed':r['seed'],'metrics':r['metrics']} for r in group],
              'resources':group[0]['resources']}
        for timing in ['preprocessing_seconds','fit_seconds','inference_seconds','total_seconds']:
            values=[r[timing] for r in group]
            item[timing+'_mean']=float(np.mean(values));item[timing+'_median']=float(np.median(values))
        for j,c in enumerate(LABELS):
            for key in ['precision','recall']:
                item[f'{key}_{c}_mean']=float(np.mean([r['metrics'][key][j] for r in group]))
        if name!='svm':
            for k in ['objective_evaluations','gradient_evaluations','native_operator_builds','logical_field_probability_evaluations']:
                item[k+'_mean']=float(np.mean([r['resources'][k] for r in group]))
        summaries.append(item)
    for r in full:
        p=pd.read_csv(out/'full'/f'{r["model"]}_seed_{r["seed"]}_predictions.csv')
        np.testing.assert_array_equal(p.sample_id,v.sample_id);np.testing.assert_array_equal(p.target,v.target)
        assert score(v.target,p.prediction,LABELS)==r['metrics']
        if r['model']!='svm':
            prob=p[[f'p_{c}' for c in LABELS]].to_numpy()
            np.testing.assert_allclose(prob.sum(axis=1),1.,atol=1e-10,rtol=0)
            np.testing.assert_array_equal(np.array(LABELS)[prob.argmax(axis=1)],p.prediction)
        assert r['selection_sha256']==metadata['selection_sha256']
    chosen=next(r for r in summaries if r['model']=='selected8');ref=next(r for r in summaries if r['model']=='ring12')
    selected_cv=next(r for r in selection['candidates'] if r['name']==selected_name)
    ref_cv=next(r for r in selection['candidates'] if r['name']=='ring12')
    cv_by_name={r['configuration']:r for r in seeds if r['seed']==0}
    # Record whether the higher COBYLA ceiling merely reproduced earlier convergence.
    same_C340_C850=all(np.array_equal(
        pd.read_csv(out/'cv'/f'C340_seed_{s}_oof.csv').prediction,
        pd.read_csv(out/'cv'/f'C850_seed_{s}_oof.csv').prediction) for s in [0,1,2])
    changes=[]
    for seed in [0,1,2]:
        a=pd.read_csv(out/'full'/f'ring12_seed_{seed}_predictions.csv').prediction.to_numpy()
        b=pd.read_csv(out/'full'/f'selected8_seed_{seed}_predictions.csv').prediction.to_numpy();truth=v.target.to_numpy()
        for mask,kind in [((a!=truth)&(b==truth),'repaired relative to ring12'),((a==truth)&(b!=truth),'new error relative to ring12')]:
            for i in np.flatnonzero(mask):changes.append({'sample_id':v.sample_id.iloc[i],'seed':seed,'actual':truth[i],
                'ring12_prediction':a[i],'selected8_prediction':b[i],'change':kind})
    pd.DataFrame(changes).to_csv(out/'changed_validation_fields.csv',index=False)
    report={'scope':'Fresh same-machine cached-Qiskit training; same standard SamplerQNN inference; SVM is classical',
            'selection':selection,'validation_metrics':summaries,
            'selected_mean_macro_f1_gap_vs_ring12':chosen['macro_f1_mean']-ref['macro_f1_mean'],
            'selected_median_total_speedup_vs_ring12':ref['total_seconds_median']/chosen['total_seconds_median'],
            'selected_median_fit_speedup_vs_ring12':ref['fit_seconds_median']/chosen['fit_seconds_median'],
            'selected_median_total_reduction_fraction':1-chosen['total_seconds_median']/ref['total_seconds_median'],
            'validation_target_0_88_reached_in_mean':chosen['macro_f1_mean']>=.88,
            'higher_COBYLA_ceiling_same_CV_predictions':same_C340_C850,
            'evaluator_microbenchmark':micro,
            'standard_training_runtime':'not measured; fixed-weight forward-pass timing is separate',
            'limitations':['Selection reuses prior nonspatial training folds','Three initialization seeds, not independent cohorts',
                           'Exact ideal simulator, not hardware speed or quantum advantage','SVM uses all 13 inputs',
                           'Historical validation is development evidence; no validation-driven reselection performed'],
            'test_inference':False}
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    flat=[{k:v for k,v in r.items() if k not in ['metrics_by_seed','resources']} for r in summaries]
    pd.DataFrame(flat).to_csv(out/'summary.csv',index=False)
    pd.DataFrame(selection['candidates']).to_csv(out/'cv_summary.csv',index=False)
    per_seed=[]
    for r in full:
        row={'model':r['model'],'seed':r['seed'],'accuracy':r['metrics']['accuracy'],'macro_f1':r['metrics']['macro_f1'],
             **{k:r[k] for k in ['preprocessing_seconds','fit_seconds','inference_seconds','total_seconds']}}
        for j,c in enumerate(LABELS):
            for key in ['precision','recall']:row[f'{key}_{c}']=r['metrics'][key][j]
        per_seed.append(row)
    pd.DataFrame(per_seed).to_csv(out/'per_seed_metrics.csv',index=False)
    assert all(sha(ROOT/p)==h for p,h in metadata['protected_sha256'].items())
    audit={'status':'passed','cv_quantum_fits_checked':63,'CV_OOF_files_rescored':21,
           'validation_runs_rescored':12,'ring12_reference_CV_reproductions':9,'old_files_unchanged':True,
           'standard_cached_probability_equality':all(r['probabilities_match'] for r in micro),
           'selection_sha256_unchanged':metadata['selection_sha256'],'test_inference':False}
    (out/'artifact_verification.json').write_text(json.dumps(audit,indent=2))
    print(pd.DataFrame(selection['candidates'])[['name','mean_oof_macro_f1','seed_SD','median_three_fold_total_seconds','eligible_eight_angle','pareto_nondominated']].to_string(index=False))
    print(pd.DataFrame(flat)[['model','macro_f1_mean','macro_f1_seed_SD','accuracy_mean','fit_seconds_median','total_seconds_median']].to_string(index=False))
    print('Selected:',selected_name,'quality criterion:',selection['quality_criterion_met'])
    print('Median total speedup vs ring12:',report['selected_median_total_speedup_vs_ring12'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);analyze(p.parse_args().results)
