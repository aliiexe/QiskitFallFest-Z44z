"""Audit the paired derivative benchmark and produce a compact external report."""
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


def analyze(out):
    out=Path(out);meta=json.loads((out/'metadata.json').read_text());assert meta['status']=='complete'
    selection=json.loads((out/'training_only_selection.json').read_text())
    assert sha(out/'training_only_selection.json')==meta['selection_sha256']
    verified=json.loads((out/'verification.json').read_text());assert verified['status']=='passed'
    cv=json.loads((out/'cv_runs.json').read_text());seeds=json.loads((out/'cv_seed_runs.json').read_text())
    full=json.loads((out/'full_runs.json').read_text())
    tables,_,_,_=load_data();train=tables['train'];validation=tables['validation']
    assert len(cv)==27 and len(seeds)==9
    selected=selection['selected_method'];comparisons=[];loss_differences=[]
    for method in ['parameter_shift','batched_shift','operator_adjoint']:
        for seed in [0,1,2]:
            frame=pd.read_csv(out/'cv'/f'{method}_seed_{seed}_oof.csv')
            np.testing.assert_array_equal(frame.sample_id,train.sample_id)
            np.testing.assert_array_equal(frame.target,train.target)
            record=next(r for r in seeds if r['method']==method and r['seed']==seed)
            assert score(train.target,frame.prediction,LABELS)==record['metrics']
            base=pd.read_csv(out/'cv'/f'parameter_shift_seed_{seed}_oof.csv')
            np.testing.assert_array_equal(frame.prediction,base.prediction)
            for fold in [0,1,2]:
                r=next(r for r in cv if r['method']==method and r['seed']==seed and r['fold']==fold)
                b=next(r for r in cv if r['method']=='parameter_shift' and r['seed']==seed and r['fold']==fold)
                np.testing.assert_array_equal(r['initial_weights'],b['initial_weights'])
                assert r['resources']['qubits']==4 and r['resources']['trainable']==12
                assert r['resources']['two_qubit_gates']==8 and r['resources']['depth']==12
                assert r['resources']['readout_group_sizes']==[6,5,5]
                history=json.loads((out/'cv'/f'{method}_seed_{seed}_fold_{fold}_history.json').read_text())
                baseline=json.loads((out/'cv'/f'parameter_shift_seed_{seed}_fold_{fold}_history.json').read_text())
                same_calls=len(history)==len(baseline)
                max_loss_delta=float(np.max(abs(np.asarray([h['objective_bits'] for h in history])-
                                               np.asarray([h['objective_bits'] for h in baseline])))) if same_calls else None
                if max_loss_delta is not None:loss_differences.append(max_loss_delta)
                comparisons.append({'method':method,'seed':seed,'fold':fold,'same_objective_call_count':same_calls,
                                    'training_curve_max_loss_difference_bits':max_loss_delta,
                                    'final_weights_max_absolute_difference':float(np.max(abs(np.array(r['final_weights'])-np.array(b['final_weights']))))})
                calls=r['resources']['objective_evaluations'];assert calls<=100 and calls==r['resources']['gradient_evaluations']
                if method=='operator_adjoint':
                    assert r['resources']['complete_forward_probability_passes']==calls
                    assert r['resources']['reverse_sweeps']==calls
                    assert r['resources']['small_16x16_matrix_products']==52*calls
                    assert r['resources']['field_batch_matrix_products']==2*calls
                    assert r['resources']['native_operator_builds']==9
                else:
                    assert r['resources']['complete_forward_probability_passes']==25*calls
                    assert r['resources']['native_operator_builds']==25*calls
    summaries=[];per_seed=[]
    for method in sorted(set(r['method'] for r in full)):
        rows=[r for r in full if r['method']==method];assert sorted(r['seed'] for r in rows)==[0,1,2]
        item={'method':method,'macro_f1_mean':float(np.mean([r['metrics']['macro_f1'] for r in rows])),
              'macro_f1_seed_SD':float(np.std([r['metrics']['macro_f1'] for r in rows])),
              'accuracy_mean':float(np.mean([r['metrics']['accuracy'] for r in rows])),
              'gradient_calls_per_seed':[r['resources']['gradient_evaluations'] for r in rows],
              'all_historical_validation_labels_match':all(r['historical_validation_labels_match'] for r in rows)}
        for key in ['preprocessing_seconds','model_construction_seconds','fit_seconds','inference_seconds','total_seconds']:
            item[key+'_median']=float(np.median([r[key] for r in rows]))
            item[key+'_mean']=float(np.mean([r[key] for r in rows]))
        for key in ['state_preparation_seconds','fixed_gate_cache_seconds','input_batch_conversion_seconds']:
            item[key+'_median']=float(np.median([r['resources'][key] for r in rows]))
        for j,c in enumerate(LABELS):
            for key in ['precision','recall']:item[f'{key}_{c}_mean']=float(np.mean([r['metrics'][key][j] for r in rows]))
        summaries.append(item)
        for r in rows:
            frame=pd.read_csv(out/'full'/f'{method}_seed_{r["seed"]}_predictions.csv')
            np.testing.assert_array_equal(frame.sample_id,validation.sample_id)
            np.testing.assert_array_equal(frame.target,validation.target)
            assert score(validation.target,frame.prediction,LABELS)==r['metrics']
            base=pd.read_csv(out/'full'/f'parameter_shift_seed_{r["seed"]}_predictions.csv')
            np.testing.assert_array_equal(frame.prediction,base.prediction)
            assert r['selection_sha256']==meta['selection_sha256']
            row={'method':method,'seed':r['seed'],'accuracy':r['metrics']['accuracy'],'macro_f1':r['metrics']['macro_f1'],
                 'gradient_calls':r['resources']['gradient_evaluations'],'final_training_loss_bits':r['final_training_loss_bits']}
            for k in ['preprocessing_seconds','model_construction_seconds','fit_seconds','inference_seconds','total_seconds']:row[k]=r[k]
            for k in ['state_preparation_seconds','fixed_gate_cache_seconds','complete_forward_probability_passes','native_operator_builds']:row[k]=r['resources'][k]
            per_seed.append(row)
    baseline=next(r for r in summaries if r['method']=='parameter_shift')
    fast=next(r for r in summaries if r['method']==selected)
    summary={'scope':'Same exact twelve-angle/four-qubit ring, L-BFGS-B unchanged; local ideal CPU simulation',
             'profile':json.loads((out/'profile.json').read_text()),'training_only_selection':selection,
             'full_training_comparison':summaries,
             'median_fit_speedup':baseline['fit_seconds_median']/fast['fit_seconds_median'],
             'median_total_speedup':baseline['total_seconds_median']/fast['total_seconds_median'],
             'all_CV_and_validation_labels_preserved':True,
             'all_CV_objective_counts_match':all(r['same_objective_call_count'] for r in comparisons),
             'CV_training_curve_max_loss_difference_bits':max(loss_differences),
             'gradient_finite_difference_max_error':max(r['finite_difference_error'] for r in verified['gradient_checks']),
             'gradient_official_Qiskit_max_error':max(r['official_Qiskit_backward_error'] for r in verified['gradient_checks']),
             'frozen_probability_max_error':max(r['max_probability_error'] for r in verified['frozen_training_probabilities']),
             'runtime_includes_all_cache_construction':True,
             'same_inference_engine':'unchanged standard SamplerQNN/QMLSampler(shots=None)',
             'backend_note':'New matrix-adjoint and batched-shift evaluators use NumPy on actual Qiskit-generated operators; not official reverse primitives or hardware algorithms',
             'no_new_notebook':True,'test_inference':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'convergence_comparison.json').write_text(json.dumps(comparisons,indent=2))
    pd.DataFrame(selection['candidates']).to_csv(out/'CV_summary.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'full_summary.csv',index=False)
    pd.DataFrame(per_seed).to_csv(out/'per_seed_metrics.csv',index=False)
    plt.rcParams.update({'font.size':10,'figure.dpi':140,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
    for ax,key,title in zip(axes,['fit_seconds_median','total_seconds_median'],['Training, including cache construction','Total: prepare + construct + train + infer']):
        vals=[baseline[key],fast[key]]
        bars=ax.bar([0,1],vals,color=['#c38a2c','#23779d'])
        ax.bar_label(bars,labels=[f'{v:.3f} s' for v in vals],padding=4)
        ax.set(xticks=[0,1],xticklabels=['Existing\nparameter shift','Qiskit-verified\noperator adjoint'],
               ylim=(0,max(vals)*1.22),ylabel='Seconds (median across seeds 0,1,2)',title=title)
    fig.savefig(out/'runtime_comparison.png',bbox_inches='tight');plt.close(fig)
    assert all(sha(ROOT/p)==h for p,h in meta['protected_sha256'].items())
    assert all(sha(ROOT/p)==h for p,h in meta['source_sha256'].items())
    audit={'status':'passed','CV_fits_rescored':27,'OOF_prediction_files_rescored':9,
           'validation_prediction_files_rescored':len(full),'all_CV_and_validation_predicted_labels_unchanged':True,
           'work_counters_verified':True,'same_initial_points':True,'same_circuit_shape':True,
           'all_previous_files_unchanged':True,
           'demo_notebook_sha256_unchanged':sha(ROOT/'notebooks/QFF_Ring_Demo.ipynb'),
           'selection_sha256_unchanged':meta['selection_sha256'],'analysis_source_sha256':sha(Path(__file__)),
           'no_new_notebook':True,'test_inference':False}
    (out/'artifact_verification.json').write_text(json.dumps(audit,indent=2))
    print(pd.DataFrame(selection['candidates'])[['method','mean_oof_macro_f1','seed_SD','median_three_fold_total_seconds','quality_gate_passed']].to_string(index=False))
    print(pd.DataFrame(summaries)[['method','macro_f1_mean','accuracy_mean','fit_seconds_median','state_preparation_seconds_median','inference_seconds_median','total_seconds_median']].to_string(index=False))
    print('Training speedup:',summary['median_fit_speedup'],'Total speedup:',summary['median_total_speedup'])
    print('Maximum CV training-curve loss difference:',summary['CV_training_curve_max_loss_difference_bits'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);analyze(p.parse_args().results)
