"""Reproduce the release from repository files only, using Python 3.12.

python scripts/verify_final_solution.py --execute-notebook --reproduce-cv
All generated files go to a new ignored output directory. No bundled model or
organizer file is overwritten. Jupyter execution needs local kernel sockets.
"""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'outputs/.matplotlib'))
os.environ.setdefault('JUPYTER_PLATFORM_DIRS', '1')

import numpy as np
import pandas as pd
from challenge.classical import reference
from challenge.data import load_data
from challenge.metrics import score
from challenge.models import angles
from qff_solution import FinalRingClassifier
from qff_solution.demo import (ARTIFACTS, LABELS, evaluate_saved, load_verified,
                               make_preprocessing, recorded_runs, train_ring, verify_artifacts)
from qff_solution.topology import TopologyClassifier


def gradient_checks(X, y):
    cases = []
    yi = np.array([LABELS.index(c) for c in y])
    for seed in (0, 1, 2):
        _, saved = load_verified(seed)
        model = FinalRingClassifier()
        states = model.prepare_states(X)
        shift = TopologyClassifier(topology='ring')
        shift_states = shift.prepare_states(X)
        for name, weights in [('initial', np.random.default_rng(seed).uniform(-np.pi, np.pi, 12)),
                              ('trained', saved['ring'].model.weights)]:
            loss, grad = model.loss_gradient(states, yi, weights)
            shift_loss, shift_grad = shift.loss_gradient(shift_states, yi, weights)
            eps = 1e-6
            numerical = []
            for j in range(12):
                direction = np.zeros(12); direction[j] = eps
                plus = model.loss_gradient(states, yi, weights+direction, gradient=False)
                minus = model.loss_gradient(states, yi, weights-direction, gradient=False)
                numerical.append((plus-minus)/(2*eps))
            # Independent standard Qiskit forward/backward at exactly fixed weights.
            p = model.qnn.forward(angles(X), weights)
            _, dp = model.qnn.backward(angles(X), weights)
            target = p[np.arange(len(y)), yi]
            qiskit_grad = -np.mean(dp[np.arange(len(y)), yi, :]/target[:, None], axis=0)/np.log(2)
            np.testing.assert_allclose(loss, shift_loss, atol=1e-12, rtol=0)
            np.testing.assert_allclose(grad, shift_grad, atol=1e-10, rtol=0)
            np.testing.assert_allclose(grad, numerical, atol=1e-7, rtol=0)
            np.testing.assert_allclose(grad, qiskit_grad, atol=1e-8, rtol=0)
            np.testing.assert_allclose(model.adjoint_proba(states, weights), p, atol=1e-9, rtol=0)
            cases.append({'seed': seed, 'weights': name,
                          'finite_difference_max_error': float(np.max(np.abs(grad-numerical))),
                          'qiskit_gradient_max_error': float(np.max(np.abs(grad-qiskit_grad))),
                          'parameter_shift_max_error': float(np.max(np.abs(grad-shift_grad)))})
    return cases


def execute_notebooks(output):
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client.kernelspec import KernelSpecManager
    # Private temporary kernelspec uses THIS clean environment, not a globally
    # registered research kernel. The committed notebook keeps generic python3.
    kernels = output/'kernels'
    spec = kernels/'qff-release-check'
    spec.mkdir(parents=True)
    (spec/'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'QFF release check', 'language': 'python'}))
    manager = KernelSpecManager(kernel_dirs=[str(kernels)])
    results = []
    for fresh in (False, True):
        notebook = nbformat.read(ROOT/'notebooks/QFF_Final_Ring_Demo.ipynb', 4)
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                cell.outputs = []; cell.execution_count = None
                if fresh and 'RETRAIN = False' in cell.source:
                    cell.source = cell.source.replace('RETRAIN = False', 'RETRAIN = True', 1)
        client = NotebookClient(notebook, timeout=180, kernel_name='qff-release-check',
                                resources={'metadata': {'path': str(ROOT/'notebooks')}})
        from jupyter_client import KernelManager
        client.km = KernelManager(kernel_name='qff-release-check', kernel_spec_manager=manager)
        tick = perf_counter()
        client.execute(cleanup_kc=True)
        assert all(c.execution_count is not None for c in notebook.cells if c.cell_type=='code')
        assert not any(o.output_type=='error' for c in notebook.cells if c.cell_type=='code' for o in c.outputs)
        # Use a portable generic kernel name in saved outputs too.
        notebook.metadata.kernelspec = {'name':'python3', 'language':'python', 'display_name':'Python 3 (ipykernel)'}
        filename = 'fresh_training.ipynb' if fresh else 'saved_models.ipynb'
        nbformat.write(notebook, output/filename)
        results.append({'workflow': 'fresh_training' if fresh else 'saved_models',
                        'code_cells_passed': sum(c.cell_type=='code' for c in notebook.cells),
                        'wall_seconds': perf_counter()-tick})
        print('Fresh kernel passed:', filename, flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute-notebook', action='store_true')
    parser.add_argument('--reproduce-cv', action='store_true')
    args = parser.parse_args()
    assert sys.version_info[:2] == (3, 12)
    output = ROOT/'outputs'/('release_check_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    verify_artifacts()
    protected = json.loads((ARTIFACTS/'organizer_sha256.json').read_text())
    for name, digest in protected.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, name
    tables, view, manifest, _ = load_data(ROOT)
    train, val = tables['train'], tables['validation']
    raw, raw_val = train[manifest['features']].to_numpy(), val[manifest['features']].to_numpy()
    y = train.target.to_numpy()
    import qff_solution, challenge
    assert Path(qff_solution.__file__).resolve().is_relative_to(ROOT)
    assert Path(challenge.__file__).resolve().is_relative_to(ROOT)
    subset = np.concatenate([np.flatnonzero(y==crop)[:4] for crop in LABELS])
    checks = gradient_checks(view['X_train'][subset], y[subset])
    print('Gradient checks passed for initial/trained weights and all seeds.', flush=True)
    recorded = recorded_runs()
    runs = []
    for seed in (0, 1, 2):
        replay = evaluate_saved(raw_val, val.target.to_numpy(), val.sample_id.to_numpy(), seed)
        for key, metrics in replay['metrics'].items():
            expected = next(r for r in recorded if r['model']==key and r['seed']==seed)
            assert metrics == expected['metrics'], (key, seed)
        prep, model, p, times = train_ring(raw, y, raw_val, seed)
        np.testing.assert_allclose(prep.transform(raw), view['X_train'], atol=1e-10, rtol=0)
        np.testing.assert_allclose(model.model.weights, replay['models']['ring'].model.weights, atol=1e-7, rtol=0)
        np.testing.assert_allclose(p, replay['models']['ring'].predict_proba(view['X_validation']), atol=1e-7, rtol=0)
        assert score(val.target, model.labels[p.argmax(axis=1)], LABELS)==replay['metrics']['ring']
        saved_history = json.loads((ARTIFACTS/f'ring_seed_{seed}_history.json').read_text())
        np.testing.assert_allclose([h['objective_bits'] for h in model.history],
                                   [h['objective_bits'] for h in saved_history], atol=1e-7, rtol=0)
        assert (model.circuit.num_qubits, len(model.weights), model.circuit.count_ops()['cx'], model.circuit.depth())==(4,12,8,12)
        assert model.gradient_method=='operator_adjoint'
        # Re-measure the controlled old/new comparison on this SAME machine.
        tick = perf_counter(); shift_prep = make_preprocessing()
        shift_X = shift_prep.fit_transform(raw)
        shift_V = shift_prep.transform(raw_val)
        shift_preprocessing = perf_counter()-tick
        tick = perf_counter(); shift = TopologyClassifier(topology='ring')
        constructor = perf_counter()-tick
        shift.fit(shift_X, y, seed)
        tick = perf_counter(); shifted_p = shift.predict_proba(shift_V)
        infer = perf_counter()-tick
        np.testing.assert_allclose(model.model.weights, shift.model.weights, atol=1e-7, rtol=0)
        np.testing.assert_allclose(p, shifted_p, atol=1e-7, rtol=0)
        assert len(model.history)==len(shift.history)
        assert model.forward_passes==len(model.history)
        assert shift.forward_passes==25*len(shift.history)
        runs.append({'seed':seed, 'macro_f1':replay['metrics']['ring']['macro_f1'],
                     'fast':times, 'shift_fit_seconds':shift.fit_seconds,
                     'shift_total_seconds':shift_preprocessing+constructor+shift.fit_seconds+infer,
                     'gradient_calls':len(model.history), 'fast_forward_passes':model.forward_passes,
                     'shift_forward_passes':shift.forward_passes})
        print(f'Full fit verified: seed {seed}; fast {model.fit_seconds:.4f}s; shift {shift.fit_seconds:.4f}s', flush=True)
    # The 13-feature SVM is fitted once: it is deterministic, not a three-seed search.
    svm = reference().fit(raw, y)
    np.testing.assert_array_equal(svm.predict(raw_val), replay['predictions']['svm'])
    cv = []
    if args.reproduce_cv:
        assignment = pd.read_csv(ARTIFACTS/'cv_fold_assignments.csv')
        np.testing.assert_array_equal(assignment.sample_id, train.sample_id)
        for seed in (0, 1, 2):
            oof = np.empty(len(train), dtype=object)
            for fold in (0, 1, 2):
                fit = assignment.fold.to_numpy()!=fold
                held = ~fit
                prep = make_preprocessing()
                X = prep.fit_transform(raw[fit])
                V = prep.transform(raw[held])
                model = FinalRingClassifier().fit(X, y[fit], seed)
                oof[held] = model.predict(V)
            expected = pd.read_csv(ARTIFACTS/f'cv_ring_seed_{seed}_oof.csv')
            np.testing.assert_array_equal(expected.sample_id, train.sample_id)
            np.testing.assert_array_equal(oof, expected.prediction)
            cv.append({'seed':seed, 'macro_f1':score(y, oof, LABELS)['macro_f1']})
        selection = json.loads((ARTIFACTS/'training_only_selection.json').read_text())
        expected_mean = next(r['mean_oof_macro_f1'] for r in selection['candidates'] if r['method']=='operator_adjoint')
        np.testing.assert_allclose(np.mean([r['macro_f1'] for r in cv]), expected_mean, atol=1e-12, rtol=0)
        print('All training-only CV out-of-fold predictions reproduced.', flush=True)
    notebooks = execute_notebooks(output) if args.execute_notebook else []
    report = {'status':'passed', 'python':sys.version.split()[0],
              'packages':{p:version(p) for p in ['numpy','scipy','qiskit','qiskit-machine-learning','scikit-learn']},
              'gradient_checks':checks, 'full_training':runs, 'training_only_cv':cv,
              'notebooks':notebooks, 'organizer_file_hashes_passed':len(protected),
              'source_imports_from_this_checkout':True, 'all_artifact_checksums_passed':True,
              'timing_scope':'Fresh local CPU times, includes every fit cache. Imports, I/O and rendering excluded.'}
    (output/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print('PASS:', output.relative_to(ROOT), flush=True)


if __name__=='__main__':
    main()
