"""Run the preregistered 1.0 vs 0.5 vs 0.25 encoding comparison.

Reuse a VERIFIED LOCAL full-training baseline for scale 1.0. Train only the
six additional candidate fits. Preserve the starter package and notebook.
"""
import argparse
from contextlib import redirect_stdout
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from challenge.data import load_data
from challenge.metrics import score
from challenge.models import QuantumClassifier, angles
from experiments.encoding_scale.model import ScaledQuantumClassifier
from experiments.encoding_scale.analyze import analyze

LABELS = ['alfalfa', 'maize', 'wheat']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(ZoneInfo('Africa/Casablanca')).isoformat()


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()
        return len(message)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    baseline = args.baseline if args.baseline.is_absolute() else ROOT / args.baseline
    out = args.results if args.results.is_absolute() else ROOT / args.results
    baseline_meta = json.loads((baseline / 'run_metadata.json').read_text())
    assert baseline_meta['status'] == 'complete' and baseline_meta['protected_files_unchanged']
    baseline_runs = json.loads((baseline / 'runs.json').read_text())
    for relative in ['challenge/models.py', 'challenge/coverage.py', 'challenge/data.py',
                     'challenge/metrics.py', 'challenge/classical.py', 'configs/reference.json']:
        assert sha(ROOT / relative) == sha(baseline / 'source_snapshot' / relative), relative
    tables, _, manifest, _ = load_data()
    train, val = tables['train'], tables['validation']
    features = manifest['features']
    assert len(train) == 914 and len(val) == 150
    training_sha = hashlib.sha256(train.to_csv(index=False).encode()).hexdigest()
    baseline_qnn = sorted([r for r in baseline_runs if r['model']=='qnn'], key=lambda r:r['seed'])
    assert [r['seed'] for r in baseline_qnn] == [0,1,2]
    for r in baseline_qnn:
        assert r['training_rows'] == 914 and r['training_sha256'] == training_sha
        assert r['validation_sha256'] == manifest['files']['validation.csv']
        assert r['resources']['objective_evaluations'] == 100

    out.mkdir(parents=True, exist_ok=False)
    protected = {str(p.relative_to(ROOT)): sha(p) for name in ('challenge','configs','data','results','notebooks')
                 for p in (ROOT / name).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    prep = make_pipeline(StandardScaler(), PCA(4, svd_solver='full'), StandardScaler())
    tick = perf_counter()
    A = prep.fit_transform(train[features].to_numpy())
    B = prep.transform(val[features].to_numpy())
    prep_seconds = perf_counter()-tick
    joblib.dump(prep, out / 'preprocessing.joblib')
    np.savez_compressed(out / 'prepared_inputs.npz', X_train=A, X_validation=B,
                        ids_train=train.sample_id.to_numpy(dtype=str), ids_validation=val.sample_id.to_numpy(dtype=str))
    initial = {str(seed): np.random.default_rng(seed).uniform(-np.pi,np.pi,8).tolist() for seed in (0,1,2)}
    (out / 'initial_weights.json').write_text(json.dumps(initial,indent=2))
    # At scale 1 the adapter must be exactly the starter, without training.
    starter = QuantumClassifier(LABELS, max_evals=100, encoding='ry')
    identity = ScaledQuantumClassifier(LABELS, max_evals=100, scale=1.0)
    np.testing.assert_allclose(starter.qnn.forward(angles(B[:6]),initial['0']),
                               identity.qnn.forward(angles(B[:6]*identity.scale),initial['0']),atol=1e-12)
    for source in [*sorted((ROOT/'experiments/encoding_scale').glob('*.py')),
                   ROOT/'challenge/models.py', ROOT/'requirements.txt']:
        target = out / 'source_snapshot' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source,target)
    shutil.copy2(baseline/'run_metadata.json',out/'local_baseline_metadata.json')
    records = []
    for r in baseline_qnn:
        copied = {**r, 'scale': 1.0, 'origin': 'reused verified LOCAL full-training baseline',
                  'baseline_results': str(baseline.relative_to(ROOT))}
        records.append(copied)
        shutil.copy2(baseline/f'qnn_{r["seed"]}_predictions.csv',out/f'scale_1_seed_{r["seed"]}_predictions.csv')
        shutil.copy2(baseline/f'qnn_{r["seed"]}_history.json',out/f'scale_1_seed_{r["seed"]}_history.json')
    (out/'runs.json').write_text(json.dumps(records,indent=2))
    metadata = {'status': 'running', 'started_at': now(),
                'branch': subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
                'hypothesis': 'Reducing tanh input scale may reduce saturation and improve validation classification.',
                'design': {'scales':[1.0,0.5,0.25], 'seeds':[0,1,2], 'training_fields':914,'validation_fields':150,
                           'encoding':'theta_j(s) = pi/2 * (1 + tanh(s*z_j))',
                           'fixed':['same fields/order','same train-fitted PCA/scalers','same initial weights per seed',
                                    'four-qubit RY circuit','eight trainable angles','COBYLA, 100 evaluations','exact probabilities'],
                           'fresh_candidate_fits':6, 'test_inference':False},
                'provenance': {'baseline':'local run, not organizer-provided metrics',
                               'baseline_path':str(baseline.relative_to(ROOT)),
                               'baseline_runs_sha256':sha(baseline/'runs.json'),
                               'organizer_results':'separate reference only, not substituted for local results'},
                'training_sha256':training_sha,'validation_sha256':manifest['files']['validation.csv'],
                'prepared_training_array_sha256':hashlib.sha256(A.tobytes()).hexdigest(),
                'prepared_validation_array_sha256':hashlib.sha256(B.tobytes()).hexdigest(),
                'protected_file_sha256_before':protected,'identity_adapter_check':'passed at scale=1',
                'environment':baseline_meta['environment'],'package_versions':baseline_meta['package_versions']}
    meta_path = out/'experiment_metadata.json'
    meta_path.write_text(json.dumps(metadata,indent=2))
    start = perf_counter()
    try:
        with (out/'training.log').open('w',buffering=1) as log, redirect_stdout(Tee(sys.stdout,log)):
            print('Reusing 3 verified LOCAL scale=1 runs. Starting 6 candidate fits.',flush=True)
            for scale in (0.5,0.25):
                for seed in (0,1,2):
                    model = ScaledQuantumClassifier(LABELS,max_evals=100,scale=scale)
                    tick = perf_counter()
                    model.fit(A,train.target.to_numpy(),seed=seed)
                    fit_seconds = perf_counter()-tick
                    np.testing.assert_array_equal(model.model.initial_point,initial[str(seed)])
                    assert len(model.history) == model.resources()['objective_evaluations'] == 100
                    tick = perf_counter()
                    probability = model.predict_proba(B)
                    prediction = model.labels[probability.argmax(axis=1)]
                    predict_seconds = perf_counter()-tick
                    np.testing.assert_allclose(probability[:6],[model.ideal_proba(row) for row in B[:6]],atol=1e-12)
                    np.testing.assert_array_equal(prediction[:6],model.predict(B[:6]))
                    r = {'model':'qnn','scale':scale,'seed':seed,'origin':'new LOCAL encoding-scale fit',
                         'training_rows':914,'training_sha256':training_sha,
                         'validation_sha256':manifest['files']['validation.csv'],
                         'metrics':score(val.target,prediction,LABELS),
                         'fit_seconds':fit_seconds,'predict_seconds':predict_seconds,
                         'preprocessing_seconds':prep_seconds,'resources':model.resources()}
                    records.append(r)
                    tag = f'scale_{scale:g}_seed_{seed}'
                    pd.DataFrame({'sample_id':val.sample_id,'target':val.target,'prediction':prediction,
                                  **{f'p_{label}':probability[:,j] for j,label in enumerate(LABELS)}}).to_csv(out/f'{tag}_predictions.csv',index=False)
                    (out/f'{tag}_history.json').write_text(json.dumps(model.history,indent=2))
                    np.savez_compressed(out/f'{tag}_weights.npz',initial=np.asarray(initial[str(seed)]),final=model.model.weights)
                    (out/'runs.json').write_text(json.dumps(records,indent=2))
                    print(f'scale={scale:g} seed={seed} F1={r["metrics"]["macro_f1"]:.6f} accuracy={r["metrics"]["accuracy"]:.4f} recall={r["metrics"]["recall"]} fit={fit_seconds:.2f}s',flush=True)
            analyze(out,baseline)
        changed = [p for p,value in protected.items() if sha(ROOT/p)!=value]
        if changed:
            raise RuntimeError('Protected starter files changed: '+', '.join(changed))
        metadata.update(status='complete',completed_at=now(),wall_seconds=perf_counter()-start,
                        protected_files_unchanged=True,all_initializations_verified=True,
                        all_candidate_probabilities_checked_against_statevector=True)
    except BaseException:
        metadata.update(status='failed',completed_at=now(),wall_seconds=perf_counter()-start,error=traceback.format_exc())
        raise
    finally:
        meta_path.write_text(json.dumps(metadata,indent=2))
    print('Encoding comparison complete. Starter, notebook, data and organizer results unchanged.',flush=True)


if __name__=='__main__':
    main()
