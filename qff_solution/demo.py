"""Small I/O helpers shared by the notebook and release verification."""
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from challenge.models import QuantumClassifier
from challenge.metrics import score
from .model import FinalRingClassifier

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT/'artifacts/final_ring'
LABELS = ['alfalfa', 'maize', 'wheat']
NAMES = {'ring': 'Final ring QNN', 'original': 'Original QNN', 'svm': 'Classical SVM'}


def verify_artifacts():
    """Check bundled files before loading the trusted local joblib artifacts."""
    manifest = json.loads((ARTIFACTS/'checksums.json').read_text())
    for name, digest in manifest.items():
        if hashlib.sha256((ARTIFACTS/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Artifact checksum mismatch: {name}')


def recorded_runs():
    return json.loads((ARTIFACTS/'recorded_runs.json').read_text())


def load_verified(seed=0):
    """Load only published fitted preprocessing, weights and the reference SVM."""
    if seed not in (0, 1, 2):
        raise ValueError('Bundled seeds are 0, 1 and 2')
    verify_artifacts()
    models = {'ring': FinalRingClassifier(), 'original': QuantumClassifier()}
    for key, model in models.items():
        with np.load(ARTIFACTS/f'{key}_seed_{seed}_weights.npz', allow_pickle=False) as w:
            model.model = SimpleNamespace(weights=w['final'].copy())
        model.history = json.loads((ARTIFACTS/f'{key}_seed_{seed}_history.json').read_text())
    models['svm'] = joblib.load(ARTIFACTS/'svm.joblib')
    return joblib.load(ARTIFACTS/'preprocessing.joblib'), models


def expected_predictions(key, seed=0):
    name = 'svm_predictions.csv' if key == 'svm' else f'{key}_seed_{seed}_predictions.csv'
    return pd.read_csv(ARTIFACTS/name)


def evaluate_saved(raw_validation, target, ids, seed=0):
    """Compute predictions now and require agreement with archived field labels."""
    start = perf_counter()
    prep, models = load_verified(seed)
    load_seconds = perf_counter()-start
    tick = perf_counter()
    prepared = prep.transform(raw_validation)
    transform_seconds = perf_counter()-tick
    predictions, metrics, timings = {}, {}, []
    for key, model in models.items():
        tick = perf_counter()
        if key == 'svm':
            pred = model.predict(raw_validation)
        else:
            p = model.predict_proba(prepared)
            pred = model.labels[p.argmax(axis=1)]
        seconds = perf_counter()-tick
        expected = expected_predictions(key, seed)
        np.testing.assert_array_equal(ids, expected.sample_id)
        np.testing.assert_array_equal(target, expected.target)
        np.testing.assert_array_equal(pred, expected.prediction)
        if key != 'svm':
            np.testing.assert_allclose(p, expected[[f'p_{c}' for c in LABELS]], atol=1e-9, rtol=0)
        predictions[key] = pred
        metrics[key] = score(target, pred, LABELS)
        timings.append({'model': NAMES[key], 'inference_seconds_now': seconds})
    return {'models': models, 'preprocessing': prep, 'prepared': prepared,
            'predictions': predictions, 'metrics': metrics, 'timings': timings,
            'load_seconds': load_seconds, 'transform_seconds': transform_seconds}


def make_preprocessing():
    return make_pipeline(StandardScaler(), PCA(4, svd_solver='full'), StandardScaler())


def train_ring(raw_train, target_train, raw_validation, seed=0):
    """One unchanged full-data fit; validation is transformed and predicted only.

    Timing includes fresh preprocessing, model construction, ALL per-fit caches,
    optimizer work and standard Qiskit inference. Imports/file I/O are excluded.
    """
    tick = perf_counter()
    prep = make_preprocessing()
    train = prep.fit_transform(raw_train)
    validation = prep.transform(raw_validation)
    preprocessing_seconds = perf_counter()-tick
    tick = perf_counter()
    model = FinalRingClassifier()
    construction_seconds = perf_counter()-tick
    model.fit(train, target_train, seed=seed)
    tick = perf_counter()
    p = model.predict_proba(validation)
    inference_seconds = perf_counter()-tick
    timings = dict(preprocessing_seconds=preprocessing_seconds,
                   model_construction_seconds=construction_seconds,
                   fit_seconds=model.fit_seconds, inference_seconds=inference_seconds)
    timings['total_seconds'] = sum(timings.values())
    return prep, model, p, timings
