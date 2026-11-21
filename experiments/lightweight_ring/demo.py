"""Small saved-model loader for the simple notebook; no default training."""
import json
from pathlib import Path
from types import SimpleNamespace
import joblib
import numpy as np
import pandas as pd
from challenge.metrics import score
from experiments.entanglement_topology.model import TopologyClassifier
from experiments.lightweight_ring.model import LightweightRingClassifier
from experiments.qnn_optimization.run import sha


def make_model(results,name):
    if name=='ring12':return TopologyClassifier(topology='ring',max_evals=100)
    if name!='selected8':raise ValueError('Choose ring12 or selected8')
    config=json.loads((Path(results)/'training_only_selection.json').read_text())['selected_config']
    return LightweightRingClassifier(optimizer=config['optimizer'],max_evals=config['max_evals'])


def load_models(results,seed=0):
    results=Path(results)
    if seed not in [0,1,2]:raise ValueError('Choose seed 0, 1 or 2; do not select a seed from validation.')
    if not (results/'artifact_verification.json').is_file():raise FileNotFoundError('Run and verify the lightweight-ring experiment before loading it.')
    assert json.loads((results/'artifact_verification.json').read_text())['status']=='passed'
    manifest=json.loads((results/'demo_manifest.json').read_text())
    for relative,expected in manifest['sha256'].items():
        if not (results/relative).is_file():raise FileNotFoundError(f'Missing saved model/result: {relative}')
        if sha(results/relative)!=expected:raise ValueError(f'Saved result changed: {relative}')
    runs=json.loads((results/'runs.json').read_text());models={}
    for name in ['selected8','ring12']:
        m=make_model(results,name)
        r=next(r for r in runs if r['model']==name and r['seed']==seed)
        with np.load(results/'full'/f'{name}_seed_{seed}_weights.npz',allow_pickle=False) as w:
            m.model=SimpleNamespace(weights=w['final'].copy());m.initial_point=w['initial'].copy()
        m.history=json.loads((results/'full'/f'{name}_seed_{seed}_history.json').read_text())
        m.seed=seed;m.fit_seconds=r['fit_seconds'];m.termination=r['termination']
        m.forward_passes=r['resources']['native_operator_builds']
        m.logical_field_evaluations=r['resources']['logical_field_probability_evaluations']
        m.state_preparation_seconds=r['resources'].get('state_preparation_seconds')
        models[name]=m
    return joblib.load(results/'preprocessing.joblib'),models['selected8'],models['ring12'],joblib.load(results/'full'/f'svm_seed_{seed}.joblib')


def verify_predictions(results,validation,features,prep,light,strong,svm,seed):
    B=prep.transform(validation[features].to_numpy());checks=[]
    for name,m in [('selected8',light),('ring12',strong),('svm',svm)]:
        pred=m.predict(validation[features].to_numpy()) if name=='svm' else m.predict(B)
        old=pd.read_csv(Path(results)/'full'/f'{name}_seed_{seed}_predictions.csv')
        np.testing.assert_array_equal(validation.sample_id,old.sample_id)
        np.testing.assert_array_equal(pred,old.prediction)
        if name!='svm':
            np.testing.assert_allclose(m.predict_proba(B),old[[f'p_{c}' for c in m.labels]],atol=1e-10,rtol=0)
        checks.append({'model':name,'seed':seed,'metrics':score(validation.target,pred,['alfalfa','maize','wheat'])})
    return checks
