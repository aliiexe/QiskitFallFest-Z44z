"""Reproduce all six candidate prediction files from saved weights, without fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd
from challenge.data import load_data
from challenge.models import angles
from experiments.encoding_scale.model import ScaledQuantumClassifier


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    args=parser.parse_args()
    out=args.results
    tables,_,manifest,_=load_data()
    prep=joblib.load(out/'preprocessing.joblib')
    B=prep.transform(tables['validation'][manifest['features']].to_numpy())
    with np.load(out/'prepared_inputs.npz',allow_pickle=False) as inputs:
        np.testing.assert_array_equal(B,inputs['X_validation'])
    checks=[]
    for scale in (.5,.25):
        for seed in (0,1,2):
            tag=f'scale_{scale:g}_seed_{seed}'
            model=ScaledQuantumClassifier(scale=scale)
            with np.load(out/f'{tag}_weights.npz',allow_pickle=False) as weights:
                np.testing.assert_array_equal(weights['initial'],np.random.default_rng(seed).uniform(-np.pi,np.pi,8))
                probabilities=model.qnn.forward(angles(scale*B),weights['final'])
            saved=pd.read_csv(out/f'{tag}_predictions.csv')
            np.testing.assert_array_equal(saved.sample_id,tables['validation'].sample_id)
            np.testing.assert_allclose(probabilities,saved[[f'p_{crop}' for crop in model.labels]].to_numpy(),atol=1e-12,rtol=0)
            np.testing.assert_array_equal(model.labels[probabilities.argmax(axis=1)],saved.prediction)
            checks.append({'scale':scale,'seed':seed,'validation_rows':150,'status':'passed',
                           'weights_sha256':hashlib.sha256((out/f'{tag}_weights.npz').read_bytes()).hexdigest()})
    (out/'saved_weight_verification.json').write_text(json.dumps({'status':'passed','fits':checks,'test_inference':False},indent=2))
    print('Saved preprocessing and weights reproduce all 900 candidate validation probability rows and labels.')


if __name__=='__main__':
    main()
