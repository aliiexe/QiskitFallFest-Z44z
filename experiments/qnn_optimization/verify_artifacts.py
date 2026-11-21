"""Reconstruct every saved QNN from parameters and replay its validation output."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd
from challenge.data import load_data
from challenge.metrics import score
from experiments.qnn_optimization.model import NativeQiskitClassifier


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('results',type=Path)
    out=parser.parse_args().results
    tables,_,manifest,_=load_data();v=tables['validation']
    prep=joblib.load(out/'preprocessing.joblib');B=prep.transform(v[manifest['features']].to_numpy())
    runs=json.loads((out/'runs.json').read_text());checks=[]
    for r in runs:
        c=r['config'];tag=f'{r["configuration"]}_seed_{r["seed"]}'
        m=NativeQiskitClassifier(reps=c['entangling_blocks'],optimizer=c['optimizer'],max_evals=c['max_evals'])
        path=out/f'{tag}_weights.npz'
        with np.load(path,allow_pickle=False) as weights:
            np.testing.assert_array_equal(weights['initial'],np.random.default_rng(r['seed']).uniform(-np.pi,np.pi,len(m.weights)))
            m.model=SimpleNamespace(weights=weights['final'].copy())
        p=m.predict_proba(B);saved=pd.read_csv(out/f'{tag}_predictions.csv')
        np.testing.assert_array_equal(saved.sample_id,v.sample_id)
        np.testing.assert_allclose(p,saved[[f'p_{crop}' for crop in m.labels]].to_numpy(),atol=1e-10,rtol=0)
        pred=m.labels[p.argmax(axis=1)]
        np.testing.assert_array_equal(pred,saved.prediction)
        assert score(v.target,pred,list(m.labels))==r['metrics']
        checks.append({'configuration':r['configuration'],'seed':r['seed'],'rows':150,'status':'passed',
                       'weights_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    metadata=json.loads((out/'metadata.json').read_text())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in metadata['protected_sha256'].items())
    result={'status':'passed','models':len(checks),'validation_rows_reproduced':len(checks)*150,
            'checks':checks,'protected_files_unchanged':True,'test_inference':False}
    (out/'artifact_verification.json').write_text(json.dumps(result,indent=2))
    print(f'PASSED: all {len(checks)} saved QNNs reproduce probabilities, labels and metrics; protected files unchanged.')


if __name__=='__main__': main()
