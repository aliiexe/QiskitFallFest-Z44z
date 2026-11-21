"""Verify the new circuit, exact probabilities and parameter-shift derivatives."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from qiskit.circuit.library import real_amplitudes
from qiskit.quantum_info import Operator
from challenge.data import load_data
from experiments.qnn_optimization.run import preprocessing,sha
from experiments.lightweight_ring.model import LightweightRingClassifier,RING_EDGES


def verify(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    tables,_,manifest,_=load_data();t=tables['train']
    X=preprocessing().fit_transform(t[manifest['features']].to_numpy())
    m=LightweightRingClassifier(optimizer='L-BFGS-B')
    w=np.random.default_rng(0).uniform(-np.pi,np.pi,8)
    library=real_amplitudes(4,reps=1,entanglement=list(RING_EDGES))
    library=library.assign_parameters(dict(zip(tuple(library.parameters),w)))
    np.testing.assert_allclose(Operator(m.ansatz.assign_parameters(dict(zip(m.weights,w)))).data,
                               Operator(library).data,atol=1e-12,rtol=0)
    states=m.prepare_states(X[:9]);yi=np.array([list(m.labels).index(c) for c in t.target.iloc[:9]])
    p=m.statevector_proba(states,w,count=False)
    direct=np.array([m.ideal_proba(row,w) for row in X[:9]])
    sampler=m.qnn.forward(__import__('challenge.models',fromlist=['angles']).angles(X[:9]),w)
    np.testing.assert_allclose(p,direct,atol=1e-12,rtol=0)
    np.testing.assert_allclose(p,sampler,atol=1e-10,rtol=0)
    _,gradient=m.loss_gradient(states,yi,w);eps=1e-6;finite=[]
    for j in range(8):
        a=w.copy();b=w.copy();a[j]+=eps;b[j]-=eps
        finite.append((m.loss_gradient(states,yi,a,False)-m.loss_gradient(states,yi,b,False))/(2*eps))
    np.testing.assert_allclose(gradient,finite,atol=2e-7,rtol=0)
    resources=m.resources()
    assert resources['qubits']==4 and resources['trainable']==8 and resources['two_qubit_gates']==4
    assert resources['entangling_blocks']==1 and resources['readout_group_sizes']==[6,5,5]
    report={'status':'passed','gradient_max_error':float(np.max(abs(gradient-finite))),
            'sampler_max_error':float(np.max(abs(p-sampler))),'library_operator_matches':True,
            'resources':resources,'verified_sources':{str(p.relative_to(ROOT)):sha(p) for p in
              [ROOT/'experiments/lightweight_ring/model.py',ROOT/'experiments/lightweight_ring/protocol.json']}}
    (out/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);verify(p.parse_args().results)
