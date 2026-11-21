"""Verify generalized width against Qiskit and benchmark the actual training path."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from qiskit.circuit.library import real_amplitudes
from qiskit.quantum_info import Operator
from challenge.data import load_data
from experiments.entanglement_topology.model import TopologyClassifier
from experiments.feature_representation.models import representation
from experiments.quantum_representation.model import RepresentationClassifier
from experiments.qnn_optimization.run import sha


def verify(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'verification.json').exists():raise FileExistsError('Existing verification is preserved')
    tables,_,manifest,_=load_data();t=tables['train'];F=manifest['features']
    records=[]
    for n in (4,6):
        prep=representation(f'pca{n}');X=prep.fit_transform(t[F].to_numpy(),t.target)
        m=RepresentationClassifier(n);w=np.random.default_rng(0).uniform(-np.pi,np.pi,3*n)
        edges=[(a,b) for a in range(n) for b in range(a+1,n)]
        library=real_amplitudes(n,reps=2,entanglement=edges)
        library=library.assign_parameters(dict(zip(tuple(library.parameters),w)))
        np.testing.assert_allclose(Operator(m.ansatz.assign_parameters(dict(zip(m.weights,w)))).data,
                                   Operator(library).data,atol=1e-12,rtol=0)
        states=m.prepare_states(X[:9]);yi=np.array([list(m.labels).index(c) for c in t.target.iloc[:9]])
        p=m.statevector_proba(states,w,count=False)
        direct=np.array([m.ideal_proba(row,w) for row in X[:9]])
        sampler=m.qnn.forward(m.encode(X[:9]),w)
        np.testing.assert_allclose(p,direct,atol=1e-12,rtol=0)
        # Match the prior native study's 1e-10 sampler tolerance: QMLSampler
        # parameter binding introduces tiny (~1e-12) rounding differences.
        np.testing.assert_allclose(p,sampler,atol=1e-10,rtol=0)
        loss,grad=m.loss_gradient(states,yi,w);eps=1e-6
        finite=[]
        for j in range(len(w)):
            a=w.copy();b=w.copy();a[j]+=eps;b[j]-=eps
            finite.append((m.loss_gradient(states,yi,a,False)-m.loss_gradient(states,yi,b,False))/(2*eps))
        error=float(np.max(abs(grad-np.array(finite))))
        np.testing.assert_allclose(grad,finite,atol=2e-7,rtol=0)
        if n==4:
            old=TopologyClassifier(topology='full')
            np.testing.assert_allclose(old.statevector_proba(old.prepare_states(X[:9]),w,False),p,atol=1e-12,rtol=0)
        # One fold-sized loss/gradient is sufficient for a concrete cost estimate.
        states=m.prepare_states(X[:610]);yi=np.array([list(m.labels).index(c) for c in t.target.iloc[:610]])
        durations=[]
        for _ in range(3):
            start=perf_counter();m.loss_gradient(states,yi,w);durations.append(perf_counter()-start)
        rec={'qubits':n,'gradient_max_absolute_error':error,'sampler_max_absolute_error':float(np.max(abs(p-sampler))),
             'sampler_statevector_library_operator_agree':True,
             'four_qubit_historical_model_agrees':n==4,'benchmark_rows':610,
             'loss_and_gradient_seconds':durations,'resources':m.resources()}
        records.append(rec)
        print(json.dumps(rec),flush=True)
    report={'status':'passed','checks':records,'training_only':True,
            'implementation_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'experiments/quantum_representation').glob('*.py')}}
    (out/'verification.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);verify(p.parse_args().results)
