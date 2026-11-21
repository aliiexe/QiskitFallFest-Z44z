"""Verify gates, readout, Qiskit evaluation paths and true CE gradients."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from qiskit.circuit.library import real_amplitudes
from qiskit.quantum_info import Operator
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from challenge.data import load_data
from challenge.models import QuantumClassifier,angles
from experiments.qnn_optimization.model import NativeQiskitClassifier


def verify():
    tables,_,manifest,_=load_data()
    t=tables['train']; F=manifest['features']
    X=make_pipeline(StandardScaler(),PCA(4,svd_solver='full'),StandardScaler()).fit_transform(t[F].to_numpy())
    # Training-only cases: include typical fields, each crop, and extreme component values.
    positions=np.unique(np.r_[np.arange(9),np.argmax(X,axis=0),np.argmin(X,axis=0)])
    rows=X[positions]
    report={'status':'passed','historical_independent_prototype_equivalence':'NOT VERIFIED: archive not available locally',
            'test_data_used':False,'training_rows_checked':positions.tolist(),'models':[]}
    for reps in (1,2):
        model=NativeQiskitClassifier(reps=reps)
        states=model.prepare_states(rows)
        sampler_error=state_error=unitary_error=gradient_error=0.
        for seed in (0,1,2):
            w=np.random.default_rng(seed).uniform(-np.pi,np.pi,len(model.weights))
            cached=model.statevector_proba(states,w,count=False)
            sampler=model.qnn.forward(angles(rows),w)
            direct=np.array([model.ideal_proba(row,w) for row in rows])
            np.testing.assert_allclose(cached,sampler,atol=1e-10,rtol=0)
            np.testing.assert_allclose(cached,direct,atol=1e-12,rtol=0)
            sampler_error=max(sampler_error,float(abs(cached-sampler).max()))
            state_error=max(state_error,float(abs(cached-direct).max()))
            library=real_amplitudes(4,reps=reps,entanglement='linear',parameter_prefix='theta')
            u=Operator(library.assign_parameters(dict(zip(library.parameters,w)))).data
            manual=Operator(model.ansatz.assign_parameters(dict(zip(model.weights,w)))).data
            np.testing.assert_allclose(u,manual,atol=1e-12,rtol=0)
            unitary_error=max(unitary_error,float(abs(u-manual).max()))
            small=states[:3]; yi=np.array([0,1,2])
            _,g=model.loss_gradient(small,yi,w)
            eps=1e-5; numeric=[]
            for j in range(len(w)):
                a,b=w.copy(),w.copy();a[j]+=eps;b[j]-=eps
                numeric.append((model.loss_gradient(small,yi,a,False)-model.loss_gradient(small,yi,b,False))/(2*eps))
            np.testing.assert_allclose(g,numeric,atol=2e-7,rtol=2e-6)
            gradient_error=max(gradient_error,float(abs(g-numeric).max()))
            if reps==1:
                original=QuantumClassifier(encoding='ry')
                np.testing.assert_allclose(original.qnn.forward(angles(rows),w),sampler,atol=1e-12,rtol=0)
        assert list(np.bincount(model.outcomes))==[6,5,5]
        assert len(model.weights)==(8 if reps==1 else 12)
        report['models'].append({'entangling_blocks':reps,'parameters':len(model.weights),
                                 'qubits':4,'depth':model.circuit.depth(),
                                 'two_qubit_gates':sum(len(g.qubits)==2 for g in model.circuit.data),
                                 'max_sampler_probability_error':sampler_error,
                                 'max_direct_statevector_error':state_error,
                                 'max_library_unitary_error':unitary_error,
                                 'max_cross_entropy_gradient_error':gradient_error,
                                 'parameter_order':[str(p) for p in model.weights],
                                 'gates':[{'gate':g.operation.name,'qubits':[model.circuit.find_bit(q).index for q in g.qubits],
                                           'parameters':[str(p) for p in g.operation.params]} for g in model.circuit.data]})
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();r=verify()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(r,indent=2))
    print(json.dumps({**r,'models':[{k:v for k,v in m.items() if k!='gates'} for m in r['models']]},indent=2))
