"""Verify all ordered topologies, Model-D equivalence and loss gradients."""
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from challenge.data import load_data
from challenge.models import angles
from experiments.qnn_optimization.model import NativeQiskitClassifier
from experiments.entanglement_topology.model import TopologyClassifier,EDGES


def verify():
    tables,_,manifest,_=load_data();t=tables['train']
    X=make_pipeline(StandardScaler(),PCA(4,svd_solver='full'),StandardScaler()).fit_transform(t[manifest['features']].to_numpy())
    ids=np.unique(np.r_[np.arange(9),np.argmax(X,axis=0),np.argmin(X,axis=0)])
    rows=X[ids]
    report={'status':'passed','training_rows_checked':ids.tolist(),'validation_or_test_used':False,'models':[]}
    reference=NativeQiskitClassifier(reps=2,optimizer='L-BFGS-B',max_evals=100)
    for name,edges in EDGES.items():
        model=TopologyClassifier(topology=name)
        states=model.prepare_states(rows)
        sampler_error=state_error=unitary_error=gradient_error=0.
        expected_gates=[]
        for bank in range(3):
            expected_gates.extend([('ry',[q],[str(model.weights[bank*4+q])]) for q in range(4)])
            if bank<2: expected_gates.extend([('cx',list(edge),[]) for edge in edges])
        actual=[(g.operation.name,[model.ansatz.find_bit(q).index for q in g.qubits],
                 [str(p) for p in g.operation.params]) for g in model.ansatz.data]
        assert actual==expected_gates
        assert len(model.weights)==12 and model.circuit.num_qubits==4
        assert list(np.bincount(model.outcomes))==[6,5,5]
        for seed in (0,1,2):
            w=np.random.default_rng(seed).uniform(-np.pi,np.pi,12)
            cached=model.statevector_proba(states,w,count=False)
            sampler=model.qnn.forward(angles(rows),w)
            direct=np.array([model.ideal_proba(row,w) for row in rows])
            np.testing.assert_allclose(cached,sampler,atol=1e-10,rtol=0)
            np.testing.assert_allclose(cached,direct,atol=1e-12,rtol=0)
            sampler_error=max(sampler_error,float(abs(cached-sampler).max()))
            state_error=max(state_error,float(abs(cached-direct).max()))
            library=real_amplitudes(4,reps=2,entanglement=list(edges),parameter_prefix='theta')
            u=Operator(library.assign_parameters(dict(zip(library.parameters,w)))).data
            explicit=Operator(model.ansatz.assign_parameters(dict(zip(model.weights,w)))).data
            np.testing.assert_allclose(u,explicit,atol=1e-12,rtol=0)
            unitary_error=max(unitary_error,float(abs(u-explicit).max()))
            if name=='linear':
                np.testing.assert_allclose(reference.qnn.forward(angles(rows),w),sampler,atol=1e-12,rtol=0)
            small=states[:3];yi=np.array([0,1,2]);_,g=model.loss_gradient(small,yi,w)
            eps=1e-5;numeric=[]
            for j in range(12):
                a,b=w.copy(),w.copy();a[j]+=eps;b[j]-=eps
                numeric.append((model.loss_gradient(small,yi,a,False)-model.loss_gradient(small,yi,b,False))/(2*eps))
            np.testing.assert_allclose(g,numeric,atol=2e-7,rtol=2e-6)
            gradient_error=max(gradient_error,float(abs(g-numeric).max()))
        report['models'].append({'topology':name,'qubits':4,'trainable':12,
                                 'two_qubit_gates':sum(len(g.qubits)==2 for g in model.circuit.data),
                                 'depth':model.circuit.depth(),'ordered_edges_per_block':[list(e) for e in edges],
                                 'parameter_order':[str(p) for p in model.weights],
                                 'max_sampler_probability_error':sampler_error,'max_direct_statevector_error':state_error,
                                 'max_library_unitary_error':unitary_error,'max_cross_entropy_gradient_error':gradient_error})
    report['linear_matches_verified_model_D']=True
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();r=verify();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
