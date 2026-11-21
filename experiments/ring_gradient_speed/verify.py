"""Numerical and native-Qiskit checks before any gradient-method selection."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from qiskit.quantum_info import Statevector,Operator
from challenge.data import load_data
from challenge.models import angles
from experiments.entanglement_topology.model import TopologyClassifier
from experiments.qnn_optimization.run import preprocessing,sha
from experiments.ring_gradient_speed.model import BatchedRingClassifier


def signature(model):
    return [(i.operation.name,[model.circuit.find_bit(q).index for q in i.qubits],
             [str(p) for p in i.operation.params]) for i in model.circuit.data]


def verify(out):
    out=Path(out)
    assert json.loads((out/'profile.json').read_text())['status']=='passed'
    if (out/'verification.json').exists():raise FileExistsError('Existing verification preserved')
    tables,_,manifest,_=load_data();t=tables['train'];X=preprocessing().fit_transform(t[manifest['features']].to_numpy())
    yi=np.array([['alfalfa','maize','wheat'].index(c) for c in t.target.iloc[:6]])
    baseline=TopologyClassifier(topology='ring')
    variants={name:BatchedRingClassifier(name) for name in ['batched_shift','operator_adjoint']}
    base_states=baseline.prepare_states(X[:6]);states={k:m.prepare_states(X[:6]) for k,m in variants.items()}
    for m in variants.values():assert signature(m)==signature(baseline)
    cases=[]
    for seed in [0,1,2]:
        cases.append((f'initial_seed_{seed}',np.random.default_rng(seed).uniform(-np.pi,np.pi,12)))
        with np.load(ROOT/f'outputs/entanglement_topology_2026-11-21/ring_seed_{seed}_weights.npz',allow_pickle=False) as w:
            cases.append((f'trained_seed_{seed}',w['final'].copy()))
    records=[]
    for name,w in cases:
        loss,g=baseline.loss_gradient(base_states,yi,w)
        p=baseline.statevector_proba(base_states,w,count=False)
        sampler=baseline.qnn.forward(angles(X[:6]),w)
        _,qiskit_dp=baseline.qnn.backward(angles(X[:6]),w)
        official=-np.mean(qiskit_dp[np.arange(6),yi,:]/sampler[np.arange(6),yi,None],axis=0)/np.log(2)
        np.testing.assert_allclose(g,official,atol=2e-7,rtol=0)
        for method,m in variants.items():
            value,gradient=m.loss_gradient(states[method],yi,w)
            np.testing.assert_allclose(value,loss,atol=1e-12,rtol=0)
            np.testing.assert_allclose(gradient,g,atol=2e-7,rtol=0)
            np.testing.assert_allclose(m.statevector_proba(states[method],w,count=False),p,atol=1e-12,rtol=0)
            eps=1e-6;fd=[]
            for j in range(12):
                a=w.copy();b=w.copy();a[j]+=eps;b[j]-=eps
                fd.append((m.loss_gradient(states[method],yi,a,False)-m.loss_gradient(states[method],yi,b,False))/(2*eps))
            np.testing.assert_allclose(gradient,fd,atol=2e-7,rtol=0)
            if method=='operator_adjoint':
                _,prefix=m._operator_tape(w)
                qiskit=Operator(m.ansatz.assign_parameters(dict(zip(m.weights,w)))).data
                np.testing.assert_allclose(prefix[-1],qiskit,atol=1e-12,rtol=0)
                np.testing.assert_allclose(m.adjoint_proba(states[method],w),p,atol=1e-12,rtol=0)
            records.append({'case':name,'method':method,'loss_error':float(abs(value-loss)),
                            'parameter_shift_gradient_error':float(np.max(abs(gradient-g))),
                            'finite_difference_error':float(np.max(abs(gradient-fd))),
                            'official_Qiskit_backward_error':float(np.max(abs(gradient-official)))})
    # Verify that the batch math handles complex amplitudes rather than relying
    # on the real-valued RY inputs, and preserves clipped CE derivatives.
    rng=np.random.default_rng(123);z=rng.normal(size=(6,16))+1j*rng.normal(size=(6,16));z/=np.linalg.norm(z,axis=1)[:,None]
    complex_states=[Statevector(row) for row in z];w=cases[0][1]
    loss,g=baseline.loss_gradient(complex_states,yi,w)
    for m in variants.values():
        value,grad=m.loss_gradient(z,yi,w)
        np.testing.assert_allclose(value,loss,atol=1e-12,rtol=0);np.testing.assert_allclose(grad,g,atol=2e-7,rtol=0)
    zeros=np.zeros((3,16),dtype=complex);zeros[:,0]=1;zero_labels=np.ones(3,dtype=int);zero_weights=np.zeros(12)
    loss,g=baseline.loss_gradient([Statevector(row) for row in zeros],zero_labels,zero_weights)
    for m in variants.values():
        value,grad=m.loss_gradient(zeros,zero_labels,zero_weights)
        np.testing.assert_allclose(value,loss,atol=1e-12,rtol=0);np.testing.assert_allclose(grad,g,atol=1e-12,rtol=0)
    # Frozen weights: every training field agrees with actual SamplerQNN output.
    all_states={k:m.prepare_states(X) for k,m in variants.items()};frozen=[]
    for name,w in cases:
        official=baseline.qnn.forward(angles(X),w)
        for method,m in variants.items():
            p=m.statevector_proba(all_states[method],w,count=False)
            np.testing.assert_allclose(p,official,atol=1e-10,rtol=0)
            if method=='operator_adjoint':
                np.testing.assert_allclose(m.adjoint_proba(all_states[method],w),official,atol=1e-10,rtol=0)
            frozen.append({'case':name,'method':method,'fields':len(X),'max_probability_error':float(np.max(abs(p-official)))})
    # Training-only warm gradient microbenchmark. Cache time is NOT omitted from
    # the later complete fits; this table isolates steady-state derivative work.
    all_y=np.array([['alfalfa','maize','wheat'].index(c) for c in t.target])
    micro=[]
    for name,m in [('parameter_shift',baseline),*variants.items()]:
        tick=perf_counter();s=m.prepare_states(X);cache_seconds=perf_counter()-tick
        timings=[]
        for _ in range(5):
            tick=perf_counter();m.loss_gradient(s,all_y,cases[0][1]);timings.append(perf_counter()-tick)
        micro.append({'method':name,'cache_construction_seconds':cache_seconds,
                      'loss_gradient_seconds':timings,'median_loss_gradient_seconds':float(np.median(timings))})
    sources=[p for p in (ROOT/'experiments/ring_gradient_speed').glob('*') if p.suffix in ['.py','.json']]
    report={'status':'passed','scope':'training data only; no validation inference',
            'same_circuit_signature':True,'gradient_checks':records,'frozen_training_probabilities':frozen,
            'complex_state_check':True,'clipped_zero_probability_check':True,
            'microbenchmark':micro,'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources}}
    (out/'verification.json').write_text(json.dumps(report,indent=2))
    print('Maximum finite-difference error:',max(r['finite_difference_error'] for r in records))
    print('Maximum official Qiskit gradient error:',max(r['official_Qiskit_backward_error'] for r in records))
    print('Maximum frozen probability error:',max(r['max_probability_error'] for r in frozen))
    print('Training-only gradient timings:',micro)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);verify(p.parse_args().results)
