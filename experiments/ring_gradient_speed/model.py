"""Same ring circuit/optimizer; batched shift and matrix-adjoint CPU derivatives.

Gate operators come from Qiskit. The matrix-adjoint evaluator is an independent
NumPy implementation, not an official Qiskit reverse-gradient primitive.
SamplerQNN inference remains unchanged. No hand-coded qubit indexing is used.
"""
from time import perf_counter
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from challenge.metrics import probabilities
from experiments.entanglement_topology.model import TopologyClassifier


class BatchedRingClassifier(TopologyClassifier):
    def __init__(self,method='batched_shift',max_evals=100):
        if method not in ['batched_shift','operator_adjoint']:raise ValueError(method)
        super().__init__(topology='ring',max_evals=max_evals)
        self.gradient_method=method
        self.bound_operator_builds=0
        self.gate_cache_builds=0
        self.gate_cache_seconds=0.
        self.batch_conversion_seconds=0.
        self.small_matrix_products=0
        self.batch_matrix_products=0
        self.reverse_sweeps=0
        self._gate_specs=None
        self._readout=(self.outcomes[:,None]==np.arange(3)[None,:]).astype(float)

    def prepare_states(self,X):
        # Use the original Qiskit input-state construction unchanged.
        states=super().prepare_states(X)
        tick=perf_counter();batch=np.asarray([s.data for s in states],dtype=complex)
        self.batch_conversion_seconds+=perf_counter()-tick
        self.input_cache_bytes=batch.nbytes
        if self.gradient_method=='operator_adjoint' and self._gate_specs is None:
            tick=perf_counter();self._build_gate_cache();self.gate_cache_seconds=perf_counter()-tick
        return batch

    def _build_gate_cache(self):
        # RY(theta) = cos(theta/2)*I + sin(theta/2)*J, with I/J obtained
        # from actual full-register Qiskit circuits, not handwritten gate arrays.
        self._identity=Operator(QuantumCircuit(4)).data
        self.gate_cache_builds=1
        ry_basis={};cx_matrices={};specs=[]
        for instruction in self.ansatz.data:
            op=instruction.operation
            qubits=tuple(self.ansatz.find_bit(q).index for q in instruction.qubits)
            if op.name=='ry':
                parameter=op.params[0]
                if parameter not in self.weights:raise ValueError('Unexpected RY expression')
                q=qubits[0]
                if q not in ry_basis:
                    c=QuantumCircuit(4);c.ry(np.pi,q)
                    # Remove the finite-precision cos(pi/2) diagonal term.
                    ry_basis[q]=(Operator(c).data-np.cos(np.pi/2)*self._identity)/np.sin(np.pi/2)
                    self.gate_cache_builds+=1
                specs.append(('ry',self.weights.index(parameter),ry_basis[q]))
            elif op.name=='cx':
                if qubits not in cx_matrices:
                    c=QuantumCircuit(4);c.append(op,qubits)
                    cx_matrices[qubits]=Operator(c).data;self.gate_cache_builds+=1
                specs.append(('cx',None,cx_matrices[qubits]))
            else:raise ValueError('This verified derivative handles the unchanged RY/CX ring only')
        assert len(specs)==20 and sum(s[0]=='ry' for s in specs)==12
        self._gate_specs=specs
        self.gate_cache_bytes=self._identity.nbytes+sum(x.nbytes for x in ry_basis.values())+sum(x.nbytes for x in cx_matrices.values())

    def statevector_proba(self,states,weights,count=True):
        # Vectorized control: exactly the prior Qiskit bound-ansatz Operator,
        # but all independent field states evolve in a single matrix product.
        operator=Operator(self.ansatz.assign_parameters(dict(zip(self.weights,weights)))).data
        amplitudes=np.asarray(states)@operator.T
        p=np.abs(amplitudes)**2@self._readout
        if count:
            self.forward_passes+=1;self.logical_field_evaluations+=len(states)
            self.bound_operator_builds+=1;self.batch_matrix_products+=1
        return probabilities(p,len(states),3)

    def _operator_tape(self,weights):
        gates=[];prefixes=[self._identity]
        for kind,index,basis in self._gate_specs:
            if kind=='ry':
                c,s=np.cos(weights[index]/2),np.sin(weights[index]/2)
                u=c*self._identity+s*basis
                du=.5*(-s*self._identity+c*basis)
            else:u,du=basis,None
            gates.append((u,du,index));prefixes.append(u@prefixes[-1])
        return gates,prefixes

    def adjoint_proba(self,states,weights):
        _,prefix=self._operator_tape(weights)
        return probabilities(np.abs(states@prefix[-1].T)**2@self._readout,len(states),3)

    def loss_gradient(self,states,y_index,weights,gradient=True):
        if self.gradient_method=='batched_shift':
            return super().loss_gradient(states,y_index,weights,gradient)
        # Differentiate the exact same clipped, base-two CE. Aggregate the
        # field-state cotangent to a 16x16 operator cotangent before reversing
        # the gate product; the reverse sweep then avoids per-field gate loops.
        gates,prefix=self._operator_tape(weights)
        psi=states@prefix[-1].T
        p=probabilities(np.abs(psi)**2@self._readout,len(states),3)
        target=p[np.arange(len(states)),y_index]
        safe=np.clip(target,1e-10,None)
        loss=float(-np.log2(safe).mean())
        self.forward_passes+=1;self.logical_field_evaluations+=len(states)
        self.small_matrix_products+=len(gates);self.batch_matrix_products+=1
        if not gradient:return loss
        mask=self.outcomes[None,:]==y_index[:,None]
        lam=-np.where(mask,psi,0.)/(len(states)*np.log(2)*safe[:,None])
        lam[target<=1e-10]=0.  # Same derivative clipping as the verified trainer.
        cotangent=lam.T@states.conj()
        self.batch_matrix_products+=1
        grad=np.zeros(len(self.weights))
        for j in range(len(gates)-1,-1,-1):
            u,du,index=gates[j]
            if du is not None:
                grad[index]=2*np.real(np.vdot(cotangent,du@prefix[j]))
                self.small_matrix_products+=1
            cotangent=u.conj().T@cotangent
            self.small_matrix_products+=1
        self.reverse_sweeps+=1
        return loss,grad

    def resources(self):
        r=super().resources()
        return {**r,'gradient_method':self.gradient_method,
                'training_engine':'NumPy batched state multiplication of Qiskit Operators' if self.gradient_method=='batched_shift'
                                  else 'Qiskit-generated RY/CX operator bases + NumPy matrix-product reverse differentiation',
                'native_operator_builds':self.bound_operator_builds+self.gate_cache_builds,
                'bound_full_ansatz_operator_builds':self.bound_operator_builds,
                'qiskit_gate_cache_operator_builds':self.gate_cache_builds,
                'complete_forward_probability_passes':self.forward_passes,
                'probability_passes_per_loss_and_gradient':25 if self.gradient_method=='batched_shift' else 1,
                'reverse_sweeps':self.reverse_sweeps,'small_16x16_matrix_products':self.small_matrix_products,
                'field_batch_matrix_products':self.batch_matrix_products,
                'fixed_gate_cache_seconds':self.gate_cache_seconds,'input_batch_conversion_seconds':self.batch_conversion_seconds,
                'state_preparation_seconds':getattr(self,'state_preparation_seconds',None),
                'input_amplitude_cache_bytes':getattr(self,'input_cache_bytes',0),
                'gate_cache_bytes':getattr(self,'gate_cache_bytes',0),
                'work_note':'Adjoint forward counts exclude separately reported reverse matrix products; not equivalent hardware circuit counts',
                'prediction_engine':'unchanged standard SamplerQNN + QMLSampler(shots=None)'}
