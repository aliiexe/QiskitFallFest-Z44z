"""Width-controlled extension of the verified native full-connectivity QNN."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.primitives import QMLSampler
from challenge.metrics import probabilities
from experiments.entanglement_topology.model import TopologyClassifier


class RepresentationClassifier(TopologyClassifier):
    def __init__(self, qubits=4, max_evals=100):
        if qubits not in (4,6):
            raise ValueError('This predefined study tests only four and six qubits')
        super().__init__(topology='full',max_evals=max_evals)
        self.qubits=qubits
        if qubits==6:
            self.inputs=tuple(ParameterVector('x',qubits))
            self.weights=tuple(ParameterVector('theta',3*qubits))
            self.encoding_circuit=QuantumCircuit(qubits)
            self.ansatz=QuantumCircuit(qubits)
            for q in range(qubits):self.encoding_circuit.ry(self.inputs[q],q)
            for bank in range(3):
                for q in range(qubits):self.ansatz.ry(self.weights[bank*qubits+q],q)
                if bank<2:
                    for control in range(qubits):
                        for target in range(control+1,qubits):self.ansatz.cx(control,target)
            self.circuit=self.encoding_circuit.compose(self.ansatz)
            self.outcomes=np.arange(2**qubits)%len(self.labels)
            self.qnn=SamplerQNN(circuit=self.circuit,sampler=QMLSampler(shots=None),
                                input_params=self.inputs,weight_params=self.weights,
                                interpret=lambda outcome:int(outcome)%len(self.labels),
                                output_shape=len(self.labels))

    def encode(self,X):
        X=np.asarray(X,dtype=float)
        if X.ndim!=2 or X.shape[1]!=self.qubits or not np.isfinite(X).all():
            raise ValueError(f'Expected a finite (n,{self.qubits}) prepared view')
        return (np.pi/2)*(1+np.tanh(X))

    def prepare_states(self,X):
        return [Statevector.from_instruction(self.encoding_circuit.assign_parameters(dict(zip(self.inputs,row))))
                for row in self.encode(X)]

    def predict_proba(self,X):
        return probabilities(self.qnn.forward(self.encode(X),self.model.weights),len(X),len(self.labels))

    def bound_circuit(self,row,weights=None):
        if weights is None:weights=self.model.weights
        bind=dict(zip(self.inputs,self.encode(np.asarray(row).reshape(1,self.qubits))[0]))
        bind.update(zip(self.weights,weights))
        return self.circuit.assign_parameters(bind)

    def resources(self):
        # Do not call the organizer's resource method: its qubit count is fixed at 4.
        return {'qubits':self.qubits,'trainable':len(self.weights),'frozen_quantum':0,
                'depth':self.circuit.depth(),'two_qubit_gates':self.circuit.count_ops().get('cx',0),
                'one_qubit_gates':self.circuit.count_ops().get('ry',0),
                'state_dimension':2**self.qubits,'readout_group_sizes':np.bincount(self.outcomes).tolist(),
                'objective_evaluations':len(self.history),
                'gradient_evaluations':sum(h['gradient_norm'] is not None for h in self.history),
                'probability_passes_per_loss_and_gradient':1+2*len(self.weights),
                'native_operator_builds':self.forward_passes,
                'logical_field_probability_evaluations':self.logical_field_evaluations,
                'topology':'full','entangling_blocks':2,'ry_parameter_banks':3,
                'ordered_cx_edges_per_block':[[a,b] for a in range(self.qubits) for b in range(a+1,self.qubits)],
                'optimizer':'L-BFGS-B','encoding':'RY(pi/2 * (1+tanh(z)))',
                'compilation':'logical RY/CX; no hardware routing or noise',
                'training_engine':'Qiskit Statevector.evolve(Qiskit Operator(bound ansatz)); fixed input-state cache',
                'prediction_engine':'Qiskit SamplerQNN + QMLSampler(shots=None)'}
