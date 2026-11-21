"""One-block, eight-angle ring: reuse the verified native Qiskit trainer."""
from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.primitives import QMLSampler
from experiments.qnn_optimization.model import NativeQiskitClassifier

RING_EDGES=((0,1),(1,2),(2,3),(3,0))


class LightweightRingClassifier(NativeQiskitClassifier):
    def __init__(self,labels=('alfalfa','maize','wheat'),optimizer='COBYLA',max_evals=100):
        super().__init__(labels=labels,reps=1,optimizer=optimizer,max_evals=max_evals)
        ansatz=QuantumCircuit(4)
        for q in range(4):ansatz.ry(self.weights[q],q)
        for a,b in RING_EDGES:ansatz.cx(a,b)
        for q in range(4):ansatz.ry(self.weights[4+q],q)
        self.ansatz=ansatz
        self.circuit=self.encoding_circuit.compose(ansatz)
        self.qnn=SamplerQNN(circuit=self.circuit,sampler=QMLSampler(shots=None),
                            input_params=self.inputs,weight_params=self.weights,
                            interpret=lambda outcome:int(outcome)%len(self.labels),
                            output_shape=len(self.labels))

    def resources(self):
        return {**super().resources(),'topology':'ring',
                'ordered_cx_edges_per_block':[list(e) for e in RING_EDGES],
                'probability_passes_per_loss_gradient':17 if self.optimizer_name=='L-BFGS-B' else 1,
                'state_preparation_seconds':getattr(self,'state_preparation_seconds',None)}
