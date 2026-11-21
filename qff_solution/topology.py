"""Change only Model D's ordered CX connectivity; reuse its verified training."""
from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.primitives import QMLSampler
from .cached_training import NativeQiskitClassifier

EDGES = {
    'linear': ((0,1),(1,2),(2,3)),
    'ring': ((0,1),(1,2),(2,3),(3,0)),
    'full': ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)),
}


class TopologyClassifier(NativeQiskitClassifier):
    def __init__(self, labels=('alfalfa','maize','wheat'), topology='linear', max_evals=100):
        if topology not in EDGES:
            raise ValueError('Expected linear, ring or full topology')
        super().__init__(labels=labels,reps=2,optimizer='L-BFGS-B',max_evals=max_evals)
        self.topology = topology
        ansatz = QuantumCircuit(4)
        for bank in range(3):
            for q in range(4):
                ansatz.ry(self.weights[4*bank+q],q)
            if bank < 2:
                for control,target in EDGES[topology]:
                    ansatz.cx(control,target)
        self.ansatz = ansatz
        self.circuit = self.encoding_circuit.compose(ansatz)
        self.qnn = SamplerQNN(circuit=self.circuit,sampler=QMLSampler(shots=None),
                              input_params=self.inputs,weight_params=self.weights,
                              interpret=lambda outcome:int(outcome)%len(self.labels),
                              output_shape=len(self.labels))

    def resources(self):
        return {**super().resources(),'topology':self.topology,
                'ordered_cx_edges_per_block':[list(edge) for edge in EDGES[self.topology]],
                'probability_passes_per_loss_and_gradient':25}
