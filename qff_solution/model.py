"""Public model with the verified fast gradient selected explicitly."""
from .adjoint import BatchedRingClassifier


class FinalRingClassifier(BatchedRingClassifier):
    """Four qubits, twelve RY weights, two ring CX blocks; exact CPU simulation.

    fit/predict take the four training-fitted, standardized PCA inputs.
    Inference uses the original Qiskit SamplerQNN. Only the training evaluator
    changes: Qiskit-generated operator bases and NumPy matrix-adjoint gradients.
    """

    def __init__(self, max_evals=100):
        super().__init__(method='operator_adjoint', max_evals=max_evals)
