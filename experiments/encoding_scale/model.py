"""A fixed input-scale adapter around the unchanged starter QNN.

The starter applies angle(z) = pi/2 * (1 + tanh(z)). Scaling its prepared
inputs here gives angle_s(z) = pi/2 * (1 + tanh(s*z)). Nothing else changes.
"""
import numpy as np
from challenge.models import QuantumClassifier


class ScaledQuantumClassifier(QuantumClassifier):
    def __init__(self, labels=('alfalfa', 'maize', 'wheat'), max_evals=100, scale=1.0):
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('scale must be positive and finite')
        self.scale = float(scale)
        super().__init__(labels=labels, max_evals=max_evals, encoding='ry')

    def fit(self, X_train, y_train, seed=0):
        return super().fit(np.asarray(X_train, dtype=float) * self.scale, y_train, seed=seed)

    def predict_proba(self, X):
        return super().predict_proba(np.asarray(X, dtype=float) * self.scale)

    def bound_circuit(self, row, weights=None):
        return super().bound_circuit(np.asarray(row, dtype=float) * self.scale, weights=weights)

    def resources(self):
        return {**super().resources(), 'fixed_input_scale': self.scale}
