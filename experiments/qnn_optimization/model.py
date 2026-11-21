"""Native Qiskit circuit/optimizer experiments extending the organizer QNN.

Quantum evolution uses Qiskit Statevector and Operator, never handwritten gates.
SamplerQNN remains the prediction interface and an independent runtime check.
Only fixed input states are cached; each parameter setting builds its operator
from the actual Qiskit circuit. Gradients use the exact RY parameter-shift rule.
"""
from time import perf_counter
from types import SimpleNamespace
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, Operator
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.primitives import QMLSampler
from challenge.models import QuantumClassifier, angles
from challenge.metrics import probabilities


def circuit_parts(reps=1):
    if reps not in (1, 2):
        raise ValueError('This controlled protocol tests one or two entangling blocks.')
    x, theta = ParameterVector('x', 4), ParameterVector('theta', 4*(reps+1))
    encoding, ansatz = QuantumCircuit(4), QuantumCircuit(4)
    for q in range(4):
        encoding.ry(x[q], q)
    # Explicit ordering: RY bank -> forward linear CX bank -> ... -> final RY bank.
    for layer in range(reps+1):
        for q in range(4):
            ansatz.ry(theta[4*layer+q], q)
        if layer < reps:
            for q in range(3):
                ansatz.cx(q, q+1)
    return encoding, ansatz, tuple(x), tuple(theta)


class BudgetExhausted(Exception):
    pass


class NativeQiskitClassifier(QuantumClassifier):
    def __init__(self, labels=('alfalfa','maize','wheat'), reps=1,
                 optimizer='COBYLA', max_evals=100):
        # Reuse organizer construction for the one-block model.
        super().__init__(labels=labels, max_evals=max_evals, encoding='ry')
        self.reps, self.optimizer_name = reps, optimizer
        self.encoding_circuit, self.ansatz, inputs, weights = circuit_parts(reps)
        if reps == 2:
            self.circuit = self.encoding_circuit.compose(self.ansatz)
            self.inputs, self.weights = inputs, weights
            self.qnn = SamplerQNN(circuit=self.circuit, sampler=QMLSampler(shots=None),
                                  input_params=self.inputs, weight_params=self.weights,
                                  interpret=lambda outcome:int(outcome)%len(self.labels),
                                  output_shape=len(self.labels))
        else:
            # Use the organizer's actual symbols with the identical explicit subcircuits.
            self.encoding_circuit = self.encoding_circuit.assign_parameters(dict(zip(inputs,self.inputs)))
            self.ansatz = self.ansatz.assign_parameters(dict(zip(weights,self.weights)))
        self.forward_passes = 0
        self.logical_field_evaluations = 0

    def prepare_states(self, X):
        return [Statevector.from_instruction(self.encoding_circuit.assign_parameters(dict(zip(self.inputs,row))))
                for row in angles(X)]

    def statevector_proba(self, states, weights, count=True):
        operator = Operator(self.ansatz.assign_parameters(dict(zip(self.weights,weights))))
        p = np.asarray([np.bincount(self.outcomes,weights=state.evolve(operator).probabilities(),
                                    minlength=len(self.labels)) for state in states])
        if count:
            self.forward_passes += 1
            self.logical_field_evaluations += len(states)
        return probabilities(p,len(states),len(self.labels))

    def loss_gradient(self, states, y_index, weights, gradient=True):
        p = self.statevector_proba(states,weights)
        target_p = p[np.arange(len(y_index)),y_index]
        safe = np.clip(target_p,1e-10,None)  # Same clipping and log base as organizer loss.
        loss = float(-np.log2(safe).mean())
        if not gradient:
            return loss
        grad = np.zeros(len(weights))
        for j in range(len(weights)):
            plus, minus = np.array(weights,copy=True),np.array(weights,copy=True)
            plus[j] += np.pi/2
            minus[j] -= np.pi/2
            dp = (self.statevector_proba(states,plus)-self.statevector_proba(states,minus))/2
            derivative = dp[np.arange(len(y_index)),y_index]
            derivative = np.where(target_p>1e-10,derivative,0.)
            grad[j] = -np.mean(derivative/safe)/np.log(2)
        return loss,grad

    def fit(self, X_train, y_train, seed=0):
        if self.optimizer_name not in ('COBYLA','L-BFGS-B'):
            raise ValueError('Unknown optimizer')
        if set(y_train)!=set(self.labels):
            raise ValueError('Every declared crop must appear in training')
        self.seed = int(seed)
        self.initial_point = np.random.default_rng(seed).uniform(-np.pi,np.pi,len(self.weights))
        self.forward_passes = self.logical_field_evaluations = 0
        start = perf_counter()
        states = self.prepare_states(X_train)
        self.state_preparation_seconds = perf_counter()-start
        self.history, self.iterations = [],0
        yi = np.array([np.flatnonzero(self.labels==v)[0] for v in y_train])
        self.best_loss, self.best_weights = np.inf,self.initial_point.copy()
        use_gradient = self.optimizer_name=='L-BFGS-B'

        def objective(w):
            if len(self.history)>=self.max_evals:
                raise BudgetExhausted()
            result = self.loss_gradient(states,yi,w,gradient=use_gradient)
            loss = result[0] if use_gradient else result
            if loss < self.best_loss:
                self.best_loss,self.best_weights = loss,np.array(w,copy=True)
            self.history.append({'evaluation':len(self.history)+1,'objective_bits':float(loss),
                                 'seconds':perf_counter()-start,
                                 'gradient_norm':float(np.linalg.norm(result[1])) if use_gradient else None,
                                 'probability_passes':self.forward_passes,
                                 'logical_field_evaluations':self.logical_field_evaluations})
            return result

        def callback(_):
            self.iterations += 1

        options = {'maxiter':self.max_evals}
        if use_gradient:
            options.update(maxfun=self.max_evals,ftol=1e-9,gtol=1e-5,maxls=20)
        try:
            result = minimize(objective,self.initial_point,method=self.optimizer_name,
                              jac=True if use_gradient else None,callback=callback,options=options)
            self.termination = {'message':str(result.message),'success':bool(result.success),
                                'nfev':int(result.nfev),'njev':int(result.njev) if use_gradient else 0,
                                'nit':int(result.nit) if hasattr(result,'nit') else self.iterations}
            # Match optimizer's returned point for COBYLA; use its optimum for L-BFGS-B too.
            final = result.x
        except BudgetExhausted:
            self.termination = {'message':'Strict objective-evaluation cap reached; returned best training loss',
                                'success':False,'nfev':len(self.history),
                                'njev':len(self.history) if use_gradient else 0,'nit':self.iterations}
            final = self.best_weights
        self.model = SimpleNamespace(weights=np.asarray(final))
        self.fit_seconds = perf_counter()-start
        return self

    def resources(self):
        return {**super().resources(),'entangling_blocks':self.reps,
                'ry_parameter_banks':self.reps+1,'optimizer':self.optimizer_name,
                'gradient_evaluations':sum(h['gradient_norm'] is not None for h in self.history),
                'native_operator_builds':self.forward_passes,
                'logical_field_probability_evaluations':self.logical_field_evaluations,
                'training_engine':'Qiskit Statevector.evolve(Qiskit Operator(bound ansatz)); fixed input-state cache',
                'prediction_engine':'original Qiskit SamplerQNN + QMLSampler(shots=None)'}
