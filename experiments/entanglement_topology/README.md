# Experiment 1: Entanglement topology optimization

Reference: verified standalone Model D, research commit `603aa09`. All research
stays on `research/quantum-model-optimization`. No organizer files, notebook,
datasets, or previous experiment files are modified. No eight-qubit, data
re-uploading or hybrid experiment is included.

## Hypothesis and ordered connectivity

Keeping the four prepared inputs, 12 RY angles, optimizer and readout fixed,
changing which qubits directly interact may help maize-wheat separation.
Denser connectivity may also worsen optimization, class balance or cost.
PCA order does not encode physical adjacency, and more CX gates do not
automatically imply a more useful quantum representation.

Every topology uses three RY parameter banks separated by two CX blocks.
Data enters once through the original four RY angles. Edges below are
control -> target, executed left to right in BOTH blocks:

| Topology | Ordered CX edges per block | Total CX | Logical depth |
| --- | --- | ---: | ---: |
| Linear | 0->1, 1->2, 2->3 | 6 | 9 |
| Ring | 0->1, 1->2, 2->3, 3->0 | 8 | 12 |
| Full | 0->1, 0->2, 0->3, 1->2, 1->3, 2->3 | 12 | 13 |

These are logical, unrouted counts, including input encoding in depth.
The chosen full and ring orders/directions are explicit experimental designs;
connectivity alone does not uniquely specify a circuit.

## Fair comparison

The protocol is recorded before selection/evaluation in protocol.json.
All topologies use all 914 training fields for the main fits and the same
150 validation fields. Every model has four qubits, four PCA inputs,
12 trainable angles, the original tanh encoding, modulo-3 readout and exact
probabilities. Native Model-D optimization is inherited unchanged: L-BFGS-B,
100 objective/gradient evaluation cap, ftol=1e-9, gtol=1e-5, maxls=20.
Initial weights match EXACTLY across topologies for seed 0, 1 or 2.

Selection uses the previous three fixed stratified TRAINING folds, with
scaling/PCA refitted inside each fold. For every topology, all three seeds
are fitted in all folds: 27 CV fits. Each seed produces predictions for all
914 training fields while each field is held out. Select by the mean of
the three pooled out-of-fold macro-F1 scores; break ties with fewer CX gates.
Lock selection before scoring validation. Then refit all nine predefined
topology/seed configurations on all training rows and report every result.
Training spatial identifiers are unavailable, so selection CV is not spatial.
The validation set has been inspected in previous project experiments.

Every gradient evaluation costs 25 probability passes for every topology.
Actual convergence can require different numbers of evaluations. Each pass
also involves a different CX count/depth; report evaluations, gradients,
logical field probability evaluations, runtime and termination status.
The Qiskit input-state cache is the same for all candidates. It accelerates
classical simulation and does not demonstrate quantum hardware speedup.

## Verification and provenance

The separate class extends the verified NativeQiskitClassifier. Quantum
evolution remains Qiskit Statevector/Operator; prediction remains native
SamplerQNN/QMLSampler. The organizer data loader and metrics are reused.

Tests check exact gate order, 12-parameter order, class readout, four-qubit
resources, explicit-versus-library unitaries with the SAME ordered edges,
direct-versus-cached statevectors, SamplerQNN probabilities and cross-entropy
gradients versus finite differences. The linear implementation is checked
against Model D. Fresh linear fits must reproduce its saved final weights,
all validation predictions/metrics, and seed-0 training OOF predictions.

## Reproduce from the repository root

Use a NEW results directory when fitting again. Existing runs are not overwritten.

```bash
.venv/bin/python experiments/entanglement_topology/verify.py --output outputs/my_topology_run/verification.json
MPLCONFIGDIR=/private/tmp/qiskit-baseline-mpl .venv/bin/python experiments/entanglement_topology/run.py --results outputs/my_topology_run
```

Verification uses only training examples, including extreme PCA values.
Source, data, notebook and earlier experiment hashes are checked before and
after fitting. Protocol, folds, initialization/final weights, CV outcomes,
probabilities, predictions, loss histories, timing and resources are saved.
