# Phase 1: native Qiskit optimization and depth

Research branch: `research/quantum-model-optimization`, based on `7786178`.
The notebook's pre-existing edits are preserved and excluded from research commits.

## Repository audit

Reviewed README, the complete two-page challenge brief, HINTS, EVALUATION,
COVERAGE, all eight challenge modules, both configs, all notebook cell sources,
organizer summaries, the locally reproduced baseline, and the encoding-scale study.

The organizer code remains the reference: `load_data` verifies data and split
hashes; `reference` supplies the frozen 13-feature SVM; `score` supplies all crop
metrics. QuantumClassifier supplies the original RY encoding, circuit and
SamplerQNN prediction interface. The submission adapter remains untouched.

Constraints: exact local simulation, at most eight qubits; 914 training fields,
150 validation fields, 221 unlabeled test fields; seeds 0/1/2; macro-F1 primary;
per-crop metrics, resources, runtime and a classical control required. The main
rubric is 30 performance / 40 analysis / 30 design and reproducibility.

Findings motivating this experiment:

- The original QNN has only eight angles and stops after 100 COBYLA evaluations.
  Its saved losses were still improving close to that limit.
- The current circuit has one forward CX bank and two trainable RY banks.
  We call this one entangling block, not one RY bank.
- The outcome-modulo-three readout has 6/5/5 basis states per crop, not three
  directly measured crop qubits. All states contribute to predictions.
- Four PCA components retain most variance, but that alone does not establish
  that every crop-discriminative feature is preserved.
- The earlier scaling study improved wheat recall while reducing mean macro-F1;
  retain the original tanh encoding for this controlled experiment.
- The SVM receives 13 inputs and the QNN four. This is a whole-pipeline baseline,
  not an isolated quantum-contribution test. The 11-parameter tiny classical
  head is a limited control and differs from the original QNN by 37.5% in size.
- Installed CrossEntropyLoss.gradient assumes a softmax-logit setting, whereas
  SamplerQNN returns measured probabilities. Our gradient is explicitly the
  derivative of mean negative log2 probability; finite differences verify it.

## Architectures and hypothesis

All candidates use four qubits and the original angle rule. A/B/E have two RY
banks and one forward CX chain (8 angles, 3 CX, depth 6). C/D have three RY banks
and two forward CX chains (12 angles, 6 CX, depth 9). Inputs enter once.

The two-block gate sequence is built explicitly; its full unitary was verified
against Qiskit's real_amplitudes(reps=2, entanglement='linear'), rather than
assuming the library setting has the intended ordering.

Hypothesis: gradient-based optimization or additional circuit capacity can
improve the standalone QNN. A longer-COBYLA control checks whether stopping
the original optimizer early accounts for the difference.

| ID | Entangling blocks | Optimizer | Maximum objective evaluations |
| --- | ---: | --- | ---: |
| A | 1 | COBYLA | 100 |
| B | 1 | L-BFGS-B | 100 |
| C | 2 | COBYLA | 100 |
| D | 2 | L-BFGS-B | 100 |
| E | 1 | COBYLA | 1700 |

An RY parameter-shift gradient needs 2*p additional probability passes. Thus
one evaluation with its gradient costs 17 passes for B and 25 for D. E matches
B's MAXIMUM probability-pass budget; optimizers may converge earlier. Equal
objective caps are not equal quantum-work budgets. Count every actual pass.

## Qiskit implementation and verification

`NativeQiskitClassifier` extends the organizer class. All quantum evolution is
performed by Qiskit Statevector and Operator. Only the unchanged input states
are cached. No handwritten gate simulator is used to train or predict.
Prediction retains the original SamplerQNN/QMLSampler path. NumPy is used for
class aggregation and loss/gradient arithmetic, not quantum gate simulation.

Three seeds on typical and extreme TRAINING inputs passed:

- Cached evaluation versus direct full-circuit Statevector: error <= 4.45e-16.
- Cached evaluation versus SamplerQNN: error <= 4.33e-12. The sampler's count
  compatibility representation approximates probabilities to dyadic numbers;
  the verification tolerance is 1e-10.
- Explicit circuit versus library ansatz unitary: maximum error 0.
- Parameter-shift cross-entropy gradient versus finite differences: <= 4.8e-11.
- Parameter order, gate order, all three class outputs, and resource counts checked.

The earlier independent two-layer NumPy prototype and its claimed 0.8807/0.8885
scores remain historically unverified. Its ZIP was not present in the project,
Downloads, or the indicated home folder when inspected. Native tests here do
NOT establish equivalence with that unavailable source.

## Selection and evaluation

Protocol is fixed in protocol.json before scoring validation. Use stratified
three-fold training-only CV, random state 2026 and training seed 0, to select a
configuration by pooled out-of-fold macro-F1. Refit every scaler/PCA inside
each fold. Training spatial group identifiers are not supplied, so these folds
are not spatial; report this limitation. Run all predeclared configurations on
all training rows with seeds 0/1/2, and report all validation results afterward.
No test predictions, validation-based tuning or best-seed selection.

Reproduce verification from the repository root:

```bash
.venv/bin/python experiments/qnn_optimization/verify.py --output outputs/qnn_optimization_2026-11-21/verification.json
```
