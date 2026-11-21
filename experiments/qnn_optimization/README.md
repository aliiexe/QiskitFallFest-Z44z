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
MPLCONFIGDIR=/private/tmp/qiskit-baseline-mpl .venv/bin/python experiments/qnn_optimization/run.py --results outputs/qnn_optimization_2026-11-21
MPLCONFIGDIR=/private/tmp/qiskit-baseline-mpl .venv/bin/python experiments/qnn_optimization/analyze.py outputs/qnn_optimization_2026-11-21
.venv/bin/python experiments/qnn_optimization/verify_artifacts.py outputs/qnn_optimization_2026-11-21
```

Use a NEW output directory for a new training run; the runner refuses to
overwrite prior training records. Verification creates the initial directory.
Analysis and artifact verification can be rerun without fitting anything.

## Locally executed Phase-1 results

Fifteen training-fold fits selected D before validation evaluation; fifteen
additional full-training fits evaluated all five predefined configurations
with seeds 0/1/2. The complete training/verification stage took 51.07 seconds
on this Mac with the shared native-Qiskit caching engine. This speed comes
from classical simulation reuse; it is not quantum hardware speedup.

| ID | Training-only OOF F1 (seed 0) | Validation mean F1 +/- seed SD | Accuracy | Mean total seconds | Mean probability passes |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 0.778649 | 0.855694 +/- 0.006027 | 85.56% | 0.559 | 100.0 |
| B | 0.857304 | 0.864420 +/- 0.006883 | 86.44% | 1.946 | 470.3 |
| C | 0.870396 | 0.827363 +/- 0.037186 | 82.89% | 0.580 | 100.0 |
| D | **0.871754** | **0.880663 +/- 0.009427** | **88.00%** | 4.215 | 1041.7 |
| E | 0.784189 | 0.860148 +/- 0.001452 | 86.00% | 2.479 | 522.3 |

All A predictions and metrics match the earlier locally reproduced original
SamplerQNN baseline exactly. Timing changed because the training evaluator
caches the input states and evaluates a Qiskit-generated ansatz Operator;
compare runtime among these A-E runs, not against a different simulator path.

D's validation seed scores were 0.887329, 0.867331 and 0.887329. The independently
executed original SVM scored 0.879740, accuracy 88.00%, with 217 support vectors.
Thus D's mean F1 is only 0.000923 above the SVM and its mean accuracy is equal.
This is not evidence of quantum advantage. The earlier unavailable prototype
reported similar numbers; matching a reported score does not verify its source.

### What improved and what did not

- B versus A changes optimizer only: observed mean F1 gain +0.008726, with more
  logical probability work. All B fits converged under their gradient tolerance.
- C versus A changes depth only at 100 COBYLA evaluations: mean F1 fell by
  0.028332 and seed sensitivity increased. Additional capacity alone did not help.
- D versus C uses the same larger circuit with gradient training: +0.053301 F1.
  D versus B adds capacity under gradient training: +0.016243 F1.
- E lets the original COBYLA model continue, up to B's maximum work budget.
  It converged at 277-732 evaluations rather than reaching the 1700 cap.
  Mean F1 improved modestly to 0.860148, below B and D. More COBYLA evaluations
  therefore did not recover the entire observed D gain in this protocol.
- All D seeds improved over their corresponding A seed; all D fits converged
  at 38-48 objective/gradient evaluations, representing 950-1200 probability
  passes per seed. Objective count alone would understate that work.

| Model | Alfalfa recall | Maize recall | Wheat recall | Wheat precision |
| --- | ---: | ---: | ---: | ---: |
| A | 88.67% | 88.00% | 80.00% | 88.86% |
| D | 91.33% | 85.33% | 87.33% | 80.86% |
| SVM | 94.00% | 86.00% | 84.00% | 84.00% |

D found more wheat but made more false wheat predictions. It also improved
alfalfa recall and the precision of alfalfa/maize predictions, while losing
some maize recall. Across the repeated three-seed validation decisions, it
repaired 24 A errors and introduced 13 new errors: a net 11 correct decisions.
Those decisions come from the same 150 fields, not 450 independent fields.

A concrete check connects this experiment to the earlier saturation study:
the low-NDVI wheat field `r2_614f27a98e1657c5` is now classified as wheat in
all three D seeds, while A missed it in every seed. But all three nearby
extreme-input maize examples (`...7ca21d`, `...c718d1`, `...5020df`) are also
called wheat by every D seed. This fixes the wheat example without establishing
that the model can separate those nearly identical encoded inputs. The full
IDs and predictions are in field_examples.json.

### Uncertainty and scope

The paired 18-spatial-block validation bootstrap (2000 draws, RNG 2026) gave:

- D minus A macro-F1: +0.024969, percentile interval [-0.002980, +0.072257].
- D minus SVM macro-F1: +0.000923, interval [-0.041868, +0.038160].

Both intervals include zero. Observed validation improvements do not establish
superiority on new regions or fields. Selection CV uses only initialization
seed 0 and lacks training spatial groups; D's OOF margin over C is small.
The validation set has also been inspected in earlier project work. The test
set remains untouched for model fitting, tuning and prediction in this phase.

The chosen model is a standalone QNN, with four qubits, 12 trainable angles,
six CX gates, depth 9 and no learned classical classifier. Shared PCA/scaling
has 99 transformation coefficients under the notebook's accounting, for 111
total fitted transformation/circuit values. The 11-parameter original classical
control is within 8.3% of D's quantum count but is still a very restrictive
one-hidden-unit architecture; beating it does not prove a quantum contribution.

## Preserved evidence and next priorities

Save protocol, selection timestamp/hash, fold assignments, OOF predictions,
every model's initial/final parameters, complete optimization histories,
full validation probabilities/predictions, resource counts, convergence status,
timings, selected-model field changes, and verification reports in the result
directory. Model and notebook integrity is checked by pre/post hashes.
Artifact replay also reconstructs all 15 full-training models from saved
preprocessing and weights, reproducing 2250 validation probability rows,
predicted labels and metrics. No refitting is needed for that verification.

Next Phase-1 questions should remain training-selected: entanglement topology
at fixed parameter count, then a justified representation comparison. A
different data re-uploading architecture belongs to Phase 2. Eight qubits is
an upper limit, not a target. Hybrid fusion belongs to Phase 3 after comparing
standalone quantum candidates. No slides or hybrid were produced in this phase.
