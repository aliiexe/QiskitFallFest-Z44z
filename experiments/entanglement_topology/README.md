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
MPLCONFIGDIR=/private/tmp/qiskit-baseline-mpl .venv/bin/python experiments/entanglement_topology/analyze.py outputs/my_topology_run
```

Verification uses only training examples, including extreme PCA values.
Source, data, notebook and earlier experiment hashes are checked before and
after fitting. Protocol, folds, initialization/final weights, CV outcomes,
probabilities, predictions, loss histories, timing and resources are saved.

## Locally executed results, 21 November 2026

All 27 training-fold fits and nine full-training fits were executed locally
using Qiskit. The fitting/selection/verification stage took 163.95 seconds,
excluding later plotting and artifact replay. All nine full fits converged
under the gradient tolerance, below the 100-evaluation cap. Fresh linear fits
reproduce prior Model D weights, predictions and metrics; its seed-0 OOF
predictions also reproduce the previous training-fold experiment.

| Topology | Mean training-only OOF F1 | Validation F1 seed 0 | Seed 1 | Seed 2 | Mean F1 +/- seed SD | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear | 0.856768 | 0.887329 | 0.867331 | 0.887329 | 0.880663 +/- 0.009427 | 88.00% |
| Ring | 0.852669 | 0.894346 | 0.894346 | 0.894346 | 0.894346 +/- 0.000000 | 89.33% |
| Full | **0.868800** | 0.874835 | 0.874835 | 0.867763 | 0.872477 +/- 0.003334 | 87.11% |

**Training-only CV selected full. Ring had the best observed validation result.**
The selection record was locked before validation and is not rewritten to call
ring the CV winner. The study did not establish a training-selected improvement
over Model D on validation. Ring remains an exploratory candidate requiring
confirmation; none of the previous model or submission implementations changed.

The three ring fits produced identical predicted labels on all 150 validation
fields, not just equal scores. Their training-fold results were less stable:
ring's seed-specific pooled OOF F1 scores were 0.834305, 0.866027 and 0.857675.
Full's were 0.870125, 0.867758 and 0.868518. All-data initialization stability is
therefore different from behavior when training on different field subsets.
The training subsets contain 609-610 fields and have fold-specific PCA fits;
this study does not isolate why the CV ranking and validation ranking differ.

## Crop trade-offs

| Topology | Alfalfa recall | Maize recall | Wheat recall | Alfalfa precision | Maize precision | Wheat precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear | 91.33% | 85.33% | 87.33% | 91.95% | 92.11% | 80.86% |
| Ring | 92.00% | 84.00% | 92.00% | 92.00% | 97.67% | 80.70% |
| Full | 89.33% | 84.00% | 88.00% | 88.76% | 96.23% | 78.57% |

Across the three repeated validation runs, ring repaired nine linear mistakes
and introduced three new mistakes, a net six correct decisions. Five distinct
fields were repaired in at least one seed; one distinct field was newly wrong.
Ring's mean F1 changes versus the matching linear seed were +0.007017,
+0.027015 and +0.007017. Thus the observed validation gain occurred in 3/3 seeds.
Full repaired seven decisions and introduced eleven: it improved in only 1/3
paired seeds and had lower mean validation F1 than linear.

Concrete examples from the real validation data:

- Wheat `r2_0790857e45338bc4`, NDVI 0.235449: linear predicts maize for every
  seed; ring predicts wheat for every seed.
- Wheat `r2_89f882698714ecba`, NDVI 0.277658: same consistent repair.
- Maize `r2_40b6d4ef6686da1a`, NDVI 0.279126: linear predicts maize correctly
  in every seed; ring incorrectly predicts wheat in every seed.

Similar NDVI values do not establish identical 13-feature or PCA inputs. These
examples illustrate actual decision trade-offs, not a proven causal explanation
of how a particular CX edge acts. Confusion matrices show wheat->maize errors
falling from 10 to 3 decisions, while maize->wheat errors rise from 19 to 21.
Counts pooled over seeds describe the SAME fields, not 450 independent fields.

## Work, convergence and uncertainty

| Topology | Mean gradient evaluations | Mean probability passes | Mean preprocess+fit+predict seconds | CX gates | Depth |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear | 41.67 | 1041.67 | 4.511 | 6 | 9 |
| Ring | 56.67 | 1416.67 | 6.252 | 8 | 12 |
| Full | 62.00 | 1550.00 | 6.826 | 12 | 13 |

All configurations have the same 25 passes per loss/gradient call. Denser
connectivity did not save time and did not guarantee a better classifier.
Their mean final training losses were similar (approximately 0.382, 0.384,
0.381 nats for linear/ring/full). Lower training cross-entropy does not
guarantee higher validation macro-F1: the metrics measure different things
and refer to different field sets. Convergence is not proof of a global optimum.

Paired spatial-block resampling of the 18 validation blocks, 2000 draws,
RNG 2026, gives exploratory two-sided percentile intervals:

- Ring minus linear mean F1: +0.013683, interval [-0.018543, +0.046175].
- Full minus linear mean F1: -0.008186, interval [-0.057815, +0.025699].

Both include zero. Zero seed SD for ring does not imply zero uncertainty on
new fields. Intervals are conditional on these fitted models, without a
multiple-comparison correction. Training CV is not spatial, validation has
been inspected in prior work, and no hidden test labels or test predictions
were used. No quantum advantage or universal topology ranking is claimed.
Results apply to these explicit directed gate orders and fixed optimizer budget.

## Saved evidence and review

The result directory contains protocol/environment/source hashes, unchanged
prior-file hashes, all fold assignments, 27 CV records and parameter sets,
nine full-model initial/final parameter files, all predictions/probabilities,
loss histories, per-seed/summary metrics, paired errors, field examples,
convergence status, timings and native Qiskit circuit diagrams. Analysis
reconstructs all nine models from saved weights and preprocessing, reproducing
1350 validation probability rows and labels, and verifies gradient pass counts.

Figures show connectivity, exact native circuits, CV selection, validation
scores/runtime/gradient work, per-crop metrics, confusion matrices, every
seed's training curves and paired error changes. No slides, eight-qubit models,
data re-uploading or hybrid models were started.
