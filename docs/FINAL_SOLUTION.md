# Final ring solution: configuration and reproducibility

Start with `notebooks/QFF_Final_Ring_Demo.ipynb`. It follows the organizer notebook's 18-cell flow (10 code cells), with saved-model inference by default and optional retraining at the end. It uses only this clone and the unchanged `requirements.txt`.

## Frozen configuration

| Item | Setting |
| --- | --- |
| Data | All 914 training fields; official 150-field validation for final reporting |
| Preprocessing | Training-fitted StandardScaler → PCA(4, full SVD) → StandardScaler |
| Feature encoding | Four RY gates; `(π/2) × (1 + tanh(z))` |
| Ansatz | RY(4) → ring CX → RY(4) → ring CX → RY(4) |
| Ring order | CX(0,1), CX(1,2), CX(2,3), CX(3,0), in each block |
| Resources | 4 qubits, 12 trainable angles, 8 CX, logical depth 12; no routing |
| Readout | Basis-state integer modulo 3 → alfalfa, maize, wheat (groups of 6/5/5) |
| Loss | Mean `−log2(clip(p_true, 1e-10, None))`; zero derivative at/below the clipping threshold |
| Initial angles | `np.random.default_rng(seed).uniform(-π, π, 12)`, seeds 0/1/2 |
| Optimizer | SciPy L-BFGS-B; analytic Jacobian, no bounds |
| Budget | maxiter=100, maxfun=100, strict cap of 100 loss/gradient calls |
| Tolerances | ftol=1e-9, gtol=1e-5, maxls=20 |
| Training | Qiskit-generated RY/CX operators; NumPy matrix-adjoint derivatives |
| Inference | Original standard SamplerQNN, QMLSampler(shots=None) |

The strict-budget fallback returns the best training-loss weights. One loss/gradient call costs different amounts: parameter-shift uses 25 full probability passes, while matrix-adjoint uses one plus a reverse sweep. These CPU work counters are **not hardware circuit executions**. The reverse computation uses 52 small 16×16 matrix products plus two field-batch products per loss/gradient call. Per-fit input and gate caches are constructed inside training time.

## Why it is faster

Parameter-shift recomputes probabilities after increasing and decreasing each of 12 angles. Our trainer instead forms the circuit operator from Qiskit's RY/CX matrices, propagates all input states in a batch, and differentiates the product backwards. Aggregating across fields before reversing avoids many Python loops. The prediction circuit remains identical.

This specialized **classical ideal simulator** is not quantum hardware acceleration, an IBM service or an official Qiskit reverse-gradient primitive. Dense operators grow exponentially with qubit count; the derivative is restricted to the verified four-qubit RY/CX ring. It is checked against finite differences, exact parameter-shift and independent Qiskit forward/backward calculations.

## Selection and evidence

Earlier training-only topology CV selected **full**, while ring had the highest observed validation F1. Releasing ring documents our strongest observed validation candidate; it does not make ring the topology-CV winner.

For gradient selection, three methods used the same three training-only stratified folds (split seed 2026) and initialization seeds 0/1/2: 27 fits. These internal folds are **not spatial CV**. The fastest engine was eligible only if pooled out-of-fold labels matched baseline for every seed and mean final loss matched within 1e-7 bits. All three methods achieved pooled OOF macro-F1 0.8526688 (seed SD 0.0134253); matrix-adjoint was fastest. No validation scores were used for that choice.

After selection was locked, paired full-training fits preserved all official validation labels for every seed: F1 0.8943463, accuracy 134/150. Zero score variation for three seeds does not establish universal stability. No architecture/tolerance was retuned for this release. Private test labels were not used.

## Artifacts and implementation origin

`artifacts/final_ring/provenance.json` records source paths, original hashes, package versions and research commit **56d406a9fe67588ac58d4909c46b4dbf3ff6af46**. Release starts from main **510f8626fcfb7448097b89892aa795a1ace2710b**, with selected files copied, not research history merged.

- `ring_seed_{0,1,2}_weights.npz`: initial/final weights from the verified matrix-adjoint fits.
- `preprocessing.joblib`: common training-fitted PCA pipeline, identical for all seeds.
- `original_seed_{0,1,2}_weights.npz`: original eight-angle **linear** QNN, equivalent cached COBYLA training, 100 calls. These replay the unchanged organizer model's inference; they are not the lightweight eight-angle ring experiment.
- `svm.joblib`: deterministic StandardScaler + RBF SVC, C=100, gamma=0.01, all 13 features.
- `*_predictions.csv`: actual field IDs, labels and QNN probabilities for exact replay.
- `*_history.json`: recorded training losses/times, including seed-0 parameter-shift for the same 12-angle model.
- `recorded_runs.json`: curated metrics, timing stages, resources and seeds. Session names distinguish the paired gradient benchmark from original-QNN/SVM replay.
- `training_only_selection.json`, `cv_fold_assignments.csv`, `cv_ring_seed_*_oof.csv`: selection rule, fixed folds and OOF evidence; fresh CV replay is available.
- `gradient_verification.json`: the original training-only numerical and Qiskit derivative audit.
- `organizer_sha256.json`: all 59 main files except the deliberately updated README; includes the original notebook, data, requirements and source.
- `checksums.json`: every bundled artifact is checked before loading the included joblib models. Joblib is Python serialization; use artifacts from a trusted clone.

The three migrated modules retain the verified algorithms with only imports changed to `qff_solution`. The public `FinalRingClassifier` explicitly selects `operator_adjoint`; the experimental base class's older default is not used. Saved inference is never presented as fresh training.

## Timing scope

Recorded medians across seeds 0/1/2, seconds:

| Same 12-angle ring | Training, including caches | Total |
| --- | ---: | ---: |
| Cached parameter-shift | 5.604531 | 5.704809 |
| Matrix-adjoint | 0.082123 | 0.177378 |

Total = per-run preprocessing + model construction + training + standard Qiskit inference on 150 fields. Median totals need not equal the sum of stage medians. Approximately 68× training / 32× total improvements describe this recorded machine/session. Qiskit input-state construction dominates remaining training time.

Original-QNN/SVM archived timings came from a separate session on the same machine, excluding model construction. Original-QNN replay used cached input states rather than the organizer's slower NeuralNetworkClassifier loop. The notebook distinguishes these scopes, saved loading, fresh inference and optional new fits. Imports, I/O, kernel startup and rendering are excluded from model times; Run All includes these extra costs.

## Reproduce

Follow README installation instructions with Python 3.12. In VS Code select the clone's `.venv` kernel, then Restart Kernel → Run All. Change `FIELD_INDEX` for another validation field. Enable `RETRAIN` only to regenerate all three fits in a separate ignored output folder.

```bash
.venv/bin/python -m pip check
.venv/bin/python scripts/verify_final_solution.py --reproduce-cv --execute-notebook
```

The verifier checks artifact/organizer hashes; derivatives at initial and trained weights for every seed; fixed-weight Qiskit probabilities; all saved metrics; fresh fast and parameter-shift fits for all seeds; matching weights, loss curves and work counts; a deterministic SVM refit; training-only OOF predictions; and saved/fresh notebook workflows, each in a new kernel. It uses its current interpreter through a private temporary kernelspec, not a globally registered research kernel. Reports and executed copies go in a new ignored `outputs/release_check_*` directory. The command does not merge or change bundled models.
