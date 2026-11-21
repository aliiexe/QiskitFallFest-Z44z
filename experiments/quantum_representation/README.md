# Standalone quantum representation decision

This is the conditional follow-up to the completed classical PCA investigation,
not a hybrid experiment. All predictive trainable weights are quantum RY angles;
PCA and optional component selection are preprocessing. No classical classifier
head, validation inference, test inference, eight-qubit circuit or data re-uploading
was introduced. Original files and earlier results remain unchanged.

## Why these candidates?

Training-only classical comparisons showed useful information beyond the first
four PCs: nested-SVM OOF macro-F1 .893912 → .909496 → .910559 for 4/6/8 PCs.
An inexpensive fold-selected four-PC view also improved that classical model.
These findings justified a small quantum comparison, not assuming the classical
gain transfers to the quantum classifier.

1. `first4_q4`: first four PCs, four qubits; the verified full-connectivity reference.
2. `selected4_q4`: select four of the first eight PCs with training-fold ANOVA;
   same circuit and same initialization as the reference.
3. `first6_q6`: first six PCs, six qubits; larger native Qiskit circuit.

Full connectivity was selected by the earlier training-only topology CV. Ring's
.8943 validation macro-F1 is historical evidence and did not select this topology.
Linear, ring and full models/results are all preserved.

## Fixed design and interpretation

Three previous stratified training folds (split seed 2026), all initialization
seeds 0,1,2: 27 fresh CV fits. Every scaler, PCA and optional selector is fitted
inside its training fold. Selection uses mean seed-specific pooled OOF macro-F1,
with exact ties favoring fewer qubits then the first-four reference. This is an
adaptive follow-up on reused training folds, not an unbiased final evaluation.
Stratification does not test geographic transfer.

The angle map is unchanged: `angle = pi/2 * (1+tanh(z))`. Quantum evolution uses
actual Qiskit Statevector and Operator; predictions use SamplerQNN/QMLSampler
with exact probabilities. The inherited verified parameter-shift derivative
optimizes cross-entropy in bits using L-BFGS-B: max 100 objective/gradient calls,
maxiter/maxfun 100, ftol 1e-9, gtol 1e-5, maxls 20, no bounds. Seeds initialize
`default_rng(seed).uniform(-pi,pi,3*n)`. Same-width initial points are identical;
the first 12 values also match across widths, although their gate roles differ.

Circuit (every RY bank contains one independently trained angle per qubit):

```text
Standardized PCs → fixed RY input encoding
                 → RY bank → full CX block → RY bank → full CX block → RY bank
                 → exact basis probabilities → basis integer modulo 3 → crop
```

| Width | State dimension | Trained angles | CNOTs | Logical depth | Passes per gradient | Readout groups |
|---|---:|---:|---:|---:|---:|---|
| Four | 16 | 12 | 12 | 13 | 25 | 6 / 5 / 5 |
| Six | 64 | 18 | 30 | 19 | 37 | 22 / 21 / 21 |

The six-qubit comparison changes representation, parameter count, connectivity,
readout proportions and runtime. It does not isolate information retention as
cleanly as the selected-four control. Equal call caps are not equal physical
work. Resources exclude routing, hardware noise and finite-shot sampling.
The selected-four inputs retain increasing PC order: folds 0 and 1 map PCs
2,3,4,8 to qubits 0,1,2,3; fold 2 keeps PCs 1,2,3,4. This changes which feature
reaches which wire as well as which feature is retained. Ordered CNOTs and the
readout are not permutation-invariant, so the failure does not prove PC8 useless.

## Completed results

| Candidate | Seed 0 F1 | Seed 1 F1 | Seed 2 F1 | Mean ± seed SD | Three-fold fit time, mean |
|---|---:|---:|---:|---:|---:|
| First4 / 4q | .870125 | .867758 | .868518 | **.868800 ± .000987** | 12.43 s |
| Selected4 / 4q | .854243 | .862588 | .845964 | .854265 ± .006787 | 8.17 s |
| First6 / 6q | .849574 | .812810 | .829415 | .830600 ± .015032 | 34.98 s |

Seed SD describes initialization variation on the same folds, not uncertainty
about future fields. First4 wins for every seed. The six-qubit mean is lower by
.038200 macro-F1 and costs 2.81 times as much CV training time.

| Candidate | Alfalfa precision/recall | Maize precision/recall | Wheat precision/recall |
|---|---|---|---|
| First4 / 4q | 91.71% / 91.09% | 81.23% / 84.02% | 87.33% / 85.96% |
| Selected4 / 4q | 86.31% / 89.47% | 78.63% / 85.40% | 90.15% / 83.53% |
| First6 / 6q | 85.44% / 89.74% | 79.34% / 74.38% | 84.94% / 85.33% |

Selected4 repairs 53 decisions and introduces 86 errors across the three seeds;
First6 repairs 118 and introduces 206. Each field appears once per seed, so
these are repeated decisions, not distinct-field counts. Selected4 reduces
maize→wheat decisions 108→66, but maize→alfalfa rises 8→40 and wheat→maize rises
126→143. Its higher wheat precision comes with lower wheat recall: it is more
cautious about predicting wheat, and misses more real wheat fields.

First6 increases maize→wheat 108→144 and maize→alfalfa 8→42. Wheat→maize falls
126→115, but wheat→alfalfa increases 53→72, so total wheat recall still declines.
Real example: wheat `r2_0786546ce39aeece` (NDVI .2780) changes maize→wheat under
First6 in every seed. Conversely, maize `r2_7f615dc6fd9fbe65` (NDVI .3015) changes
maize→wheat in every seed. The complete identities/probabilities are saved.

All 27 fits stop by the projected-gradient tolerance before the 100-call cap.
Average calls per fold are 56.7 / 36.7 / 53.2. These are local stopping conditions,
not proof of global optima. Six-qubit training loss is worse too; overfitting
alone does not explain the result. Representation/circuit/readout mismatch and
local optima are possible explanations, not experimentally isolated causes.
Loss-curve spikes are trial line-search evaluations, not necessarily accepted
optimizer steps or evidence of a broken gradient.

## Decision

Keep the first-four-PC, four-qubit, full-connectivity reference. Three complete
full-training fits on all 914 fields (seeds 0,1,2) and their preprocessing/weights
are saved separately. Their weights exactly reproduce the historical full model.
No new validation scores were generated and ring was not promoted from validation.

The experiment establishes that discarding later PCs loses useful classical
predictive information. It does not establish that six qubits solve that problem,
that all later PCs are noise, or that quantum advantage exists.

Eight qubits are deferred: the nested-SVM gain from six to eight is only .001063,
and the first width expansion failed here. Data re-uploading remains a reasonable
future hypothesis: encode additional inputs at a later circuit stage while
retaining four qubits. It would change gate placement and depth and requires a
separate preregistered training-only comparison; it is not an observed improvement.
Replacing PC1 with a univariately selected later PC was tested here and did not
improve the QNN. A different supervised representation is also unproven.

## Figures and checks

`01`: bars are mean OOF macro-F1, markers are seeds; right panel measures total
training time for one three-fold pass. `02`: recall is the share of real crop
fields found, precision the share of that crop's predictions that are correct.
`03`: confusion-matrix rows are actual crops, columns predictions; the diagonal
is correct, off-diagonal cells are errors. Counts pool the same fields over seeds.
`04`: training-loss curves, lower is better; these are not held-out performance.
`05`: resource/cost table. `06`: actual descriptive PC values for two wheat fields
illustrate both a repaired and a new error in the classical investigation.

Verification compares Qiskit library and explicit circuit operators, direct
statevector and SamplerQNN probabilities (tolerance 1e-10), and all parameter
gradients against finite differences (max error < 6e-10). Nine reference CV fits
reproduce historical weights/metrics/OOF predictions. All nine new OOF files are
rescored; call counters and identical starts are checked. Prior protected files,
including notebook edits and historical outputs, retain their SHA-256 hashes.

## Reproduction

From the project root, use a fresh destination:

```sh
MPLCONFIGDIR=/private/tmp/qiskit-baseline-mpl .venv/bin/python experiments/quantum_representation/verify.py outputs/quantum_representation_reproduction
MPLCONFIGDIR=/private/tmp/qiskit-baseline-mpl .venv/bin/python experiments/quantum_representation/run.py --results outputs/quantum_representation_reproduction
MPLCONFIGDIR=/private/tmp/qiskit-baseline-mpl .venv/bin/python experiments/quantum_representation/analyze.py outputs/quantum_representation_reproduction
```

The runner checks the branch and verification hashes and refuses to overwrite
previous records. Runtime benchmark estimated 3–6 minutes; measured experiment
time was 196.0 s for CV, final fits and saving (plots/verification excluded).
Results are in `outputs/quantum_representation_2026-11-21/`.
