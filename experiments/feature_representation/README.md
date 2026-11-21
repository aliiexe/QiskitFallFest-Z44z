# Feature representation investigation

Training-only study, 21 November 2026. Original files and historical experiments
are preserved. No new validation or test predictions are made.

## Question and design

PCA orders numerical variation, not crop discrimination. Does retaining more
than four components improve predictions on held-out training fields?

Use 914 fields and all 13 inputs: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9,
B11, B12, NDVI. Class counts: 247 alfalfa, 242 maize, 425 wheat.
Three repeats of stratified three-fold CV use split seeds 2026, 2027, 2028.
The first repeat exactly matches the previous quantum study's folds.
Every scaler, PCA and supervised component selector is fitted inside its fold.

Three learners test robustness: fixed logistic regression; organizer SVM settings;
and nested SVM tuning (C=1,10,100; gamma=.01,.1,scale; three inner folds).
The nested procedure includes preprocessing inside each inner split. The primary
comparison uses the mean repeat-specific pooled out-of-fold (OOF) macro-F1.
This is model-selection evidence, not an untouched final generalization estimate.
Repeats share fields; their SD is partition variation, not an independent-sample SE.
Folds are stratified, not geographic; spatial generalization remains untested.

## Results

| Representation | Logistic | Fixed SVM | Nested SVM |
|---|---:|---:|---:|
| First 4 PCs | .863450 | .895865 | .893912 |
| First 6 PCs | .877822 | .908208 | .909496 |
| First 8 PCs | .896598 | .911961 | .910559 |
| All 13 PCs | .912667 | .919919 | .909307 |
| Original 13 features, standardized | .892496 | .911888 | .913041 |
| PCs 5–8 only | .372572 | .530603 | .615794 |
| Select 4 of first 8 PCs within folds | .877227 | .899478 | .900739 |

Four/six/eight PCs explain 99.1236% / 99.7050% / 99.9322% of standardized input
variance. PCs 5–8 explain only 0.8086% together, but prediction improves when
they are retained. The six-PC gain over four occurs in all repeats for all three
learners. Eight adds only .001063 over six for nested SVM; this is weak evidence
for paying for eight qubits, although logistic regression benefits substantially.

For nested SVM, wheat recall rises .8784 → .9043 → .9067; maize recall rises
.8884 → .8994 → .8994. Across three OOF passes of the same fields, six PCs repair
102 previously wrong decisions and introduce 59 newly wrong ones (43 net fixes).
Maize→wheat decisions fall 70→61 and wheat→maize 92→69. These are repeated
decisions, not 2,742 distinct fields. See the changed-prediction CSV for identities.

Real example: wheat field `r2_0786546ce39aeece` (NDVI .2780) and maize field
`r2_603a372b00c27a56` (NDVI .2019) are repaired in all three repeats by six PCs.
Wheat `r2_1da35aa4b575dc80` (NDVI .2729) becomes wrong in all three repeats.
Similar NDVI does not imply the same crop: a classifier needs the other inputs.

PCs 2,3,4 are selected in all nine outer training folds; the fourth selected input
is PC8 five times, PC1 three times, PC7 once. Selection is fold-local, never a
single supervised selection applied before CV. This justifies a four-qubit
representation control as well as a six-qubit candidate.

The full-PCA transformation is invertible (numerically verified), so raw13 versus
pca13 score differences reflect geometry, regularization and component rescaling,
not information loss. PCA coefficient signs are arbitrary. PC5 is dominated by a
B9 contrast, PC6 includes NDVI, PC8 contrasts B11 with B12; coefficients are not
causal feature importance. Band calibration and cloud processing are undocumented.

The primary classical selection is raw13, but this is a diagnostic control, not a
hybrid candidate or an instruction to use 13 qubits. Classical gains do not imply
that a quantum classifier can exploit the same information.

## Reading the figures

1. `01_complete_pca_variance`: bars show each PC's variance on a logarithmic axis;
   the cumulative curve measures reconstruction, not classification quality.
2. `02_original_feature_distributions`: each box is a crop's middle 50%; its line
   is the median; points show extreme fields. Axes differ between features.
3. `03_class_separation`: each dot is one training field. Color overlap shows why
   no two-dimensional view alone can settle predictive value.
4. `04_late_component_loadings`: red/blue indicate opposite sides of a band
   contrast; large absolute coefficients indicate contribution to that PC.
5. `05_components_vs_cv_performance`: higher OOF macro-F1 is better; bars show
   partition SD. Macro-F1 gives each crop equal weight, despite unequal counts.
6. `06_representation_controls`: PC5–8 alone, fold-selected PCs and all-input
   controls help distinguish new information from a change in model geometry.
7. `07_crop_metrics`: recall asks how many real fields of a crop were found;
   precision asks how often predictions of that crop were right.
8. `08_training_oof_confusion`: rows are true crops; columns are predictions;
   off-diagonal cells are errors. Each field appears once per repeat.

## Conditional quantum decision

Test four first PCs/four qubits, fold-selected four of eight PCs/four qubits, and
six first PCs/six qubits. Retain full connectivity from training-only topology
selection. Ring's .8943 validation result remains historical; it does not select
the architecture. Defer eight qubits and data re-uploading. Keep the angle map,
optimizer settings, folds, seeds and objective-evaluation budget fixed. Increasing
width changes parameter count, gate count and readout grouping, so it is not a
pure representation ablation; the selected-four control is cleaner.

## Reproduce

From the project root, using the existing virtual environment:

```sh
MPLCONFIGDIR=/private/tmp/qiskit-baseline-mpl .venv/bin/python experiments/feature_representation/run.py --results outputs/feature_representation_reproduction
MPLCONFIGDIR=/private/tmp/qiskit-baseline-mpl .venv/bin/python experiments/feature_representation/analyze.py outputs/feature_representation_reproduction
```

The runner refuses to overwrite an existing destination. Saved records include
all fold configurations, inner-grid scores, OOF predictions, timings, package
versions and SHA-256 hashes protecting prior files. Analysis rescored all 63 OOF
files (57,582 repeated decisions). Full-training descriptive PCA is used only
for plots, not for CV preprocessing. Results: `outputs/feature_representation_2026-11-21/`.
