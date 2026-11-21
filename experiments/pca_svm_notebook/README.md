# Second research notebook: PCA/SVM feature study

Deliverable: **`notebooks/QFF_PCA_SVM_Feature_Research.ipynb`**.
It adapts the first notebook's exact 25-cell type/order layout, seven sections,
restrained style and four figure slots. The first notebook is pinned by SHA-256
and never overwritten. User clarification identified this classical study,
rather than the full-connectivity QNN's CV precision comparison.

## Exact experiment and interpretation

Source: `experiments/feature_representation/`, originally committed as `233f2ee`.
Results: `outputs/feature_representation_2026-11-21/`.
The focus is existing **nested-SVM pca4/pca6/pca8**, not a new quantum architecture.
Three repeats of three outer stratified training folds use seeds 2026–2028.
Each outer training portion runs the existing nine-combination C/gamma grid in
three inner folds, with the entire preprocessing pipeline fitted inside each.

Four to six PCs improves mean precision: alfalfa .902680→.913678,
maize .861258→.896969, wheat .908425→.914382. Mean OOF macro-F1 rises
.893912→.909496; eight reaches .910559. Wheat precision falls slightly in
repeat 1 (.913151→.911905), so all-crop mean gains are not all-repeat gains.
The full study's primary classical selection was raw13 (.913041); six PCs are
an information-retention comparison, not a claimed universal winner.

The four figures are: classical pipeline plus **unchanged actual Qiskit ring
reference**; actual inner-CV tuning scores; three training-only confusion panels;
and one real validation field's SVM decision scores. No QNN training-loss curve
or calibrated SVM probability is invented. SVC remains `probability=False`.

## Run in VS Code

Choose **Crop challenge** or the project's `.venv/bin/python`, leave `saved`,
then **Restart Kernel → Run All**. `quick` demonstrates fitting on 120 fields
and a disjoint 180-field training holdout. `full` exactly reproduces the primary
three-view repeated nested-CV experiment. All modes use training data for tuning.

The original archive saved a full-training pca4 SVM but no equivalent pca6/pca8
checkpoints. The notebook loads pca4 and freshly fits the other two using the
predefined training-only grid for the individual-field explorer. These new demo
fits are labeled; they do not generate an aggregate validation ranking. Change
`FIELD_INDEX` and `DEMO_VIEW` to explore a real validation field. SVM scores are
not probabilities, do not need to sum to one, and may lie outside [0,1].

Every run creates a new session directory under `outputs/pca_svm_notebook/`.
Earlier notebooks, models, data and output files are checked for preservation.
Generated session folders and extra executed copies are not committed.

## Verification and maintenance

```sh
.venv/bin/python experiments/pca_svm_notebook/execute.py --modes quick,full,saved
```

Each mode starts a fresh existing Python 3.12 kernel. Only saved mode writes the
delivered notebook; the other modes save separate copies. Full verification
compares all 27 outer folds' selected parameters, nine inner-grid scores and
held-out predictions to the archive, then recomputes all nine pooled OOF metrics. Default
mode independently rescores all nine primary OOF files. The original ring notebook
and its maintenance/verification files are untouched.

The builder reads the first `.ipynb` as its actual template and refuses to
overwrite the second notebook once created. Preserve personal notebook edits
before any explicit maintenance regeneration.
