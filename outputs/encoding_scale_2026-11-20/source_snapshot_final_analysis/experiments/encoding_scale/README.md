# Controlled tanh encoding-scale experiment

Question: does reducing tanh saturation improve crop classification under the
starter's original training budget?

The experiment changes only the fixed number `s` in

`angle_s(z) = (pi / 2) * (1 + tanh(s * z))`.

The original is `s=1`. The two candidates are `s=0.5` and `s=0.25`.
Smaller scales distinguish extreme inputs more strongly, while reducing the
angle differences for inputs close to zero. A mathematical improvement in
extreme-input distinguishability does not guarantee improved classification.

## Fixed protocol

- All 914 released training fields, in their original order.
- The same 150 validation fields; no test predictions or test-label access.
- StandardScaler -> full-SVD PCA(4) -> StandardScaler, fitted on training only.
- The original four-qubit RY circuit, eight trainable angles, modulo-3 readout.
- Fresh initialization for each seed 0, 1 and 2. The exact starting weights
  are identical across scales for a given seed and saved in the results.
- COBYLA with 100 objective evaluations and exact probabilities (`shots=None`).
- Validation macro-F1 is the primary measure. Also report accuracy, precision,
  recall for every crop, confusion matrices, and preprocessing/fit/predict time.
- Report every candidate and seed. Do not select the best seed.

The unchanged baseline is executed locally through `challenge.coverage.train`.
Its results are compared with the supplied organizer reference in `results/`;
organizer results are not substituted for local observations. The separate
adapter in `model.py` rescales inputs passed to the original classifier. Neither
`challenge/models.py` nor the notebook is edited.

## Reproduce

Run from the repository root, using NEW output directories:

```bash
MPLCONFIGDIR=/private/tmp/qiskit-baseline-mpl .venv/bin/python experiments/full_baseline_914/run.py --results outputs/my_full_baseline_914
MPLCONFIGDIR=/private/tmp/qiskit-baseline-mpl .venv/bin/python experiments/encoding_scale/run.py --baseline outputs/my_full_baseline_914 --results outputs/my_encoding_scale
```

The second command verifies and reuses the three local scale-1 fits, then trains
six new candidate fits. It refuses to overwrite an existing destination. The
Matplotlib cache path is optional and may be changed to another writable path.

Regenerate the experiment figures and summaries without fitting models:

```bash
MPLCONFIGDIR=/private/tmp/qiskit-baseline-mpl .venv/bin/python experiments/encoding_scale/analyze.py outputs/my_encoding_scale --baseline outputs/my_full_baseline_914
MPLCONFIGDIR=/private/tmp/qiskit-baseline-mpl .venv/bin/python experiments/encoding_scale/plot_seed_comparison.py outputs/my_encoding_scale --seed 1
.venv/bin/python experiments/encoding_scale/verify_saved.py outputs/my_encoding_scale
```

## Saved evidence

- `experiment_metadata.json`: design, provenance, environment, input hashes,
  source preservation checks, and probability/initialization verification.
- `runs.json`, `per_seed_metrics.csv`, `summary.json`, `summary.csv`: every
  observed result, means, population seed standard deviations, and resources.
- Prediction CSVs for all nine fits, paired field changes, and encoding values
  for the actual extreme fields that motivated the experiment.
- Optimization histories, candidate initial/final weights, fitted preprocessing,
  prepared input arrays, source snapshots, and an execution log.
- Figures for the encoding rule, macro-F1/runtime, crop recall/precision,
  confusion matrices, training loss, and paired spatial-block uncertainty.

Seed standard deviation measures sensitivity to initialization. The optional
paired bootstrap resamples the 18 validation spatial blocks 2,000 times, keeping
all scales and seeds paired. Its two-sided percentile intervals are conditional
on these fitted models, are not corrected for comparing two candidates, and do
not replace final-test evaluation. The validation set is used for model selection.

Confusion counts pooled across seeds represent 450 predictions on the SAME
150 fields, not 450 independent fields. Recall percentages average across seeds.
The SVM has all 13 features; it is a whole-pipeline reference, not a matched test
of whether the quantum component provides an advantage.

## Observed local results: 20 November 2026

The complete unchanged baseline was executed locally first. Its per-seed QNN,
SVM, and control macro-F1/accuracy values exactly matched the supplied organizer
reference. All scale-1 results below reuse those local fits. Six additional
fits were executed locally for the two scaling candidates.

| Scale | Seed 0 F1 | Seed 1 F1 | Seed 2 F1 | Mean F1 +/- seed SD | Mean accuracy | Mean total seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 (original) | 0.847300 | 0.858608 | 0.861174 | 0.855694 +/- 0.006027 | 85.56% | 48.20 |
| 0.5 | 0.853872 | 0.826533 | 0.853160 | 0.844522 +/- 0.012723 | 84.44% | 48.36 |
| 0.25 | 0.789739 | 0.243015 | 0.177499 | 0.403418 +/- 0.274477 | 49.33% | 49.60 |

| Scale | Alfalfa recall | Maize recall | Wheat recall | Wheat precision |
| --- | ---: | ---: | ---: | ---: |
| 1.0 | 88.67% | 88.00% | 80.00% | 88.86% |
| 0.5 | 84.67% | 80.67% | 88.00% | 81.71% |
| 0.25 | 27.33% | 24.67% | 96.00% | 43.83% |

The local SVM reference achieved macro-F1 0.879740 and accuracy 88.00% for
every seed. The tiny classical control achieved mean macro-F1 0.581371 with
seed SD 0.105862. The main scale comparison concerns the QNN alone.

### Interpretation

- Saturation was observed in the actual extreme fields. Smaller scales made
  their quantum input angles more distinguishable, as expected mathematically.
- The extreme wheat field `r2_614f27a98e1657c5` remained incorrectly classified
  in all nine fits. Its predicted labels are saved in the actual-field encoding
  table. The aggregate wheat-recall gains did not repair this motivating example.
- Scale 0.5 improved wheat recall in all three paired seeds, but lost alfalfa
  and maize recall on average. It repaired 17 incorrect predictions and
  introduced 22 new mistakes across the three runs: five fewer correct decisions.
- Scale 0.25 mostly shifted predictions toward wheat in seeds 1 and 2.
  Seed 1 found 49/50 wheat fields but found 0/50 alfalfa and 6/50 maize fields.
  Its impressive wheat recall therefore concealed poor three-crop classification.
- The same circuit and 100-evaluation budget took approximately the same time
  for every scale. There was no resource or speed benefit from reducing scale.
- The observed scale-0.5 mean F1 difference was -0.01117. Its paired spatial
  interval was [-0.05720, +0.03503], so this small dataset does not resolve the
  sign of the population difference under that resampling method. Scale 0.25's
  corresponding interval was [-0.53406, -0.34572].
- These findings apply to the tested architecture, seeds, dataset and budget.
  They do not prove that gentler encoding can never work with other training
  settings. They provide no evidence of quantum advantage over the SVM.

Recommendation for the current project: retain the original scale 1.0. Present
the scaling study as a completed controlled experiment and report its crop-level
trade-offs and initialization sensitivity. No starter setting was changed.

Results are saved separately under `outputs/full_baseline_914_2026-11-20/` and
`outputs/encoding_scale_2026-11-20/`. Candidate fitting took 294.66 seconds
including verification/analysis. Both completed metadata files confirm that
the starter source, notebook, data and organizer results remained unchanged.
