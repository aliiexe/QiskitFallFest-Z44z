# Ring-only educational notebook

Open **`notebooks/QFF_Quantum_Crop_Research.ipynb`**. This is the deliverable:
a compact notebook explaining and running the verified four-qubit ring QNN.
It has exactly four figures: actual Qiskit circuit, loss curve, confusion matrix
and a real field's predicted probabilities. A brief text/table recap records all
earlier experiments without constructing or comparing other models.

In VS Code select **Crop challenge** or the project `.venv/bin/python` kernel,
leave `RUN_MODE='saved'`, and **Restart Kernel → Run All**. Change `FIELD_INDEX`
and `FIELD_SEED` to explore actual validation fields.

Modes: `saved` restores ring checkpoints and verifies historical predictions;
`quick` trains ring on 120 training fields and evaluates on 180 disjoint training
fields with a 20-call cap; `full` refits ring on all 914 fields with a 100-call
cap and checks historical validation predictions. Seeds default to 0,1,2.
Saved and fresh results are labeled separately; quick scores are not compared
with full scores. Full's training-CV selection versus ring's .8943 historical
validation mean remains explicit. No new architecture is introduced.

Every run writes a new unique folder under `outputs/research_notebook/` and
checks hashes protecting earlier source, notebook, data and experiment outputs.
`artifact_manifest.json` pins the ring archives. Model code is imported from the
existing `TopologyClassifier`; there is no duplicated classifier implementation.

Fresh-kernel verification commands, from the project root:

```sh
.venv/bin/python experiments/research_notebook/execute.py --mode saved
.venv/bin/python experiments/research_notebook/execute.py --mode quick
.venv/bin/python experiments/research_notebook/execute.py --mode full
```

Only saved mode writes executed outputs into the delivered notebook. Quick/full
checks save separate executed copies. Large generated session folders are not
committed. `build_notebook.py` is a maintenance source; explicit regeneration
requires `--replace-generated` and clears outputs, so preserve personal edits.
