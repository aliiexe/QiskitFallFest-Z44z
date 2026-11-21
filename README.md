<p>
  <img src="assets/emsi-logo.webp" alt="EMSI — École Marocaine des Sciences de l’Ingénieur" height="58">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/quantum-africa-logo.png" alt="Quantum Africa" height="58">
</p>

# Moroccan Crop Classification Challenge

**Qiskit Fall Fest · EMSI · Quantum Africa**

**Start with [QFF_Final_Ring_Demo.ipynb](notebooks/QFF_Final_Ring_Demo.ipynb)** — our final four-qubit solution. It loads verified weights, draws the circuit, evaluates all three seeds and predicts real validation fields. Optional fast retraining is at the end. No hardware credentials are required.

The [organizers' notebook](notebooks/Qiskit_QuantumAfrica_Crop_Challenge.ipynb), original implementation, data and reference results remain unchanged. These are the only two notebooks in this clean release. Older research notebooks and experiments remain on the research branch. See the [student brief (PDF)](Student_Challenge_Brief.pdf) for the challenge rules.

## Install and run in VS Code

Use Python **3.12**, VS Code's Python extension and Jupyter extension:

```bash
git clone --branch release/qff-final-demo https://github.com/aliiexe/QiskitFallFest-Z44z.git
cd QiskitFallFest-Z44z
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Open this folder in VS Code. Open **`notebooks/QFF_Final_Ring_Demo.ipynb`**, click **Select Kernel → Python Environments → `.venv`**, then **Restart Kernel → Run All**. All data and models are included. The pinned environment includes Qiskit 2.4.2 and Qiskit Machine Learning 0.9.1. Default Run All performs fresh inference, without training.

To retrain, set `RETRAIN = True` in the final code cell. It fits seeds 0/1/2 on all 914 training fields, checks validation predictions and saves separate timestamped artifacts under ignored `outputs/`. Bundled models are never overwritten.

## Our final model and results

**4 qubits · 4 PCA inputs · 12 trainable RY angles · 2 ring blocks · 8 CX gates · L-BFGS-B.** Encoding and the three-class readout match the starter. The fast matrix-adjoint trainer uses **classical simulation with Qiskit-generated operators and NumPy calculations**. It is not quantum hardware acceleration or an official Qiskit reverse-gradient primitive.

Validation on the same 150 fields after training on all 914 fields, seeds 0/1/2:

| Model | Mean macro-F1 | Mean accuracy |
| --- | ---: | ---: |
| Final 12-angle ring QNN | **0.8943** | **89.33%** |
| Original 8-angle linear QNN | 0.8557 | 85.56% |
| Classical RBF SVM, all 13 features | 0.8797 | 88.00% |

At seed 0, the ring finds 46/50 wheat fields versus 42/50 for SVM, but finds one fewer maize and one fewer alfalfa field. Its wheat precision falls to 80.70% versus SVM's 84.00%. The notebook shows every crop's precision/recall, confusion matrices and all three seeds. SVM is deterministic.

**Selection differs from reporting:** earlier training-only topology CV selected full connectivity; ring is our strongest *observed validation* candidate. The gradient engine was selected separately using training-only CV, preserving all baseline out-of-fold labels (mean macro-F1 **0.8527**). All seeds are reported; seed 0 is just the fixed demonstration seed. These are development-set findings, not unseen-test scores or quantum advantage.

The recorded paired benchmark for the **same 12-angle ring** reduced median total CPU time from **5.705 s** to **0.177 s**; training alone fell from **5.605 s** to **0.082 s**, including all per-fit caches. Inference remains standard Qiskit SamplerQNN. Imports, saved-file loading and plotting are excluded. Timings depend on the machine. Original-QNN/SVM timings came from a separate session; the original-QNN replay used an equivalent cached evaluator, not the organizer's slower training loop.

## Reproduce and inspect

```bash
.venv/bin/python -m pip check
.venv/bin/python scripts/verify_final_solution.py --reproduce-cv --execute-notebook
```

The verifier checks gradients against Qiskit and numerical differences, replays saved metrics, retrains all seeds with both gradient methods, reproduces training-only CV, and executes saved/fresh notebook workflows in separate fresh kernels. Generated reports go under ignored `outputs/release_check_*`; allow about a minute. See [configuration and reproducibility details](docs/FINAL_SOLUTION.md).

| Location | Purpose |
| --- | --- |
| `notebooks/QFF_Final_Ring_Demo.ipynb` | Primary executable demonstration |
| `notebooks/Qiskit_QuantumAfrica_Crop_Challenge.ipynb` | Unchanged organizer reference |
| `qff_solution/` | Verified trainer and small loading helpers |
| `artifacts/final_ring/` | Weights, fitted PCA/SVM, predictions, histories, CV evidence, configuration, checksums |
| `requirements.txt` | Unchanged pinned dependencies; sufficient for this solution |
| `data/`, `challenge/`, `configs/`, `results/` | Unchanged organizer files |

## Dataset in context

Ouassanouan et al. (2026) describe around **10,000 surveyed agricultural parcels** across Morocco, with crop and irrigation labels and associated satellite observations. This challenge uses a prepared **three-crop cohort from Doukkala**, with fixed training, validation and test splits. Read the [dataset paper](https://www.nature.com/articles/s41597-026-06993-y) for the wider collection methodology.

<p align="center">
  <a href="https://www.nature.com/articles/s41597-026-06993-y/figures/1">
    <img src="assets/morocco-study-regions.webp" alt="Figure 1: Moroccan study regions, surveyed agricultural parcels and mean annual NDVI for 2024" width="430">
  </a>
</p>

**Image source:** Figure 1 in Ouassanouan, Y., Elfarkh, J., Grich, S., Liblab, A. & Chehbouni, A., *Crop and irrigation types ground-truth dataset for Moroccan agricultural regions*, **Scientific Data 13, 746 (2026)**. [Original figure](https://www.nature.com/articles/s41597-026-06993-y/figures/1) · [DOI](https://doi.org/10.1038/s41597-026-06993-y). © The Author(s) 2026, reproduced unchanged under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). The map describes the source dataset's coverage, not the challenge splits.

## Data source

Adapted from Ouassanouan et al., [Moroccan Crop and Irrigation Dataset v2](https://doi.org/10.6084/m9.figshare.28486022.v2), under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Filtering, daily medians, spatial splitting and training-only PCA were applied. Labels surveyed on 23–27 April are **assumed** to apply to imagery from 25 April 2024; crop persistence is not independently verified. Cloud handling and band calibration are undocumented. Results describe this local Doukkala cohort.

## License

The code and original challenge documentation are available under the [MIT License](LICENSE). Third-party materials retain their own terms:

- **Dataset and derived data:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see [data attribution](data/README.md).
- **Paper figure**, including its reproduction in the brief: [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/), credited above.
- **EMSI and Quantum Africa logos:** belong to their respective owners and are excluded from the MIT license.

Dependencies retain their respective licenses.
