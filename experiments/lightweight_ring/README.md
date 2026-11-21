# Lightweight eight-angle ring benchmark and simple demonstration

The completed benchmark found a large cached-training speed improvement over the
twelve-angle ring, but **did not meet the .88–.89 validation macro-F1 target**.
Retain the twelve-angle ring as the strongest observed validation candidate and
the eight-angle ring as a lightweight comparison. Full connectivity remains the
winner of the earlier training-only topology study; these are separate claims.

## Controlled experiment

New circuit: four PCA inputs, four qubits, original RY angle map, two trained RY
banks (eight angles), one ordered ring CX block `(0,1),(1,2),(2,3),(3,0)`, original
modulo-three readout. It has four CX gates and logical depth seven.
It imports the existing verified native Qiskit trainer and exact CE gradient;
no hand-written gate simulator or optimizer was introduced. Probabilities,
library circuit operator and all parameter gradients were verified (maximum
finite-difference error 5.83e-10).

Same three training folds (split seed 2026), all initialization seeds 0,1,2,
fold-fitted preprocessing. COBYLA caps 100/340/850; L-BFGS-B caps 20/50/100.
An eight-angle loss/gradient requires 17 probability passes, versus one for
COBYLA. C340/L20 and C850/L50 have matching maximum pass budgets. Actual work
and early stopping are recorded, not inferred from the cap alone.

| Candidate | Mean training OOF macro-F1 | Seed SD | Median total time for 3 folds |
|---|---:|---:|---:|
| COBYLA 100 | .843000 | .006173 | 1.459 s |
| COBYLA 340 | .845782 | .001657 | 2.581 s |
| COBYLA 850 | .845782 | .001657 | 2.609 s |
| L-BFGS-B 20 | .846096 | .002100 | 3.032 s |
| L-BFGS-B 50 | .845782 | .001657 | 3.109 s |
| L-BFGS-B 100 | .845782 | .001657 | 3.185 s |
| Twelve-angle ring, L-BFGS-B 100 | .852669 | .013425 | 12.800 s |

Before validation, select the fastest eight-angle candidate within .01 macro-F1
of the best quantum CV mean, with worst-seed F1 no more than .01 below the
twelve-angle reference's worst seed. This locked **C100**. It was not the highest
CV score. C340/C850 converged before 340 and had identical CV predictions;
larger L-BFGS-B caps also stopped early without a benefit.

The candidate grid and threshold were fixed in `protocol.json` before fitting.
This is adaptive research on reused, stratified/nonspatial training folds, not an
independent generalization estimate. The one-point CV tolerance did not translate
to a one-point validation gap. Only the selected eight-angle candidate was then
evaluated on validation; no budget/seed was selected or extended from that result.

## Fresh same-machine validation comparison

All quantum models use the SAME cached Qiskit training engine and SAME standard
SamplerQNN inference. Times include preprocessing + fit + inference, with input
state preparation included in fit. SVM uses its original 13-feature frozen
configuration; preprocessing, SVC fitting and inference are timed separately.

| Model | Seed 0 F1 | Seed 1 F1 | Seed 2 F1 | Mean F1 ± seed SD | Mean accuracy | Median total time |
|---|---:|---:|---:|---:|---:|---:|
| Eight-angle ring C100 | .868555 | .868555 | .845155 | .860755 ± .011031 | 86.00% | .572 s |
| Twelve-angle ring | .894346 | .894346 | .894346 | .894346 ± .000000 | 89.33% | 5.734 s |
| Original eight-angle LINEAR, cached | .847300 | .858608 | .861174 | .855694 ± .006027 | 85.56% | .574 s |
| Original SVM | .879740 | .879740 | .879740 | .879740 ± .000000 | 88.00% | .006 s |

The new lightweight ring is approximately **10.02× faster** than twelve-angle
ring end-to-end under this cached method, but loses **.033591 macro-F1**. It is
about as fast as the original eight-angle linear model under that same cached
method; speed is not an improvement uniquely caused by ring connectivity.
Three SVM rows measure repeated deterministic fits, not random-seed uncertainty.

| Model | Alfalfa precision / recall | Maize precision / recall | Wheat precision / recall |
|---|---|---|---|
| Eight-angle ring | 88.24% / 90.00% | 93.94% / 83.33% | 78.88% / 84.67% |
| Twelve-angle ring | 92.00% / 92.00% | 97.67% / 84.00% | 80.70% / 92.00% |
| SVM | 88.68% / 94.00% | 91.49% / 86.00% | 84.00% / 84.00% |

Eight-angle seed 2 is weaker (.8452), rather than matching seeds 0/1 (.8686).
Every selected eight-angle seed scores below the matching twelve-angle seed.
`changed_validation_fields.csv` records the exact repairs/new errors. No claim
is made about validation scores of unselected budgets; those were not evaluated.

### Runtime interpretation

The historical ~6.25 s twelve-angle timing is not substituted for today's
measurements. Current median twelve-angle preprocessing/fit/inference are
.00125 / 5.63798 / .09493 s; eight-angle .00107 / .48521 / .08507 s.
Medians of individual stages need not sum to the median total.

The separate fixed-weight 914-field microbenchmark compares one probability
pass, not optimizer training. Median standard SamplerQNN vs cached-pass seconds:
eight-ring .4785/.0043; twelve-ring .5460/.0043; original-linear .4707/.0043.
Cached-pass measurements exclude the one-time encoded-state preparation, which
is recorded separately and included in real fitting. Predictions are not cached.
This does NOT measure standard optimizer training runtime or hardware speed.
No quantum advantage or geographic-transfer claim is supported.

## Reproduce and inspect

From the project root, use a fresh destination:

```sh
.venv/bin/python experiments/lightweight_ring/verify.py outputs/my_lightweight_ring
.venv/bin/python experiments/lightweight_ring/run.py --results outputs/my_lightweight_ring
.venv/bin/python experiments/lightweight_ring/analyze.py outputs/my_lightweight_ring
```

The benchmark took 113.8 s locally. It saves the preregistered protocol, folds,
all 63 quantum CV fits, three deterministic SVM CV fits, initial/final weights,
histories, work counters, all 12 full-training validation runs, preprocessing,
SVM checkpoints, runtime samples, source versions and preservation hashes.
Analysis rescored 21 OOF files and 12 validation files. Historical twelve-angle
CV/full fits and original-QNN metrics reproduce. Source/data/old outputs are unchanged.

The main teaching deliverable is `notebooks/QFF_Ring_Demo.ipynb`, derived from
the organizers' actual 18-cell notebook, not the longer research notebooks.
It loads verified twelve- and eight-angle ring models plus SVM by default,
shows their real results and one real field, and puts retraining behind `RETRAIN=False`.
Detailed research logs remain outside the demonstration.
