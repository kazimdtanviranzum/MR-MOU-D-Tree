# MR-MOU-D-Tree

This repository contains a fully reproducible pipeline that extends the simulation
metamodel decision tree framework applies it to measles and rubella (MR) immunization intervention
selection across the eight divisions of Bangladesh:

1. an **uncertainty-aware deep ensemble metamodel** that returns a full predictive
distribution over each outcome, separates epistemic from aleatoric variance, and is
**recalibrated with split conformal prediction**;
2. a **constrained multiobjective, risk-adjusted decision rule** that first requires a
district to reach a programmatic immunity target (robustly, under vaccine efficacy
uncertainty) and only then minimizes the Conditional Value at Risk (CVaR) of the cost
per DALY averted, with a **denominator floor** that removes the ICER explosion at
near-elimination coverage;
3. a **two-level decision boundary stability analysis**: a district-resampling bootstrap
and a **nested bootstrap** that resamples simulation replicates, retrains the
ensemble, relabels every district and refits the tree; and
4. a **baseline comparison** against Gaussian process (kriging), random forest, gradient
boosting and quantile regression forest metamodels under a strict
train / calibration / test / out-of-distribution split.

\---

## Data provenance: which inputs are real

The division-level structure of this study is built from **real, publicly reported
Bangladesh data**, not generic assumptions:

|Input|Source|
|-|-|
|Division populations (all 8 divisions)|Bangladesh Bureau of Statistics, **Population and Housing Census 2022** (PEC-adjusted)|
|Share of national under-5 population|derived from the Census 2022 counts|
|Division urbanicity|Census 2022|
|Division measles incidence per million (2026)|**WHO Disease Outbreak News DON598 (4 April 2026)** and WHO SEARO situation reports|
|National MR1 = 0.86, MR2 = 0.807 coverage anchors|**Coverage Evaluation Survey 2023** (DGHS/EPI)|
|Measles R0 = 15.9 (SD 1.6)|Guerra et al. (2017), systematic review|
|Vaccine efficacy, disability weights, unit costs|WHO position papers, GBD 2019, Gavi/campaign costing benchmarks|

Division transmission (`r0\_mult`) and case-fatality (`cfr\_mult`) multipliers are derived
from the real incidence and urbanicity data above. See `tables/table\_divisions.csv` and
`tables/table\_data\_sources.csv`.

**What is simulated rather than observed:** the 1000 individual districts are sampled
around the real national coverage anchors with division mixes drawn from the real census
population shares, because district-level coverage line lists are not public. Intervention
effect sizes are drawn from literature-based distributions rather than fixed constants.
The transmission dynamics are simulated. To analyze a specific district set, replace the
sampled coverage vectors in `build\_populations()` (`src/run\_pipeline.py`) with observed
district coverage; every other input is already real and the rest of the pipeline is
unchanged.

\---

## Repository structure

```
MR-MOU-D-Tree/
  reproduce.sh             one command that regenerates every artifact
  src/
    config.py              parameters, real division data, costs, seeds, split design
    epi\_model.py           stochastic SEIR model, stochastic effectiveness functions
    run\_pipeline.py        end-to-end pipeline (stages 1-9)
    run\_checkpointed.py    resumable stage-by-stage driver for constrained machines
    render\_equations.py    writes equations/equations.tex and preview images
    build\_manuscript.py    assembles the Word manuscript (native editable equations)
  data/                    generated scenarios, simulation outcomes, population objectives
  results/                 results\_summary.json (machine-readable headline numbers)
  requirements.txt
  LICENSE                  MIT
  README.md
```

## Installation

```bash
python -m venv venv \&\& source venv/bin/activate
pip install -r requirements.txt
```

`pandoc` must be on the PATH: the manuscript builder uses it to convert the LaTeX in
`equations/equations.tex` into **native, editable Office Math (OMML)** equation objects,
so equations in the .docx can be edited in Word rather than being flat images.

## Reproducing everything (one command)

```bash
bash reproduce.sh
```

That runs, in order, `render\_equations.py`, `run\_pipeline.py` and `build\_manuscript.py`,
regenerating `data/`, `figures/`, `tables/`, `results/` and `manuscript/`.

Reproducibility guarantees:

* `GLOBAL\_SEED = 20260707` seeds every stochastic operation (`src/config.py`).
* `SPLIT\_SEED = 424242` seeds the train/calibration/test split, independently of the
simulation, so the split does not move when simulation settings change.
* Design sizes: 400 scenario centroids, 60 replications per scenario and intervention,
25 ensemble members, 1000 districts, 300 district-bootstrap replicates, 60 nested
bootstrap replicates.
* Split: 191 train / 70 calibration / 87 test / 52 out-of-distribution
(OOD = baseline MR1 >= 0.92).

A full run takes roughly 20-40 minutes on one CPU core; the nested bootstrap dominates
the runtime. On machines with a per-process time limit, `src/run\_checkpointed.py` runs
the same pipeline stage-by-stage with an on-disk checkpoint:

```bash
cd src
python run\_checkpointed.py 1 2 4 build\_pops 5 6
python run\_checkpointed.py 7simple 7nested:10   # repeat until 60/60
python run\_checkpointed.py 7finalize 8 9 final
```

Setting `MRMOU\_QUICK=1` runs a reduced-size smoke test in seconds. It exercises every
code path but does **not** reproduce the published numbers.

## Headline results

* Deep ensemble metamodel: mean test R² = **0.974** (min 0.908); mean R² on the
out-of-distribution high-coverage set = **0.476**, which bounds the coverage range over
which the recommendations should be read.
* Conformal recalibration moves the worst-case DALY interval coverage from **0.839** to
within **0.954-0.966** of the nominal 0.90 level.
* Baseline comparison: point accuracy is closely matched across the nonlinear learners
(GBM 0.975, deep ensemble 0.974, quantile RF 0.970, RF 0.969, GP 0.964); the GP (0.904)
and the recalibrated deep ensemble (0.908) both attain near-nominal 90% coverage.
* The primary boundary (baseline MR1) is **not sharp**: the district-resampling bootstrap
gives a median of **0.851** with a 95% interval of **0.777 to 0.862**, about eight
coverage points wide.
* The **nested bootstrap** moves the boundary only within **0.851 to 0.861** (width
0.010): boundary uncertainty is driven by *which districts are observed*, not by
metamodel or simulation error. Collect more district coverage data, not more simulation.
* **4.8%** of districts change recommendation once uncertainty is propagated (8.7% near
the boundary vs 3.1% away from it). Using the **joint** rather than independent
cost-DALY distribution changes only **0.6%**.
* CHW outreach is selected in **28.8%** of districts under the default transmission-scaled
target, rising to **60.2%** under a strict 0.95 elimination target — it is chosen where
it is the *feasible* option that reaches the immunity target, not because it is cheap,
despite rarely being on the cost-versus-cases Pareto frontier.
* Decision-rule sensitivity: the recommendation is robust to the robustness probability,
CVaR risk level, willingness to pay and budget cap; **only the immunity target moves it
materially**.
* Equity: Gini **0.068**, max/min ratio **1.50** across divisions. A maximin objective
raises the least-served division from 720 to 828 cases averted per 100k.

See `results/results\_summary.json` for exact values.

## License

Released under the MIT License (see `LICENSE`).



## 

