# DSA4212-Proj
 
A controlled comparison of **Gaussian-process Bayesian optimisation (GP-BO)** and **CMA-ES** on two benchmark functions (sphere and ellipsoid) when only some of the input variables actually matter.
 
The central question: *how do the two optimisers behave as the ambient dimension `D` and the effective dimension `d` change, and how sensitive is the answer to the evaluation budget and to the GP kernel?*
 
---
 
## Problem setup
 
All objectives are minimised over the box `[-5, 5]^D` with optimum `x* = 1` (value `0`).
 
| Objective | Definition (first `d` of `D` coordinates are relevant) |
| --- | --- |
| `sphere` | `f(x) = Σ_{i=1..d} (x_i − 1)²` |
| `ellipsoid` | `f(x) = Σ_{i=1..d} w_i (x_i − 1)²`, with `w = geomspace(1, 100, d)` |
 
 
### Experiments
 
Four experiment families, each over four `(D, d)` pairs (defined in `src/config.py`):
 
| Experiment | Objective | `(D, d)` pairs | What it tests |
| --- | --- | --- | --- |
| `sphere_full` | sphere | (2,2), (5,5), (10,10), (20,20) | Scaling when every variable matters |
| `ellipsoid_full` | ellipsoid | (2,2), (5,5), (10,10), (20,20) | Same, with ill-conditioning |
| `sphere_effective` | sphere | (20,2), (20,5), (20,10), (20,20) | Fixed `D = 20`, shrinking effective dimension |
| `ellipsoid_effective` | ellipsoid | (20,2), (20,5), (20,10), (20,20) | Same, with ill-conditioning |
 
`(20,20)` appears in both the `*_full` and `*_effective` families. Because the problem and seeds are identical, those runs are used as a **reproducibility cross-check** and de-duplicated in the main analysis.
 
### Algorithms
 
**GP-BO** (`src/bo.py`)
- 8 scrambled-Sobol initial points.
- `scikit-learn` GP with `ConstantKernel × Matern(ν = 2.5)`, `normalize_y=True`, `alpha = 1e-6`.
- Kernel: **ARD** (one length scale per input dimension, the default) or **isotropic** (single length scale).
- The GP is re-conditioned on all data at every iteration; hyperparameters are re-optimised by marginal-likelihood MLE (L-BFGS-B) only at `n = 8, 18, 28, …` and warm-started in between.
- Acquisition: Expected Improvement (`xi = 0`), maximised over 512 random candidates followed by 2 local L-BFGS-B refinements (35 iterations max).
- A deterministic non-duplicate fallback guards against degenerate EI.
**CMA-ES** (`src/cma_es.py`)
- `cma` library, started from the origin with `sigma0 = 2.0`, bounds `[-5, 5]`.
- All internal stopping criteria are disabled so that every run uses its full budget.
- A final partial generation is evaluated and recorded but **not** told to the strategy, so the budget is exact.
Both optimisers share the same `Recorder` (`src/record.py`), which enforces the budget, re-evaluates the objective with bounds checks, and logs `x`, `value`, `best_so_far` and wall-clock time for every evaluation.
 
---
 
## Repository layout
 
```
.
├── authors.md                  # Group members
├── requirements.txt            # Pinned dependencies
├── analyze.py                  # Per-phase summary + convergence plots for results/raw
├── configs/
│   └── followup_plan.json      # Budget / robustness / extension plan
├── src/
│   ├── config.py               # Settings dataclass and EXPERIMENTS table
│   ├── objectives.py           # sphere / ellipsoid with input validation
│   ├── bo.py                   # GP-BO (EI, Matern-5/2, ARD or isotropic)
│   ├── cma_es.py               # CMA-ES wrapper with exact budget accounting
│   ├── record.py               # Evaluation recorder
│   ├── runner.py               # CLI: run_one() and the pilot runner
│   ├── followups/              # Budget, robustness and extension phases
│   │   ├── run_experiments.py
│   │   ├── summarize_results.py
│   │   ├── validate_results.py
│   │   └── test_followups.py
│   ├── investigations/         # Diagnostics (no change to optimiser behaviour)
│   │   ├── common.py
│   │   ├── validate_archives.py
│   │   ├── compare_kernels.py
│   │   ├── gp_diagnostics.py
│   │   └── reproducibility.py
│   └── analysis/
│       └── consolidate_results.py   # Validates and merges every phase
        └── last_improvement.py 
├── notebooks/                  # One thin notebook per experiment family
├── tests/test_project.py       # Unit tests
├── results/                    # All outputs (see below)
└── draft_runs/                 # Early exploratory notebooks and results (not used in the final analysis)
```
 
### Results layout
 
Every run produces a `.csv` (full evaluation trace) and a `.json` (metadata and final value) named
`{experiment}_D{D}_d{d}_{bo|cma}_{ard|isotropic|standard}_b{budget}_s{seed}`.
 
| Path | Contents |
| --- | --- |
| `results/raw/` | **Pilot**: budget 100, seeds 0–4, ARD BO and CMA-ES |
| `results/budget/<experiment>/raw/` | Budgets **250 and 500**, seeds 0–4 (ARD BO and CMA-ES) |
| `results/robustness/<experiment>/raw/` | **Isotropic-kernel** BO, budget 100, seeds 0–4 |
| `results/extension/<experiment>/raw/` | Seeds **5–9**, budget 100 (brings the pilot to 10 seeds) |
| `results/figures/` | Pilot plots and `summary.csv` from `analyze.py` |
| `results/investigations/` | Validation, kernel comparison, GP length-scale diagnostics, reproducibility cross-checks |
| `results/consolidated/` | **Final merged tables and figures** (start here) |
 
Key files in `results/consolidated/`: `pilot_summary.csv`, `extension_ten_seed_summary.csv`, `budget_summary.csv`, `kernel_summary.csv`, the matching `*_paired.csv` files (seed-matched differences), `all_runs.csv`, `unique_runs.csv`, `validation_report.json`, and `figures/` (`pilot_convergence.png`, `budget_sensitivity.png`, `kernel_sensitivity.png`).
 
---
 
## Installation
 
Python 3.11+ is required (the pinned NumPy 2.3 needs it; development was done on 3.13/3.14).
 
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
 
Pinned versions: `numpy 2.3.5`, `scipy 1.17.0`, `scikit-learn 1.8.0`, `cma 4.4.0`, `pandas 2.2.3`, `matplotlib 3.10.8`, `pytest 9.0.2`, `nbformat 5.10.4`, `notebook 7.4.7`.
 
All commands below are run from the **repository root**.
 
---
 
## Reproducing the experiments
 
### 1. Pilot (budget 100, seeds 0–4)
 
```bash
python -m src.runner --experiment sphere_full          # or ellipsoid_full / sphere_effective / ellipsoid_effective / all
python analyze.py --raw results/raw --figures results/figures
```
 
`src.runner` options: `--experiment`, `--phase {pilot,budget,robustness}`, `--budget`, `--seeds`, `--kernel {ard,isotropic}`, `--outdir`, `--algorithms {bo,cma}`, `--max-cases` (smoke test).
 
The notebooks in `notebooks/` do the same thing, one experiment family per notebook (`sphere_full`, `ellipsoid_full`, `sphere_effective`, `ellipsoid_effective`).
 
### 2. Follow-up phases
 
Defined in `configs/followup_plan.json`:
 
| Phase | Budget(s) | Seeds | Algorithms / kernel |
| --- | --- | --- | --- |
| `budget` | 250, 500 | 0–4 | BO (ARD), CMA-ES |
| `robustness` | 100 | 0–4 | BO (isotropic) |
| `extension` | 100 | 5–9 | BO (ARD), CMA-ES |
 
```bash
# Preview the planned jobs without running anything
python src/followups/run_experiments.py --phase budget --experiment sphere_full --dry-run
 
# Run (existing complete outputs are skipped; use --overwrite to replace them)
python src/followups/run_experiments.py --phase budget     --experiment sphere_full
python src/followups/run_experiments.py --phase robustness --experiment sphere_full
python src/followups/run_experiments.py --phase extension  --experiment sphere_full
# repeat for ellipsoid_full, sphere_effective, ellipsoid_effective
 
# Validate and summarise a phase
python src/followups/validate_results.py  --phase budget
python src/followups/summarize_results.py --phase budget
```

 
### 3. Consolidate everything
 
```bash
python -m src.analysis.consolidate_results             # strict: fails if any expected run is missing or invalid
python -m src.analysis.consolidate_results --allow-incomplete   # partial, clearly flagged outputs
```
 
This re-validates **every** raw trace (budget, indexing, bounds, objective recomputation from `x`, best-so-far monotonicity, metadata consistency), checks coverage against the full expected design, and writes the tables and figures into `results/consolidated/`. It does not rerun any optimiser.
 
The committed dataset passes: `validation_report.json`
 
### 4. Investigations (diagnostics)
 
```bash
python -m src.investigations.validate_archives                 # audit every CSV/JSON pair
python -m src.investigations.compare_kernels                   # seed-matched ARD vs isotropic
python -m src.investigations.gp_diagnostics --objective sphere --D 20 --d 20 --seed 0
python -m src.investigations.reproducibility --objective sphere --algorithm bo [--rerun]
python src/analysis/last_improvement.py                        # reproduce Table 4
```
 
- `gp_diagnostics` re-runs one BO case while recording the fitted length scales and log-marginal-likelihood at every fit (via a temporary wrapper; the acquisition policy is unchanged). Length scales are descriptive and not proof that the relevant variables were identified.
- `reproducibility` confirms that the `(20,20)` full and effective runs are bit-identical (all 5 seeds, for both BO and CMA-ES, on both objectives).
- `last_improvement` reproduces Table 4 from the saved 500-evaluation GP-BO ARD traces (seeds 0–4). For each run, it identifies the final evaluation at which best_so_far strictly decreases, then calculates the median and maximum across five seeds. Results are saved to results/consolidated/last_improvement_table.csv. No optimisers are rerun.

### 5. Tests
 
```bash
pytest tests src/followups/test_followups.py
```
 
The tests cover objective values and invariance to irrelevant coordinates, the EI function, exact budget accounting, bit-for-bit determinism of BO under a fixed seed, and CMA-ES budget handling, plus follow-up job counts and validator behaviour.
 
---
 
## Experimental design
 
- **Seeds:** 10 per condition (0–4 pilot, 5–9 extension). Runs are **paired by seed** across algorithms (and across kernels), so BO-vs-CMA and ARD-vs-isotropic differences are computed seed by seed.
- **Metric:** final best objective value (lower is better) after the full budget; medians and quartiles are reported because the distributions are heavy-tailed.
- **Total:** 720 validated runs across all phases.
- **Statistics:** `analyze.py` uses a percentile bootstrap CI on the mean (descriptive, not a significance test).
