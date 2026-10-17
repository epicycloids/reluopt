# Real-Data Solver Comparison

**Documentation:** [Overview](../README.md) · [Method & API](method.md) · [Synthetic Benchmarks](benchmarks.md) · [Real-Data Comparison](real-data.md) · [AdamW Initialization](adamw-initialization.md)

**On this page:**
- [Key Takeaways](#key-takeaways)
- [Results](#results)
- [Data](#data)
- [Solvers and Objectives](#solvers-and-objectives)
- [Hyperparameter Selection and Timing Protocol](#hyperparameter-selection-and-timing-protocol)
- [Reproducing the Benchmarks](#reproducing-the-benchmarks)

---

## Key Takeaways

This benchmark compares `reluopt` (CPU and CUDA) against two sampled activation-cone convex ReLU solvers ([SCNN](https://github.com/pilancilab/scnn) and [adelie](https://github.com/JamesYang007/adelie)) and three first-order neural network optimizers (SGD with momentum, Adam, and AdamW) across four public regression datasets under a 15-second per-fit time budget:

1. **Outperforms sampled activation-cone solvers on every dataset.**
   `reluopt` achieves substantially lower training objectives (at fixed $`\lambda_{\mathrm{mean}} = 0.001`$) and lower held-out test RMSE than both SCNN and adelie on all four datasets. SCNN and adelie optimize over a fixed random sample of 128 hyperplane activation patterns and must enforce linear inequality constraints to stay inside those activation cones; within 15 seconds, SCNN hits the time limit on all 36 runs and adelie hits its inner solver limit or time limit on 27 of 36 runs, leaving constraint violations and higher prediction error. By searching directly over continuous unit vectors and exact ReLU activations, `reluopt` avoids cone-feasibility approximations.

2. **Reaches lower path-regularized training objectives than SGD and Adam.**
   At a fixed penalty ($`\lambda_{\mathrm{mean}} = 0.001`$), `reluopt` achieves the lowest penalized training objective of all tested methods on all four datasets, confirming that column generation with L-BFGS direction refinement is an effective optimizer for the convex ReLU objective.

3. **Competitive test error with lower tuning cost and no learning-rate instability.**
   On held-out test RMSE, `reluopt` matches or outperforms all gradient methods on White Wine (where AdamW overfits at 0.837 RMSE vs. 0.700 for `reluopt`) and avoids SGD's divergence on California Housing and Superconductivity (where SGD encountered nonfinite losses in 12 of 18 California Housing trials and 9 of 18 Superconductivity trials). Adam and AdamW achieve slightly lower test RMSE on Concrete, California Housing, and Superconductivity; as shown in [AdamW Initialization](adamw-initialization.md), combining AdamW's learned directions with a `reluopt` output-weight refit improves test RMSE relative to the AdamW initializer across all four datasets. Because `reluopt` only tunes $`\lambda`$ (3 values per split) rather than a grid of learning rates and weight-decay values, its total tuning time is roughly half that of Adam and AdamW.

---

## Results

Measured on 2026-10-15 with `reluopt` 0.4.0 on Modal workers (4 CPU cores, 32 GiB RAM, and one NVIDIA L4 GPU per worker) across 3 grouped train/validation/test splits per dataset (336 total fits; 84 validation-selected models). Each fit has a 15-second soft time budget.

In the tables below, **SCNN CUDA (corrected AL)** denotes `pyscnn`'s augmented Lagrangian solver with a bug fix to its multiplier update described in [Solvers and Objectives](#solvers-and-objectives).

### Held-Out Test RMSE

Mean ± sample standard deviation across the 3 splits, in original target units (lower is better). Per-split RMSE and $`R^2`$ values are recorded in [`benchmarks/real-results.json`](../benchmarks/real-results.json).

| Method | Concrete (MPa) | White Wine | California (USD) | Superconductivity (K) |
| --- | --- | --- | --- | --- |
| `reluopt` CPU | 5.08 ± 0.78 | **0.700 ± 0.026** | 54,779 ± 962 | 13.52 ± 0.29 |
| `reluopt` CUDA | 5.09 ± 0.68 | **0.700 ± 0.028** | 54,007 ± 982 | 13.10 ± 0.20 |
| SCNN CUDA (corrected AL) | 6.82 ± 1.03 | 0.710 ± 0.034 | 61,172 ± 1,006 | 14.38 ± 0.27 |
| adelie CPU | 7.87 ± 2.20 | 0.717 ± 0.034 | 70,293 ± 1,669 | 24.87 ± 2.02 |
| SGD CUDA | 5.08 ± 0.85 | 0.701 ± 0.029 | 86,331 ± 47,512 | 12.95 ± 0.17 |
| Adam CUDA | 5.00 ± 0.67 | 0.702 ± 0.028 | 53,339 ± 349 | 12.89 ± 0.31 |
| AdamW CUDA | **4.99 ± 0.60** | 0.837 ± 0.041 | **52,489 ± 920** | **12.76 ± 0.19** |

### Fixed-Penalty Training Objective ($`\lambda_{\mathrm{mean}} = 0.001`$)

Mean training objective $`\frac{1}{2n}\lVert f(X)-y\rVert_2^2 + \lambda_{\mathrm{mean}}\lVert\beta\rVert_1`$ at $`\lambda_{\mathrm{mean}} = 0.001`$ across the 3 splits (lower is better). For SGD and Adam, the learning rate with the lowest training objective at $`\lambda_{\mathrm{mean}} = 0.001`$ is reported. AdamW is omitted because decoupled weight decay does not optimize this objective. All methods are scored by evaluating their resulting ReLU network $`f(X) = \sum_j \beta_j (X u_j)_+`$ on the standardized training data.

| Method | Concrete | White Wine | California | Superconductivity |
| --- | ---: | ---: | ---: | ---: |
| `reluopt` CPU | 0.02975 | 0.17008 | 0.12461 | 0.07025 |
| `reluopt` CUDA | **0.02973** | **0.16378** | **0.12285** | **0.06399** |
| SCNN CUDA (corrected AL) | 0.06798 | 0.30696 | 0.15784 | 0.09443 |
| adelie CPU | 0.10475 | 0.31475 | 0.44380 | 0.26848 |
| SGD CUDA | 0.03167 | 0.17192 | 0.23597 | 0.06700 |
| Adam CUDA | 0.03149 | 0.17259 | 0.12548 | 0.06615 |

### Runtime

Each cell reports **selected fit time / total tuning time** in seconds (median across the 3 splits). Total tuning time sums all hyperparameter trials for that split, including process startup and validation scoring.

| Method | Trials / split | Concrete | White Wine | California | Superconductivity |
| --- | ---: | ---: | ---: | ---: | ---: |
| `reluopt` CPU | 3 | 9.65 / 36.0 | 3.95 / 45.4 | 15.73 / 50.3 | 15.32 / 56.2 |
| `reluopt` CUDA | 3 | 12.25 / 40.7 | 1.83 / 45.6 | 15.08 / 40.5 | 15.08 / 46.6 |
| SCNN CUDA (corrected AL) | 3 | 15.03 / 59.3 | 15.04 / 59.9 | 15.11 / 59.1 | 15.13 / 59.8 |
| adelie CPU | 3 | 7.27 / 17.6 | 15.04 / 30.3 | 14.69 / 23.5 | 21.10 / 75.6 |
| SGD CUDA | 6 | 5.24 / 64.3 | 15.01 / 124.1 | 15.01 / 77.0 | 15.02 / 79.2 |
| Adam CUDA | 6 | 5.40 / 64.3 | 15.01 / 123.0 | 15.01 / 122.2 | 15.01 / 123.5 |
| AdamW CUDA | 4 | 4.42 / 38.9 | 14.59 / 81.2 | 15.00 / 81.3 | 15.01 / 81.9 |

### Peak Memory and Solver Termination

Peak memory across the 12 validation-selected fits per method (in MiB), along with termination counts across all attempted hyperparameter trials:

| Method | Peak process RSS (MiB) | Peak Torch GPU (MiB) | Total trials | Time-limit stops | Numerical / inner-limit stops | Other stops |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `reluopt` CPU | 4,789–4,931 | — | 36 | 24 | 0 | 12 |
| `reluopt` CUDA | 5,529–5,609 | 10.0–103.8 | 36 | 16 | 0 | 20 |
| SCNN CUDA (corrected AL) | 5,555–5,650 | 11.8–87.5 | 36 | 36 | 0 | 0 |
| adelie CPU | 4,771–4,852 | — | 36 | 13 | 14 | 9 |
| SGD CUDA | 5,662–5,694 | 17.5–28.7 | 72 | 33 | 21 | 18 |
| Adam CUDA | 5,673–5,703 | 17.5–28.8 | 72 | 54 | 0 | 18 |
| AdamW CUDA | 5,673–5,698 | 17.5–28.8 | 48 | 26 | 0 | 22 |

Notes on termination and cone feasibility:
- **Process RSS** includes the Python interpreter and imported libraries (including PyTorch on all workers).
- **Other stops** comprise `heuristic_gap` (10 CPU, 9 CUDA), `no_violator` (1 CPU, 6 CUDA), `no_new_column` (1 CPU, 2 CUDA), `column_budget` (2 CUDA), and `max_iter` (1 CUDA) for `reluopt`; native convergence (`converged_native_criterion`) for `adelie` (9 trials); and the 1,000-epoch cap (`max_epochs`) for SGD (18), Adam (18), and AdamW (22).
- **adelie** reached its internal quadratic-program (`pinball`) iteration limit in 14 of 36 trials.
- **SGD** encountered nonfinite loss values in 21 of 72 trials—12 of 18 trials on California Housing (9 at learning rate `0.1` and 3 at `0.01`) and 9 of 18 trials on Superconductivity (all at learning rate `0.1`)—retaining the best finite checkpoint recorded prior to divergence.
- **Cone-constraint violations in SCNN and adelie:** Because SCNN and adelie replace the ReLU nonlinearity with linear variables constrained to match sampled activation masks $`D_j = \mathrm{diag}(s_j)`$, an iterate that stops before reaching full feasibility has cone-constraint violations, and its linear cone-program output differs from the actual ReLU network prediction $`\sum_j \beta_j (X u_j)_+`$. Across all 36 trials, the maximum constraint violation and maximum prediction discrepancy on standardized training data were `0.0433` and `0.1129` for SCNN, and `0.0158` and `0.0158` for adelie.

---

## Data

| Dataset | Observations ($`n`$) | Features ($`d`$) | Target unit | Source |
| --- | ---: | ---: | --- | --- |
| Concrete Compressive Strength | 1,030 | 8 | MPa | [Yeh (1998), UCI](https://doi.org/10.24432/C5PK67) |
| White Wine Quality | 4,898 | 11 | Score (0–10) | [Cortez et al. (2009), UCI](https://doi.org/10.24432/C56S3T) |
| California Housing | 20,640 | 8 | USD | [Pace and Barry (1997)](https://doi.org/10.1016/S0167-7152(96)00140-X); [Figshare archive](https://doi.org/10.6084/m9.figshare.3829992.v2) |
| Superconductivity | 21,263 | 81 | K | [Hamidieh (2018), UCI](https://doi.org/10.24432/C53P47) |

All four datasets are distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and are downloaded and checksum-verified automatically by `benchmarks/datasets.py`.

**Splitting and preprocessing (`seeds = 0, 1, 2`):**
1. Observations with identical raw feature vectors are grouped together so duplicate feature rows never cross split boundaries.
2. Groups are randomly partitioned into approximately 60% training, 20% validation, and 20% test observations.
3. Missing values (if any) are mean-imputed, features are standardized to zero mean and unit variance, and the target is standardized to zero mean and unit variance using **training-set statistics only**.
4. A constant column of `1.0` is appended to $`X`$ so every solver can learn hidden-unit biases subject to the same norm penalty. Held-out predictions are rescaled back to original target units to compute test RMSE.

---

## Solvers and Objectives

On the standardized training data $`(X, y)`$, the shared path-regularized ReLU objective is

```math
\frac{1}{2n} \left\lVert \sum_{j=1}^P \beta_j (X u_j)_+ - y \right\rVert_2^2 + \lambda_{\mathrm{mean}} \sum_{j=1}^P \lvert \beta_j \rvert, \qquad \lVert u_j \rVert_2 = 1.
```

- **`reluopt` (`relu-cpu`, `relu-cuda`):** Optimizes this objective directly with `lam = n * lambda_mean`, `max_cols = 256`, `max_iter = 80`, `final_verify = True`, and `final_diagnostics = False`.
- **SCNN (`scnn-cuda`) and adelie (`adelie`):** Optimize the sampled activation-cone convex program of [Pilanci and Ergen (2020)](https://proceedings.mlr.press/v119/pilanci20a.html) and [Mishkin, Sahiner, and Pilanci (2022)](https://proceedings.mlr.press/v162/mishkin22a.html). Both receive the same 128 randomly sampled hyperplane activation patterns (allowing up to 256 signed ReLU units). `adelie` solves the cone-constrained group Lasso via block coordinate descent ([Yang and Hastie, 2024](https://arxiv.org/abs/2405.08631)).
- **Multiplier-update correction in SCNN:** `pyscnn` 0.0.10 solves the cone-constrained problem via an augmented Lagrangian method with FISTA inner steps. For an inequality constraint $`g_j(w) \le 0`$ with nonnegative multiplier $`\mu_j`$ and penalty parameter $`\rho > 0`$, the standard Rockafellar augmented-Lagrangian multiplier update is $`\mu_j^{\mathrm{new}} = \max\{0,\, \mu_j + \rho\, g_j(w)\}.`$

  In `pyscnn` 0.0.10, `g_j(w)` is pre-clipped to $`\max\{g_j(w), 0\}`$ before updating $`\mu_j`$, which prevents multipliers from ever decreasing once positive, even when $`g_j(w) < 0`$ is strictly satisfied. Our benchmark adapter ([`benchmarks/convex_solvers.py`](../benchmarks/convex_solvers.py)) passes the signed value $`g_j(w)`$ into the update so multipliers can decay back toward zero while retaining `pyscnn`'s FISTA inner solver, augmented objective, and stopping criteria.
- **SGD, Adam, and AdamW:** Train 256-unit two-layer ReLU networks $`f(X) = (X W^\top)_+ a`$ with minibatch size 256 for up to 1,000 epochs. SGD (momentum 0.9) and [Adam](https://arxiv.org/abs/1412.6980) minimize $`\frac{1}{2n}\lVert f(X)-y\rVert_2^2 + \frac{\lambda_{\mathrm{mean}}}{2}(\lVert W\rVert_F^2 + \lVert a\rVert_2^2)`$ from balanced initializations ($`\lVert w_j\rVert_2 = |a_j|`$), whose global minimum coincides with the path-regularized objective above ([Pilanci and Ergen, 2020](https://proceedings.mlr.press/v119/pilanci20a.html)). [AdamW](https://arxiv.org/abs/1711.05101) trains with decoupled weight decay on the unpenalized mean squared error.

---

## Hyperparameter Selection and Timing Protocol

For each dataset, split seed (`0, 1, 2`), and method:
- **Hyperparameter grids:**
  - `reluopt`, SCNN, adelie: $`\lambda_{\mathrm{mean}} \in \{10^{-4}, 10^{-3}, 10^{-2}\}`$ (3 trials).
  - SGD: $`\lambda_{\mathrm{mean}} \in \{10^{-4}, 10^{-3}, 10^{-2}\} \times \text{lr} \in \{10^{-2}, 10^{-1}\}`$ (6 trials).
  - Adam: $`\lambda_{\mathrm{mean}} \in \{10^{-4}, 10^{-3}, 10^{-2}\} \times \text{lr} \in \{10^{-3}, 10^{-2}\}`$ (6 trials).
  - AdamW: $`\text{lr} \in \{10^{-3}, 10^{-2}\} \times \text{weight\_decay} \in \{10^{-4}, 10^{-2}\}`$ (4 trials).
- **Model selection:** Every trial runs in an isolated subprocess with access only to the training split. The configuration achieving the lowest validation MSE is selected and evaluated once on the test split.
- **Timing:** Library imports and CUDA context initialization happen before the fit timer starts. Fit time covers data placement, optimization, and parameter extraction.

---

## Reproducing the Benchmarks

From a repository checkout:

```bash
uv sync --extra dev --extra compute

# Quick smoke run on Modal (single dataset and seed, 2 s budget)
uv run --extra compute modal run -m benchmarks.modal_real --dataset concrete \
  --seeds 0 --width 32 --budget 2 --smoke

# Full benchmark suite on Modal
uv run --extra compute modal run --detach -m benchmarks.modal_real

# Collect results from the Modal Volume and print summary tables
uv run --extra compute modal run -m benchmarks.modal_real --collect-only \
  --run-id RUN_ID --output runs/real-comparison.json
uv run python -m benchmarks.report_real runs/real-comparison.json \
  --compact-output benchmarks/real-results.json
```

To run locally without Modal, use the pinned environment in [`benchmarks/requirements-real.txt`](../benchmarks/requirements-real.txt) (which uses Python 3.12 and NumPy `<2` as required by `adelie`):

```bash
uv run --no-project --python 3.12 \
  --with-requirements benchmarks/requirements-real.txt python -m benchmarks.real --help
```

Full trajectories are archived in the [v0.6.0 release assets](https://github.com/epicycloids/reluopt/releases/tag/v0.6.0), and compact per-trial metrics are committed in [`benchmarks/real-results.json`](../benchmarks/real-results.json).
