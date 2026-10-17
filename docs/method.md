# Method and API Reference

**Documentation:** [Overview](../README.md) · [Method & API](method.md) · [Synthetic Benchmarks](benchmarks.md) · [Real-Data Comparison](real-data.md) · [AdamW Initialization](adamw-initialization.md)

**On this page:**
- [Problem Formulation](#problem-formulation)
- [Optimization Algorithm](#optimization-algorithm)
- [Initialize from a Fitted Network](#initialize-from-a-fitted-network)
- [API Reference (`solve_rcg`)](#api-reference-solve_rcg)
- [Execution Backend Options](#execution-backend-options)
- [Returned Dictionary and `ReLUModel`](#returned-dictionary-and-relumodel)

---

## Problem Formulation

Given an input matrix $`X \in \mathbb{R}^{n \times d}`$, target vector $`y \in \mathbb{R}^n`$, and regularization parameter $`\lambda \ge 0`$, `solve_rcg` fits a scalar-output two-layer ReLU network

```math
f(X) = \sum_{j=1}^P \beta_j (X u_j)_+, \qquad \lVert u_j \rVert_2 \le 1,
```

by minimizing

```math
\frac{1}{2} \left\lVert \sum_{j=1}^P \beta_j (X u_j)_+ - y \right\rVert_2^2 + \lambda \sum_{j=1}^P \lvert \beta_j \rvert.
```

Each hidden unit consists of a unit-norm input weight vector $`u_j \in \mathbb{R}^d`$ ($`\lVert u_j \rVert_2 = 1`$) and a signed scalar output weight $`\beta_j \in \mathbb{R}`$. Note that $`u`$ and $`-u`$ define distinct ReLU units because $`(X u)_+`$ and $`(-X u)_+`$ activate on opposite half-spaces.

This infinite-dimensional Lasso (or Radon-norm) formulation of two-layer neural networks was introduced by [Bengio et al. (2005)](https://papers.neurips.cc/paper_files/paper/2005/hash/0fc170ecbb8ff1afb2c6de48ea5343e7-Abstract.html) and analyzed by [Bach (2017)](https://jmlr.org/papers/v18/14-546.html). By positive homogeneity of the ReLU activation, minimizing this objective over unit directions $`u_j`$ and output weights $`\beta_j`$ is equivalent to minimizing the weight-decay objective $`\frac{1}{2}\lVert (X W^\top)_+ a - y \rVert_2^2 + \frac{\lambda}{2} (\lVert W \rVert_F^2 + \lVert a \rVert_2^2)`$ over unconstrained two-layer weights $`(W, a)`$, and has an equivalent activation-cone convex formulation ([Pilanci and Ergen, 2020](https://proceedings.mlr.press/v119/pilanci20a.html)).

### Input Scaling and Intercepts

`solve_rcg` operates directly on the supplied arrays `X` and `y`:
- **Summed loss:** The squared error is summed over the $`n`$ rows. To convert from a mean-squared-error penalty $`\lambda_{\mathrm{mean}}`$, set `lam = len(y) * lambda_mean`.
- **No implicit preprocessing:** Inputs are not centered or standardized, and no intercept is added automatically.
- **Hidden-unit biases:** Appending a constant column (e.g., `np.ones((n, 1))`) to `X` allows each hidden unit to learn a bias term. That coordinate is included in the unit-norm constraint $`\lVert u_j \rVert_2 \le 1`$ and is therefore regularized alongside the feature weights.

---

## Optimization Algorithm

`solve_rcg` uses **alternating descent conditional gradient** ([Boyd, Schiebinger, and Recht, 2017](https://doi.org/10.1137/15M1035793)), a column-generation method that alternates between adding new hidden units, solving for the output weights on the current working set, and locally refining the active hidden-unit directions.

Each outer iteration performs three steps:

### 1. Direction Search (Pricing Subproblem)

Let $`\nu = y - \sum_{j=1}^P \beta_j (X u_j)_+`$ be the current residual vector. By the first-order optimality conditions of the $`\ell_1`$-regularized problem, a new unit direction $`u`$ ($`\lVert u \rVert_2 \le 1`$) can reduce the objective whenever its ReLU activation has correlation greater than $`\lambda`$ with the residual:

```math
\lvert \nu^\top (X u)_+ \rvert > \lambda.
```

Finding the direction $`u`$ that maximizes $`\lvert \nu^\top (X u)_+ \rvert`$ is nonconvex. The solver searches both signs ($`\pm \nu`$) in parallel from multiple initializations (the linear correlation $`X^\top \nu`$, top eigenvectors of $`X^\top \mathrm{diag}(\nu)^2 X`$, perturbations of active units, and random vectors) using projected gradient ascent on the unit sphere. For large $`n`$, the ascent runs on an importance-sampled row subset (`sketch_m`) before evaluating candidate directions on all $`n`$ rows. Each round gathers up to `eff_add` candidates per sign (where `eff_add` starts at `add_per_round` and grows up to `max(add_per_round, min(32, n_active // 4))` when the previous direction-refinement step succeeded) and appends up to `2 * eff_add` distinct directions with correlation exceeding $`\lambda (1 + 10^{-4})`$.

### 2. Output-Weight Refit (Restricted Master Problem)

Given the current working set of directions $`U = [u_1, \dots, u_P]`$ and feature matrix $`A_{ij} = (x_i^\top u_j)_+`$, the output weights $`\beta \in \mathbb{R}^P`$ are updated by solving the finite-dimensional Lasso problem

```math
\min_{\beta \in \mathbb{R}^P} \frac{1}{2} \lVert A \beta - y \rVert_2^2 + \lambda \lVert \beta \rVert_1.
```

The solver maintains the Gram matrix $`A^\top A`$ incrementally and solves the Lasso subproblem using warm-started coordinate descent followed by active-set Newton steps, checking the Lasso KKT conditions against the stored features. Active units that fall into the same hyperplane chamber—that is, units with the same output-weight sign and identical binary activation pattern $`\mathbf{1}\{X u_j > 0\}`$ on the training data—are merged into a single unit by positive homogeneity.

### 3. Joint Direction Refinement

Periodically (controlled by `gated_every`), the solver jointly adjusts the directions of up to 96 active units with the largest $`|\beta_j|`$:
- **L-BFGS on the ReLU objective (default when SciPy is installed and at least 4 units are active):** Optimizes the unnormalized vectors $`z_j = |\beta_j| u_j`$ directly on $`\frac{1}{2}\lVert \sum_j \mathrm{sign}(\beta_j)(X z_j)_+ - y \rVert_2^2 + \lambda \sum_j \lVert z_j \rVert_2`$ using [L-BFGS (Liu and Nocedal, 1989)](https://doi.org/10.1007/BF01589116), allowing units to cross hyperplane boundaries and change activation patterns.
- **Fixed-mask group Lasso (NumPy-only fallback or fewer than 4 active units):** Fixes each active unit's binary activation mask $`m_j = \mathbf{1}\{X u_j \ge 0\}`$ and applies proximal gradient or block coordinate descent to the resulting group-Lasso subproblem $`\frac{1}{2}\lVert \sum_j m_j \odot (X z_j) - y \rVert_2^2 + \lambda \sum_j \lVert z_j \rVert_2`$ ([Mishkin, Sahiner, and Pilanci, 2022](https://proceedings.mlr.press/v162/mishkin22a.html)).

Refined directions are normalized back to the unit sphere, output weights $`\beta`$ are refitted on the full dataset, and the updated working set is accepted only if it does not increase the objective.

### Optimality Bounds and Stopping Criteria

To estimate how far the current model is from global optimality, the solver constructs a dual-feasible ray $`\theta \nu`$ from the residual $`\nu`$. Given upper bounds $`\mathrm{ub}_+`$ and $`\mathrm{ub}_-`$ on the directional correlations $`\max_{\lVert u \rVert_2 \le 1} (\pm \nu)^\top (X u)_+`$, scaling $`\nu`$ by $`0 \le \theta \le \lambda / \max(\mathrm{ub}_+, \mathrm{ub}_-)`$ yields a dual objective value $`D(\theta) = \theta \nu^\top y - \frac{1}{2}\theta^2 \lVert \nu \rVert_2^2`$ and a primal–dual objective gap $`\max(\text{obj} - D(\theta^*), 0)`$:
- `gap_heur` plugs the largest correlation found during pricing ($`\max(\rho_{\mathrm{lb}}, \lambda)`$) into this ray dual formula to form a heuristic objective gap estimate.
- `ub_plus` and `ub_minus` upper-bound the directional correlations $`\max_{\lVert u \rVert_2 \le 1} (\pm \nu)^\top (X u)_+`$ via a dual semidefinite relaxation of Boolean quadratic maximization ([Nesterov, 1998](https://doi.org/10.1080/10556789808805690)), and `gap_cert` is the resulting primal–dual objective gap. Because the semidefinite bound is evaluated in standard floating-point arithmetic, stopping on `gap_cert` is opt-in (`allow_numerical_bound_stop=True`).

The outer loop terminates when one of the following holds (`stop_reason`):
- `"heuristic_gap"`: `eps_heur > 0`, `it > 2`, and `gap_heur <= eps_heur * max(1.0, abs(obj))`.
- `"no_violator"`: $`\rho_{\mathrm{lb}} \le \lambda (1 + \text{price\_tol})`$ after escalating the pricing search to include Burer–Monteiro SDP rounding.
- `"no_new_column"`: pricing found directions with correlation $`> \lambda (1 + 10^{-4})`$, but all candidates duplicate existing working-set directions (cosine similarity $`> 0.9995`$).
- `"plateau"`: over the last `patience` rounds, the objective decrease satisfies $`0 \le \text{obj}_{t-\text{patience}} - \text{obj}_t \le \text{stall\_rtol} \cdot \max(1.0, |\text{obj}_t|)`$ and $`\rho_{\mathrm{lb}} \le 1.05 \lambda`$.
- `"max_iter"` / `"time_budget"`: the iteration limit `max_iter` or elapsed-time limit `time_budget` is reached.
- `"column_budget"` / `"feature_budget"`: the working set reaches `max_cols` or `max_feature_bytes` with no zero-weight columns available to prune.
- `"numerical_bound"`: `gap_cert <= eps_rel * max(1.0, abs(obj))` (only when `allow_numerical_bound_stop=True`).
- `"master_iteration_limit"`: the restricted-master Lasso solver did not reach its KKT tolerance on the current working set.

---

## Initialize from a Fitted Network

An existing two-layer ReLU network $`f(X) = (X W^\top)_+ a`$ can be passed to `solve_rcg` via `initial_U` (shape `(n_features, n_units)`) and `initial_beta` (shape `(n_units,)`). Both arguments are converted with `np.array(..., dtype=np.float64)`; for PyTorch tensors, call `.detach().cpu().numpy()` before passing them:

```python
W_np = W.detach().cpu().numpy()  # shape (n_units, n_features)
a_np = a.detach().cpu().numpy()  # shape (n_units,)

# 1. Output-weight Lasso refit only:
refit = solve_rcg(X, y, lam=lam, initial_U=W_np.T, initial_beta=a_np, max_iter=0)

# 2. Warm-started column generation and direction refinement:
warm = solve_rcg(
    X, y, lam=lam, initial_U=W_np.T, initial_beta=a_np, max_cols=W_np.shape[0] + 256
)
```

How warm-starting works:
1. **Normalization:** Exact-zero columns in `initial_U` are dropped. Nonzero columns are normalized to unit $`\ell_2`$ norm, and their norms are multiplied into `initial_beta` so that the initial network's predictions are unchanged. (If `initial_beta` is omitted, initial output weights start at zero.)
2. **Initial Lasso refit:** Before any direction search, the solver refits the output weights $`\beta`$ on the supplied directions in `float64`.
3. **Output-weight refit mode (`max_iter=0`):** Returns immediately after the initial Lasso refit without running pricing or direction refinement.
4. **Continued search (`max_iter > 0`):** Uses the refitted units to seed the working set, then runs the standard column-generation and L-BFGS refinement loop. Make sure `max_cols` (and `max_feature_bytes`, if set) is larger than the number of nonzero initial units so the solver has room to add new directions. If a lower-precision (`float32`) search step ever produces a worse `float64` objective than the initial Lasso refit, the solver retains the initial refitted model.

See [AdamW Initialization](adamw-initialization.md) for empirical results comparing `max_iter=0` output-weight refitting and continued search on trained AdamW networks.

---

## API Reference (`solve_rcg`)

```python
solve_rcg(
    X, y, lam,
    device="cpu", dtype_data=None, devices=None, threads=False,
    max_iter=100, add_per_round=8, max_cols=512, max_feature_bytes=None,
    time_budget=None, rng_seed=0, verbose=False,
    eps_heur=1e-3, price_tol=1e-3, stall_rtol=1e-7, patience=6,
    gated_every=1, gated_iters=100, gated_method="auto",
    sketch_m="auto", oracle_kwargs=None,
    certify_every=5, cert_steps=15, eps_rel=1e-4, allow_numerical_bound_stop=False,
    final_verify=True, final_diagnostics=True,
    initial_U=None, initial_beta=None,
)
```

### Main Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `X`, `y` | *(required)* | Input matrix of shape `(n_samples, n_features)` and target vector of shape `(n_samples,)` |
| `lam` | *(required)* | $`\ell_1`$ regularization strength $`\lambda \ge 0`$ multiplying $`\lVert \beta \rVert_1`$ against the summed squared error |
| `max_iter` | `100` | Maximum number of outer column-generation iterations (`0` refits output weights on `initial_U` only) |
| `add_per_round` | `8` | Base per-sign candidate count per round (`eff_add = max(add_per_round, min(32, n_active // 4))` when refinement succeeds; up to `2 * eff_add` violating directions across both signs can be added) |
| `max_cols` | `512` | Maximum number of columns retained in the working set |
| `max_feature_bytes` | `None` | Optional byte limit on the stored working-set feature matrix $`A = (X U)_+`$ |
| `time_budget` | `None` | Optional soft time limit in seconds, checked between outer iterations |
| `initial_U`, `initial_beta` | `None` | Optional initial directions `(n_features, n_units)` and signed output weights `(n_units,)` |
| `gated_every` | `1` | Run joint direction refinement every `gated_every` rounds (`0` disables refinement) |
| `gated_iters` | `100` | Maximum iterations for each joint direction refinement call |
| `sketch_m` | `"auto"` | Row sketch size for pricing and refinement (`"auto"` uses all rows when $`n \le 65{,}536`$, else $`32{,}768`$; `0` uses all rows) |
| `certify_every` | `5` | Compute the SDP optimality bound `gap_cert` every `certify_every` rounds (`0` disables intermediate bounds) |
| `final_verify` | `True` | Recompute features from `X` and refit output weights $`\beta`$ in `float64` on the CPU before returning |
| `final_diagnostics` | `True` | Run a final pricing pass and SDP bound evaluation for `rho_lb`, `gap_heur`, and `gap_cert` (`False` skips these and sets them to `None`) |
| `rng_seed` | `0` | Random seed for sketching and random search initializations |

---

## Execution Backend Options

By default (`device="cpu"`), `solve_rcg` uses NumPy `float64` arrays. Passing `device="cuda"` (or `devices=["cuda:0", "cuda:1"]`) uses PyTorch in `float32` by default for device feature storage while accumulating restricted-master statistics in `float64` and running a final `float64` output-weight refit on the original inputs (`final_verify=True`).

| Option | Description |
| --- | --- |
| `device="cpu"` | Default NumPy `float64` execution on CPU (`threads=True` enables multi-threaded shard evaluation) |
| `device="cuda"` / `"torch:cpu"` | PyTorch backend on GPU (CUDA or ROCm) or CPU; defaults to `dtype_data="float32"` |
| `devices=["cuda:0", "cuda:1"]` | Shard rows across multiple devices |
| `dtype_data="float64"` | Store sharded data and features in `float64` on the chosen backend |
| `gated_method="auto"` | Uses SciPy L-BFGS when available (switching to `"lbfgs-device"` on `float32` CUDA runs when `sketch_rows * n_features * min(n_active, 96) >= 1_000_000`), falling back to fixed-mask proximal/coordinate refinement when SciPy is unavailable or fewer than 4 units are active |
| `gated_method="lbfgs"` | Host `float64` L-BFGS direction refinement (requires SciPy) |
| `gated_method="lbfgs-device"` | SciPy L-BFGS with GPU objective and gradient evaluation (requires SciPy and CUDA) |
| `gated_method="coordinate"` / `"accelerated"` | Fixed-mask group-Lasso refinement via block coordinate descent or FISTA |
| `oracle_kwargs={"search_backend": "device"}` | Run the multi-start pricing ascent on the GPU |
| `oracle_kwargs={"search_backend": "device-graph"}` | Capture fixed-shape GPU pricing ascent steps as CUDA graphs |
| `oracle_kwargs={"spectral_method": "randomized"}` | Approximate top spectral directions via block power iteration instead of forming the $`d \times d`$ weighted covariance matrix (useful when $`d`$ is large) |

---

## Returned Dictionary and `ReLUModel`

`solve_rcg` returns a dictionary with the fitted model, parameters, and diagnostics:

| Key | Description |
| --- | --- |
| `model` | `ReLUModel(directions=U, coefficients=beta)` with a `.predict(X_new, chunk_rows=65536)` method that evaluates $`(X_{\mathrm{new}} U)_+ \beta`$ in `float64` |
| `U` | Unit-norm hidden direction matrix of shape `(n_features, n_cols)` |
| `beta` | Signed output weight vector of shape `(n_cols,)` |
| `obj` | Final objective value $`\frac{1}{2}\lVert f(X) - y \rVert_2^2 + \lambda \lVert \beta \rVert_1`$ |
| `nu` | Final training residual vector $`y - f(X)`$ |
| `n_active`, `n_cols` | Number of nonzero output weights (`beta != 0`) and total columns retained in `U` |
| `stop_reason`, `n_iter` | Termination reason string and number of completed outer iterations |
| `finite_master_kkt`, `finite_master_converged` | Maximum KKT residual and convergence boolean for the Lasso subproblem over `U` |
| `rho_lb` | Largest residual correlation $`\max_j \lvert \nu^\top (X u_j)_+ \rvert`$ found during the final pricing search (`None` if `final_diagnostics=False`) |
| `ub_plus`, `ub_minus` | Numerical SDP upper bounds on $`\max_{\lVert u \rVert_2 \le 1} (\pm \nu)^\top (X u)_+`$ (`None` if not computed) |
| `gap_heur`, `gap_cert`, `theta` | Heuristic and SDP-bound primal–dual objective gaps, plus the dual scaling factor $`\theta`$ (`None` if not computed) |
| `initialization` | Dictionary with initial/retained column counts, initial and refitted objectives, refit time, and `fallback_used` (`None` when `initial_U` is not provided) |
| `verified_float64`, `diagnostics_computed` | Booleans indicating whether the final `float64` refit and final pricing/bound diagnostics ran |
| `history`, `timings`, `time` | Per-iteration metric traces, cumulative stage timings (`master`, `gated`, `price`, `cert`, `other`), and total wall-clock time in seconds |
