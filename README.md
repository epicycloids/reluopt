# reluopt

**Documentation:** [Overview](README.md) · [Method & API](docs/method.md) · [Synthetic Benchmarks](docs/benchmarks.md) · [Real-Data Comparison](docs/real-data.md) · [AdamW Initialization](docs/adamw-initialization.md)

`reluopt` is a CPU and CUDA solver for convex two-layer ReLU regression. Following the convex neural network formulation of [Bengio et al. (2005)](https://papers.neurips.cc/paper_files/paper/2005/hash/0fc170ecbb8ff1afb2c6de48ea5343e7-Abstract.html) and [Bach (2017)](https://jmlr.org/papers/v18/14-546.html), it fits a scalar-output model

```math
f(x) = \sum_{j=1}^P \beta_j (x^\top u_j)_+, \qquad \lVert u_j \rVert_2 \le 1,
```

by minimizing the $`\ell_1`$-regularized squared-error objective

```math
\frac{1}{2} \lVert f(X) - y \rVert_2^2 + \lambda \lVert \beta \rVert_1.
```

Because the ReLU activation $`(z)_+ = \max(z, 0)`$ is positively homogeneous, constraining the hidden-unit directions $`u_j`$ to the unit $`\ell_2`$ ball and penalizing $`\lVert \beta \rVert_1`$ is equivalent to penalizing the path norm $`\sum_j |\beta_j| \lVert u_j \rVert_2`$, or to minimizing $`\frac{1}{2}\sum_j (\lVert w_j \rVert_2^2 + a_j^2)`$ over two-layer parameterizations $`f(x) = \sum_j a_j (x^\top w_j)_+`$ ([Pilanci and Ergen, 2020](https://proceedings.mlr.press/v119/pilanci20a.html)).

Rather than sampling a fixed set of activation patterns in advance, `reluopt` builds the network incrementally using alternating descent conditional gradient ([Boyd, Schiebinger, and Recht, 2017](https://doi.org/10.1137/15M1035793)): each iteration searches for new unit directions correlated with the current residual, refits all output weights $`\beta`$ via Lasso, and jointly refines the active hidden-unit directions using L-BFGS.

> **Scaling convention:** The objective sums squared errors over rows. To match a mean-squared-error penalty $`\frac{1}{2n}\lVert f(X)-y\rVert_2^2 + \lambda_{\mathrm{mean}}\lVert\beta\rVert_1`$, pass `lam = len(y) * lambda_mean`. The solver does not center or scale inputs or add an implicit intercept; to include hidden-unit biases, append a constant column to `X`.

---

## Quick Start

### Installation

Requires Python 3.10+ and NumPy. Installing with the `performance` extra adds SciPy for faster triangular solves and L-BFGS direction refinement:

```bash
uv sync --extra performance
```

### Fitting a Model

```python
import numpy as np
from reluopt import solve_rcg

rng = np.random.default_rng(0)
X = rng.standard_normal((200, 4))
y = np.maximum(X[:, 0], 0) - 0.5 * np.maximum(-X[:, 1], 0)

result = solve_rcg(X, y, lam=0.1, max_iter=30, rng_seed=0)
y_hat = result["model"].predict(X)
print(result["obj"], result["n_active"], result["stop_reason"])
```

See [`examples/fit.py`](examples/fit.py) for a runnable script and [Method & API](docs/method.md) for all solver options and return fields.

### Initializing from a Trained PyTorch / NumPy Network

You can pass the weights of an existing two-layer ReLU network $`f(X) = (X W^\top)_+ a`$ via `initial_U` and `initial_beta` as NumPy arrays (for PyTorch tensors, convert with `.detach().cpu().numpy()` first). The solver normalizes each nonzero hidden unit to unit $`\ell_2`$ norm, absorbs its magnitude into the output weight, and solves the Lasso subproblem for the output weights:

```python
W_np = W.detach().cpu().numpy()  # shape (n_units, n_features)
a_np = a.detach().cpu().numpy()  # shape (n_units,)

# Refit output weights only (Lasso calibration on fixed hidden directions):
refit = solve_rcg(X, y, lam=lam, initial_U=W_np.T, initial_beta=a_np, max_iter=0)

# Or continue optimizing directions and adding new units:
warm = solve_rcg(
    X, y, lam=lam, initial_U=W_np.T, initial_beta=a_np, max_cols=W_np.shape[0] + 256
)
```

### GPU Execution

By default, `reluopt` runs on the CPU in `float64`. When PyTorch with CUDA (or ROCm) is installed, pass `device="cuda"` to evaluate data-engine matrix products on the GPU and `oracle_kwargs={"search_backend": "device"}` to run the pricing search on the GPU:

```python
result = solve_rcg(
    X, y, lam=0.1, device="cuda", oracle_kwargs={"search_backend": "device"}
)
```

With the default `gated_method="auto"`, L-BFGS direction refinement runs on the GPU (`"lbfgs-device"`) for `float32` CUDA workloads once the refinement problem reaches `sketch_rows * n_features * min(n_active, 96) >= 1_000_000`, and on the CPU for smaller problems. Pass `gated_method="lbfgs-device"` to force GPU evaluation during L-BFGS refinement (including on smaller inputs or with `dtype_data="float64"`). See [Method & API: Execution Backend Options](docs/method.md#execution-backend-options) for multi-GPU sharding (`devices=["cuda:0", "cuda:1"]`) and CUDA graphs (`search_backend="device-graph"`).

---

## Summary of Findings

The repository includes three benchmark suites comparing `reluopt` across synthetic scaling workloads, real tabular regression datasets, and warm-started initialization from AdamW networks:

1. **Direct column generation outperforms sampled activation-cone solvers ([Real-Data Comparison](docs/real-data.md)).**
   Convex ReLU solvers such as [SCNN](https://github.com/pilancilab/scnn) and [adelie](https://github.com/JamesYang007/adelie) approximate the problem by sampling a fixed set of hyperplane activation patterns and enforcing linear cone constraints. Under a 15-second budget across four UCI/tabular regression datasets (Concrete, White Wine, California Housing, Superconductivity), SCNN reaches the time limit on all 36 runs and adelie stops early on 27 of 36 runs, leaving cone-constraint violations and substantially higher training objectives and test RMSE. By optimizing directly over continuous unit directions and exact ReLU activations, `reluopt` achieves the lowest fixed-penalty training objective of all tested methods (both convex and gradient-based) on every dataset, and substantially lower test RMSE than SCNN and adelie on all four datasets.

2. **Against first-order neural network optimizers, `reluopt` achieves lower path-regularized training loss and requires less hyperparameter tuning ([Real-Data Comparison](docs/real-data.md)).**
   At a fixed regularization strength ($`\lambda_{\mathrm{mean}} = 0.001`$), `reluopt` reaches a lower penalized training objective than SGD and Adam on all four datasets. On held-out test RMSE, `reluopt` is more reliable on smaller or noisier datasets—outperforming AdamW on White Wine (0.700 vs. 0.837 RMSE, where unregularized/decoupled AdamW overfits) and avoiding SGD's divergence on California Housing and Superconductivity—while Adam and AdamW reach slightly lower test RMSE on Concrete, California Housing, and Superconductivity. Because `reluopt` only tunes $`\lambda`$ rather than a grid of learning rates and weight-decay parameters, its total tuning time is roughly half that of Adam and AdamW.

3. **Refitting output weights on a trained AdamW network (`max_iter=0`) consistently improves test error over the AdamW initializer ([AdamW Initialization](docs/adamw-initialization.md)).**
   Using a 15-second AdamW network to initialize `reluopt` separates feature learning from output-weight calibration:
   - **Output-weight Lasso refit (`max_iter=0`):** Keeping the AdamW hidden-unit directions fixed and solving the convex Lasso problem for the output weights $`\beta`$ has a median runtime of **0.19–0.46 seconds** per dataset (`0.03–4.60 s` across individual splits), reduces the path-regularized training objective in all 36 paired runs, and **reduces held-out test RMSE relative to the 15-second AdamW initializer on all 12 dataset splits (12/12)**. Notably, it eliminates AdamW's overfitting on White Wine (reducing RMSE from 0.845 to 0.711), achieves the lowest test RMSE on California Housing (`51,447` USD), and matches 30-second AdamW on Superconductivity (`12.55` K) at half the training time.
   - **Continued direction search (`warm-rcg`):** Running 15 seconds of `reluopt` column generation and L-BFGS direction refinement from the AdamW initialization reduces the training objective further in all 36 paired runs and achieves the lowest test RMSE on Concrete (4.78 MPa).

4. **GPU acceleration speeds up medium and large problems by $`2.5\times\text{–}2.8\times`$ ([Synthetic Benchmarks](docs/benchmarks.md)).**
   On synthetic regression workloads up to $`262{,}144 \times 32`$ and $`16{,}384 \times 256`$, offloading the direction search and L-BFGS refinement to an NVIDIA L4 or A100 GPU reduces fit time from 16.0 s to 6.2 s ($`4{,}096 \times 128`$ on L4) and from 5.0 s to 1.8 s ($`262{,}144 \times 32`$ on A100), reaching final objectives within `0.012%` of the `float64` CPU solve.

---

## Documentation Guide

| Document | Contents |
| --- | --- |
| [Method & API (`docs/method.md`)](docs/method.md) | Mathematical formulation, column-generation and L-BFGS refinement algorithm, warm-start initialization, `solve_rcg` parameters, backend options, and return fields |
| [Synthetic Benchmarks (`docs/benchmarks.md`)](docs/benchmarks.md) | CPU and CUDA runtime and objective scaling across problem dimensions and target functions |
| [Real-Data Comparison (`docs/real-data.md`)](docs/real-data.md) | Held-out RMSE, fixed-penalty training objectives, runtime, and memory for `reluopt`, SCNN, adelie, SGD, Adam, and AdamW on four regression datasets |
| [AdamW Initialization (`docs/adamw-initialization.md`)](docs/adamw-initialization.md) | Paired evaluation of AdamW training, fast Lasso output-weight refitting (`max_iter=0`), warm-started `reluopt` search, and 30-second controls |

---

## Running Tests and Benchmarks

Run the test suite and local smoke benchmark:

```bash
uv sync --extra dev
uv run python -m pytest tests
uv run python -m benchmarks.run --suite smoke
```

See the individual benchmark pages for commands to reproduce the [synthetic benchmarks](docs/benchmarks.md#reproducing-the-benchmarks), [real-data solver comparison](docs/real-data.md#reproducing-the-benchmarks), and [AdamW initialization experiments](docs/adamw-initialization.md#reproducing-the-benchmarks).

## License

Code and documentation are released under the [MIT License](LICENSE). Citation metadata is provided in [`CITATION.cff`](CITATION.cff).
