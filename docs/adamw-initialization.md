# Initializing from AdamW Networks

**Documentation:** [Overview](../README.md) · [Method & API](method.md) · [Synthetic Benchmarks](benchmarks.md) · [Real-Data Comparison](real-data.md) · [AdamW Initialization](adamw-initialization.md)

**On this page:**
- [Key Takeaways](#key-takeaways)
- [Results](#results)
- [Experimental Protocol](#experimental-protocol)
- [Reproducing the Benchmarks](#reproducing-the-benchmarks)

---

## Key Takeaways

Standard gradient-based training (such as [AdamW](https://arxiv.org/abs/1711.05101)) is effective at finding informative hidden-unit directions, but decoupled weight decay does not calibrate output weights under an explicit sparsity ($`\ell_1`$ path-norm) penalty. Passing a trained AdamW network to `solve_rcg` via `initial_U` and `initial_beta` tests two ways of combining AdamW with convex optimization:

1. **Lasso output-weight refitting (`max_iter=0`) improves test RMSE over its 15-second AdamW initializer on all 12 splits (12/12).**
   Keeping the hidden-unit directions from a 15-second AdamW model fixed and solving the convex Lasso problem for the output weights $`\beta`$ has a median stage runtime of **0.19–0.46 seconds** per dataset (`0.03–4.60 s` across individual splits), lowers the path-regularized training objective in all 36 paired runs, and reduces test RMSE relative to the 15-second AdamW model on every dataset split:
   - On **White Wine**, where standalone AdamW overfits badly (`0.845` RMSE), a Lasso refit (median `0.24 s`) drops test RMSE to `0.711`.
   - On **California Housing**, `AdamW + coefficient refit` achieves the lowest test RMSE across all methods (`51,447` USD), outperforming both 15-second (`52,250` USD) and 30-second (`52,048` USD) standalone AdamW and standalone `reluopt` (`53,680` USD).
   - On **Superconductivity**, `AdamW + coefficient refit` reduces the 15-second AdamW test RMSE from `12.81` K to `12.55` K (`12.553` K), matching 30-second standalone AdamW (`12.550` K) at half the training time.

2. **Continued direction search (`warm-rcg`) lowers the penalized training objective further in all 36 paired runs.**
   Continuing from the refitted AdamW model with up to 15 seconds of `reluopt` column generation and L-BFGS direction refinement reduces the path-regularized training objective below the Lasso refit in all 36 paired cases. On test data, continued search improves over the AdamW initializer on 8 of 12 splits and improves over the Lasso refit on 5 of 12 splits—achieving the lowest test RMSE on **Concrete** (`4.78` MPa) and improving **White Wine** further to `0.702`, while slightly increasing test RMSE relative to the Lasso refit on California Housing and Superconductivity.

3. **Avoids the overfitting of longer standalone AdamW training.**
   Extending standalone AdamW from 15 seconds to 30 seconds worsens test RMSE on Concrete (`4.87` to `5.05`) and White Wine (`0.845` to `0.861`), whereas following a 15-second AdamW run with a `reluopt` output-weight refit improves test error over the 15-second AdamW model on every dataset and split.

---

## Results

Measured on 2026-10-15 with `reluopt` 0.5.0 ([commit `6d86253`](https://github.com/epicycloids/reluopt/releases/download/v0.6.0/benchmark-source-6d86253.tar.gz)) on Modal workers (4 CPU cores, 32 GiB RAM, and one NVIDIA L4 GPU per worker, CUDA `float32` with TF32 disabled and `float64` output-weight verification) across 3 grouped splits per dataset (240 total fits; 72 validation-selected models).

### Held-Out Test RMSE

Mean ± sample standard deviation across the 3 splits, in original target units (lower is better). Each row selects its own hyperparameters by validation MSE.

| Method | Concrete (MPa) | White Wine | California (USD) | Superconductivity (K) |
| --- | --- | --- | --- | --- |
| AdamW (15 s) | 4.87 ± 0.65 | 0.845 ± 0.041 | 52,250 ± 750 | 12.81 ± 0.14 |
| **AdamW + coefficient refit (`max_iter=0`)** | 4.84 ± 0.68 | 0.711 ± 0.029 | **51,447 ± 530** | **12.55 ± 0.20** |
| **AdamW + full search (`warm-rcg`, +15 s)** | **4.78 ± 0.55** | 0.702 ± 0.028 | 52,366 ± 1,366 | 12.94 ± 0.26 |
| Standalone `reluopt` (15 s) | 5.08 ± 0.71 | **0.700 ± 0.028** | 53,744 ± 448 | 13.07 ± 0.18 |
| Standalone `reluopt` (30 s) | 5.08 ± 0.71 | **0.700 ± 0.028** | 53,680 ± 512 | 13.02 ± 0.15 |
| AdamW (30 s) | 5.05 ± 0.70 | 0.861 ± 0.041 | 52,048 ± 48 | **12.55 ± 0.24** |

### Fixed-Penalty Training Objective ($`\lambda_{\mathrm{mean}} = 0.001`$)

Mean path-regularized training objective $`\frac{1}{2n}\lVert f(X)-y\rVert_2^2 + \lambda_{\mathrm{mean}}\lVert\beta\rVert_1`$ at $`\lambda_{\mathrm{mean}} = 0.001`$ across the 3 splits. The first three columns evaluate the **same** validation-selected 15-second AdamW network before refitting, after Lasso output-weight refitting (`max_iter=0`), and after warm-started `reluopt` search:

| Dataset | Initial AdamW model | + Coefficient refit (`max_iter=0`) | + Full search (`warm-rcg`) | Standalone `reluopt` (15 s) | Standalone `reluopt` (30 s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Concrete | 0.054193 | 0.036937 | **0.029698** | 0.029788 | 0.029788 |
| White Wine | 0.173623 | 0.165588 | 0.156773 | 0.162206 | **0.154562** |
| California | 0.186159 | 0.131736 | **0.122773** | 0.122826 | 0.122825 |
| Superconductivity | 0.071409 | 0.066978 | 0.064183 | 0.063859 | **0.063519** |

Across all three tested penalty levels ($`\lambda_{\mathrm{mean}} \in \{10^{-4}, 10^{-3}, 10^{-2}\}`$), Lasso coefficient refitting lowered the training objective relative to the initial AdamW network in **36/36** paired runs, and continued `reluopt` search lowered it further in **36/36** paired runs.

### Runtime

Each cell reports **total selected pipeline time / stage fit time / total tuning time** in seconds (median across the 3 splits):
- **Stage fit time** is the runtime of the `reluopt` step alone (dataset medians `0.19–0.46 s` for `max_iter=0` coefficient refitting, ranging from `0.03 s` to `4.60 s` across the 12 individual splits; `1.60–15.55 s` for warm-started search).
- **Selected pipeline time** sums the 15-second AdamW initializer fit and the `reluopt` stage fit.
- **Total tuning time** sums all hyperparameter trials needed to select the model on that split (4 AdamW trials + 3 $`\lambda_{\mathrm{mean}}`$ trials for the warm-started tracks).

| Method | Concrete | White Wine | California | Superconductivity |
| --- | ---: | ---: | ---: | ---: |
| AdamW (15 s) | 15.00 / 15.00 / 82.5 | 15.01 / 15.01 / 81.0 | 15.01 / 15.01 / 81.3 | 15.01 / 15.01 / 80.3 |
| AdamW + coefficient refit | 15.46 / **0.46** / 94.6 | 15.25 / **0.24** / 91.8 | 15.20 / **0.19** / 92.8 | 15.23 / **0.22** / 93.6 |
| AdamW + full search | 30.55 / 15.55 / 121.3 | 16.61 / 1.60 / 124.0 | 30.44 / 15.44 / 116.8 | 30.39 / 15.38 / 127.2 |
| Standalone `reluopt` (15 s) | 12.89 / 12.89 / 41.4 | 1.76 / 1.76 / 43.7 | 15.25 / 15.25 / 37.4 | 15.36 / 15.36 / 46.0 |
| Standalone `reluopt` (30 s) | 12.48 / 12.48 / 58.2 | 2.01 / 2.01 / 73.0 | 15.23 / 15.23 / 52.5 | 21.70 / 21.70 / 67.3 |
| AdamW (30 s) | 30.00 / 30.00 / 142.0 | 30.01 / 30.01 / 140.1 | 30.01 / 30.01 / 142.0 | 30.01 / 30.01 / 140.2 |

---

## Experimental Protocol

The benchmark uses the same four datasets, grouped 60/20/20 train/validation/test splits (`seeds = 0, 1, 2`), and training-only standardization described in [Real-Data Comparison: Data](real-data.md#data).

For each dataset and split:
1. **Train AdamW initializers (15 s budget):** Train 256-unit two-layer ReLU networks with AdamW across the 4-point grid $`\text{lr} \in \{10^{-3}, 10^{-2}\} \times \text{weight\_decay} \in \{10^{-4}, 10^{-2}\}`$ (with an epoch cap of $`10^6`$ so the 15-second timer governs termination), retaining the best training-MSE checkpoint from each run. Select the model with the lowest validation MSE.
2. **Run warm-started `reluopt` tracks:** Freeze the selected 15-second AdamW network and pass its weights (`initial_U`, `initial_beta`) to `solve_rcg` for each penalty $`\lambda_{\mathrm{mean}} \in \{10^{-4}, 10^{-3}, 10^{-2}\}`$:
   - **Coefficient refit (`max_iter=0`):** Refits the output weights $`\beta`$ via Lasso on the fixed AdamW directions.
   - **Full search (`warm-rcg`):** Refits $`\beta`$ and continues column generation and L-BFGS direction refinement for up to 15 seconds (`max_cols=512`).
3. **Run standalone controls:** Fit cold-start `reluopt` (`max_cols=512`) with 15-second and 30-second budgets across $`\lambda_{\mathrm{mean}} \in \{10^{-4}, 10^{-3}, 10^{-2}\}`$, and fit standalone AdamW from the same random initializations with a 30-second budget across the 4-point $`(\text{lr}, \text{weight\_decay})`$ grid.
4. **Validation selection and test scoring:** For each of the 6 methods independently, pick the hyperparameter setting with the lowest validation MSE and evaluate it once on the held-out test split.

---

## Reproducing the Benchmarks

From a repository checkout:

```bash
uv sync --extra dev --extra compute

# Run the full suite on Modal L4 workers
uv run --extra compute modal run --detach -m benchmarks.modal_hybrid

# Collect completed results from the Modal Volume and generate summary tables
uv run --extra compute modal run -m benchmarks.modal_hybrid --collect-only \
  --run-id RUN_ID --output runs/adamw-initialization.json
uv run python -m benchmarks.report_hybrid runs/adamw-initialization.json \
  --compact-output benchmarks/hybrid-results.json
```

To run locally without Modal using the pinned dependency versions in [`benchmarks/requirements-real.txt`](../benchmarks/requirements-real.txt):

```bash
uv run --no-project --python 3.12 \
  --with-requirements benchmarks/requirements-real.txt python -m benchmarks.hybrid --help
```

Compact per-trial metrics and paired comparisons are stored in [`benchmarks/hybrid-results.json`](../benchmarks/hybrid-results.json); full trajectories are archived in the [v0.6.0 release assets](https://github.com/epicycloids/reluopt/releases/tag/v0.6.0).
