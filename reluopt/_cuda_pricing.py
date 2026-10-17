"""Optional CUDA graph execution for the fixed-shape pricing ascent.

There is no compiler dependency or generated kernel. Graph construction is a
cold-call cost; an engine keeps at most two shapes, capturing a shape on its
second use so changing candidate counts initially run eagerly. Device inputs are
copied into persistent buffers before replay. The ordinary Torch path is kept
separately for workloads where capture does not amortize.
"""

from __future__ import annotations

import numpy as np


def ascent_cuda_graph(
    Xs,
    nus,
    U0,
    sgn,
    n_iters=40,
    lr=0.3,
    prune_at=8,
    keep_min=24,
    trial_mode="both",
    device=None,
    graph_cache=None,
):
    import torch

    from .adaptive_pricing import _ascent_pm_torch

    dev = torch.device(device if device is not None else Xs.device)
    if dev.type != "cuda":
        raise ValueError("device-graph search requires a CUDA device")
    if graph_cache is None:
        raise ValueError("a persistent graph_cache is required")
    if trial_mode not in {"both", "damped", "fixed", "packed", "reuse"}:
        raise ValueError(f"unknown ascent trial_mode {trial_mode!r}")
    # Search creates these signs on the host. Keep their index plan off the
    # device: nonzero() has a data-dependent output shape and cannot be captured.
    signs = sgn.detach().cpu().numpy() if torch.is_tensor(sgn) else np.asarray(sgn)
    uses_pruning = bool(prune_at and prune_at < n_iters and U0.shape[1] > 2 * keep_min)
    key = (
        str(dev), tuple(Xs.shape), tuple(U0.shape), n_iters, float(lr),
        prune_at, keep_min, trial_mode,
        signs.tobytes() if uses_pruning else None,
        str(signs.dtype) if uses_pruning else None,
        torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
    )
    with torch.no_grad(), torch.cuda.device(dev):
        entry = graph_cache.pop(key, None)
        if entry is None:
            result = _ascent_pm_torch(
                Xs, nus, U0, sgn, n_iters=n_iters, lr=lr,
                prune_at=prune_at, keep_min=keep_min,
                trial_mode=trial_mode, device=dev,
            )
            # Empty tuple marks a shape seen once and holds no device memory.
            graph_cache[key] = ()
            while len(graph_cache) > 2:
                graph_cache.popitem(last=False)
            return result
        if not entry:
            Xd = torch.as_tensor(Xs, device=dev, dtype=torch.float32).clone()
            nd = torch.as_tensor(nus, device=dev, dtype=torch.float32).clone()
            Ud = torch.as_tensor(U0, device=dev, dtype=torch.float32).clone()
            sg = torch.as_tensor(signs, device=dev, dtype=torch.float32).clone()
            sign_cols = [
                torch.as_tensor(np.flatnonzero(signs == sign), device=dev)
                for sign in (1.0, -1.0)
            ]
            kwargs = dict(
                n_iters=n_iters, lr=lr, prune_at=prune_at, keep_min=keep_min,
                trial_mode=trial_mode, device=dev,
                _return_tensors=True, _sign_cols=sign_cols,
            )
            # Initialize BLAS handles and allocator paths on the capture stream
            # before recording the fixed-shape operation sequence.
            stream = torch.cuda.Stream(device=dev)
            stream.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(stream):
                _ascent_pm_torch(Xd, nd, Ud, sg, **kwargs)
            torch.cuda.current_stream(dev).wait_stream(stream)
            torch.cuda.synchronize(dev)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                result = _ascent_pm_torch(Xd, nd, Ud, sg, **kwargs)
            # sign_cols is allocated before capture, outside the graph's
            # private memory pool. Captured indexing kernels retain its device
            # pointers, so its tensors must live as long as the graph.
            entry = (graph, Xd, nd, Ud, sg, result, sign_cols)
        else:
            graph, Xd, nd, Ud, sg, result, sign_cols = entry
            Xd.copy_(torch.as_tensor(Xs, device=dev, dtype=torch.float32))
            nd.copy_(torch.as_tensor(nus, device=dev, dtype=torch.float32))
            Ud.copy_(torch.as_tensor(U0, device=dev, dtype=torch.float32))
            sg.copy_(torch.as_tensor(signs, device=dev, dtype=torch.float32))
        graph_cache[key] = entry
        while len(graph_cache) > 2:
            graph_cache.popitem(last=False)
        entry[0].replay()
        return tuple(tensor.detach().cpu().numpy().copy() for tensor in entry[5])
