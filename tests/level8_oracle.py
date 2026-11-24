"""Independent exact oracle and deterministic grids for Level 8 regression tests.

TEST HELPER ONLY. It must never be imported by production code.

The Level 6 model is Linear(5,16)-ReLU-Linear(16,8)-ReLU-Linear(8,1): piecewise linear in
its input. Along the straight path from baseline b to input x, the logit is therefore
piecewise linear in the path parameter t, and the gradient is constant between ReLU
switching points. Integrated Gradients is then computed exactly (float64) by finding every
switching point and summing (piece length x piece gradient). No Captum, no quadrature.
"""

from __future__ import annotations

import hashlib
import itertools

import numpy as np
import torch

# Development-data ranges and API-valid bounds used to build the verification grids.
DEV_LO = np.array([20.0, 5.5, 45.0, 0.0, 0.0])
DEV_HI = np.array([45.0, 8.0, 57.0, 10.0, 1.0])
API_LO = np.array([0.0, 0.0, 30.0, 0.0, 0.0])
API_HI = np.array([200.0, 14.0, 220.0, 10.0, 1.0])
GRID_SEED = 42
N_DEV, N_WIDE, N_POOL, N_STRADDLE = 500, 100, 20000, 20
PRIOR_INJURY_RATE_DEV = 1 / 6


def params64(model: torch.nn.Module):
    sd = {k: v.detach().double().numpy() for k, v in model.state_dict().items()}
    return (
        sd["encoder.layer1.weight"], sd["encoder.layer1.bias"],
        sd["encoder.layer2.weight"], sd["encoder.layer2.bias"],
        sd["head.weight"][0], float(sd["head.bias"][0]),
    )


def logit64(p, z) -> float:
    w1, c1, w2, c2, w3, c3 = p
    return float(w3 @ np.maximum(w2 @ np.maximum(w1 @ z + c1, 0) + c2, 0) + c3)


def oracle_ig(p, x, b):
    """Exact IG of the piecewise-linear logit on the straight path b -> x. Returns
    (attribution[5], number_of_linear_pieces)."""
    w1, c1, w2, c2, w3, c3 = p
    d = x - b
    h1_0, h1_d = w1 @ b + c1, w1 @ d
    cuts = {0.0, 1.0}
    for j in range(16):
        if h1_d[j] != 0.0:
            t = -h1_0[j] / h1_d[j]
            if 0.0 < t < 1.0:
                cuts.add(t)
    first = sorted(cuts)
    final = set(first)
    for lo, hi in zip(first[:-1], first[1:]):
        mid = (lo + hi) / 2
        on1 = (h1_0 + mid * h1_d) > 0
        base2, slope2 = w2 @ (on1 * h1_0) + c2, w2 @ (on1 * h1_d)
        for k in range(8):
            if slope2[k] != 0.0:
                t = -base2[k] / slope2[k]
                if lo < t < hi:
                    final.add(t)
    cuts = sorted(final)
    total = np.zeros(5)
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        if hi <= lo:
            continue
        z = b + (lo + hi) / 2 * d
        h1 = w1 @ z + c1
        on1 = (h1 > 0).astype(float)
        h2 = w2 @ (on1 * h1) + c2
        on2 = (h2 > 0).astype(float)
        total += ((((w3 * on2) @ w2) * on1) @ w1) * (hi - lo)
    return d * total, len(cuts) - 1


def standardize(runtime, raw_rows) -> torch.Tensor:
    """Standardize rows through the authoritative ModelRuntime.preprocess."""
    rows = [
        torch.from_numpy(runtime.preprocess(dict(zip(runtime.feature_names, map(float, row)))))
        for row in raw_rows
    ]
    return torch.cat(rows, dim=0)


def build_grids(runtime, sample: dict) -> dict[str, np.ndarray]:
    """Deterministic grids. The draw order of the seeded generator is part of the contract:
    dev uniform, dev prior_injury, wide uniform, wide prior_injury, pool uniform, pool
    prior_injury. The 20 threshold-straddling points are the pool inputs with the smallest
    |logit| (the logit-0 decision boundary of the sigmoid)."""
    rng = np.random.default_rng(GRID_SEED)
    dev = rng.uniform(DEV_LO, DEV_HI, size=(N_DEV, 5))
    dev[:, 4] = (rng.random(N_DEV) < PRIOR_INJURY_RATE_DEV).astype(float)
    corners = np.array(list(itertools.product(*zip(API_LO, API_HI))), dtype=float)
    wide = rng.uniform(API_LO, API_HI, size=(N_WIDE, 5))
    wide[:, 4] = (rng.random(N_WIDE) < 0.5).astype(float)
    pool = rng.uniform(DEV_LO, DEV_HI, size=(N_POOL, 5))
    pool[:, 4] = (rng.random(N_POOL) < PRIOR_INJURY_RATE_DEV).astype(float)
    with torch.no_grad():
        logits = runtime.model(standardize(runtime, pool)).numpy()
    straddle = pool[np.argsort(np.abs(logits))[:N_STRADDLE]]
    return {
        "sample": np.array([[float(sample[n]) for n in runtime.feature_names]]),
        "dev_range": dev,
        "straddle_threshold": straddle,
        "api_corners": corners,
        "api_wide": wide,
    }


def grid_digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()
