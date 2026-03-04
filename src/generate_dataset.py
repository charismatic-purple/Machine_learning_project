from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, Any

import numpy as np


@dataclass(frozen=True)
class SyntheticDatasetConfig:
    """
    Configuration for generating an anisotropic Gaussian mixture dataset in intrinsic space.

    Key design choices:
    - Anisotropic diagonal covariance: per-dimension variances are randomly sampled.
    - Signal-variance independence: class mean directions are random and not aligned
      to any specific coordinate axes.
    """
    seed: int
    intrinsic_dim: int                 # d*
    num_classes: int = 3               # K
    samples_per_class: int = 1000      # n_c
    var_low: float = 0.5               # a
    var_high: float = 3.0              # b
    delta: float = 2.0                 # separation scale (to be calibrated)
    shuffle: bool = True               # shuffle the final dataset


def _validate_config(cfg: SyntheticDatasetConfig) -> None:
    """Validate configuration values to avoid silent errors."""
    if cfg.intrinsic_dim <= 0:
        raise ValueError("intrinsic_dim must be > 0")
    if cfg.num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    if cfg.samples_per_class <= 0:
        raise ValueError("samples_per_class must be > 0")
    if not (cfg.var_low > 0 and cfg.var_high > 0):
        raise ValueError("var_low and var_high must be > 0")
    if cfg.var_low >= cfg.var_high:
        raise ValueError("var_low must be < var_high")
    if cfg.delta <= 0:
        raise ValueError("delta must be > 0")


def generate_intrinsic_gaussian_mixture(
    cfg: SyntheticDatasetConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate an intrinsic-space dataset:
        x | y=k ~ N(mu_k, Sigma)
    where Sigma is diagonal anisotropic and mu_k are random-direction means.

    Returns:
        X: (N, d*) float64
        y: (N,) int64 in {0, ..., K-1}
        meta: dict with reproducibility info (variances, means, config)
    """
    _validate_config(cfg)
    rng = np.random.default_rng(cfg.seed)

    d = cfg.intrinsic_dim
    K = cfg.num_classes
    n_c = cfg.samples_per_class
    N = K * n_c

    # 1) Sample per-dimension variances for anisotropic diagonal covariance.
    #    sigma2[i] ~ Uniform(var_low, var_high)
    sigma2 = rng.uniform(cfg.var_low, cfg.var_high, size=d).astype(np.float64)
    sigma = np.sqrt(sigma2)

    # 2) Sample random mean directions v_k ~ N(0, I), then normalize, then scale by delta.
    #    This makes the signal direction independent from per-coordinate variance.
    means = np.zeros((K, d), dtype=np.float64)
    for k in range(K):
        v = rng.normal(loc=0.0, scale=1.0, size=d).astype(np.float64)
        norm_v = np.linalg.norm(v)
        if norm_v == 0.0:
            # Extremely unlikely, but handle defensively.
            v[0] = 1.0
            norm_v = 1.0
        v_unit = v / norm_v
        means[k] = cfg.delta * v_unit

    # 3) Generate samples for each class:
    #    x = mu_k + diag(sigma) * eps, eps ~ N(0, I)
    X = np.empty((N, d), dtype=np.float64)
    y = np.empty((N,), dtype=np.int64)

    idx = 0
    for k in range(K):
        eps = rng.normal(loc=0.0, scale=1.0, size=(n_c, d)).astype(np.float64)
        X_k = means[k] + eps * sigma  # broadcast sigma over rows
        X[idx: idx + n_c] = X_k
        y[idx: idx + n_c] = k
        idx += n_c

    # 4) Optional shuffle to remove class block ordering.
    if cfg.shuffle:
        perm = rng.permutation(N)
        X = X[perm]
        y = y[perm]

    meta: Dict[str, Any] = {
        "config": asdict(cfg),
        "sigma2": sigma2,     # store variances to reproduce / analyze anisotropy
        "means": means,       # store class means for debugging & analysis
        "N": N,
    }
    return X, y, meta


if __name__ == "__main__":
    # Minimal smoke test: generate a small dataset and print shapes.
    cfg = SyntheticDatasetConfig(
        seed=0,
        intrinsic_dim=64,
        num_classes=3,
        samples_per_class=100,
        var_low=0.5,
        var_high=3.0,
        delta=2.0,
        shuffle=True,
    )
    X, y, meta = generate_intrinsic_gaussian_mixture(cfg)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("classes:", np.unique(y))
    print("variance range:", float(meta["sigma2"].min()), float(meta["sigma2"].max()))