from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Tuple, Optional

import numpy as np


@dataclass(frozen=True)
class AmbientEmbeddingConfig:
    """
    Configuration for embedding intrinsic vectors into a fixed ambient space.

    Core idea:
      Z = X @ A.T + noise
    where A has orthonormal columns (A^T A = I).
    """
    seed: int
    ambient_dim: int = 768             # D
    noise_sigma: float = 0.1           # sigma_epsilon
    store_projection_matrix: bool = True


def _validate_inputs(X: np.ndarray, cfg: AmbientEmbeddingConfig) -> None:
    """Validate shapes and parameters to avoid silent numeric/shape errors."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array with shape (N, d_star)")
    if cfg.ambient_dim <= 0:
        raise ValueError("ambient_dim must be > 0")
    if cfg.noise_sigma < 0:
        raise ValueError("noise_sigma must be >= 0")
    d_star = X.shape[1]
    if d_star > cfg.ambient_dim:
        raise ValueError(
            f"intrinsic_dim (d*={d_star}) cannot exceed ambient_dim (D={cfg.ambient_dim})"
        )


def _orthonormal_columns_matrix(
    rng: np.random.Generator,
    ambient_dim: int,
    intrinsic_dim: int
) -> np.ndarray:
    """
    Build a matrix A in R^{ambient_dim x intrinsic_dim} with orthonormal columns.

    We sample a random Gaussian matrix G and compute a QR factorization:
      G = Q R
    Q has orthonormal columns (when using reduced mode).
    """
    G = rng.normal(loc=0.0, scale=1.0, size=(ambient_dim, intrinsic_dim)).astype(np.float64)

    # QR decomposition: Q shape (ambient_dim, intrinsic_dim), R shape (intrinsic_dim, intrinsic_dim)
    Q, R = np.linalg.qr(G, mode="reduced")

    # Optional sign correction for determinism (QR is unique up to column sign).
    # This makes A stable across environments given the same random seed.
    diag = np.diag(R)
    signs = np.sign(diag)
    signs[signs == 0] = 1.0
    Q = Q * signs

    return Q.astype(np.float64)


def embed_to_ambient(
    X_intrinsic: np.ndarray,
    cfg: AmbientEmbeddingConfig,
    A: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Embed intrinsic vectors X (N, d*) into ambient space (N, D) using an orthonormal projection.

    Args:
        X_intrinsic: intrinsic vectors, shape (N, d*)
        cfg: AmbientEmbeddingConfig
        A: Optional precomputed projection matrix, shape (D, d*)

    Returns:
        Z_ambient: shape (N, D)
        meta: dict containing config and (optionally) the projection matrix A
    """
    _validate_inputs(X_intrinsic, cfg)
    rng = np.random.default_rng(cfg.seed)

    N, d_star = X_intrinsic.shape
    D = cfg.ambient_dim

    # If A is not provided, generate it from the seed.
    if A is None:
        A = _orthonormal_columns_matrix(rng, ambient_dim=D, intrinsic_dim=d_star)
    else:
        # Validate supplied A.
        if A.shape != (D, d_star):
            raise ValueError(f"Provided A must have shape ({D}, {d_star}), got {A.shape}")
        # Lightweight orthonormality check (tolerant).
        # We do not fail hard unless it's clearly wrong.
        gram = A.T @ A
        if not np.allclose(gram, np.eye(d_star), atol=1e-6, rtol=1e-6):
            raise ValueError("Provided A does not have orthonormal columns (A^T A != I)")

    # Project into ambient space: (N, d*) @ (d*, D) -> (N, D)
    Z = X_intrinsic @ A.T

    # Add isotropic ambient noise.
    if cfg.noise_sigma > 0:
        noise = rng.normal(loc=0.0, scale=cfg.noise_sigma, size=(N, D)).astype(np.float64)
        Z = Z + noise

    meta: Dict[str, Any] = {
        "config": asdict(cfg),
        "N": N,
        "intrinsic_dim": d_star,
        "ambient_dim": D,
        "noise_sigma": cfg.noise_sigma,
    }

    if cfg.store_projection_matrix:
        meta["A"] = A  # store for reproducibility / debugging / reporting

    return Z.astype(np.float64), meta


if __name__ == "__main__":
    # Minimal smoke test for Phase 3.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 64))  # N=5, d*=64

    cfg = AmbientEmbeddingConfig(seed=42, ambient_dim=768, noise_sigma=0.1)
    Z, meta = embed_to_ambient(X, cfg)

    print("X shape:", X.shape)
    print("Z shape:", Z.shape)
    print("A shape:", meta["A"].shape if "A" in meta else None)

    # Sanity check: A^T A should be close to I
    A = meta["A"]
    gram = A.T @ A
    print("Max |A^T A - I|:", float(np.max(np.abs(gram - np.eye(A.shape[1])))))