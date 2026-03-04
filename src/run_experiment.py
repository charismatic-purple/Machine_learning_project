from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

from src.generate_dataset import SyntheticDatasetConfig, generate_intrinsic_gaussian_mixture
from src.embed_data import AmbientEmbeddingConfig, embed_to_ambient
from src.models import train_and_eval_logreg, train_and_eval_linear_svm, train_and_eval_knn


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Phase 4 experiment configuration.

    Core design:
    - Generate intrinsic data in d* with anisotropic diagonal covariance
    - Embed into ambient D=768 via orthonormal projection + noise
    - For each observed dimension d, fit PCA on training only, transform train/test
    - Standardize features (fit on train only)
    - Train models and log results
    """
    ambient_dim: int = 768
    noise_sigma: float = 0.1

    intrinsic_dims: tuple[int, ...] = (64, 256, 768)
    # Calibrated deltas from your run:
    deltas: Dict[int, float] = None  # set in __post_init__-like pattern below

    observed_dims: tuple[int, ...] = (16, 32, 64, 128, 256, 512, 768)

    num_classes: int = 3
    samples_per_class: int = 1000
    var_low: float = 0.5
    var_high: float = 3.0

    test_size: float = 0.2
    seeds: tuple[int, ...] = tuple(range(20))

    results_path: str = "../results/raw/experiment_results.csv"

    def with_defaults(self) -> "ExperimentConfig":
        """Return a copy with default deltas set if not provided."""
        if self.deltas is not None:
            return self
        # Default calibrated deltas (from your calibration output)
        return ExperimentConfig(
            ambient_dim=self.ambient_dim,
            noise_sigma=self.noise_sigma,
            intrinsic_dims=self.intrinsic_dims,
            deltas={64: 2.0, 256: 2.25, 768: 2.5},
            observed_dims=self.observed_dims,
            num_classes=self.num_classes,
            samples_per_class=self.samples_per_class,
            var_low=self.var_low,
            var_high=self.var_high,
            test_size=self.test_size,
            seeds=self.seeds,
            results_path=self.results_path
        )


def ensure_parent_dir(filepath: str) -> None:
    """Create parent directory if it does not exist."""
    parent = os.path.dirname(filepath)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def distance_concentration_metric(X: np.ndarray, max_points: int, seed: int) -> dict:
    """
    Compute distance concentration diagnostics on a subset of points.

    Returns:
        nn_mean: mean nearest-neighbor distance (excluding self)
        far_mean: mean farthest distance (excluding self)
        contrast: (far_mean - nn_mean) / nn_mean

    Notes:
    - Uses Euclidean distance.
    - Uses a subset to keep computation reasonable.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    m = min(max_points, n)
    idx = rng.choice(n, size=m, replace=False)
    Xs = X[idx]

    D = pairwise_distances(Xs, metric="euclidean")

    # Nearest neighbor: exclude self-distances by setting diagonal to +inf
    D_nn = D.copy()
    np.fill_diagonal(D_nn, np.inf)
    nn = np.min(D_nn, axis=1)

    # Farthest neighbor: exclude self-distances by setting diagonal to -inf
    D_far = D.copy()
    np.fill_diagonal(D_far, -np.inf)
    far = np.max(D_far, axis=1)

    nn_mean = float(np.mean(nn))
    far_mean = float(np.mean(far))
    contrast = float((far_mean - nn_mean) / max(nn_mean, 1e-12))

    return {"nn_mean": nn_mean, "far_mean": far_mean, "contrast": contrast}


def run_experiment(cfg: ExperimentConfig) -> pd.DataFrame:
    cfg = cfg.with_defaults()
    ensure_parent_dir(cfg.results_path)

    rows: List[Dict[str, Any]] = []

    for seed in tqdm(cfg.seeds, desc="Seeds"):
        for d_star in tqdm(cfg.intrinsic_dims, desc=f"Intrinsic dims (seed {seed})", leave=False):
            delta = cfg.deltas[d_star]

            # 1) Generate intrinsic dataset
            ds_cfg = SyntheticDatasetConfig(
                seed=seed,
                intrinsic_dim=d_star,
                num_classes=cfg.num_classes,
                samples_per_class=cfg.samples_per_class,
                var_low=cfg.var_low,
                var_high=cfg.var_high,
                delta=delta,
                shuffle=True
            )
            X_intrinsic, y, _meta = generate_intrinsic_gaussian_mixture(ds_cfg)

            # 2) Embed into ambient space (D=768)
            emb_cfg = AmbientEmbeddingConfig(
                seed=seed + 1000,  # offset to decouple dataset seed and embedding seed
                ambient_dim=cfg.ambient_dim,
                noise_sigma=cfg.noise_sigma,
                store_projection_matrix=False  # keep results light; seed reproduces A
            )
            Z, _emb_meta = embed_to_ambient(X_intrinsic, emb_cfg)

            # 3) Train/test split (stratified)
            Z_train, Z_test, y_train, y_test = train_test_split(
                Z, y,
                test_size=cfg.test_size,
                random_state=seed,
                stratify=y
            )

            # 4) For each observed dimension: PCA(train only) -> transform -> scale(train only)
            for d_obs in tqdm(cfg.observed_dims,
                              desc=f"PCA dims (d*={d_star})",
                              leave=False):
                # PCA fit only on training set to prevent leakage
                pca = PCA(n_components=d_obs, random_state=seed)
                Xtr = pca.fit_transform(Z_train)
                Xte = pca.transform(Z_test)

                # Compute an SNR proxy based on how much variance PCA captures vs what remains.
                # This uses only the training set PCA fit (no leakage).
                explained = float(np.sum(pca.explained_variance_))  # sum of retained eigenvalues

                # Total variance in the training data (trace of covariance) can be approximated by
                # summing variances across original dimensions.
                # We compute it directly from Z_train to remain robust across sklearn versions.
                total_var = float(np.sum(np.var(Z_train, axis=0, ddof=1)))
                remaining = max(total_var - explained, 1e-12)

                snr_proxy = explained / remaining

                # Standardize features (important for LR and SVM); fit only on train
                scaler = StandardScaler(with_mean=True, with_std=True)
                Xtr = scaler.fit_transform(Xtr)
                Xte = scaler.transform(Xte)

                # Curse-of-dimensionality diagnostic computed on training data in the reduced space.
                conc = distance_concentration_metric(Xtr, max_points=400, seed=seed + d_obs)

                # 5) Train + evaluate models
                lr_train, lr_test = train_and_eval_logreg(Xtr, y_train, Xte, y_test, seed)
                svm_train, svm_test = train_and_eval_linear_svm(Xtr, y_train, Xte, y_test, seed)
                knn_train, knn_test = train_and_eval_knn(Xtr, y_train, Xte, y_test, k=11)


                # 6) Log results
                rows.append({
                    "seed": seed,
                    "intrinsic_dim": d_star,
                    "delta": delta,
                    "ambient_dim": cfg.ambient_dim,
                    "noise_sigma": cfg.noise_sigma,
                    "observed_dim": d_obs,
                    "model": "logreg",
                    "train_accuracy": lr_train,
                    "test_accuracy": lr_test,
                    "generalization_gap": lr_train - lr_test,
                    "pca_explained_var": explained,
                    "pca_total_var": total_var,
                    "pca_remaining_var": remaining,
                    "snr_proxy": snr_proxy,
                    "nn_mean": conc["nn_mean"],
                    "far_mean": conc["far_mean"],
                    "distance_contrast": conc["contrast"],
                })
                rows.append({
                    "seed": seed,
                    "intrinsic_dim": d_star,
                    "delta": delta,
                    "ambient_dim": cfg.ambient_dim,
                    "noise_sigma": cfg.noise_sigma,
                    "observed_dim": d_obs,
                    "model": "linear_svm",
                    "train_accuracy": svm_train,
                    "test_accuracy": svm_test,
                    "generalization_gap": svm_train - svm_test,
                    "pca_explained_var": explained,
                    "pca_total_var": total_var,
                    "pca_remaining_var": remaining,
                    "snr_proxy": snr_proxy,
                    "nn_mean": conc["nn_mean"],
                    "far_mean": conc["far_mean"],
                    "distance_contrast": conc["contrast"],
                })
                rows.append({
                    "seed": seed,
                    "intrinsic_dim": d_star,
                    "delta": delta,
                    "ambient_dim": cfg.ambient_dim,
                    "noise_sigma": cfg.noise_sigma,
                    "observed_dim": d_obs,
                    "model": "knn_k11",
                    "train_accuracy": knn_train,
                    "test_accuracy": knn_test,
                    "generalization_gap": knn_train - knn_test,
                    "pca_explained_var": explained,
                    "pca_total_var": total_var,
                    "pca_remaining_var": remaining,
                    "snr_proxy": snr_proxy,
                    "nn_mean": conc["nn_mean"],
                    "far_mean": conc["far_mean"],
                    "distance_contrast": conc["contrast"],
                })


    df = pd.DataFrame(rows)
    df.to_csv(cfg.results_path, index=False)
    print(f"\nSaved results to: {cfg.results_path}")
    return df


if __name__ == "__main__":
    cfg = ExperimentConfig()
    run_experiment(cfg)