from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.generate_dataset import SyntheticDatasetConfig, generate_intrinsic_gaussian_mixture


@dataclass(frozen=True)
class CalibrationConfig:
    """
    Configuration for calibrating delta (class separation scale).

    We search for a delta that yields non-trivial intrinsic-space classification accuracy,
    targeting a band such as [0.70, 0.85].
    """
    intrinsic_dims: Tuple[int, ...] = (64, 256, 768)
    num_classes: int = 3
    samples_per_class: int = 1000
    var_low: float = 0.5
    var_high: float = 3.0
    test_size: float = 0.2
    seeds: Tuple[int, ...] = tuple(range(10))  # number of repeats for stability
    delta_grid: Tuple[float, ...] = (
        0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0,
        2.25, 2.5, 2.75, 3.0, 3.5, 4.0
    )
    target_low: float = 0.70
    target_high: float = 0.85
    target_mid: float = 0.775  # preferred center within band


def _fit_eval_logreg(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    test_size: float,
) -> float:
    """
    Train and evaluate a multinomial logistic regression on intrinsic-space features.

    Returns test accuracy.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )

    # Note: We keep the model simple. If convergence warnings appear for 768 dims,
    # increase max_iter.
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        random_state=seed
    )
    clf.fit(X_train, y_train)
    acc = float(clf.score(X_test, y_test))
    return acc


def calibrate_delta_for_dim(
    d_star: int,
    cfg: CalibrationConfig,
) -> Dict[str, object]:
    """
    For a fixed intrinsic dimension d*, search delta_grid and select a delta
    whose mean accuracy across seeds is within [target_low, target_high].
    If multiple deltas satisfy, choose the one closest to target_mid.
    If none satisfy, choose the delta minimizing distance to the band.
    """
    results_per_delta: List[Tuple[float, float, float]] = []
    # Each entry: (delta, mean_acc, std_acc)

    for delta in cfg.delta_grid:
        accs = []
        for seed in cfg.seeds:
            ds_cfg = SyntheticDatasetConfig(
                seed=seed,
                intrinsic_dim=d_star,
                num_classes=cfg.num_classes,
                samples_per_class=cfg.samples_per_class,
                var_low=cfg.var_low,
                var_high=cfg.var_high,
                delta=delta,
                shuffle=True,
            )
            X, y, _meta = generate_intrinsic_gaussian_mixture(ds_cfg)
            acc = _fit_eval_logreg(X, y, seed=seed, test_size=cfg.test_size)
            accs.append(acc)

        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
        results_per_delta.append((delta, mean_acc, std_acc))

    # 1) Filter deltas that land inside the target band.
    inside = [r for r in results_per_delta if cfg.target_low <= r[1] <= cfg.target_high]

    if inside:
        # Choose the delta with mean accuracy closest to target_mid
        chosen = min(inside, key=lambda r: abs(r[1] - cfg.target_mid))
        reason = "inside_target_band"
    else:
        # If nothing lands inside the band, choose delta closest to the band
        # Distance is 0 if inside; otherwise distance to nearest endpoint.
        def band_distance(mean_acc: float) -> float:
            if mean_acc < cfg.target_low:
                return cfg.target_low - mean_acc
            if mean_acc > cfg.target_high:
                return mean_acc - cfg.target_high
            return 0.0

        chosen = min(results_per_delta, key=lambda r: band_distance(r[1]))
        reason = "closest_to_target_band"

    delta_star, mean_star, std_star = chosen

    return {
        "intrinsic_dim": d_star,
        "chosen_delta": delta_star,
        "chosen_mean_acc": mean_star,
        "chosen_std_acc": std_star,
        "selection_reason": reason,
        "all_results": results_per_delta,
        "target_band": (cfg.target_low, cfg.target_high),
        "target_mid": cfg.target_mid,
        "seeds": cfg.seeds,
        "delta_grid": cfg.delta_grid,
    }


def calibrate_all(cfg: CalibrationConfig) -> Dict[int, Dict[str, object]]:
    """Calibrate delta for all intrinsic dimensions and return a dict keyed by d*."""
    out: Dict[int, Dict[str, object]] = {}
    for d_star in cfg.intrinsic_dims:
        out[d_star] = calibrate_delta_for_dim(d_star, cfg)
    return out


if __name__ == "__main__":
    cfg = CalibrationConfig()
    results = calibrate_all(cfg)

    print("\n=== Delta Calibration Summary ===")
    for d_star, info in results.items():
        print(
            f"d*={d_star:>3} | chosen Δ={info['chosen_delta']:<4} "
            f"| mean acc={info['chosen_mean_acc']:.3f} "
            f"| std={info['chosen_std_acc']:.3f} "
            f"| reason={info['selection_reason']}"
        )


    # Optional: print the full grid for one dimension for inspection
    example_dim = cfg.intrinsic_dims[0]
    print(f"\n--- Full grid results for d*={example_dim} ---")
    for delta, mean_acc, std_acc in results[example_dim]["all_results"]:
        print(f"Δ={delta:<4} | mean={mean_acc:.3f} | std={std_acc:.3f}")