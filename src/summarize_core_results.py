from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SummaryConfig:
    raw_results_path: str = "../results/raw/experiment_results.csv"
    output_dir: str = "../results/derived"
    ci_z: float = 1.96  # 95% confidence interval using normal approximation


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_raw_results(path: str) -> pd.DataFrame:
    """Load raw results and validate required columns."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw results file not found: {path}")

    df = pd.read_csv(path)

    required_cols = {
        "seed",
        "intrinsic_dim",
        "observed_dim",
        "model",
        "test_accuracy",
        "generalization_gap",
        "snr_proxy",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in raw results: {missing}")

    return df


def aggregate_core_metrics(df: pd.DataFrame, ci_z: float) -> pd.DataFrame:
    """
    Aggregate the core metrics across seeds for each condition.

    Output level:
        model x intrinsic_dim x observed_dim
    """
    grouped = (
        df.groupby(["model", "intrinsic_dim", "observed_dim"], dropna=False)
        .agg(
            n=("seed", "count"),

            test_accuracy_mean=("test_accuracy", "mean"),
            test_accuracy_std=("test_accuracy", "std"),

            generalization_gap_mean=("generalization_gap", "mean"),
            generalization_gap_std=("generalization_gap", "std"),

            snr_proxy_mean=("snr_proxy", "mean"),
            snr_proxy_std=("snr_proxy", "std"),
        )
        .reset_index()
    )

    # Fill NaN std values in case n=1
    for col in [
        "test_accuracy_std",
        "generalization_gap_std",
        "snr_proxy_std",
    ]:
        grouped[col] = grouped[col].fillna(0.0)

    # Standard errors
    grouped["test_accuracy_se"] = grouped["test_accuracy_std"] / np.sqrt(grouped["n"])
    grouped["generalization_gap_se"] = grouped["generalization_gap_std"] / np.sqrt(grouped["n"])
    grouped["snr_proxy_se"] = grouped["snr_proxy_std"] / np.sqrt(grouped["n"])

    # Confidence intervals
    grouped["test_accuracy_ci_low"] = grouped["test_accuracy_mean"] - ci_z * grouped["test_accuracy_se"]
    grouped["test_accuracy_ci_high"] = grouped["test_accuracy_mean"] + ci_z * grouped["test_accuracy_se"]

    grouped["generalization_gap_ci_low"] = grouped["generalization_gap_mean"] - ci_z * grouped["generalization_gap_se"]
    grouped["generalization_gap_ci_high"] = grouped["generalization_gap_mean"] + ci_z * grouped["generalization_gap_se"]

    grouped["snr_proxy_ci_low"] = grouped["snr_proxy_mean"] - ci_z * grouped["snr_proxy_se"]
    grouped["snr_proxy_ci_high"] = grouped["snr_proxy_mean"] + ci_z * grouped["snr_proxy_se"]

    grouped = grouped.sort_values(["model", "intrinsic_dim", "observed_dim"]).reset_index(drop=True)
    return grouped


def summarize_optimal_dimension(core_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each model x intrinsic_dim, find the observed dimension that maximizes mean test accuracy.

    If there are ties, choose the smallest observed dimension among tied maxima.
    """
    rows = []

    for (model, intrinsic_dim), sub in core_df.groupby(["model", "intrinsic_dim"], dropna=False):
        sub_sorted = sub.sort_values(["test_accuracy_mean", "observed_dim"], ascending=[False, True]).reset_index(drop=True)
        best = sub_sorted.iloc[0]

        rows.append({
            "model": model,
            "intrinsic_dim": intrinsic_dim,
            "optimal_dimension": int(best["observed_dim"]),
            "peak_test_accuracy": float(best["test_accuracy_mean"]),
            "peak_test_accuracy_ci_low": float(best["test_accuracy_ci_low"]),
            "peak_test_accuracy_ci_high": float(best["test_accuracy_ci_high"]),
            "generalization_gap_at_optimum": float(best["generalization_gap_mean"]),
            "snr_proxy_at_optimum": float(best["snr_proxy_mean"]),
            "dimension_ratio_optimal_over_intrinsic": float(best["observed_dim"] / intrinsic_dim),
            "dimension_error": int(best["observed_dim"] - intrinsic_dim),
        })

    out = pd.DataFrame(rows).sort_values(["model", "intrinsic_dim"]).reset_index(drop=True)
    return out


def compute_sensitivity_measures(core_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sensitivity measures for each model x intrinsic_dim.

    Measures included:
    - average absolute slope of test accuracy vs observed dimension
    - max increase slope
    - max decrease slope
    - peak sharpness: peak accuracy minus accuracy at largest dimension
    - gap growth: generalization gap at largest dimension minus smallest dimension
    """
    rows = []

    for (model, intrinsic_dim), sub in core_df.groupby(["model", "intrinsic_dim"], dropna=False):
        sub = sub.sort_values("observed_dim").reset_index(drop=True)

        x = sub["observed_dim"].to_numpy(dtype=float)
        y_acc = sub["test_accuracy_mean"].to_numpy(dtype=float)
        y_gap = sub["generalization_gap_mean"].to_numpy(dtype=float)

        if len(x) < 2:
            avg_abs_slope = np.nan
            max_increase_slope = np.nan
            max_decrease_slope = np.nan
        else:
            dx = np.diff(x)
            dy = np.diff(y_acc)
            slopes = dy / dx

            avg_abs_slope = float(np.mean(np.abs(slopes)))
            max_increase_slope = float(np.max(slopes))
            max_decrease_slope = float(np.min(slopes))

        peak_idx = int(np.argmax(y_acc))
        peak_dim = int(x[peak_idx])
        peak_acc = float(y_acc[peak_idx])

        smallest_dim_acc = float(y_acc[0])
        largest_dim_acc = float(y_acc[-1])

        peak_minus_smallest_dim = peak_acc - smallest_dim_acc
        peak_minus_largest_dim = peak_acc - largest_dim_acc

        gap_growth = float(y_gap[-1] - y_gap[0])

        rows.append({
            "model": model,
            "intrinsic_dim": intrinsic_dim,
            "avg_abs_accuracy_slope": avg_abs_slope,
            "max_increase_slope": max_increase_slope,
            "max_decrease_slope": max_decrease_slope,
            "peak_dimension": peak_dim,
            "peak_accuracy": peak_acc,
            "peak_minus_smallest_dim_accuracy": peak_minus_smallest_dim,
            "peak_minus_largest_dim_accuracy": peak_minus_largest_dim,
            "gap_growth_smallest_to_largest_dim": gap_growth,
            "accuracy_drop_after_peak": peak_acc - largest_dim_acc
        })

    out = pd.DataFrame(rows).sort_values(["model", "intrinsic_dim"]).reset_index(drop=True)
    return out


def summarize_global_model_rankings(core_df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional compact table: one row per model, averaged across all intrinsic and observed dimensions.
    Useful for a high-level overview, but not the main scientific table.
    """
    out = (
        core_df.groupby("model", dropna=False)
        .agg(
            mean_test_accuracy_overall=("test_accuracy_mean", "mean"),
            mean_generalization_gap_overall=("generalization_gap_mean", "mean"),
            mean_snr_proxy_overall=("snr_proxy_mean", "mean"),
        )
        .reset_index()
        .sort_values("mean_test_accuracy_overall", ascending=False)
        .reset_index(drop=True)
    )
    return out


def main() -> None:
    cfg = SummaryConfig()
    ensure_dir(cfg.output_dir)

    raw_df = load_raw_results(cfg.raw_results_path)

    core_df = aggregate_core_metrics(raw_df, ci_z=cfg.ci_z)
    optimal_df = summarize_optimal_dimension(core_df)
    sensitivity_df = compute_sensitivity_measures(core_df)
    ranking_df = summarize_global_model_rankings(core_df)

    core_path = os.path.join(cfg.output_dir, "core_metrics_by_model_intrinsic_dimension.csv")
    optimal_path = os.path.join(cfg.output_dir, "optimal_dimension_summary.csv")
    sensitivity_path = os.path.join(cfg.output_dir, "sensitivity_summary.csv")
    ranking_path = os.path.join(cfg.output_dir, "global_model_ranking_summary.csv")

    core_df.to_csv(core_path, index=False)
    optimal_df.to_csv(optimal_path, index=False)
    sensitivity_df.to_csv(sensitivity_path, index=False)
    ranking_df.to_csv(ranking_path, index=False)

    print("Saved derived summaries:")
    print(f" - {core_path}")
    print(f" - {optimal_path}")
    print(f" - {sensitivity_path}")
    print(f" - {ranking_path}")


if __name__ == "__main__":
    main()