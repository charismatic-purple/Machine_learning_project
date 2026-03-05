from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class AnalysisConfig:
    """
    Configuration for Phase 5: aggregation and plotting.
    """
    raw_results_path: str = "../results/raw/experiment_results.csv"
    summary_path: str = "../results/aggregated/results_summary.csv"
    figures_dir: str = "../figures"

    ci_z: float = 1.96  # 95% confidence interval multiplier for normal approx


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_results(path: str) -> pd.DataFrame:
    """Load raw experiment results CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw results file not found: {path}")
    df = pd.read_csv(path)
    required_cols = {
        "seed", "intrinsic_dim", "observed_dim", "model",
        "train_accuracy", "test_accuracy", "generalization_gap",
        "snr_proxy", "distance_contrast"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in results CSV: {missing}")
    return df


def aggregate_results(df: pd.DataFrame, ci_z: float = 1.96) -> pd.DataFrame:
    """
    Aggregate raw results across seeds using a pandas-version-stable approach.

    Produces columns:
      test_mean, test_std, test_se, test_ci_low, test_ci_high
      train_mean, train_std, train_se, train_ci_low, train_ci_high
      gap_mean, gap_std, gap_se, gap_ci_low, gap_ci_high
      n  (count per group)
    grouped by intrinsic_dim, observed_dim, model.
    """
    group_cols = ["intrinsic_dim", "observed_dim", "model"]

    # Core aggregation: mean/std/count for each metric
    g = df.groupby(group_cols, dropna=False).agg(
        n=("seed", "count"),

        test_mean=("test_accuracy", "mean"),
        test_std=("test_accuracy", "std"),

        train_mean=("train_accuracy", "mean"),
        train_std=("train_accuracy", "std"),

        gap_mean=("generalization_gap", "mean"),
        gap_std=("generalization_gap", "std"),
        snr_mean=("snr_proxy", "mean"),
        snr_std=("snr_proxy", "std"),
        contrast_mean=("distance_contrast", "mean"),
        contrast_std=("distance_contrast", "std"),
    ).reset_index()

    # Replace NaN std (happens when n=1) with 0.0
    for col in ["test_std", "train_std", "gap_std"]:
        g[col] = g[col].fillna(0.0)

    # Compute standard error
    g["test_se"] = g["test_std"] / np.sqrt(g["n"])
    g["train_se"] = g["train_std"] / np.sqrt(g["n"])
    g["gap_se"] = g["gap_std"] / np.sqrt(g["n"])

    # Compute 95% CI
    g["test_ci_low"] = g["test_mean"] - ci_z * g["test_se"]
    g["test_ci_high"] = g["test_mean"] + ci_z * g["test_se"]

    g["train_ci_low"] = g["train_mean"] - ci_z * g["train_se"]
    g["train_ci_high"] = g["train_mean"] + ci_z * g["train_se"]

    g["gap_ci_low"] = g["gap_mean"] - ci_z * g["gap_se"]
    g["gap_ci_high"] = g["gap_mean"] + ci_z * g["gap_se"]

    g["snr_std"] = g["snr_std"].fillna(0.0)
    g["contrast_std"] = g["contrast_std"].fillna(0.0)

    g["snr_se"] = g["snr_std"] / np.sqrt(g["n"])
    g["contrast_se"] = g["contrast_std"] / np.sqrt(g["n"])

    g["snr_ci_low"] = g["snr_mean"] - ci_z * g["snr_se"]
    g["snr_ci_high"] = g["snr_mean"] + ci_z * g["snr_se"]

    g["contrast_ci_low"] = g["contrast_mean"] - ci_z * g["contrast_se"]
    g["contrast_ci_high"] = g["contrast_mean"] + ci_z * g["contrast_se"]

    # Sort for plotting
    g = g.sort_values(["model", "intrinsic_dim", "observed_dim"]).reset_index(drop=True)
    return g


def save_summary(df_summary: pd.DataFrame, path: str) -> None:
    """Save summary CSV."""
    ensure_dir(os.path.dirname(path))
    df_summary.to_csv(path, index=False)


def plot_metric_vs_dimension(
    df_summary: pd.DataFrame,
    metric_prefix: str,
    ylabel: str,
    title: str,
    output_path: str,
    use_ci: bool = True
) -> None:
    """
    Plot a metric (mean + CI or mean + std) vs observed_dim.
    Separate curves by intrinsic_dim, and separate figures by model.

    metric_prefix should be one of:
    - "test" -> uses columns test_mean, test_ci_low, test_ci_high
    - "gap"  -> uses columns gap_mean, gap_ci_low, gap_ci_high
    """
    ensure_dir(os.path.dirname(output_path))

    models = sorted(df_summary["model"].unique())
    intrinsic_dims = sorted(df_summary["intrinsic_dim"].unique())

    # We'll create one figure per model to keep plots clean for the paper.
    for model in models:
        sub = df_summary[df_summary["model"] == model]

        plt.figure()
        for d_star in intrinsic_dims:
            s2 = sub[sub["intrinsic_dim"] == d_star].sort_values("observed_dim")
            x = s2["observed_dim"].to_numpy()
            y = s2[f"{metric_prefix}_mean"].to_numpy()

            if use_ci:
                y_low = s2[f"{metric_prefix}_ci_low"].to_numpy()
                y_high = s2[f"{metric_prefix}_ci_high"].to_numpy()
                plt.fill_between(x, y_low, y_high, alpha=0.2)
            else:
                y_std = s2[f"{metric_prefix}_std"].to_numpy()
                plt.fill_between(x, y - y_std, y + y_std, alpha=0.2)

            plt.plot(x, y, marker="o", label=f"intrinsic d*={d_star}")

        plt.xlabel("Observed dimension (after PCA)")
        plt.ylabel(ylabel)
        plt.title(f"{title} — model: {model}")
        plt.legend()
        plt.tight_layout()

        # Save per-model figure (e.g., accuracy_vs_dimension_logreg.png)
        base, ext = os.path.splitext(output_path)
        model_path = f"{base}_{model}{ext}"
        plt.savefig(model_path, dpi=300)
        plt.close()


def plot_combined_three_curves(
    df_summary: pd.DataFrame,
    model: str,
    y_col: str,
    y_ci_low: str,
    y_ci_high: str,
    ylabel: str,
    title: str,
    output_path: str
) -> None:
    """
    One plot, one model, three curves (intrinsic dims), with CI bands.
    """
    ensure_dir(os.path.dirname(output_path))

    sub = df_summary[df_summary["model"] == model]
    intrinsic_dims = sorted(sub["intrinsic_dim"].unique())

    plt.figure()
    for d_star in intrinsic_dims:
        s2 = sub[sub["intrinsic_dim"] == d_star].sort_values("observed_dim")
        x = s2["observed_dim"].to_numpy()
        y = s2[y_col].to_numpy()
        lo = s2[y_ci_low].to_numpy()
        hi = s2[y_ci_high].to_numpy()

        plt.fill_between(x, lo, hi, alpha=0.2)
        plt.plot(x, y, marker="o", label=f"intrinsic d*={d_star}")

    plt.xlabel("Observed dimension (after PCA)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_all_models_mean_curve_from_raw(
        df_raw: pd.DataFrame,
        y_col: str,
        ylabel: str,
        title: str,
        output_path: str,
        filter_intrinsic_dim: int | None = None,
) -> None:
    """
    Plot mean metric vs observed_dim with one curve per model,
    aggregated across seeds. Optionally filter to a single intrinsic_dim.

    This is the full-run version of the smoke-test diagnostic plots.
    """
    ensure_dir(os.path.dirname(output_path))

    df = df_raw.copy()

    if filter_intrinsic_dim is not None:
        df = df[df["intrinsic_dim"] == filter_intrinsic_dim]

    if df.empty:
        raise ValueError("No data available for the requested filter settings.")

    summary = (
        df.groupby(["observed_dim", "model"], dropna=False)[y_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["model", "observed_dim"])
    )

    plt.figure()
    for model in sorted(summary["model"].unique()):
        sub = summary[summary["model"] == model].sort_values("observed_dim")
        plt.plot(sub["observed_dim"], sub["mean"], marker="o", label=model)

    xlabel = "Observed dimension (after PCA)"
    if filter_intrinsic_dim is not None:
        xlabel += f"  (intrinsic d*={filter_intrinsic_dim})"

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_phase5(cfg: AnalysisConfig) -> None:
    """Run Phase 5 end-to-end: load -> aggregate -> save -> plot."""
    df_raw = load_results(cfg.raw_results_path)
    df_summary = aggregate_results(df_raw, ci_z=cfg.ci_z)



    df_summary = pd.read_csv("../results/aggregated/results_summary.csv")

    print("Columns:", df_summary.columns.tolist())
    print("Unique models:", df_summary["model"].unique())
    print("Unique intrinsic dims:", df_summary["intrinsic_dim"].unique())
    print("Unique observed dims:", df_summary["observed_dim"].unique())

    # Check whether contrast columns exist and are non-null
    contrast_cols = [c for c in df_summary.columns if "contrast" in c]
    print("Contrast-related columns:", contrast_cols)

    if "contrast_mean" in df_summary.columns:
        print("Non-null contrast_mean rows:", df_summary["contrast_mean"].notna().sum())

    # Check if the model you plotted actually exists
    target_model = "logreg"  # or whatever you passed into plot_combined_three_curves
    print("Rows for model:", target_model, "=>", (df_summary["model"] == target_model).sum())



    save_summary(df_summary, cfg.summary_path)

    ensure_dir(cfg.figures_dir)

    # Plot 1: Test accuracy vs observed dimension
    plot_metric_vs_dimension(
        df_summary=df_summary,
        metric_prefix="test",
        ylabel="Test accuracy",
        title="Accuracy vs observed dimension",
        output_path=os.path.join(cfg.figures_dir, "accuracy_vs_dimension.png"),
        use_ci=True
    )

    # Plot 2: Generalization gap vs observed dimension
    plot_metric_vs_dimension(
        df_summary=df_summary,
        metric_prefix="gap",
        ylabel="Generalization gap (train - test)",
        title="Generalization gap vs observed dimension",
        output_path=os.path.join(cfg.figures_dir, "generalization_gap_vs_dimension.png"),
        use_ci=True
    )

    # Main combined paper figure: logistic regression accuracy curves (3 intrinsic dims)
    plot_combined_three_curves(
        df_summary=df_summary,
        model="logreg",
        y_col="test_mean",
        y_ci_low="test_ci_low",
        y_ci_high="test_ci_high",
        ylabel="Test accuracy",
        title="Test accuracy vs observed dimension (Logistic Regression)",
        output_path=os.path.join(cfg.figures_dir, "paper_main_accuracy_logreg.png")
    )

    # SNR proxy vs dimension (same layout)
    plot_combined_three_curves(
        df_summary=df_summary,
        model="logreg",
        y_col="snr_mean",
        y_ci_low="snr_ci_low",
        y_ci_high="snr_ci_high",
        ylabel="SNR proxy (explained variance / remaining variance)",
        title="SNR proxy vs observed dimension (Logistic Regression pipeline)",
        output_path=os.path.join(cfg.figures_dir, "snr_proxy_vs_dimension.png")
    )

    # Curse of dimensionality: distance contrast vs dimension
    plot_combined_three_curves(
        df_summary=df_summary,
        model="logreg",
        y_col="contrast_mean",
        y_ci_low="contrast_ci_low",
        y_ci_high="contrast_ci_high",
        ylabel="Distance contrast (relative)",
        title="Curse of dimensionality: distance concentration vs dimension",
        output_path=os.path.join(cfg.figures_dir, "distance_contrast_vs_dimension.png")
    )

    # Full-run: "smoke-style" plots (all models on one axis), aggregated across seeds
    plot_all_models_mean_curve_from_raw(
        df_raw=df_raw,
        y_col="test_accuracy",
        ylabel="Mean test accuracy",
        title="Full experiment: Accuracy vs observed dimension (all models)",
        output_path=os.path.join(cfg.figures_dir, "accuracy_vs_dimension_all_models.png"),
        filter_intrinsic_dim=None,  # aggregate across all intrinsic dims
    )

    plot_all_models_mean_curve_from_raw(
        df_raw=df_raw,
        y_col="generalization_gap",
        ylabel="Mean generalization gap (train - test)",
        title="Full experiment: Generalization gap vs observed dimension (all models)",
        output_path=os.path.join(cfg.figures_dir, "generalization_gap_vs_dimension_all_models.png"),
        filter_intrinsic_dim=None,
    )


    print(f"Saved summary CSV to: {cfg.summary_path}")
    print(f"Saved figures to: {cfg.figures_dir}/")
    print("Note: figures are saved per model (suffix _logreg / _linear_svm).")


if __name__ == "__main__":
    cfg = AnalysisConfig()
    run_phase5(cfg)