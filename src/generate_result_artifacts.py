# src/generate_results_artifacts.py
# Generates figures + LaTeX tables from derived CSVs

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DERIVED_DIR = "../results/derived"
FIG_DIR = "../final-figures/figures"
TABLE_DIR = "../final-figures/tables"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)


def load_data():
    core = pd.read_csv(os.path.join(DERIVED_DIR, "core_metrics_by_model_intrinsic_dimension.csv"))
    optimal = pd.read_csv(os.path.join(DERIVED_DIR, "optimal_dimension_summary.csv"))
    sensitivity = pd.read_csv(os.path.join(DERIVED_DIR, "sensitivity_summary.csv"))
    return core, optimal, sensitivity


# -------------------------
# 📈 FIGURE 1 — GAP VS DIMENSION
# -------------------------
def plot_gap_vs_dimension(core):
    for d_star in sorted(core["intrinsic_dim"].unique()):
        sub = core[core["intrinsic_dim"] == d_star]

        plt.figure()

        for model in sub["model"].unique():
            m = sub[sub["model"] == model].sort_values("observed_dim")

            plt.plot(
                m["observed_dim"],
                m["generalization_gap_mean"],
                marker="o",
                label=model
            )

        plt.xlabel("Observed Dimension")
        plt.ylabel("Generalization Gap")
        plt.title(f"Generalization Gap vs Dimension (d*={d_star})")
        plt.legend()
        plt.tight_layout()

        plt.savefig(f"{FIG_DIR}/gap_vs_dimension_dstar_{d_star}.png", dpi=300)
        plt.close()


# -------------------------
# 📈 FIGURE 2 — SNR VS ACCURACY
# -------------------------
def plot_snr_vs_accuracy(core):
    plt.figure()

    for model in core["model"].unique():
        m = core[core["model"] == model]

        plt.scatter(
            m["snr_proxy_mean"],
            m["test_accuracy_mean"],
            label=model,
            alpha=0.7
        )

    plt.xlabel("SNR Proxy")
    plt.ylabel("Test Accuracy")
    plt.title("SNR vs Accuracy")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{FIG_DIR}/snr_vs_accuracy.png", dpi=300)
    plt.close()


# -------------------------
# 📈 FIGURE 3 — INTRINSIC VS OPTIMAL
# -------------------------
def plot_intrinsic_vs_optimal(optimal):
    plt.figure()

    for model in optimal["model"].unique():
        m = optimal[optimal["model"] == model]

        plt.scatter(
            m["intrinsic_dim"],
            m["optimal_dimension"],
            label=model
        )

    # diagonal reference
    x = [64, 256, 768]
    plt.plot(x, x, linestyle="--")

    plt.xlabel("Intrinsic Dimension")
    plt.ylabel("Optimal Dimension")
    plt.title("Intrinsic vs Optimal Dimension")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{FIG_DIR}/intrinsic_vs_optimal.png", dpi=300)
    plt.close()


# -------------------------
# 📈 FIGURE 4 — DROP AFTER PEAK
# -------------------------
def plot_drop_after_peak(sensitivity):
    plt.figure()

    for i, model in enumerate(sensitivity["model"].unique()):
        m = sensitivity[sensitivity["model"] == model]

        plt.bar(
            [i],
            m["peak_minus_largest_dim_accuracy"].mean()
        )

    plt.xticks(range(len(sensitivity["model"].unique())), sensitivity["model"].unique(), rotation=45)
    plt.ylabel("Accuracy Drop")
    plt.title("Performance Drop After Peak")
    plt.tight_layout()

    plt.savefig(f"{FIG_DIR}/drop_after_peak.png", dpi=300)
    plt.close()


# -------------------------
# 📈 FIGURE 5 — SENSITIVITY
# -------------------------
def plot_sensitivity(sensitivity):
    plt.figure()

    for i, model in enumerate(sensitivity["model"].unique()):
        m = sensitivity[sensitivity["model"] == model]

        plt.bar(
            [i],
            m["avg_abs_accuracy_slope"].mean()
        )

    plt.xticks(range(len(sensitivity["model"].unique())), sensitivity["model"].unique(), rotation=45)
    plt.ylabel("Avg Sensitivity")
    plt.title("Model Sensitivity to Dimensionality")
    plt.tight_layout()

    plt.savefig(f"{FIG_DIR}/sensitivity.png", dpi=300)
    plt.close()


def plot_gap_vs_dimension_combined(core):
    intrinsic_dims = sorted(core["intrinsic_dim"].unique())

    fig, axes = plt.subplots(
        1, len(intrinsic_dims),
        figsize=(16, 4.8),
        sharex=True,
        sharey=True
    )

    if len(intrinsic_dims) == 1:
        axes = [axes]

    for ax, d_star in zip(axes, intrinsic_dims):
        sub = core[core["intrinsic_dim"] == d_star]

        for model in sorted(sub["model"].unique()):
            m = sub[sub["model"] == model].sort_values("observed_dim")

            ax.plot(
                m["observed_dim"],
                m["generalization_gap_mean"],
                marker="o",
                label=model
            )

        ax.set_title(rf"$d^* = {d_star}$")
        ax.set_xlabel("Observed Dimension")

    axes[0].set_ylabel("Generalization Gap")

    # One shared legend for the whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.08),
        frameon=True
    )

    plt.tight_layout(rect=[0, 0.14, 1, 1])
    plt.savefig(f"{FIG_DIR}/gap_vs_dimension_combined.png", dpi=300, bbox_inches="tight")
    plt.close()


# -------------------------
# 📊 LATEX TABLE HELPERS
# -------------------------

MODEL_NAME_MAP = {
    "knn_k11": r"kNN ($k=11$)",
    "linear_svm": "Linear SVM",
    "logreg": "Logistic Regression",
    "mlp_base": "MLP Base",
    "mlp_reg": "MLP Regularized",
    "mlp_wide": "MLP Wide",
}

INTRINSIC_ORDER = [64, 256, 768]


def format_model_name(model: str) -> str:
    return MODEL_NAME_MAP.get(model, model)


def write_latex_table(tabular_body: str, caption: str, label: str, filename: str) -> None:
    """
    Wrap a tabular body inside a full LaTeX table environment and save it.
    """
    latex = f"""\\begin{{table}}[h]
\\centering
{tabular_body}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{table}}
"""
    with open(f"{TABLE_DIR}/{filename}", "w") as f:
        f.write(latex)


def build_three_column_table(df: pd.DataFrame, value_col: str, caption: str, label: str, filename: str,
                             float_fmt: str = "{:.3f}") -> None:
    """
    Build a LaTeX table with columns:
      Model | d*=64 | d*=256 | d*=768
    using one value column.
    """
    pivot = df.pivot(index="model", columns="intrinsic_dim", values=value_col)

    # Reorder rows and columns
    row_order = [m for m in MODEL_NAME_MAP.keys() if m in pivot.index]
    pivot = pivot.reindex(index=row_order, columns=INTRINSIC_ORDER)

    lines = []
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{$d^*=64$} & \textbf{$d^*=256$} & \textbf{$d^*=768$} \\")
    lines.append(r"\midrule")

    for model in pivot.index:
        vals = []
        for d_star in INTRINSIC_ORDER:
            val = pivot.loc[model, d_star]
            if pd.isna(val):
                vals.append("--")
            else:
                vals.append(float_fmt.format(val))
        lines.append(f"{format_model_name(model)} & {vals[0]} & {vals[1]} & {vals[2]} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    tabular_body = "\n".join(lines)
    write_latex_table(tabular_body, caption, label, filename)


def build_combined_optimal_dimension_table(optimal: pd.DataFrame) -> None:
    """
    Build the combined table:
      optimal dimension (dimension error)
    in each cell.
    """
    df = optimal.copy()

    # Compute dimension error if not already present
    if "dimension_error" not in df.columns:
        df["dimension_error"] = df["optimal_dimension"] - df["intrinsic_dim"]

    row_order = [m for m in MODEL_NAME_MAP.keys() if m in df["model"].unique()]

    lines = []
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{$d^*=64$} & \textbf{$d^*=256$} & \textbf{$d^*=768$} \\")
    lines.append(r"\midrule")

    for model in row_order:
        sub = df[df["model"] == model].set_index("intrinsic_dim")
        vals = []
        for d_star in INTRINSIC_ORDER:
            if d_star not in sub.index:
                vals.append("--")
            else:
                opt_dim = int(sub.loc[d_star, "optimal_dimension"])
                err = int(sub.loc[d_star, "dimension_error"])
                vals.append(f"{opt_dim} ({err})")

        lines.append(f"{format_model_name(model)} & {vals[0]} & {vals[1]} & {vals[2]} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    tabular_body = "\n".join(lines)
    write_latex_table(
        tabular_body=tabular_body,
        caption=(
            r"Optimal observed dimensionality (with dimension error in parentheses) "
            r"for each model across intrinsic dimensionalities. The value in parentheses "
            r"indicates $d_{\mathrm{opt}} - d^*$, where negative values mean the optimal "
            r"representation is lower than the intrinsic dimensionality."
        ),
        label="tab:optimal_dimension_combined",
        filename="optimal_dimension_combined.tex"
    )


def generate_tables(core: pd.DataFrame, optimal: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    """
    Generate all LaTeX tables in the new compact format.
    """

    # 1) Combined optimal dimension table
    build_combined_optimal_dimension_table(optimal)

    # 2) Peak performance table
    build_three_column_table(
        df=optimal,
        value_col="peak_test_accuracy",
        caption="Peak test accuracy achieved by each model at its optimal observed dimensionality for each intrinsic dimension $d^*$.",
        label="tab:peak_performance",
        filename="peak_performance.tex",
        float_fmt="{:.3f}"
    )

    # 3) Dimension ratio table
    build_three_column_table(
        df=optimal,
        value_col="dimension_ratio_optimal_over_intrinsic",
        caption=r"Ratio between optimal observed dimensionality and intrinsic dimensionality ($d_{\mathrm{opt}} / d^*$) for each model.",
        label="tab:dimension_ratio",
        filename="dimension_ratio.tex",
        float_fmt="{:.2f}"
    )

    # 4) Gap growth table
    build_three_column_table(
        df=sensitivity,
        value_col="gap_growth_smallest_to_largest_dim",
        caption="Growth of the generalization gap from the smallest to the largest observed dimensionality for each model and intrinsic dimension.",
        label="tab:gap_growth",
        filename="gap_growth.tex",
        float_fmt="{:.3f}"
    )

    # 5) Sensitivity table
    build_three_column_table(
        df=sensitivity,
        value_col="avg_abs_accuracy_slope",
        caption="Average sensitivity of test accuracy to changes in observed dimensionality for each model and intrinsic dimension.",
        label="tab:sensitivity",
        filename="sensitivity_summary.tex",
        float_fmt="{:.4f}"
    )


# -------------------------
# MAIN
# -------------------------
def main():
    core, optimal, sensitivity = load_data()

    plot_gap_vs_dimension_combined(core)
    plot_snr_vs_accuracy(core)
    plot_intrinsic_vs_optimal(optimal)
    plot_drop_after_peak(sensitivity)
    plot_sensitivity(sensitivity)

    generate_tables(core, optimal, sensitivity)

    print("All figures and tables generated.")


if __name__ == "__main__":
    main()