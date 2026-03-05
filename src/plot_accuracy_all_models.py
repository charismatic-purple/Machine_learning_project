import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load aggregated results (use full experiment results, not smoke test)
df = pd.read_csv("../results/raw/experiment_results.csv")

# Aggregate across seeds
summary = (
    df.groupby(["observed_dim", "model"])["test_accuracy"]
    .agg(["mean", "std"])
    .reset_index()
)

# Plot style
sns.set_style("whitegrid")
plt.figure(figsize=(8,5))

# Define consistent colors
palette = {
    "logreg": "#1f77b4",
    "linear_svm": "#ff7f0e",
    "knn_k11": "#2ca02c",
    "mlp_base": "#d62728",
    "mlp_reg": "#9467bd",
    "mlp_wide": "#8c564b",
}

# Plot each model
for model in sorted(summary["model"].unique()):

    sub = summary[summary["model"] == model].sort_values("observed_dim")

    plt.plot(
        sub["observed_dim"],
        sub["mean"],
        marker="o",
        label=model,
        color=palette.get(model, None),
        linewidth=2
    )

plt.xlabel("Observed Dimension (after PCA)", fontsize=12)
plt.ylabel("Mean Test Accuracy", fontsize=12)
plt.title("Model Performance vs Representation Dimension", fontsize=14)

plt.legend(title="Model", fontsize=10)

plt.tight_layout()

plt.savefig("../figures/accuracy_vs_dimension_all_models.png", dpi=300)
plt.close()

print("Saved figures/accuracy_vs_dimension_all_models.png")