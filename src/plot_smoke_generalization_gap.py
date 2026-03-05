import pandas as pd
import matplotlib.pyplot as plt

# Load smoke test results
df = pd.read_csv("../results/raw/experiment_results_nn_smoke.csv")

# Aggregate across seeds
summary = (
    df.groupby(["observed_dim", "model"])["generalization_gap"]
    .agg(["mean", "std"])
    .reset_index()
)

# Plot
plt.figure(figsize=(7,5))

for model in sorted(summary["model"].unique()):

    sub = summary[summary["model"] == model].sort_values("observed_dim")

    plt.plot(
        sub["observed_dim"],
        sub["mean"],
        marker="o",
        label=model
    )

plt.xlabel("Observed Dimension (after PCA)")
plt.ylabel("Generalization Gap (train - test)")
plt.title("Smoke Test: Generalization Gap vs Dimension")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("../figures/smoke_generalization_gap_vs_dimension.png", dpi=300)
plt.close()

print("Saved figures/smoke_generalization_gap_vs_dimension.png")