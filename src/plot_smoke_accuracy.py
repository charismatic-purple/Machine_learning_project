import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../results/raw/experiment_results_nn_smoke.csv")
g = df.groupby(["observed_dim", "model"])["test_accuracy"].agg(["mean", "std"]).reset_index()
for m in sorted(g["model"].unique()):
    s = g[g["model"] == m].sort_values("observed_dim")
    plt.plot(s["observed_dim"], s["mean"], marker="o", label=m)
plt.xlabel("Observed dimension (after PCA)"); plt.ylabel("Mean test accuracy"); plt.title("Smoke test: Accuracy vs observed dimension")
plt.legend(); plt.tight_layout(); plt.savefig("../figures/smoke_accuracy_vs_dimension.png", dpi=300); plt.close()
print("Saved ../figures/smoke_accuracy_vs_dimension.png")