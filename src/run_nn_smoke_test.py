from __future__ import annotations

import os
import pandas as pd

from src.run_experiment import ExperimentConfig, run_experiment


if __name__ == "__main__":
    # Smoke test configuration: small but covers the full 7 observed dims.
    cfg = ExperimentConfig(
        seeds=(0, 1),
        intrinsic_dims=(64,),
        observed_dims=(16, 32, 64, 128, 256, 512, 768),
        results_path="../results/raw/experiment_results_nn_smoke.csv",
        run_neural_nets=True
    )

    df = run_experiment(cfg)
    print("\nSmoke test completed.")
    print("Rows:", len(df))
    print("Saved:", cfg.results_path)