import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.generate_dataset import (
    SyntheticDatasetConfig,
    generate_intrinsic_gaussian_mixture
)

from src.embed_data import (
    AmbientEmbeddingConfig,
    embed_to_ambient
)


def evaluate_logistic_regression(X, y, seed=0):
    """
    Train logistic regression and return train and test accuracy.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y
    )

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        random_state=seed
    )

    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    return train_acc, test_acc


def run_sanity_check():

    # Use calibrated parameters
    intrinsic_dim = 64
    delta = 2.0

    dataset_cfg = SyntheticDatasetConfig(
        seed=0,
        intrinsic_dim=intrinsic_dim,
        num_classes=3,
        samples_per_class=1000,
        var_low=0.5,
        var_high=3.0,
        delta=delta,
        shuffle=True
    )

    X_intrinsic, y, _ = generate_intrinsic_gaussian_mixture(dataset_cfg)

    print("\nIntrinsic dataset")
    print("-------------------")
    train_acc, test_acc = evaluate_logistic_regression(X_intrinsic, y)
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy : {test_acc:.3f}")

    # Embed to ambient space
    embed_cfg = AmbientEmbeddingConfig(
        seed=42,
        ambient_dim=768,
        noise_sigma=0.1
    )

    Z_ambient, _ = embed_to_ambient(X_intrinsic, embed_cfg)

    print("\nAmbient dataset (768D)")
    print("----------------------")
    train_acc, test_acc = evaluate_logistic_regression(Z_ambient, y)
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy : {test_acc:.3f}")


if __name__ == "__main__":
    run_sanity_check()