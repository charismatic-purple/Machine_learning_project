from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def train_and_eval_logreg(X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          seed: int) -> tuple[float, float]:
    """
    Train Logistic Regression and return (train_acc, test_acc).

    Note:
    - We intentionally avoid using the 'multi_class' constructor argument
      to remain compatible with older scikit-learn versions.
    """
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=3000,
        random_state=seed
    )
    clf.fit(X_train, y_train)
    train_acc = float(clf.score(X_train, y_train))
    test_acc = float(clf.score(X_test, y_test))
    return train_acc, test_acc


def train_and_eval_linear_svm(X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray,
                              seed: int) -> tuple[float, float]:
    """
    Train Linear SVM and return (train_acc, test_acc).

    We use LinearSVC for speed and stability in high dimensions.
    """
    clf = LinearSVC(
        C=1.0,
        max_iter=10000,
        random_state=seed
    )
    clf.fit(X_train, y_train)
    train_acc = float(clf.score(X_train, y_train))
    test_acc = float(clf.score(X_test, y_test))
    return train_acc, test_acc

from sklearn.neighbors import KNeighborsClassifier

def train_and_eval_knn(X_train, y_train, X_test, y_test, k: int) -> tuple[float, float]:
    """
    Train kNN and return (train_acc, test_acc).
    """
    clf = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    clf.fit(X_train, y_train)
    train_acc = float(clf.score(X_train, y_train))
    test_acc = float(clf.score(X_test, y_test))
    return train_acc, test_acc