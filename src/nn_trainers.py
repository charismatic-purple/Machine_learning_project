from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.nn_utils import (
    set_global_seed,
    get_device,
    create_dataloaders,
    evaluate_accuracy,
    train_model,
)

from src.nn_models import (
    MLPBaseline,
    MLPRegularized,
    MLPWide,
)


@dataclass(frozen=True)
class NNTrainConfig:
    """
    Training hyperparameters shared across NN variants.
    """
    batch_size: int = 128
    lr: float = 1e-3
    max_epochs: int = 100
    patience: int = 10
    val_size: float = 0.2  # split inside training set


def _prepare_splits(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    val_size: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create train_sub / val split from the training set only.
    Uses stratification to keep class proportions stable.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        random_state=seed,
        stratify=y_train
    )
    return X_tr, X_val, y_tr, y_val


def _to_test_loader(X_test: np.ndarray, y_test: np.ndarray, batch_size: int = 512):
    """
    Build a DataLoader for evaluation (no shuffling).
    """
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


def train_eval_mlp_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    cfg: NNTrainConfig
) -> Dict[str, Any]:
    """
    Train and evaluate the baseline MLP.
    Returns metrics for logging into the experiment CSV.
    """
    set_global_seed(seed)
    device = get_device()

    X_tr, X_val, y_tr, y_val = _prepare_splits(X_train, y_train, seed, cfg.val_size)

    train_loader, val_loader = create_dataloaders(
        X_tr, y_tr, X_val, y_val, batch_size=cfg.batch_size
    )
    test_loader = _to_test_loader(X_test, y_test)

    model = MLPBaseline(input_dim=X_train.shape[1], num_classes=int(np.max(y_train)) + 1)

    model, best_val_acc, best_epoch = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=cfg.lr,
        weight_decay=0.0,
        max_epochs=cfg.max_epochs,
        patience=cfg.patience
    )

    train_acc = evaluate_accuracy(model, train_loader, device)
    test_acc = evaluate_accuracy(model, test_loader, device)

    return {
        "model": "mlp_base",
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "generalization_gap": float(train_acc - test_acc),
        "best_val_accuracy": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "device": str(device),
    }


def train_eval_mlp_regularized(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    cfg: NNTrainConfig,
    dropout: float = 0.2,
    weight_decay: float = 1e-4
) -> Dict[str, Any]:
    """
    Train and evaluate the regularized MLP (dropout + weight decay).
    """
    set_global_seed(seed)
    device = get_device()

    X_tr, X_val, y_tr, y_val = _prepare_splits(X_train, y_train, seed, cfg.val_size)

    train_loader, val_loader = create_dataloaders(
        X_tr, y_tr, X_val, y_val, batch_size=cfg.batch_size
    )
    test_loader = _to_test_loader(X_test, y_test)

    model = MLPRegularized(
        input_dim=X_train.shape[1],
        num_classes=int(np.max(y_train)) + 1,
        dropout=dropout
    )

    model, best_val_acc, best_epoch = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=cfg.lr,
        weight_decay=weight_decay,
        max_epochs=cfg.max_epochs,
        patience=cfg.patience
    )

    train_acc = evaluate_accuracy(model, train_loader, device)
    test_acc = evaluate_accuracy(model, test_loader, device)

    return {
        "model": "mlp_reg",
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "generalization_gap": float(train_acc - test_acc),
        "best_val_accuracy": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "dropout": float(dropout),
        "weight_decay": float(weight_decay),
        "device": str(device),
    }


def train_eval_mlp_wide(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    cfg: NNTrainConfig
) -> Dict[str, Any]:
    """
    Train and evaluate the wide MLP (capacity sweep).
    """
    set_global_seed(seed)
    device = get_device()

    X_tr, X_val, y_tr, y_val = _prepare_splits(X_train, y_train, seed, cfg.val_size)

    train_loader, val_loader = create_dataloaders(
        X_tr, y_tr, X_val, y_val, batch_size=cfg.batch_size
    )
    test_loader = _to_test_loader(X_test, y_test)

    model = MLPWide(input_dim=X_train.shape[1], num_classes=int(np.max(y_train)) + 1)

    model, best_val_acc, best_epoch = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=cfg.lr,
        weight_decay=0.0,
        max_epochs=cfg.max_epochs,
        patience=cfg.patience
    )

    train_acc = evaluate_accuracy(model, train_loader, device)
    test_acc = evaluate_accuracy(model, test_loader, device)

    return {
        "model": "mlp_wide",
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "generalization_gap": float(train_acc - test_acc),
        "best_val_accuracy": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "device": str(device),
    }