import random
import numpy as np
import torch

import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def set_global_seed(seed: int) -> None:
    """
    Set seeds for reproducibility across numpy, python, and PyTorch.

    This ensures that:
    - dataset shuffling is deterministic
    - weight initialization is deterministic
    - GPU operations (if used) are reproducible
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Make CUDA deterministic (slightly slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Select computation device automatically.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def create_dataloaders(
    X_train,
    y_train,
    X_val,
    y_val,
    batch_size: int = 128,
):
    """
    Convert numpy arrays to PyTorch DataLoaders.
    """

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader


def evaluate_accuracy(model, loader, device):
    """
    Compute classification accuracy of a model on a DataLoader.
    """

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for X, y in loader:

            X = X.to(device)
            y = y.to(device)

            logits = model(X)

            preds = torch.argmax(logits, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    max_epochs: int = 100,
    patience: int = 10,
):
    """
    Train a model with early stopping.

    Returns:
        best_model
        best_val_accuracy
        best_epoch
    """

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val_acc = 0
    best_epoch = 0
    best_state = None

    patience_counter = 0

    model.to(device)

    for epoch in range(max_epochs):

        model.train()

        for X, y in train_loader:

            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(X)

            loss = criterion(logits, y)

            loss.backward()

            optimizer.step()

        val_acc = evaluate_accuracy(model, val_loader, device)

        if val_acc > best_val_acc:

            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)

    return model, best_val_acc, best_epoch