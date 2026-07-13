"""A small multilayer perceptron, implemented in plain NumPy.

Not using a deep learning framework keeps the pre-softmax logits
directly accessible, which is what the calibration code needs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((y.shape[0], n_classes))
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class MLP:
    """One-hidden-layer MLP with ReLU activation and a linear (logit) output layer."""

    n_in: int
    n_hidden: int
    n_out: int
    seed: int = 0
    weight_decay: float = 1e-5
    history: dict = field(default_factory=lambda: {"train_loss": [], "train_acc": []})

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        # He initialization, appropriate for ReLU hidden units.
        self.W1 = rng.normal(0, np.sqrt(2.0 / self.n_in), size=(self.n_in, self.n_hidden))
        self.b1 = np.zeros(self.n_hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / self.n_hidden), size=(self.n_hidden, self.n_out))
        self.b2 = np.zeros(self.n_out)

    def logits(self, X: np.ndarray) -> np.ndarray:
        h = np.maximum(0.0, X @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def _forward_cache(self, X: np.ndarray):
        z1 = X @ self.W1 + self.b1
        h = np.maximum(0.0, z1)
        z2 = h @ self.W2 + self.b2
        return z1, h, z2

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 400,
        lr: float = 0.05,
    ) -> "MLP":
        """Full-batch gradient descent.

        Trained for a lot of epochs with only light L2 regularization.
        The point is to get a model that's accurate but not necessarily
        well calibrated, since that's the regime post-hoc calibration
        is meant to fix.
        """
        n = X.shape[0]
        y_oh = _one_hot(y, self.n_out)

        for epoch in range(epochs):
            z1, h, z2 = self._forward_cache(X)
            probs = _softmax(z2)

            loss = -np.mean(np.sum(y_oh * np.log(probs + 1e-12), axis=1))
            loss += self.weight_decay * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
            acc = float(np.mean(np.argmax(z2, axis=1) == y))
            self.history["train_loss"].append(loss)
            self.history["train_acc"].append(acc)

            # Backprop.
            dz2 = (probs - y_oh) / n
            dW2 = h.T @ dz2 + 2 * self.weight_decay * self.W2
            db2 = dz2.sum(axis=0)

            dh = dz2 @ self.W2.T
            dz1 = dh * (z1 > 0)
            dW1 = X.T @ dz1 + 2 * self.weight_decay * self.W1
            db1 = dz1.sum(axis=0)

            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2

        return self
