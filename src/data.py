"""Loads and splits the digits dataset.

Uses sklearn's built-in `digits` dataset (10 classes, 8x8 grayscale
images) so the project runs offline with no external downloads.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


@dataclass
class Splits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_calib: np.ndarray
    y_calib: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    n_classes: int


def load_splits(seed: int = 0) -> Splits:
    """Load the digits dataset and split it into train / calibration / test.

    The calibration split is used to fit the temperature parameter
    (Guo et al., 2017); the test split is only used for final evaluation.
    """
    X, y = load_digits(return_X_y=True)
    X = X / 16.0  # pixel values are in [0, 16]; rescale to [0, 1]
    X = X - X.mean(axis=0, keepdims=True)

    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y
    )
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest
    )

    return Splits(
        X_train=X_train, y_train=y_train,
        X_calib=X_calib, y_calib=y_calib,
        X_test=X_test, y_test=y_test,
        n_classes=int(y.max()) + 1,
    )
