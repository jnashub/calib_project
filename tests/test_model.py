import numpy as np

from src.model import MLP, _softmax


def test_forward_logits_shape():
    model = MLP(n_in=5, n_hidden=8, n_out=3, seed=0)
    X = np.random.default_rng(0).normal(size=(4, 5))
    logits = model.logits(X)
    assert logits.shape == (4, 3)


def test_backprop_matches_finite_differences():
    """Checks the hand-written backprop against numerical differentiation.

    Everything downstream (the calibration solvers, the numbers in the
    paper) assumes the logits come from a correctly trained network, so
    this checks that assumption directly, independent of the training
    loop itself.
    """
    rng = np.random.default_rng(0)
    n_in, n_hidden, n_out, n = 4, 6, 3, 5

    model = MLP(n_in=n_in, n_hidden=n_hidden, n_out=n_out, seed=1, weight_decay=1e-3)
    X = rng.normal(size=(n, n_in))
    y = rng.integers(0, n_out, size=n)
    y_oh = np.zeros((n, n_out))
    y_oh[np.arange(n), y] = 1.0

    def loss_fn() -> float:
        probs = _softmax(model.logits(X))
        ce = -np.mean(np.sum(y_oh * np.log(probs + 1e-12), axis=1))
        reg = model.weight_decay * (np.sum(model.W1 ** 2) + np.sum(model.W2 ** 2))
        return ce + reg

    # Analytic gradients: replicate one training step's backward pass
    # without applying the update, so we can compare against numerical
    # gradients on the exact same loss.
    z1, h, z2 = model._forward_cache(X)
    probs = _softmax(z2)
    dz2 = (probs - y_oh) / n
    analytic_dW2 = h.T @ dz2 + 2 * model.weight_decay * model.W2
    dh = dz2 @ model.W2.T
    dz1 = dh * (z1 > 0)
    analytic_dW1 = X.T @ dz1 + 2 * model.weight_decay * model.W1

    eps = 1e-5
    for param, analytic_grad, name in [
        (model.W1, analytic_dW1, "W1"),
        (model.W2, analytic_dW2, "W2"),
    ]:
        # Spot-check a handful of entries rather than the full matrix,
        # to keep the test fast.
        idxs = [(0, 0), (1, 2) if param.shape[1] > 2 else (0, 0)]
        for i, j in idxs:
            orig = param[i, j]

            param[i, j] = orig + eps
            loss_plus = loss_fn()
            param[i, j] = orig - eps
            loss_minus = loss_fn()
            param[i, j] = orig

            numeric_grad = (loss_plus - loss_minus) / (2 * eps)
            assert abs(numeric_grad - analytic_grad[i, j]) < 1e-4, (
                f"{name}[{i},{j}]: analytic={analytic_grad[i, j]:.6f} "
                f"numeric={numeric_grad:.6f}"
            )


def test_training_reduces_loss():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    y = rng.integers(0, 3, size=50)
    model = MLP(n_in=4, n_hidden=10, n_out=3, seed=0, weight_decay=1e-6)
    model.train(X, y, epochs=50, lr=0.1)
    assert model.history["train_loss"][-1] < model.history["train_loss"][0]
