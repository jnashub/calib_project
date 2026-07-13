import numpy as np
import pytest

from src.calibration import (
    expected_calibration_error,
    fit_temperature_optimizer,
    fit_temperature_picard,
    fit_temperature_steffensen,
    softmax,
)


def _synthetic_logits(seed=0, n=500, k=5, scale=3.0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, size=n)
    base = rng.normal(0, 1.0, size=(n, k))
    # Boost the true class's logit so the model is reasonably accurate.
    base[np.arange(n), y] += scale
    return base, y


def test_ece_is_zero_for_perfectly_calibrated_predictions():
    # Construct labels whose true generating probability equals the
    # model's stated confidence: y ~ Bernoulli(p), predicted probs = [1-p, p].
    # By construction this model is calibrated, so ECE should be ~0
    # up to finite-sample binning noise.
    rng = np.random.default_rng(0)
    n = 20000
    p = rng.uniform(0.5, 1.0, size=n)
    y = (rng.uniform(size=n) < p).astype(int)
    probs = np.stack([1 - p, p], axis=1)

    ece, mce, _ = expected_calibration_error(probs, y, n_bins=20)
    assert ece < 0.02  # allow for sampling noise


def test_softmax_rows_sum_to_one():
    logits = np.random.default_rng(0).normal(size=(50, 7))
    probs = softmax(logits)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(50), atol=1e-10)


def test_temperature_solvers_agree():
    logits, y = _synthetic_logits()
    T_opt = fit_temperature_optimizer(logits, y)
    picard = fit_temperature_picard(logits, y, max_iter=2000, tol=1e-12)
    steffensen = fit_temperature_steffensen(logits, y, max_iter=100, tol=1e-12)

    assert T_opt == pytest.approx(picard.T, abs=1e-3)
    assert T_opt == pytest.approx(steffensen.T, abs=1e-6)


def test_steffensen_converges_faster_than_naive_picard():
    logits, y = _synthetic_logits()
    picard = fit_temperature_picard(logits, y, max_iter=2000, tol=1e-10)
    steffensen = fit_temperature_steffensen(logits, y, max_iter=100, tol=1e-10)

    assert steffensen.n_iter < picard.n_iter


def test_picard_raises_on_degenerate_input():
    # All-zero logits make S(beta) = 0 for every beta -> undefined update.
    logits = np.zeros((10, 3))
    y = np.zeros(10, dtype=int)
    with pytest.raises(RuntimeError):
        fit_temperature_picard(logits, y, max_iter=5)
