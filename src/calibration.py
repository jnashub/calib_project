"""ECE/MCE metrics and three ways to fit the temperature-scaling parameter.

fit_temperature_optimizer: the usual way, just minimize NLL(T) with a
scalar optimizer. This is the baseline/ground truth.

fit_temperature_picard: same problem, but derived as a fixed-point
equation and solved with plain Picard iteration. Related to iterative
scaling methods for exponential-family MLE (Darroch & Ratcliff 1972;
Della Pietra et al. 1997). Converges, just slowly (linear rate).

fit_temperature_steffensen: same fixed-point map, but with Steffensen /
Aitken acceleration (Aitken 1926) so it converges quadratically instead.

Full derivation is in paper/paper.tex, Section 3.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, y: np.ndarray, T: float) -> float:
    probs = softmax(logits / T)
    return float(-np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12)))


def expected_calibration_error(
    probs: np.ndarray, y: np.ndarray, n_bins: int = 15
) -> tuple[float, float, list[dict]]:
    """ECE and MCE with equal-width confidence bins.

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
    MCE = max_b |acc(B_b) - conf(B_b)|
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y)
    ece = 0.0
    mce = 0.0
    bins = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        if hi == bin_edges[-1]:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        count = int(mask.sum())
        if count == 0:
            bins.append({"lo": lo, "hi": hi, "count": 0, "acc": None, "conf": None})
            continue
        acc = float(correct[mask].mean())
        conf = float(confidences[mask].mean())
        gap = abs(acc - conf)
        ece += (count / n) * gap
        mce = max(mce, gap)
        bins.append({"lo": lo, "hi": hi, "count": count, "acc": acc, "conf": conf})

    return ece, mce, bins


def expected_calibration_error_equal_mass(
    probs: np.ndarray, y: np.ndarray, n_bins: int = 10
) -> tuple[float, float]:
    """Same thing, but bins have equal counts instead of equal width.

    Equal-width bins can end up almost empty when confidence is
    concentrated near 1, which makes MCE noisy since it's a max over
    bins. This version sidesteps that. Used it to double check whether
    an MCE increase after calibration was a real effect or just bin
    noise (see results/metrics.json and the paper's discussion section).
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y).astype(float)
    order = np.argsort(confidences)

    n = len(y)
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    ece = 0.0
    mce = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = order[lo:hi]
        if len(idx) == 0:
            continue
        acc = float(correct[idx].mean())
        conf = float(confidences[idx].mean())
        gap = abs(acc - conf)
        ece += (len(idx) / n) * gap
        mce = max(mce, gap)

    return ece, mce


def fit_temperature_optimizer(logits: np.ndarray, y: np.ndarray) -> float:
    """Minimize NLL(T) directly with Brent's method. This is the baseline."""
    result = minimize_scalar(
        lambda T: nll(logits, y, T), bounds=(0.05, 20.0), method="bounded",
        options={"xatol": 1e-10},
    )
    return float(result.x)


def _fixed_point_map(logits: np.ndarray, y: np.ndarray):
    """Builds h(beta) = beta * A / S(beta), used by both fixed-point solvers.

    Pulled out into one place so the Picard and Steffensen solvers are
    guaranteed to iterate the exact same map instead of each defining
    their own copy of A, S, h (which is what an earlier version did,
    and it's an easy way to introduce a bug if you ever change one and
    forget the other).
    """
    A = float(np.sum(logits[np.arange(len(y)), y]))

    def S(beta: float) -> float:
        p = softmax(beta * logits)
        return float(np.sum(p * logits))

    def h(beta: float) -> float:
        s_val = S(beta)
        if s_val <= 0:
            raise RuntimeError("S(beta) <= 0; fixed-point update undefined.")
        return beta * A / s_val

    return h


def _contraction_factor(h, beta: float, eps: float = 1e-4) -> float:
    """|h'(beta)| estimated with a central finite difference."""
    return float(abs((h(beta + eps) - h(beta - eps)) / (2 * eps)))


@dataclass
class PicardResult:
    T: float
    beta_trajectory: list[float]
    nll_trajectory: list[float]
    n_iter: int
    contraction_factor: float


def fit_temperature_picard(
    logits: np.ndarray,
    y: np.ndarray,
    beta_init: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> PicardResult:
    """Solve for T with plain Picard iteration on the fixed-point map.

    Let beta = 1/T. Setting d NLL(beta) / d beta = 0 gives

        A := sum_i z_{i, y_i}  =  sum_i sum_c softmax(beta * z_i)_c * z_{i,c} =: S(beta)

    Rearranged, this is a fixed point: beta = h(beta) = beta * A / S(beta),
    solved here by iterating beta_{k+1} = h(beta_k).

    This is basically the same update rule as Generalized/Improved
    Iterative Scaling for exponential-family log-likelihoods, just
    applied to a single shared parameter instead of one per feature.
    Convergence is linear, at rate |h'(beta*)| (Banach fixed-point
    theorem) -- see the paper for the derivation and why that rate ends
    up close to 1 here.
    """
    h = _fixed_point_map(logits, y)

    beta = beta_init
    traj = [beta]
    nll_traj = [nll(logits, y, 1.0 / beta)]

    for _ in range(max_iter):
        beta_new = h(beta)
        traj.append(beta_new)
        nll_traj.append(nll(logits, y, 1.0 / beta_new))
        if abs(beta_new - beta) < tol:
            beta = beta_new
            break
        beta = beta_new

    return PicardResult(
        T=1.0 / beta,
        beta_trajectory=traj,
        nll_trajectory=nll_traj,
        n_iter=len(traj) - 1,
        contraction_factor=_contraction_factor(h, beta),
    )


def fit_temperature_steffensen(
    logits: np.ndarray,
    y: np.ndarray,
    beta_init: float = 1.0,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> PicardResult:
    """Same fixed-point map as above, sped up with Steffensen's method.

    Plain Picard iteration here converges only linearly (see
    fit_temperature_picard), which is slow when |h'(beta*)| is close
    to 1, which it is in our experiments. Steffensen's method fixes
    this by applying Aitken's delta-squared trick to each triple of
    iterates, which gets rid of the leading error term and gives
    quadratic convergence without ever computing h'.

    One step, given h:
        p0 = beta_k
        p1 = h(p0)
        p2 = h(p1)
        beta_{k+1} = p0 - (p1 - p0)^2 / (p2 - 2*p1 + p0)
    """
    h = _fixed_point_map(logits, y)

    beta = beta_init
    traj = [beta]
    nll_traj = [nll(logits, y, 1.0 / beta)]

    for _ in range(max_iter):
        p0 = beta
        p1 = h(p0)
        p2 = h(p1)
        denom = p2 - 2 * p1 + p0
        beta_new = p2 if abs(denom) < 1e-14 else p0 - (p1 - p0) ** 2 / denom
        traj.append(beta_new)
        nll_traj.append(nll(logits, y, 1.0 / beta_new))
        if abs(beta_new - beta) < tol:
            beta = beta_new
            break
        beta = beta_new

    return PicardResult(
        T=1.0 / beta,
        beta_trajectory=traj,
        nll_trajectory=nll_traj,
        n_iter=len(traj) - 1,
        contraction_factor=_contraction_factor(h, beta),
    )
