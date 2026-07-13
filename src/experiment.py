"""Runs the whole pipeline: train a model, calibrate it three ways, save
figures and metrics.

Run with:  python -m src.experiment
Outputs:   paper/figures/*.png, results/metrics.json
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .calibration import (
    expected_calibration_error,
    expected_calibration_error_equal_mass,
    fit_temperature_optimizer,
    fit_temperature_picard,
    fit_temperature_steffensen,
    nll,
    softmax,
)
from .data import load_splits
from .model import MLP

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def reliability_diagram(probs, y, title, path, n_bins=10):
    ece, mce, bins = expected_calibration_error(probs, y, n_bins=n_bins)
    centers, accs, confs = [], [], []
    for b in bins:
        if b["count"] == 0:
            continue
        centers.append((b["lo"] + b["hi"]) / 2)
        accs.append(b["acc"])
        confs.append(b["conf"])

    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.bar(centers, accs, width=1.0 / n_bins, edgecolor="black",
           color="#3b6ea5", alpha=0.85, label="Accuracy")
    ax.scatter(confs, accs, color="#d1495b", zorder=5, s=18, label="Bin (conf, acc)")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title}\nECE = {ece:.4f}, MCE = {mce:.4f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return ece, mce


def plot_convergence(picard_result, steffensen_result, T_opt, path):
    beta_star = 1.0 / T_opt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))

    for result, label, color in [
        (picard_result, "Picard (naive)", "#3b6ea5"),
        (steffensen_result, "Picard + Steffensen", "#d1495b"),
    ]:
        Ts = 1.0 / np.array(result.beta_trajectory)
        axes[0].plot(range(len(Ts)), Ts, marker="o", ms=3, color=color, label=label)

        err = np.abs(np.array(result.beta_trajectory) - beta_star)
        err[err == 0] = 1e-16
        axes[1].semilogy(range(len(err)), err, marker="o", ms=3, color=color, label=label)

    axes[0].axhline(T_opt, linestyle="--", color="gray", label=f"Optimizer $T^\\star$={T_opt:.5f}")
    axes[0].set_xlabel("Iteration $k$")
    axes[0].set_ylabel("$T_k = 1/\\beta_k$")
    axes[0].set_title("Temperature trajectory\n(first 10 iterations)")
    axes[0].legend(fontsize=7)
    axes[0].set_xlim(0, max(10, steffensen_result.n_iter + 2))

    axes[1].set_xlabel("Iteration $k$")
    axes[1].set_ylabel("$|\\beta_k - \\beta^\\star|$ (log scale)")
    axes[1].set_title("Convergence: naive vs. accelerated\n(first 10 iterations)")
    axes[1].legend(fontsize=7)
    axes[1].set_xlim(0, max(10, steffensen_result.n_iter + 2))

    # Panels 0 and 1 zoom into the first 10 iterations so the Steffensen
    # curve is actually visible, but that crops naive Picard mid-flight and
    # makes it look stuck. This third panel shows its whole run so it's
    # clear it does converge, just slowly.
    err_full = np.abs(np.array(picard_result.beta_trajectory) - beta_star)
    err_full[err_full == 0] = 1e-16
    axes[2].semilogy(range(len(err_full)), err_full, color="#3b6ea5",
                      label=f"Picard (naive), all {picard_result.n_iter} iters")
    axes[2].set_xlabel("Iteration $k$")
    axes[2].set_ylabel("$|\\beta_k - \\beta^\\star|$ (log scale)")
    axes[2].set_title("Naive Picard, full trajectory\n(confirms it does converge, just slowly)")
    axes[2].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    splits = load_splits(seed=0)

    model = MLP(n_in=splits.X_train.shape[1], n_hidden=128, n_out=splits.n_classes,
                seed=0, weight_decay=1e-7)
    model.train(splits.X_train, splits.y_train, epochs=1500, lr=0.15)

    logits_calib = model.logits(splits.X_calib)
    logits_test = model.logits(splits.X_test)

    train_acc = float(np.mean(np.argmax(model.logits(splits.X_train), axis=1) == splits.y_train))
    test_acc = float(np.mean(np.argmax(logits_test, axis=1) == splits.y_test))

    # Uncalibrated baseline (T = 1).
    probs_test_raw = softmax(logits_test)
    ece_raw, mce_raw = reliability_diagram(
        probs_test_raw, splits.y_test, "Before calibration (T=1)",
        os.path.join(FIG_DIR, "reliability_before.png"),
    )
    nll_raw = nll(logits_test, splits.y_test, T=1.0)

    # Fit T three separate ways, all using only the calibration split.
    T_opt = fit_temperature_optimizer(logits_calib, splits.y_calib)
    picard = fit_temperature_picard(logits_calib, splits.y_calib, beta_init=1.0,
                                     max_iter=800, tol=1e-10)
    steffensen = fit_temperature_steffensen(logits_calib, splits.y_calib, beta_init=1.0,
                                             max_iter=50, tol=1e-10)
    T_picard = picard.T
    T_steffensen = steffensen.T

    # Now apply the calibrated T and re-evaluate on the held-out test split.
    probs_test_cal = softmax(logits_test / T_opt)
    ece_cal, mce_cal = reliability_diagram(
        probs_test_cal, splits.y_test, f"After calibration (T={T_opt:.3f})",
        os.path.join(FIG_DIR, "reliability_after.png"),
    )
    nll_cal = nll(logits_test, splits.y_test, T=T_opt)

    plot_convergence(picard, steffensen, T_opt, os.path.join(FIG_DIR, "convergence.png"))

    # Check whether the MCE increase above is real or just a sparse-bin
    # artifact, by recomputing it with equal-mass bins instead.
    _, mce_raw_eqmass = expected_calibration_error_equal_mass(probs_test_raw, splits.y_test)
    _, mce_cal_eqmass = expected_calibration_error_equal_mass(probs_test_cal, splits.y_test)

    # Repeat the whole thing (split, train, calibrate) over a few seeds so
    # the headline ECE numbers aren't just one lucky split.
    seed_eces_before, seed_eces_after = [], []
    for seed in range(5):
        s = load_splits(seed=seed)
        m = MLP(n_in=s.X_train.shape[1], n_hidden=128, n_out=s.n_classes,
                 seed=seed, weight_decay=1e-7)
        m.train(s.X_train, s.y_train, epochs=1500, lr=0.15)
        lc, lt = m.logits(s.X_calib), m.logits(s.X_test)
        t_s = fit_temperature_optimizer(lc, s.y_calib)
        e_before, _, _ = expected_calibration_error(softmax(lt), s.y_test)
        e_after, _, _ = expected_calibration_error(softmax(lt / t_s), s.y_test)
        seed_eces_before.append(e_before)
        seed_eces_after.append(e_after)

    metrics = {
        "n_train": int(len(splits.y_train)),
        "n_calib": int(len(splits.y_calib)),
        "n_test": int(len(splits.y_test)),
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "T_optimizer": T_opt,
        "T_picard": T_picard,
        "T_steffensen": T_steffensen,
        "T_agreement_abs_diff_picard": abs(T_opt - T_picard),
        "T_agreement_abs_diff_steffensen": abs(T_opt - T_steffensen),
        "picard_n_iter": picard.n_iter,
        "picard_contraction_factor": picard.contraction_factor,
        "steffensen_n_iter": steffensen.n_iter,
        "steffensen_contraction_factor": steffensen.contraction_factor,
        "ece_before": ece_raw,
        "mce_before": mce_raw,
        "mce_before_equal_mass": mce_raw_eqmass,
        "nll_before": nll_raw,
        "ece_after": ece_cal,
        "mce_after": mce_cal,
        "mce_after_equal_mass": mce_cal_eqmass,
        "nll_after": nll_cal,
        "multi_seed_ece_before_mean": float(np.mean(seed_eces_before)),
        "multi_seed_ece_before_std": float(np.std(seed_eces_before)),
        "multi_seed_ece_after_mean": float(np.mean(seed_eces_after)),
        "multi_seed_ece_after_std": float(np.std(seed_eces_after)),
        "multi_seed_n_seeds": 5,
    }

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
