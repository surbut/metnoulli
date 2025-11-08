"""
End-to-end MetNoulli latent pathway simulation and training pipeline.

Steps:
    1. Simulate metastasis probabilities with latent pathways.
    2. Sample binary metastasis observations + event times.
    3. Produce a Noulli-style hot start for ψ via spectral clustering.
    4. Train the MetNoulli latent model with Adam.
    5. Align recovered pathways with ground truth and report diagnostics.

Run this file directly:
    python metnoulli_latent_pipeline.py --plot
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import SpectralClustering

import torch

# Ensure we can import project modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from metnoulli_latent_simulation import (  # type: ignore  # noqa: E402
    LatentPathwaySimulationConfig,
    plot_simulation_outputs,
    simulate_metastasis_probabilities,
)
from metnoulli_latent_model import (  # type: ignore  # noqa: E402
    MetNoulliBatch,
    MetNoulliLatentModel,
    train_metnoulli_model,
)


@dataclass
class PreparedDataset:
    Y: np.ndarray
    event_times: np.ndarray
    covariates: np.ndarray
    baseline_logit: np.ndarray
    sites: np.ndarray
    time: np.ndarray


def prepare_simulated_dataset(
    simulation: Dict[str, np.ndarray], rng_seed: int = 123
) -> PreparedDataset:
    """Sample metastasis indicators and event times from simulator output."""
    rng = np.random.default_rng(rng_seed)

    probs = simulation["probs"]  # (N, D, T)
    Y = rng.binomial(1, probs)

    # First-hit index per patient/site, else censored at final bin
    event_times = np.full(probs.shape[:2], probs.shape[2] - 1, dtype=int)
    for i in range(probs.shape[0]):
        for s in range(probs.shape[1]):
            hits = np.where(Y[i, s] == 1)[0]
            if hits.size > 0:
                event_times[i, s] = hits[0]

    baseline_prob = simulation["baseline_prob"]
    baseline_logit = np.log(baseline_prob / (1 - baseline_prob))

    return PreparedDataset(
        Y=Y,
        event_times=event_times,
        covariates=simulation["covariates"],
        baseline_logit=baseline_logit,
        sites=simulation["sites"],
        time=simulation["time"],
    )


def hot_start_from_counts(dataset: PreparedDataset, n_pathways: int) -> np.ndarray:
    """Estimate ψ with spectral clustering and contrast configuration."""
    Y_avg = dataset.Y.mean(axis=0)  # (D, T)
    similarity = np.corrcoef(Y_avg)
    similarity = np.nan_to_num(similarity, nan=0.0)
    affinity = (similarity + 1.0) / 2.0

    spectral = SpectralClustering(
        n_clusters=n_pathways,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=1,
    )
    clusters = spectral.fit_predict(affinity)

    psi_config = {
        "in_cluster": 1.0,
        "out_cluster": -2.0,
        "noise_in": 0.1,
        "noise_out": 0.01,
    }

    rng = np.random.default_rng(42)
    psi_init = np.zeros((n_pathways, dataset.sites.size))
    for k in range(n_pathways):
        membership = clusters == k
        psi_init[k, membership] = psi_config["in_cluster"] + psi_config["noise_in"] * rng.standard_normal(
            membership.sum()
        )
        psi_init[k, ~membership] = psi_config["out_cluster"] + psi_config["noise_out"] * rng.standard_normal(
            (~membership).sum()
        )

    return psi_init


def align_components(
    phi_true: np.ndarray,
    phi_est: np.ndarray,
    theta_est: np.ndarray,
    psi_est: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align estimated latent components with ground truth (permute + sign)."""
    k = phi_true.shape[0]
    corr_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            corr_matrix[i, j] = np.corrcoef(phi_true[i].ravel(), phi_est[j].ravel())[0, 1]

    row_ind, col_ind = linear_sum_assignment(-np.abs(corr_matrix))
    phi_aligned = phi_est[col_ind].copy()
    theta_aligned = theta_est[:, col_ind].copy()
    psi_aligned = psi_est[col_ind].copy()
    corr_aligned = corr_matrix[row_ind, col_ind].copy()

    for idx, corr in enumerate(corr_aligned):
        if corr < 0:
            phi_aligned[idx] *= -1
            theta_aligned[:, idx] *= -1
            psi_aligned[idx] *= -1
            corr_aligned[idx] *= -1

    return phi_aligned, theta_aligned, psi_aligned, corr_aligned


def _mask_at_risk(
    event_times: np.ndarray,
    probs_true: np.ndarray,
    probs_est: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute RMSE and confusion matrix restricted to at-risk windows."""
    N, D, T = probs_true.shape
    at_risk_mask = np.zeros_like(probs_true, dtype=bool)
    for i in range(N):
        for d in range(D):
            cutoff = event_times[i, d]  # first event index or censoring
            at_risk_mask[i, d, : cutoff + 1] = True  # inclusive of event bin

    y_true = (probs_true > 0.5).astype(int)[at_risk_mask]
    y_pred_probs = probs_est[at_risk_mask]
    y_pred = (y_pred_probs >= threshold).astype(int)

    rmse = float(np.sqrt(np.mean((probs_true[at_risk_mask] - probs_est[at_risk_mask]) ** 2)))

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    return {
        "rmse": rmse,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def compute_diagnostics(
    phi_true: np.ndarray,
    theta_true: np.ndarray,
    probs_true: np.ndarray,
    phi_est: np.ndarray,
    theta_est: np.ndarray,
    probs_est: np.ndarray,
    event_times: np.ndarray,
) -> Dict[str, float]:
    """Return summary statistics for recovery accuracy."""
    phi_corr = [np.corrcoef(phi_true[k].ravel(), phi_est[k].ravel())[0, 1] for k in range(phi_true.shape[0])]
    theta_mse = np.mean((theta_true - theta_est) ** 2, axis=2)
    probs_rmse = float(np.sqrt(np.mean((probs_true - probs_est) ** 2)))
    at_risk_metrics = _mask_at_risk(event_times, probs_true, probs_est)

    return {
        "phi_corr": np.array(phi_corr),
        "theta_median_mse": np.median(theta_mse, axis=1),
        "probs_rmse": probs_rmse,
        "at_risk_rmse": at_risk_metrics["rmse"],
        "confusion_matrix": (at_risk_metrics["tp"], at_risk_metrics["fp"], at_risk_metrics["fn"], at_risk_metrics["tn"]),
    }


def main(plot: bool = False) -> None:
    config = LatentPathwaySimulationConfig(
        n_patients=300,
        n_sites=5,
        n_pathways=3,
        n_time=40,
    )

    simulation = simulate_metastasis_probabilities(config)
    if plot:
        plot_simulation_outputs(simulation)

    dataset = prepare_simulated_dataset(simulation)
    psi_init = hot_start_from_counts(dataset, n_pathways=config.n_pathways)

    baseline_logit_tensor = torch.tensor(dataset.baseline_logit, dtype=torch.float32)
    covariate_tensor = torch.tensor(dataset.covariates, dtype=torch.float32)

    model = MetNoulliLatentModel(
        n_patients=dataset.Y.shape[0],
        n_sites=dataset.Y.shape[1],
        n_time=dataset.Y.shape[2],
        n_pathways=config.n_pathways,
        n_covariates=covariate_tensor.shape[1],
        baseline_logit=baseline_logit_tensor,
        covariates=covariate_tensor,
        psi_init=psi_init,
    )

    batch = MetNoulliBatch(
        Y=torch.tensor(dataset.Y, dtype=torch.float32),
        event_times=torch.tensor(dataset.event_times, dtype=torch.long),
        covariates=covariate_tensor,
    )

    results = train_metnoulli_model(model, batch, n_epochs=1000, learning_rate=5e-3, verbose=True)
    print(f"Final training loss: {results['losses'][-1]:.4f}")

    # Align components
    phi_aligned, theta_aligned, psi_aligned, corr_aligned = align_components(
        simulation["phi"],
        results["phi"],
        results["theta"],
        results["psi"],
    )

    diagnostics = compute_diagnostics(
        simulation["phi"],
        simulation["theta"],
        simulation["probs"],
        phi_aligned,
        theta_aligned,
        results["probs"],
        dataset.event_times,
    )

    print("\n=== Diagnostic summary ===")
    print("Aligned φ correlations:", np.round(corr_aligned, 3))
    print("Median θ MSE per patient (first 5):", np.round(diagnostics["theta_median_mse"][:5], 4))
    print("Global probability RMSE:", round(diagnostics["probs_rmse"], 4))
    print("At-risk probability RMSE:", round(diagnostics["at_risk_rmse"], 4))

    tp, fp, fn, tn = diagnostics["confusion_matrix"]
    print(f"Confusion matrix (TP, FP, FN, TN): ({tp}, {fp}, {fn}, {tn})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MetNoulli latent pathway pipeline.")
    parser.add_argument("--plot", action="store_true", help="Show simulation plots.")
    args = parser.parse_args()

    main(plot=args.plot)

