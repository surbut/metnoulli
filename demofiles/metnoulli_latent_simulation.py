"""
MetNoulli Latent Pathway Simulation
-----------------------------------

This script synthesizes time-varying latent metastasis pathways to illustrate how
an Aladynoulli-style factorization can model metastatic cascades.

Key ideas:
    • Each latent pathway k encodes a time-evolving pattern of site involvement φ_k,s(t).
    • Each patient i expresses pathways through time-varying mixture weights θ_i,k(t).
    • Site-specific metastasis probabilities combine latent pathways and covariate effects.

Run directly to generate plots demonstrating:
    1. Latent pathway evolution across metastatic sites.
    2. Patient-specific pathway weights and resulting metastasis probabilities.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.special import softmax, expit


@dataclass
class LatentPathwaySimulationConfig:
    n_patients: int = 6
    n_sites: int = 5
    n_pathways: int = 3
    n_time: int = 40
    base_logit: float = -2.5
    covariate_scale: float = 0.6
    random_seed: int = 42
    baseline_min: float = 0.05
    baseline_max: float = 0.35


def _generate_time_grid(n_time: int) -> np.ndarray:
    return np.linspace(0, 1, n_time)  # normalized follow-up


def _make_site_baselines(config: LatentPathwaySimulationConfig) -> np.ndarray:
    """
    Create site-specific prevalence curves (probabilities) over time.

    Each site follows a scaled logistic rise with different timing and peak.
    """
    rng = np.random.default_rng(config.random_seed + 3)
    time = _generate_time_grid(config.n_time)
    baselines = np.zeros((config.n_sites, config.n_time))

    for s in range(config.n_sites):
        midpoint = rng.uniform(0.2, 0.8)
        slope = rng.uniform(6.0, 12.0)
        raw_curve = expit(slope * (time - midpoint))
        raw_curve -= raw_curve.min()
        raw_curve /= raw_curve.max() + 1e-8
        peak = rng.uniform(config.baseline_min, config.baseline_max)
        baselines[s] = np.clip(raw_curve * peak, 1e-5, 0.95)

    return baselines


def _make_pathway_signatures(config: LatentPathwaySimulationConfig) -> np.ndarray:
    """
    Construct φ_k,s(t): latent pathway log-odds adjustments across metastatic sites.

    Strategy:
        • Assign each pathway a dominant subset of target sites (with wrap-around).
        • Generate positive bumps for dominant sites and small negative deviations for others.
        • Center each time slice so effects add around zero, allowing baseline to govern level.
    """
    rng = np.random.default_rng(config.random_seed)
    time = _generate_time_grid(config.n_time)
    centers = np.linspace(0.2, 0.8, config.n_sites)
    widths = np.linspace(0.06, 0.12, config.n_sites)

    phi = np.zeros((config.n_pathways, config.n_sites, config.n_time))

    dominant_size = max(1, config.n_sites // config.n_pathways)

    for k in range(config.n_pathways):
        dominant_sites = [(k * dominant_size + j) % config.n_sites for j in range(dominant_size)]
        effects = np.zeros((config.n_sites, config.n_time))

        for site_idx in dominant_sites:
            mean = centers[site_idx] + rng.uniform(-0.05, 0.05)
            scale = widths[site_idx]
            signal = np.exp(-0.5 * ((time - mean) / scale) ** 2)
            peak = rng.uniform(1.0, 1.6)
            effects[site_idx] = peak * signal

        for site_idx in range(config.n_sites):
            if site_idx not in dominant_sites:
                mean = centers[site_idx] + rng.uniform(-0.05, 0.05)
                scale = widths[site_idx] * rng.uniform(1.5, 2.0)
                signal = np.exp(-0.5 * ((time - mean) / scale) ** 2)
                trough = rng.uniform(0.2, 0.5)
                effects[site_idx] -= trough * signal

        effects -= effects.mean(axis=0, keepdims=True)
        phi[k] = effects

    return phi


def _make_patient_weights(
    config: LatentPathwaySimulationConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct λ_i,k(t) and θ_i,k(t):
        • λ's follow smooth trajectories with patient-level random effects.
        • θ = softmax(λ) gives mixture weights over pathways.
    """
    rng = np.random.default_rng(config.random_seed + 7)
    time = _generate_time_grid(config.n_time)

    # basis functions produce smooth evolution (sin/cos plus linear trend)
    basis = np.stack(
        [
            np.ones_like(time),
            np.sin(np.pi * time),
            np.cos(np.pi * time),
            np.sin(2 * np.pi * time),
        ],
        axis=0,
    )

    coefficients = rng.normal(
        loc=0.0,
        scale=0.8,
        size=(config.n_patients, config.n_pathways, basis.shape[0]),
    )
    # encourage interpretable ordering (pathway 0 early, 1 mid, 2 late)
    timing_bias = np.array([-1.0, 0.0, 1.0])

    lam = np.zeros((config.n_patients, config.n_pathways, config.n_time))
    for i in range(config.n_patients):
        for k in range(config.n_pathways):
            lam[i, k] = coefficients[i, k] @ basis + timing_bias[k] * (time - 0.5) * 2

    theta = softmax(lam, axis=1)
    return lam, theta


def _simulate_covariates(config: LatentPathwaySimulationConfig) -> np.ndarray:
    rng = np.random.default_rng(config.random_seed + 21)
    return rng.normal(size=(config.n_patients, 2))  # two covariates


def simulate_metastasis_probabilities(
    config: LatentPathwaySimulationConfig,
) -> Dict[str, np.ndarray]:
    """
    Combine latent pathways and patient weights to produce metastasis probabilities.
    """
    rng = np.random.default_rng(config.random_seed + 99)
    sites = [f"Site_{i+1}" for i in range(config.n_sites)]
    time = _generate_time_grid(config.n_time)

    baseline_prob = _make_site_baselines(config)
    baseline_logit = np.log(baseline_prob / (1 - baseline_prob))

    phi = _make_pathway_signatures(config)
    lam, theta = _make_patient_weights(config)
    covariates = _simulate_covariates(config)

    # site-specific covariate loads and pathway sensitivities
    cov_beta = rng.normal(scale=config.covariate_scale, size=(config.n_sites, covariates.shape[1]))
    pathway_scaler = rng.normal(loc=1.0, scale=0.2, size=config.n_sites)

    # latent pathways contribution: (patients × sites × time)
    pathway_effect = np.einsum("ikt,kst->ist", theta, phi)
    pathway_effect *= pathway_scaler[None, :, None]

    # covariate contribution replicated across time
    cov_effect = covariates @ cov_beta.T  # (patients × sites)
    cov_effect = cov_effect[:, :, None]

    logits = baseline_logit[None, :, :] + config.base_logit + pathway_effect + cov_effect
    probs = expit(logits)

    return {
        "sites": np.array(sites),
        "time": time,
        "baseline_prob": baseline_prob,
        "phi": phi,
        "lam": lam,
        "theta": theta,
        "covariates": covariates,
        "logits": logits,
        "probs": probs,
    }


def plot_simulation_outputs(simulation: Dict[str, np.ndarray]) -> None:
    sites = simulation["sites"]
    time = simulation["time"]
    phi = simulation["phi"]
    theta = simulation["theta"]
    probs = simulation["probs"]
    baseline_prob = simulation["baseline_prob"]

    sns.set_style("whitegrid")
    cmap = sns.color_palette("viridis", phi.shape[1])

    # ------------------------------------------------------------------ #
    # Latent pathway signatures: one subplot per pathway
    # ------------------------------------------------------------------ #
    fig_pathways, pathway_axes = plt.subplots(
        1, phi.shape[0], figsize=(5 * phi.shape[0], 4), sharey=True
    )
    if phi.shape[0] == 1:
        pathway_axes = [pathway_axes]

    for k, ax in enumerate(pathway_axes):
        for s_idx, site in enumerate(sites):
            ax.plot(
                time,
                phi[k, s_idx],
                color=cmap[s_idx],
                label=site if k == 0 else None,
                linewidth=2,
            )
        ax.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.6)
        ax.set_title(f"Latent Pathway {k+1}")
        ax.set_xlabel("Normalized Time")
        if k == 0:
            ax.set_ylabel("Log-odds deviation φ_k,s(t)")
    pathway_axes[0].legend(loc="upper left", ncol=1)
    fig_pathways.tight_layout()

    # Baseline prevalence curves
    fig_baseline, ax_base = plt.subplots(figsize=(10, 4))
    for s_idx, site in enumerate(sites):
        ax_base.plot(time, baseline_prob[s_idx], color=cmap[s_idx], linewidth=2, label=site)
    ax_base.set_title("Site-specific Baseline Prevalence Curves")
    ax_base.set_xlabel("Normalized Time")
    ax_base.set_ylabel("Baseline metastasis probability")
    ax_base.legend(loc="upper left", ncol=2)
    fig_baseline.tight_layout()

    # ------------------------------------------------------------------ #
    # Patient-specific mixture weights (select a subset)
    # ------------------------------------------------------------------ #
    n_patients_to_plot = min(4, theta.shape[0])
    fig_patients, axes_patients = plt.subplots(
        2, 2, figsize=(12, 8), sharex=True, sharey=True
    )
    axes_patients = axes_patients.flatten()
    for idx in range(n_patients_to_plot):
        ax = axes_patients[idx]
        for k in range(theta.shape[1]):
            ax.plot(
                time,
                theta[idx, k],
                linewidth=2,
                label=f"Pathway {k+1}" if idx == 0 else None,
            )
        ax.set_title(f"Patient {idx+1}")
        ax.set_xlabel("Normalized Time")
        ax.set_ylabel("θ_i,k(t)")
    if theta.shape[0] < 4:
        for ax in axes_patients[theta.shape[0] :]:
            ax.axis("off")
    handles, labels = axes_patients[0].get_legend_handles_labels()
    fig_patients.legend(handles, labels, loc="upper center", ncol=theta.shape[1])
    fig_patients.tight_layout(rect=[0, 0, 1, 0.95])

    # ------------------------------------------------------------------ #
    # Example site probabilities across patients
    # ------------------------------------------------------------------ #
    fig_probs, ax_probs = plt.subplots(figsize=(10, 4))
    site_idx = 2  # highlight site 3
    for i in range(probs.shape[0]):
        ax_probs.plot(
            time,
            probs[i, site_idx],
            linewidth=2 if i == 0 else 1,
            alpha=0.8,
            label=f"Patient {i+1}",
        )
    ax_probs.set_title(f"Metastasis Probability Trajectories – {sites[site_idx]}")
    ax_probs.set_xlabel("Normalized Time")
    ax_probs.set_ylabel("P(metastasis)")
    ax_probs.set_ylim(0, 1)
    ax_probs.legend(loc="upper left")
    fig_probs.tight_layout()

    # ------------------------------------------------------------------ #
    # Cumulative incidence curves for highlighted site
    # ------------------------------------------------------------------ #
    fig_cum, ax_cum = plt.subplots(figsize=(10, 4))
    cumulative = probs[:, site_idx, :].copy()
    cumulative = 1 - np.cumprod(1 - cumulative, axis=1)
    for i in range(probs.shape[0]):
        ax_cum.plot(
            time,
            cumulative[i],
            linewidth=2 if i == 0 else 1,
            alpha=0.8,
            label=f"Patient {i+1}",
        )
    ax_cum.set_title(f"Cumulative Metastasis Risk – {sites[site_idx]}")
    ax_cum.set_xlabel("Normalized Time")
    ax_cum.set_ylabel("Cumulative probability")
    ax_cum.set_ylim(0, 1)
    ax_cum.legend(loc="upper left")
    fig_cum.tight_layout()

    plt.show()


def run_demo(config: LatentPathwaySimulationConfig = LatentPathwaySimulationConfig()) -> None:
    simulation = simulate_metastasis_probabilities(config)
    plot_simulation_outputs(simulation)


if __name__ == "__main__":
    run_demo()

