"""
Integration: Bayesian Metastasis Model with aladynoulli Pipeline

This script shows how to:
1. Load DFCI melanoma metastasis data
2. Extract molecular signatures (if available)
3. Fit the Bayesian Metastasis Transition Model
4. Make site-specific predictions
5. Compare with existing Markov chain results
"""

import numpy as np
import pandas as pd
import torch
import sys
import os
from typing import Dict, List, Optional

# Assuming aladynoulli is in path
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision')

from bayesian_metastasis_transition_model import BayesianMetastasisTransitionModel


def load_dfci_metastasis_data(
    data_path: str = None
) -> Dict:
    """
    Load DFCI melanoma metastasis data.
    
    Expected format:
    - patient_id: unique patient identifier
    - site: metastatic site (Adrenal, Bone, Brain, Liver, LN, Lung, Peritoneal)
    - time_to_met: time to first metastasis at site (months)
    - event: 1 if metastasis occurred, 0 if censored
    - baseline_time: time 0 for patient
    
    Returns:
    --------
    data_dict : Dict
        Contains patient_data, sites, and optional molecular features
    """
    print("="*80)
    print("LOADING DFCI MELANOMA METASTASIS DATA")
    print("="*80)
    
    # TODO: Replace with actual data loading
    # For now, create example structure
    
    print("\n⚠️  Using synthetic data structure as placeholder")
    print("   Replace this with actual DFCI data loading\n")
    
    sites = ['Adrenal', 'Bone', 'Brain', 'Liver', 'LN', 'Lung', 'Peritoneal']
    n_patients = 611
    
    # Example: Create patient-level data structure
    patient_records = []
    
    for patient_id in range(n_patients):
        for site in sites:
            # Synthetic data - replace with actual annotations
            if np.random.rand() < 0.25:  # ~25% met rate per site
                time_to_met = np.random.exponential(20) + 1  # months
                time_to_met = min(time_to_met, 84)  # censor at 84 months
                event = 1 if time_to_met < 84 else 0
            else:
                time_to_met = 84  # censored at 84 months
                event = 0
            
            patient_records.append({
                'patient_id': patient_id,
                'site': site,
                'time_to_met': time_to_met,
                'event': event
            })
    
    patient_data = pd.DataFrame(patient_records)
    
    print(f"✅ Loaded data:")
    print(f"   Patients: {n_patients}")
    print(f"   Sites: {len(sites)}")
    print(f"   Total records: {len(patient_data)}")
    print(f"   Events: {patient_data['event'].sum()}")
    
    # Event rates by site
    print("\n📊 Event rates by site:")
    for site in sites:
        site_data = patient_data[patient_data['site'] == site]
        event_rate = site_data['event'].mean()
        n_events = site_data['event'].sum()
        print(f"   {site:12s}: {event_rate:.2%} ({n_events}/{len(site_data)})")
    
    return {
        'patient_data': patient_data,
        'sites': sites,
        'n_patients': n_patients
    }


def integrate_with_molecular_signatures(
    patient_data: pd.DataFrame,
    signature_path: str = None
) -> Optional[np.ndarray]:
    """
    Integrate molecular signatures (gene expression, mutations, etc.)
    
    This could come from:
    1. Gene expression signatures (like your disease signatures)
    2. Mutation profiles
    3. Copy number alterations
    4. Other molecular features
    
    Returns:
    --------
    molecular_features : np.ndarray
        Molecular features per patient (N, K)
    """
    print("\n" + "="*80)
    print("LOADING MOLECULAR SIGNATURES")
    print("="*80)
    
    # TODO: Load actual molecular data
    # Could be from RNA-seq, WES, targeted panels, etc.
    
    print("\n⚠️  No molecular features loaded")
    print("   Model will use baseline + clinical covariates only\n")
    
    # For demonstration: create synthetic features
    n_patients = patient_data['patient_id'].nunique()
    K = 10  # number of molecular features
    
    # Example features that might be relevant for metastasis:
    # - Gene expression signatures (proliferation, immune, etc.)
    # - Key mutations (BRAF, NRAS, KIT, etc.)
    # - Tumor mutation burden
    # - PD-L1 expression
    # - etc.
    
    molecular_features = np.random.randn(n_patients, K)
    
    print(f"✅ Generated {K} synthetic molecular features")
    print(f"   Shape: {molecular_features.shape}")
    
    return molecular_features


def extract_patient_covariates(
    patient_data: pd.DataFrame
) -> np.ndarray:
    """
    Extract patient-level covariates (age, sex, stage, etc.)
    """
    print("\n" + "="*80)
    print("EXTRACTING PATIENT COVARIATES")
    print("="*80)
    
    n_patients = patient_data['patient_id'].nunique()
    
    # TODO: Load actual patient characteristics
    # Examples:
    # - Age at diagnosis
    # - Sex
    # - Stage at diagnosis
    # - Primary site thickness
    # - Ulceration
    # - Mitotic rate
    
    # For now: synthetic covariates
    age = np.random.normal(60, 15, size=n_patients)  # years
    sex = np.random.binomial(1, 0.5, size=n_patients)  # 0=F, 1=M
    stage = np.random.choice([2, 3, 4], size=n_patients, p=[0.3, 0.4, 0.3])
    
    patient_covariates = np.column_stack([age, sex, stage])
    
    print(f"✅ Extracted {patient_covariates.shape[1]} patient covariates:")
    print(f"   - Age (mean: {age.mean():.1f})")
    print(f"   - Sex (% male: {sex.mean():.1%})")
    print(f"   - Stage distribution: {np.bincount(stage.astype(int))}")
    
    return patient_covariates


def fit_metastasis_model(
    patient_data: pd.DataFrame,
    sites: List[str],
    molecular_features: Optional[np.ndarray] = None,
    patient_covariates: Optional[np.ndarray] = None,
    use_site_interactions: bool = True
) -> BayesianMetastasisTransitionModel:
    """
    Fit the Bayesian Metastasis Transition Model.
    """
    print("\n" + "="*80)
    print("FITTING BAYESIAN METASTASIS TRANSITION MODEL")
    print("="*80)
    
    # Initialize model
    K = molecular_features.shape[1] if molecular_features is not None else 0
    P = patient_covariates.shape[1] if patient_covariates is not None else 0
    
    model = BayesianMetastasisTransitionModel(
        sites=sites,
        K=K,
        P=P,
        use_site_interactions=use_site_interactions
    )
    
    print(f"\n✅ Model initialized:")
    print(f"   Sites: {len(sites)}")
    print(f"   Molecular features: {K}")
    print(f"   Patient covariates: {P}")
    print(f"   Site interactions: {use_site_interactions}")
    
    # Fit model
    print("\n📈 Fitting model...")
    model.fit_maximum_likelihood(
        patient_data,
        molecular_features=molecular_features,
        patient_covariates=patient_covariates,
        verbose=True
    )
    
    print("\n✅ Model fitted successfully")
    
    return model


def analyze_site_specific_risks(
    model: BayesianMetastasisTransitionModel,
    patient_data: pd.DataFrame,
    molecular_features: Optional[np.ndarray] = None,
    patient_covariates: Optional[np.ndarray] = None
):
    """
    Analyze site-specific metastasis risks and patterns.
    """
    print("\n" + "="*80)
    print("SITE-SPECIFIC RISK ANALYSIS")
    print("="*80)
    
    # 1. Baseline risks by site
    print("\n1️⃣ Baseline risks (logit scale):")
    for site in model.sites:
        print(f"   {site:12s}: α = {model.alpha[site]:6.3f}, β = {model.beta[site]:7.4f}")
    
    # 2. Predicted risks at different time points
    print("\n2️⃣ Predicted risks at different time points:")
    time_points = [6, 12, 24, 36, 60]  # months
    
    # Use median patient as example
    n_patients = patient_data['patient_id'].nunique()
    median_idx = n_patients // 2
    
    example_features = molecular_features[median_idx, :] if molecular_features is not None else None
    example_covariates = patient_covariates[median_idx, :] if patient_covariates is not None else None
    
    for t in time_points:
        print(f"\n   At {t} months:")
        risks = model.predict_all_sites(
            time=t,
            molecular_features=example_features,
            patient_covariates=example_covariates
        )
        for site, risk in risks.items():
            print(f"     {site:12s}: {risk:.4f}")
    
    # 3. Site-site interaction patterns (if enabled)
    if model.use_site_interactions:
        print("\n3️⃣ Site-site interaction effects:")
        print("   (How metastasis at one site affects risk at another)")
        
        for target_site in model.sites[:3]:  # show first 3 as example
            print(f"\n   Target site: {target_site}")
            psi_effects = model.psi[target_site]
            for i, source_site in enumerate(model.sites):
                if abs(psi_effects[i]) > 0.01:  # show non-trivial effects
                    print(f"     {source_site:12s} → {target_site}: {psi_effects[i]:+.3f}")


def compare_risk_groups(
    model: BayesianMetastasisTransitionModel,
    molecular_features: Optional[np.ndarray] = None,
    patient_covariates: Optional[np.ndarray] = None
):
    """
    Compare metastasis trajectories between high-risk and low-risk groups.
    """
    print("\n" + "="*80)
    print("COMPARING RISK GROUPS")
    print("="*80)
    
    if molecular_features is None:
        print("\n⚠️  No molecular features available")
        print("   Using random patient groups for demonstration\n")
        n_patients = 100
        molecular_features = np.random.randn(n_patients, 10)
    
    # Define risk groups based on some criterion
    # Example: split by first principal component of molecular features
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(molecular_features).flatten()
    
    # High vs low risk based on PC1
    high_risk_idx = pc1 > np.median(pc1)
    low_risk_idx = ~high_risk_idx
    
    high_risk_features = molecular_features[high_risk_idx, :]
    low_risk_features = molecular_features[low_risk_idx, :]
    
    print(f"✅ Identified risk groups:")
    print(f"   High risk: {high_risk_idx.sum()} patients")
    print(f"   Low risk: {low_risk_idx.sum()} patients")
    
    # Compare trajectories
    model.compare_patient_groups(
        group1_features=high_risk_features,
        group2_features=low_risk_features,
        group1_name="High Risk",
        group2_name="Low Risk",
        max_time=60.0
    )


def validate_predictions(
    model: BayesianMetastasisTransitionModel,
    patient_data: pd.DataFrame,
    molecular_features: Optional[np.ndarray] = None,
    patient_covariates: Optional[np.ndarray] = None
):
    """
    Validate model predictions against observed data.
    """
    print("\n" + "="*80)
    print("MODEL VALIDATION")
    print("="*80)
    
    # For each site, compare predicted vs observed prevalence over time
    print("\n📊 Predicted vs Observed Prevalence by Site:")
    
    time_bins = [(0, 12), (12, 24), (24, 36), (36, 60), (60, 84)]
    
    for site in model.sites:
        print(f"\n{site}:")
        site_data = patient_data[patient_data['site'] == site]
        
        for t_start, t_end in time_bins:
            # Observed
            in_window = (site_data['time_to_met'] >= t_start) & (site_data['time_to_met'] < t_end)
            obs_events = (in_window & (site_data['event'] == 1)).sum()
            obs_total = in_window.sum()
            obs_prev = obs_events / obs_total if obs_total > 0 else 0
            
            # Predicted (at midpoint)
            t_mid = (t_start + t_end) / 2
            # Use average patient characteristics
            pred_risk = model.predict_site_risk(site, time=t_mid)
            
            print(f"  [{t_start:2d}-{t_end:2d}m]: Obs={obs_prev:.3f}, Pred={pred_risk:.3f}, " +
                  f"N={obs_total}")


def main_pipeline():
    """
    Main integration pipeline for metastasis modeling.
    """
    print("\n" + "="*80)
    print("🔬 BAYESIAN METASTASIS MODELING PIPELINE")
    print("="*80)
    print("\nAdapting aladynoulli framework for cancer metastasis prediction")
    print("DFCI Melanoma Cohort (N=611)")
    print("="*80)
    
    # Step 1: Load data
    data_dict = load_dfci_metastasis_data()
    patient_data = data_dict['patient_data']
    sites = data_dict['sites']
    
    # Step 2: Load molecular features
    molecular_features = integrate_with_molecular_signatures(patient_data)
    
    # Step 3: Extract patient covariates
    patient_covariates = extract_patient_covariates(patient_data)
    
    # Step 4: Fit model
    model = fit_metastasis_model(
        patient_data,
        sites,
        molecular_features=molecular_features,
        patient_covariates=patient_covariates,
        use_site_interactions=True
    )
    
    # Step 5: Analyze site-specific risks
    analyze_site_specific_risks(
        model,
        patient_data,
        molecular_features=molecular_features,
        patient_covariates=patient_covariates
    )
    
    # Step 6: Compare risk groups
    compare_risk_groups(
        model,
        molecular_features=molecular_features,
        patient_covariates=patient_covariates
    )
    
    # Step 7: Validate predictions
    validate_predictions(
        model,
        patient_data,
        molecular_features=molecular_features,
        patient_covariates=patient_covariates
    )
    
    # Step 8: Generate predictions for clinical use
    print("\n" + "="*80)
    print("CLINICAL PREDICTION EXAMPLE")
    print("="*80)
    
    # Example: New patient
    new_patient_features = molecular_features[0, :]
    new_patient_covariates = patient_covariates[0, :]
    
    print("\n🏥 Risk assessment for new patient:")
    print("\n1-year risks:")
    risks_1yr = model.predict_all_sites(
        time=12,
        molecular_features=new_patient_features,
        patient_covariates=new_patient_covariates
    )
    
    # Sort by risk
    sorted_risks = sorted(risks_1yr.items(), key=lambda x: x[1], reverse=True)
    for site, risk in sorted_risks:
        risk_pct = risk * 100
        bar = '█' * int(risk_pct / 2)
        print(f"  {site:12s}: {risk_pct:5.2f}% {bar}")
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE")
    print("="*80)
    
    print("\n📝 Next steps for production:")
    print("1. Load actual DFCI melanoma data")
    print("2. Incorporate real molecular features (mutations, expression)")
    print("3. Add competing risks (death without metastasis)")
    print("4. Implement full Bayesian MCMC for uncertainty quantification")
    print("5. Cross-validate on held-out patients")
    print("6. Compare with standard Cox models")
    print("7. Develop clinical decision support tool")
    
    return model, patient_data


if __name__ == "__main__":
    model, data = main_pipeline()
    
    print("\n✅ Model object and data returned for further analysis")
