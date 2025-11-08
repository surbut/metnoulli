"""
Bayesian Metastasis Transition Model (BMTM)

Adaptation of the Bayesian Pathway Transition Model for metastatic progression.
Predicts site-specific metastasis risk based on molecular signatures and patient characteristics.

Model:
  π_i,s(t) = κ_s · sigmoid(η_i,s(t))
  
  η_i,s(t) = α_s + β_s · t + 
             Σ_k γ_k,s · x_i,k +               # Signature/molecular effects
             Σ_m ψ_s,m · I(met_m present) +    # Site-site interactions
             G_i^T · Γ_s                        # Genetic/clinical effects

where:
  - s indexes metastatic sites
  - x_i,k are molecular features (e.g., gene expression, mutations)
  - I(met_m present) indicates presence of metastasis at site m
  - G_i are patient-level covariates (age, sex, etc.)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, beta as beta_dist
from scipy.special import expit  # sigmoid function
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class BayesianMetastasisTransitionModel:
    """
    Bayesian model for site-specific metastasis prediction.
    
    Key features:
    1. Site-specific baseline risks
    2. Molecular signature effects per site
    3. Site-site interaction effects (metastasis at one site affects risk at others)
    4. Time-dependent progression
    5. Patient-level covariates
    """
    
    def __init__(
        self,
        sites: List[str],  # Metastatic site names
        K: int = 0,  # Number of molecular features (e.g., gene signatures)
        P: int = 0,  # Number of patient-level covariates
        use_site_interactions: bool = True,  # Model site-site dependencies
        device: str = 'cpu'
    ):
        """
        Initialize the Bayesian Metastasis Transition Model.
        
        Parameters:
        -----------
        sites : List[str]
            Names of metastatic sites (e.g., ['Bone', 'Brain', 'Liver', ...])
        K : int
            Number of molecular features/signatures
        P : int
            Number of patient covariates (age, sex, etc.)
        use_site_interactions : bool
            Whether to model dependencies between metastatic sites
        device : str
            Device for computation
        """
        self.sites = sites
        self.n_sites = len(sites)
        self.K = K
        self.P = P
        self.use_site_interactions = use_site_interactions
        self.device = device
        
        # Site name to index mapping
        self.site_to_idx = {site: i for i, site in enumerate(sites)}
        
        # Model parameters (to be learned via MCMC)
        self.alpha = {}  # Baseline log-odds per site: {site: float}
        self.beta = {}   # Time effect per site: {site: float}
        self.gamma = {}  # Molecular effects per site: {site: np.ndarray(K,)}
        self.psi = {}    # Site-site interaction matrix: {site: np.ndarray(n_sites,)}
        self.Gamma = {}  # Patient covariate effects per site: {site: np.ndarray(P,)}
        self.kappa = {}  # Site-specific calibration: {site: float}
        
        # Initialize with default values
        for site in sites:
            self.alpha[site] = 0.0
            self.beta[site] = 0.0
            self.kappa[site] = 1.0
            if K > 0:
                self.gamma[site] = np.zeros(K)
            if use_site_interactions:
                self.psi[site] = np.zeros(self.n_sites)
            if P > 0:
                self.Gamma[site] = np.zeros(P)
    
    def compute_transition_logit(
        self,
        site: str,
        time: float,
        molecular_features: Optional[np.ndarray] = None,
        met_status: Optional[np.ndarray] = None,
        patient_covariates: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute transition logit η_i,s(t) for metastasis to a specific site.
        
        Parameters:
        -----------
        site : str
            Target metastatic site
        time : float
            Time since diagnosis or baseline
        molecular_features : np.ndarray, optional
            Molecular signature values (K,)
        met_status : np.ndarray, optional
            Binary vector of metastasis presence at each site (n_sites,)
        patient_covariates : np.ndarray, optional
            Patient-level covariates (P,)
            
        Returns:
        --------
        logit : float
            Transition logit η_i,s(t)
        """
        # Baseline + time trend
        logit = self.alpha[site] + self.beta[site] * time
        
        # Molecular signature effects
        if molecular_features is not None and self.K > 0:
            logit += np.dot(self.gamma[site], molecular_features)
        
        # Site-site interactions (how existing mets affect risk at this site)
        if met_status is not None and self.use_site_interactions:
            logit += np.dot(self.psi[site], met_status)
        
        # Patient covariates
        if patient_covariates is not None and self.P > 0:
            logit += np.dot(self.Gamma[site], patient_covariates)
        
        return logit
    
    def predict_site_risk(
        self,
        site: str,
        time: float,
        molecular_features: Optional[np.ndarray] = None,
        met_status: Optional[np.ndarray] = None,
        patient_covariates: Optional[np.ndarray] = None
    ) -> float:
        """
        Predict probability of metastasis to a specific site.
        
        Parameters:
        -----------
        site : str
            Target metastatic site
        time : float
            Time since diagnosis
        molecular_features : np.ndarray, optional
            Molecular features (K,)
        met_status : np.ndarray, optional
            Current metastasis status (n_sites,)
        patient_covariates : np.ndarray, optional
            Patient covariates (P,)
            
        Returns:
        --------
        probability : float
            Predicted probability of metastasis to site
        """
        logit = self.compute_transition_logit(
            site, time, molecular_features, met_status, patient_covariates
        )
        
        # Apply calibration and sigmoid
        probability = self.kappa[site] * expit(logit)
        
        return probability
    
    def predict_all_sites(
        self,
        time: float,
        molecular_features: Optional[np.ndarray] = None,
        met_status: Optional[np.ndarray] = None,
        patient_covariates: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Predict metastasis risk for all sites simultaneously.
        
        Returns:
        --------
        site_risks : Dict[str, float]
            Predicted risk for each site
        """
        site_risks = {}
        for site in self.sites:
            site_risks[site] = self.predict_site_risk(
                site, time, molecular_features, met_status, patient_covariates
            )
        return site_risks
    
    def predict_trajectory(
        self,
        molecular_features: Optional[np.ndarray] = None,
        patient_covariates: Optional[np.ndarray] = None,
        max_time: float = 84.0,  # months
        time_steps: int = 50
    ) -> pd.DataFrame:
        """
        Predict metastasis risk trajectory over time for all sites.
        
        Parameters:
        -----------
        molecular_features : np.ndarray, optional
            Patient molecular features (K,)
        patient_covariates : np.ndarray, optional
            Patient covariates (P,)
        max_time : float
            Maximum follow-up time (months)
        time_steps : int
            Number of time points to evaluate
            
        Returns:
        --------
        trajectory_df : pd.DataFrame
            Risk predictions over time (columns: time, site1, site2, ...)
        """
        times = np.linspace(0, max_time, time_steps)
        trajectories = {site: [] for site in self.sites}
        trajectories['time'] = times
        
        # Initialize met status (no mets at start)
        met_status = np.zeros(self.n_sites)
        
        for t in times:
            site_risks = self.predict_all_sites(
                t, molecular_features, met_status, patient_covariates
            )
            
            for site, risk in site_risks.items():
                trajectories[site].append(risk)
        
        return pd.DataFrame(trajectories)
    
    def fit_maximum_likelihood(
        self,
        patient_data: pd.DataFrame,
        molecular_features: Optional[np.ndarray] = None,
        patient_covariates: Optional[np.ndarray] = None,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        verbose: bool = True
    ):
        """
        Fit model using maximum likelihood estimation (simplified version).
        
        Parameters:
        -----------
        patient_data : pd.DataFrame
            Columns: patient_id, site, time_to_met, event (1=met, 0=censored)
        molecular_features : np.ndarray, optional
            Molecular features per patient (N, K)
        patient_covariates : np.ndarray, optional
            Patient covariates (N, P)
        learning_rate : float
            Learning rate for gradient descent
        max_iter : int
            Maximum iterations
        verbose : bool
            Print progress
        """
        # Site-specific models
        for site in self.sites:
            site_data = patient_data[patient_data['site'] == site]
            
            if len(site_data) == 0:
                continue
            
            # Empirical baseline (proportion with metastasis)
            n_events = site_data['event'].sum()
            n_total = len(site_data)
            
            if n_events > 0:
                # Initialize with empirical log-odds
                self.alpha[site] = np.log(n_events / (n_total - n_events + 1))
            else:
                self.alpha[site] = -2.0  # Low baseline
            
            # Estimate time effect (simple regression)
            if n_events > 5:
                events = site_data[site_data['event'] == 1]
                avg_time = events['time_to_met'].mean()
                self.beta[site] = 0.01 / avg_time if avg_time > 0 else 0.01
            
            if verbose:
                print(f"Site: {site}")
                print(f"  Events: {n_events}/{n_total}")
                print(f"  α = {self.alpha[site]:.3f}")
                print(f"  β = {self.beta[site]:.4f}")
        
        # TODO: Full gradient descent for molecular and interaction effects
        # For now, initialize randomly
        if molecular_features is not None:
            for site in self.sites:
                self.gamma[site] = np.random.normal(0, 0.1, size=self.K)
        
        if self.use_site_interactions:
            for site in self.sites:
                self.psi[site] = np.random.normal(0, 0.05, size=self.n_sites)
        
        if patient_covariates is not None:
            for site in self.sites:
                self.Gamma[site] = np.random.normal(0, 0.1, size=self.P)
    
    def fit_bayesian_mcmc(
        self,
        patient_data: pd.DataFrame,
        molecular_features: Optional[np.ndarray] = None,
        patient_covariates: Optional[np.ndarray] = None,
        n_iterations: int = 5000,
        burn_in: int = 2000,
        thin: int = 5,
        verbose: bool = True
    ):
        """
        Fit model using MCMC sampling (placeholder for full Bayesian inference).
        
        This would use Stan, PyMC, or custom Gibbs/Metropolis-Hastings.
        """
        print("Full MCMC implementation coming soon!")
        print("Using maximum likelihood for now...")
        
        # Fall back to ML estimation
        self.fit_maximum_likelihood(
            patient_data, molecular_features, patient_covariates, verbose=verbose
        )
    
    def plot_site_trajectories(
        self,
        molecular_features: Optional[np.ndarray] = None,
        patient_covariates: Optional[np.ndarray] = None,
        max_time: float = 84.0,
        save_path: Optional[str] = None
    ):
        """
        Plot predicted metastasis risk trajectories for all sites.
        """
        trajectory_df = self.predict_trajectory(
            molecular_features, patient_covariates, max_time=max_time
        )
        
        plt.figure(figsize=(12, 6))
        
        for site in self.sites:
            plt.plot(trajectory_df['time'], trajectory_df[site], 
                    label=site, linewidth=2)
        
        plt.xlabel('Time (months)', fontsize=12)
        plt.ylabel('Metastasis Risk', fontsize=12)
        plt.title('Predicted Metastasis Risk Trajectories by Site', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_patient_groups(
        self,
        group1_features: np.ndarray,
        group2_features: np.ndarray,
        group1_name: str = "Group 1",
        group2_name: str = "Group 2",
        max_time: float = 84.0
    ):
        """
        Compare metastasis trajectories between two patient groups.
        """
        # Average features for each group
        avg_features1 = np.mean(group1_features, axis=0) if len(group1_features.shape) > 1 else group1_features
        avg_features2 = np.mean(group2_features, axis=0) if len(group2_features.shape) > 1 else group2_features
        
        # Predict trajectories
        traj1 = self.predict_trajectory(molecular_features=avg_features1, max_time=max_time)
        traj2 = self.predict_trajectory(molecular_features=avg_features2, max_time=max_time)
        
        # Plot comparison
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, site in enumerate(self.sites):
            if i < len(axes):
                axes[i].plot(traj1['time'], traj1[site], label=group1_name, linewidth=2)
                axes[i].plot(traj2['time'], traj2[site], label=group2_name, linewidth=2)
                axes[i].set_title(site, fontsize=12)
                axes[i].set_xlabel('Time (months)')
                axes[i].set_ylabel('Risk')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def example_melanoma_usage():
    """
    Example usage for melanoma metastasis modeling.
    """
    print("="*80)
    print("BAYESIAN METASTASIS TRANSITION MODEL - MELANOMA EXAMPLE")
    print("="*80)
    
    # Define metastatic sites (from your DFCI cohort)
    sites = ['Adrenal', 'Bone', 'Brain', 'Liver', 'LN', 'Lung', 'Peritoneal']
    
    # Initialize model
    K = 10  # Number of molecular features (e.g., gene expression signatures)
    P = 3   # Patient covariates (age, sex, stage)
    
    model = BayesianMetastasisTransitionModel(
        sites=sites,
        K=K,
        P=P,
        use_site_interactions=True
    )
    
    print(f"\n✅ Model initialized for {len(sites)} metastatic sites")
    print(f"   Sites: {', '.join(sites)}")
    print(f"   Molecular features: {K}")
    print(f"   Patient covariates: {P}")
    
    # Create synthetic patient data for demonstration
    print("\n" + "-"*80)
    print("GENERATING SYNTHETIC DATA")
    print("-"*80)
    
    n_patients = 611  # Your cohort size
    
    # Synthetic patient data
    patient_records = []
    for patient_id in range(n_patients):
        # Each patient may have multiple site observations
        for site in sites:
            # Simulate time to metastasis (exponential distribution)
            if np.random.rand() < 0.3:  # 30% develop met at this site
                time_to_met = np.random.exponential(24)  # months
                event = 1
            else:
                time_to_met = np.random.uniform(60, 84)  # censored
                event = 0
            
            patient_records.append({
                'patient_id': patient_id,
                'site': site,
                'time_to_met': time_to_met,
                'event': event
            })
    
    patient_data = pd.DataFrame(patient_records)
    
    # Synthetic molecular features (gene expression, mutations, etc.)
    molecular_features = np.random.randn(n_patients, K)
    
    # Synthetic patient covariates (age, sex, etc.)
    patient_covariates = np.random.randn(n_patients, P)
    
    print(f"✅ Created synthetic data:")
    print(f"   Patients: {n_patients}")
    print(f"   Total observations: {len(patient_data)}")
    print(f"   Events: {patient_data['event'].sum()}")
    
    # Fit model
    print("\n" + "-"*80)
    print("FITTING MODEL")
    print("-"*80)
    
    model.fit_maximum_likelihood(patient_data, molecular_features, patient_covariates)
    
    print("\n✅ Model fitted")
    
    # Make predictions for an example patient
    print("\n" + "-"*80)
    print("EXAMPLE PREDICTIONS")
    print("-"*80)
    
    example_patient_features = molecular_features[0, :]
    example_patient_covariates = patient_covariates[0, :]
    
    # Predict at 12 months
    risks_12mo = model.predict_all_sites(
        time=12.0,
        molecular_features=example_patient_features,
        patient_covariates=example_patient_covariates
    )
    
    print("\nPredicted metastasis risk at 12 months:")
    for site, risk in risks_12mo.items():
        print(f"  {site:12s}: {risk:.4f}")
    
    # Predict trajectory over time
    print("\n" + "-"*80)
    print("TRAJECTORY PREDICTION")
    print("-"*80)
    
    trajectory = model.predict_trajectory(
        molecular_features=example_patient_features,
        patient_covariates=example_patient_covariates,
        max_time=60.0
    )
    
    print(f"\n✅ Predicted trajectory (shape: {trajectory.shape})")
    print(trajectory.head())
    
    # Plot trajectories
    print("\n" + "-"*80)
    print("PLOTTING TRAJECTORIES")
    print("-"*80)
    
    model.plot_site_trajectories(
        molecular_features=example_patient_features,
        patient_covariates=example_patient_covariates,
        max_time=60.0
    )
    
    print("\n" + "="*80)
    print("✅ EXAMPLE COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Load real DFCI melanoma data")
    print("2. Incorporate actual molecular features (mutations, gene expression)")
    print("3. Implement full Bayesian MCMC")
    print("4. Add competing risks (death)")
    print("5. Validate predictions with held-out data")
    print("6. Compare with standard survival models")


if __name__ == "__main__":
    example_melanoma_usage()
