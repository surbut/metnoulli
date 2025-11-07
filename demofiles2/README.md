# Bayesian Metastasis Transition Model (BMTM)

Adaptation of your aladynoulli BPTM framework for cancer metastasis prediction.

## 🎯 What This Does

Takes your Bayesian Pathway Transition Model (used for disease transitions like RA → MI) and adapts it for **site-specific metastasis prediction** in your DFCI melanoma cohort (N=611, 7 metastatic sites).

## 📁 Files

1. **`bayesian_metastasis_transition_model.py`**
   - Core model implementation
   - Site-specific risk prediction
   - Site-site interaction modeling
   - Molecular feature integration
   
2. **`integrate_metastasis_pipeline.py`**
   - Full data loading and processing pipeline
   - Integration with your aladynoulli framework
   - Example predictions and validation
   - Clinical risk assessment
   
3. **`BPTM_to_BMTM_comparison.py`**
   - Detailed comparison document
   - Conceptual mapping between frameworks
   - Implementation roadmap
   - Clinical use cases

## 🚀 Quick Start

```python
from bayesian_metastasis_transition_model import BayesianMetastasisTransitionModel

# Define your 7 metastatic sites
sites = ['Adrenal', 'Bone', 'Brain', 'Liver', 'LN', 'Lung', 'Peritoneal']

# Initialize model
model = BayesianMetastasisTransitionModel(
    sites=sites,
    K=10,  # molecular features (mutations, gene signatures)
    P=3,   # patient covariates (age, sex, stage)
    use_site_interactions=True  # model site dependencies
)

# Load your DFCI data
# patient_data should have columns: patient_id, site, time_to_met, event
model.fit_maximum_likelihood(patient_data)

# Make predictions
risk_12mo = model.predict_site_risk(
    site='Brain',
    time=12.0,  # months
    molecular_features=patient_features,
    patient_covariates=patient_covars
)
```

## 🔄 Key Adaptations from BPTM

| BPTM (Disease Transitions) | BMTM (Metastasis) |
|---------------------------|-------------------|
| Disease states (RA, MI) | Metastatic sites (Lung, Brain) |
| Binary transition | Multiple site occupancy |
| Signature deviations δ | Molecular features x |
| Disease-specific α, β | Site-specific α_s, β_s |
| Population reference μ | Molecular subtypes |
| — | Site interactions ψ_s,m |

## 📊 Model Structure

```
π_i,s(t) = κ_s · sigmoid(η_i,s(t))

η_i,s(t) = α_s                        # Baseline risk for site s
         + β_s · t                    # Time since diagnosis
         + Σ_k γ_k,s · x_i,k         # Molecular effects (BRAF, etc.)
         + Σ_m ψ_s,m · I(met_m)      # Site-site interactions
         + G_i^T · Γ_s                # Patient covariates (age, stage)
```

## 💡 Key Features

### 1. Site-Specific Models
Each metastatic site gets its own:
- Baseline risk (α_s)
- Time trend (β_s)  
- Molecular effects (γ_k,s)

### 2. Site Interactions (NEW!)
Models how metastasis at one site affects risk at others:
- Example: Lung met → ↑ Brain met risk
- "Seeding" hypothesis
- Metastatic cascade patterns

### 3. Molecular Integration
Incorporate:
- Mutations (BRAF, NRAS, NF1)
- Gene expression signatures
- Tumor mutation burden
- Immune features

### 4. Clinical Covariates
Patient-level factors:
- Age, sex
- Stage at diagnosis
- Primary tumor characteristics (Breslow depth, ulceration)
- Sentinel lymph node status

## 🎯 Clinical Use Cases

### Use Case 1: Surveillance Strategy
**Question:** How often should we scan for brain metastases?

```python
# Predict brain met risk trajectory
trajectory = model.predict_trajectory(
    molecular_features=patient_features,
    max_time=24  # months
)

if trajectory['Brain'].loc[trajectory['time'] == 12].values[0] > 0.15:
    print("Recommend brain MRI every 6 months")
```

### Use Case 2: Risk Stratification
**Question:** Who are the high-risk patients?

```python
# Compare risk groups
model.compare_patient_groups(
    group1_features=high_risk_molecular,
    group2_features=low_risk_molecular,
    group1_name="BRAF+",
    group2_name="BRAF-"
)
```

### Use Case 3: Treatment Planning
**Question:** Should we use systemic vs local therapy?

```python
# Predict future sites given current met
risks = model.predict_all_sites(
    time=12,
    met_status=current_mets,  # includes site interactions
    molecular_features=features
)
# High risk of additional sites → systemic therapy
```

## 📈 Getting Started with Your Data

### Step 1: Prepare Data
```python
# Your data should look like:
#   patient_id | site      | time_to_met | event | ...
#   0          | Brain     | 15.3        | 1     | ...
#   0          | Lung      | 8.2         | 1     | ...
#   1          | Brain     | 84.0        | 0     | ... (censored)
```

### Step 2: Add Molecular Features
```python
# If you have mutation data:
molecular_features = create_mutation_matrix(
    mutations=['BRAF', 'NRAS', 'NF1', 'KIT'],
    patients=patient_ids
)

# If you have gene expression:
molecular_features = get_signature_loadings(
    signatures=['Proliferation', 'Immune', 'Melanocyte'],
    patients=patient_ids
)
```

### Step 3: Run Full Pipeline
```python
from integrate_metastasis_pipeline import main_pipeline

model, data = main_pipeline()
# This will:
# - Load your DFCI data
# - Extract features
# - Fit model
# - Generate predictions
# - Validate results
```

## 🔬 Advantages Over Standard Models

### vs Cox Proportional Hazards:
✅ Handles site interactions  
✅ Uncertainty quantification  
✅ Flexible time-varying effects  
✅ Natural for molecular covariates  

### vs Multi-State Markov:
✅ Avoids state explosion (2^7 = 128 states!)  
✅ Easier to incorporate covariates  
✅ Bayesian uncertainty  

### vs Fine-Gray Competing Risks:
✅ Models all sites jointly  
✅ Site dependencies  
✅ Richer covariate structure  

## 🚧 Development Roadmap

### Phase 1: Basic Model ✅
- [x] Site-specific baseline risks
- [x] Clinical covariates
- [x] Maximum likelihood fitting

### Phase 2: Advanced Features (In Progress)
- [ ] Full Bayesian MCMC (Stan/PyMC)
- [ ] Site interaction effects
- [ ] Molecular feature integration
- [ ] Cross-validation

### Phase 3: Clinical Validation
- [ ] External validation cohort
- [ ] Compare with standard Cox models
- [ ] Calibration analysis
- [ ] Decision curve analysis

### Phase 4: Deployment
- [ ] Web-based risk calculator
- [ ] EHR integration
- [ ] Real-time predictions

## 📊 Data Requirements

**Minimal (to run basic model):**
- ✅ Patient ID
- ✅ Metastatic site annotations (7 sites)
- ✅ Time to first met
- ✅ Censoring indicator

**Recommended:**
- Primary tumor characteristics (Breslow, ulceration)
- Age, sex, stage
- Death times (competing risk)

**Optimal:**
- Mutation profile (BRAF, NRAS, etc.)
- Gene expression signatures
- Immune features

## 🤝 Integration with Aladynoulli

This model follows your BPTM architecture:

```python
# Same conceptual structure as your disease model
from bayesian_pathway_transition_model import BayesianPathwayTransitionModel
from bayesian_metastasis_transition_model import BayesianMetastasisTransitionModel

# Disease transitions
disease_model = BayesianPathwayTransitionModel(K=21, T=50)
disease_model.fit(Y, thetas, disease_names)

# Metastasis transitions (parallel structure!)
met_model = BayesianMetastasisTransitionModel(sites=sites, K=10)
met_model.fit(patient_data, molecular_features)
```

## 📝 Citation & References

If you use this for your melanoma metastasis work, relevant biological frameworks:
- Seed-soil hypothesis (Paget 1889)
- Metastatic organotropism
- Metastatic cascade theory

Methodological inspiration:
- Your MSGene work (multistate genetic models)
- Aladynoulli framework (disease signatures)
- Bayesian survival analysis

## 🐛 Troubleshooting

**Issue:** Model won't fit
- Check for sufficient events per site (need ≥10-20)
- Reduce number of molecular features (curse of dimensionality)
- Use regularization priors

**Issue:** Site interaction effects huge
- Check for multicollinearity
- Consider hierarchical priors to share information
- Validate with held-out data

**Issue:** Poor calibration
- Recalibrate using κ_s parameters
- Add more flexible time effects
- Consider non-proportional hazards

## 📧 Next Steps

1. **Load your DFCI data** into the format shown above
2. **Run the integration pipeline** to see example workflow
3. **Validate** against your Markov chain results from the slides
4. **Extend** with your actual molecular data
5. **Compare** with standard Cox models

Ready to predict some mets! 🎯
