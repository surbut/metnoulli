"""
COMPARISON: Disease Transition (BPTM) vs Metastasis Transition (BMTM)

This document explains how the Bayesian Pathway Transition Model framework
adapts to metastatic progression modeling.
"""

# ============================================================================
# KEY CONCEPTUAL MAPPING
# ============================================================================

"""
ALADYNOULLI (Disease Transitions)          →    METASTASIS MODEL
─────────────────────────────────────────────────────────────────────────

STATES:
- Disease presence/absence                  →    Metastatic site occupancy
- Binary: has disease or not                →    Binary per site: met or no met
- Example: RA → MI                          →    Example: Lung → Brain

DATA STRUCTURE:
- Y: (N, D, T)                              →    Patient-Site-Time observations
  N patients, D diseases, T timepoints            N patients, S sites, T timepoints
- Binary disease indicators                 →    Binary metastasis indicators
- Signature deviations δ_i,k,t             →    Molecular features x_i,k

TRANSITIONS:
- Precursor → Target disease                →    Primary → Metastatic site
- One-time transition                       →    Sequential site accumulation
- Example: Diabetes → CAD                   →    Example: No met → Lung → Brain

COVARIATES:
- Signature loadings θ_i,k,t               →    Gene expression, mutations
- Population reference μ_k,t               →    Molecular subtype features
- Genetic factors G_i                      →    Clinical factors (age, stage)

MODEL STRUCTURE:
π_i(t | d_precursor) = sigmoid(η_i(t))    →    π_i,s(t) = sigmoid(η_i,s(t))

η_i(t) = α_d + β_d·t +                     →    η_i,s(t) = α_s + β_s·t +
         Σ_k γ_k·δ_i,k,t +                            Σ_k γ_k,s·x_i,k +
         Σ_τ ω_k,τ·δ_i,k,t-τ +                        Σ_m ψ_s,m·I(met_m) +
         G_i^T·Γ_d                                     G_i^T·Γ_s
"""

# ============================================================================
# DETAILED FEATURE COMPARISON
# ============================================================================

"""
╔═══════════════════════════════════════════════════════════════════════╗
║                    FEATURE-BY-FEATURE COMPARISON                       ║
╚═══════════════════════════════════════════════════════════════════════╝

1. BASELINE RISK (α)
   ────────────────────────────────────────────────────────────────────
   Disease Model:  α_d (precursor-specific)
                   Different baseline for RA→MI vs Diabetes→MI
   
   Metastasis:     α_s (site-specific)  
                   Different baseline for Lung vs Brain vs Liver
   
   Example:        α_Brain might be lower than α_Lung
                   (brain mets less common than lung)

2. TIME DEPENDENCE (β)
   ────────────────────────────────────────────────────────────────────
   Disease Model:  β_d (time since precursor)
                   Risk increases/decreases over time since RA diagnosis
   
   Metastasis:     β_s (time since primary)
                   Risk changes over time since primary diagnosis
   
   Consideration:  Early vs late metastasis patterns
                   Some sites met early, others late

3. SIGNATURE/MOLECULAR EFFECTS (γ)
   ────────────────────────────────────────────────────────────────────
   Disease Model:  γ_k: Effect of signature k on transition
                   How does immune signature affect RA→MI?
   
   Metastasis:     γ_k,s: Effect of feature k on site s
                   How does BRAF mutation affect brain metastasis?
                   How does immune signature affect liver metastasis?
   
   Examples:       - BRAF mutation → different site tropism
                   - High proliferation → faster progression
                   - Immune-cold → more aggressive spread

4. PATHWAY/SITE INTERACTIONS (NEW!)
   ────────────────────────────────────────────────────────────────────
   Disease Model:  (could have pathway-specific γ)
   
   Metastasis:     ψ_s,m: How metastasis at site m affects risk at site s
                   
   Examples:       - Lung met → increased brain met risk
                   - Lymph node met → increased systemic spread
                   - "Seeding" hypothesis: some sites facilitate others
   
   Biological:     Reflects metastatic cascade and site-site dependencies

5. TEMPORAL PATTERNS
   ────────────────────────────────────────────────────────────────────
   Disease Model:  Lagged effects ω_k,τ
                   Signature deviations τ timepoints ago
   
   Metastasis:     Could track molecular evolution
                   Clonal dynamics during progression
   
   Note:           Less relevant if using static baseline molecular features
                   More relevant with longitudinal ctDNA or imaging

6. PATIENT COVARIATES (Γ)
   ────────────────────────────────────────────────────────────────────
   Disease Model:  Genetic risk factors, demographics
                   PRS, age, sex
   
   Metastasis:     Clinical prognostic factors
                   Age, sex, stage, primary site characteristics:
                   - Tumor thickness (Breslow depth)
                   - Ulceration
                   - Mitotic rate
                   - Sentinel lymph node status
"""

# ============================================================================
# STATISTICAL CONSIDERATIONS
# ============================================================================

"""
╔═══════════════════════════════════════════════════════════════════════╗
║                    STATISTICAL ADAPTATIONS NEEDED                      ║
╚═══════════════════════════════════════════════════════════════════════╝

1. COMPETING RISKS
   ────────────────────────────────────────────────────────────────────
   Issue:          Patients can die before metastasizing
   
   Disease Model:  Censoring handled implicitly
   
   Metastasis:     MUST account for competing risk of death
                   
   Solution:       Add death state, model jointly:
                   - π_i,s(t): Risk of met at site s
                   - π_i,death(t): Risk of death
                   - Ensure: Σ_s π_i,s(t) + π_i,death(t) ≤ 1

2. MULTI-STATE DYNAMICS
   ────────────────────────────────────────────────────────────────────
   Issue:          Patients can have multiple mets simultaneously
   
   Disease Model:  Typically one transition at a time
   
   Metastasis:     Multiple sites can be occupied
                   
   States:         - No mets (healthy)
                   - 1 met (any of 7 sites)
                   - 2 mets (any combination)
                   - ...
                   - Death
   
   Approach:       Model each site independently BUT with interactions

3. TIME SCALES
   ────────────────────────────────────────────────────────────────────
   Issue:          Different natural time scales
   
   Disease Model:  Years/decades (chronic disease progression)
   
   Metastasis:     Months (your data censored at 84 months)
                   
   Implication:    Different parameterizations, faster dynamics

4. SITE DEPENDENCIES
   ────────────────────────────────────────────────────────────────────
   Issue:          Sites are not independent
   
   Examples:       - Lung met increases brain met risk (hematogenous spread)
                   - LN met changes systemic risk profile
                   - "Seed and soil" hypothesis
   
   Modeling:       Include ψ_s,m interaction terms
                   Could use graphical models for site network

5. INTERVAL CENSORING
   ────────────────────────────────────────────────────────────────────
   Issue:          Don't observe exact metastasis time
   
   Observation:    Scans at discrete timepoints
   
   Solution:       - Interval censored survival models
                   - Account for detection lag
                   - Different scan frequencies per patient
"""

# ============================================================================
# DATA REQUIREMENTS
# ============================================================================

"""
╔═══════════════════════════════════════════════════════════════════════╗
║                         DATA YOU'LL NEED                               ║
╚═══════════════════════════════════════════════════════════════════════╝

MINIMAL (for basic model):
───────────────────────────────────────────────────────────────────────
✅ Patient ID
✅ Metastatic site annotations (7 sites)
✅ Time to first met at each site
✅ Censoring indicator
✅ Death information (time, status)

RECOMMENDED (for better predictions):
───────────────────────────────────────────────────────────────────────
✅ Primary tumor characteristics:
   - Breslow thickness
   - Ulceration status
   - Mitotic rate
   - Histologic subtype
   
✅ Clinical covariates:
   - Age at diagnosis
   - Sex
   - Stage at diagnosis
   - Prior treatments

OPTIMAL (for full model):
───────────────────────────────────────────────────────────────────────
✅ Molecular features from PRIMARY tumor:
   - RNA-seq (gene expression signatures)
   - WES (mutation profile: BRAF, NRAS, NF1, KIT, etc.)
   - Copy number alterations
   - Tumor mutation burden
   - Microsatellite stability
   
✅ Immune features:
   - Tumor-infiltrating lymphocytes
   - PD-L1 expression
   - Immune gene expression signatures
   
✅ Longitudinal data:
   - ctDNA monitoring
   - Serial imaging
   - Treatment history

GOLD STANDARD (research setting):
───────────────────────────────────────────────────────────────────────
✅ Metastatic tissue profiling:
   - Matched primary-metastasis pairs
   - Site-specific molecular profiles
   - Clonal architecture evolution
"""

# ============================================================================
# IMPLEMENTATION ROADMAP
# ============================================================================

"""
╔═══════════════════════════════════════════════════════════════════════╗
║              STEP-BY-STEP IMPLEMENTATION PLAN                          ║
╚═══════════════════════════════════════════════════════════════════════╝

PHASE 1: Basic Site-Specific Models (CURRENT)
──────────────────────────────────────────────────────────────────────
✅ Load DFCI melanoma data
✅ Fit independent models per site (no interactions)
✅ Include only baseline clinical covariates
✅ Compare with standard Kaplan-Meier

Outputs: Site-specific baseline risks, time trends

PHASE 2: Add Molecular Features
──────────────────────────────────────────────────────────────────────
□ Integrate mutation data (BRAF, NRAS, etc.)
□ Add gene expression signatures if available
□ Test site-specific molecular effects (γ_k,s)
□ Validate with cross-validation

Outputs: Personalized risk predictions based on molecular profile

PHASE 3: Site Interactions
──────────────────────────────────────────────────────────────────────
□ Add site-site interaction terms (ψ_s,m)
□ Model sequential metastasis patterns
□ Identify "hub" sites that increase systemic risk
□ Test causality with temporal ordering

Outputs: Network of metastatic site dependencies

PHASE 4: Competing Risks
──────────────────────────────────────────────────────────────────────
□ Add death as competing outcome
□ Joint modeling of metastasis + death
□ Estimate treatment effects on both endpoints
□ Risk-benefit analysis

Outputs: Net benefit predictions accounting for mortality

PHASE 5: Full Bayesian Inference
──────────────────────────────────────────────────────────────────────
□ Implement MCMC (Stan/PyMC)
□ Specify informative priors
□ Estimate parameter uncertainty
□ Generate posterior predictive distributions

Outputs: Probabilistic predictions with credible intervals

PHASE 6: Clinical Validation
──────────────────────────────────────────────────────────────────────
□ External validation cohort
□ Prospective validation
□ Compare with standard models (Cox, Fine-Gray)
□ Clinical utility analysis (decision curves)
□ Cost-effectiveness

Outputs: Validated clinical prediction tool

PHASE 7: Deployment
──────────────────────────────────────────────────────────────────────
□ Web-based risk calculator
□ Integration with EHR
□ Real-time predictions
□ Monitoring dashboard

Outputs: Clinical decision support system
"""

# ============================================================================
# KEY ADVANTAGES OF BAYESIAN APPROACH
# ============================================================================

"""
╔═══════════════════════════════════════════════════════════════════════╗
║            WHY USE BAYESIAN MODEL FOR METASTASIS?                      ║
╚═══════════════════════════════════════════════════════════════════════╝

1. UNCERTAINTY QUANTIFICATION
   ────────────────────────────────────────────────────────────────────
   Standard Models:    Point estimates only
   Bayesian:           Full posterior distribution
   
   Clinical Value:     "Your 1-year brain met risk is 15% (95% CI: 8-25%)"
                       vs "Your risk is 15%" (no uncertainty)

2. INCORPORATE PRIOR KNOWLEDGE
   ────────────────────────────────────────────────────────────────────
   Standard Models:    Start from scratch each time
   Bayesian:           Use previous studies as priors
   
   Example:            Prior knowledge that BRAF mutations
                       affect brain met risk can be incorporated

3. HANDLE SMALL SAMPLES
   ────────────────────────────────────────────────────────────────────
   Standard Models:    Unreliable with rare events
   Bayesian:           Regularization through priors
   
   Your Case:          Only 611 patients across 7 sites
                       → Some site-mutation combinations rare
                       → Priors stabilize estimates

4. HIERARCHICAL STRUCTURE
   ────────────────────────────────────────────────────────────────────
   Standard Models:    Each site independent
   Bayesian:           Share information across sites
   
   Example:            If BRAF affects Brain risk,
                       inform estimates for other sites

5. SEQUENTIAL UPDATING
   ────────────────────────────────────────────────────────────────────
   Standard Models:    Refit entire model with new data
   Bayesian:           Update posterior with new observations
   
   Clinical:           Add new patients incrementally
                       Model continuously learns

6. NATURAL FOR DECISION-MAKING
   ────────────────────────────────────────────────────────────────────
   Standard Models:    P-values, confidence intervals
   Bayesian:           Probability statements
   
   Example:            "90% probability that brain surveillance
                       is beneficial for this patient"
"""

# ============================================================================
# COMPARISON WITH STANDARD METHODS
# ============================================================================

"""
╔═══════════════════════════════════════════════════════════════════════╗
║        BMTM vs TRADITIONAL SURVIVAL MODELS                             ║
╚═══════════════════════════════════════════════════════════════════════╝

Cox Proportional Hazards:
───────────────────────────────────────────────────────────────────────
Pros:   - Simple, interpretable
        - Well-understood
        - Standard in oncology
        
Cons:   - Proportional hazards assumption often violated
        - Can't easily handle site interactions
        - No uncertainty quantification
        - Separate model per site

Fine-Gray (Competing Risks):
───────────────────────────────────────────────────────────────────────
Pros:   - Handles competing risks properly
        - Cumulative incidence interpretation
        
Cons:   - Still separate per site
        - No site interactions
        - Limited to proportional subdistribution hazards

Multi-State Markov Models:
───────────────────────────────────────────────────────────────────────
Pros:   - Natural for sequential transitions
        - Can handle site interactions
        
Cons:   - State space explosion (2^7 = 128 states!)
        - Markov assumption restrictive
        - No molecular covariates easily incorporated

BMTM (Your Approach):
───────────────────────────────────────────────────────────────────────
Pros:   ✅ Handles site interactions naturally
        ✅ Incorporates molecular features
        ✅ Uncertainty quantification
        ✅ Flexible time-varying effects
        ✅ Can add pathway structure
        ✅ Shares information across sites
        
Cons:   ⚠️  More complex to fit (MCMC)
        ⚠️  Requires larger sample for full model
        ⚠️  Interpretability vs flexibility tradeoff
        ⚠️  Computational cost

WHEN TO USE BMTM:
───────────────────────────────────────────────────────────────────────
✅ Rich molecular/genomic data available
✅ Interest in site-specific predictions
✅ Need to model site dependencies
✅ Want uncertainty quantification
✅ Building clinical decision support tool
✅ Research setting with time for development
"""

# ============================================================================
# EXAMPLE USE CASES
# ============================================================================

"""
╔═══════════════════════════════════════════════════════════════════════╗
║                    CLINICAL USE CASES                                  ║
╚═══════════════════════════════════════════════════════════════════════╝

USE CASE 1: Surveillance Strategy
──────────────────────────────────────────────────────────────────────
Patient:    45F, Stage III melanoma, BRAF V600E mutation
Question:   How often should we scan for brain metastases?

Model Use:  
- Predict π_brain(t) over next 24 months
- If π_brain(12mo) > 15% → recommend brain MRI every 6 months
- If π_brain(12mo) < 5% → less frequent imaging
- Personalized to molecular profile

Value:      Avoid unnecessary imaging (cost, anxiety)
            while catching mets early when treatable

USE CASE 2: Treatment Selection
──────────────────────────────────────────────────────────────────────
Patient:    60M, oligometastatic disease (lung met only)
Question:   Should we treat with SBRT or systemic therapy?

Model Use:
- Predict π_other_sites(t | lung_met) with interactions
- If ψ_brain,lung high → risk of brain met after lung treatment
- Inform systemic vs local treatment decision

Value:      Personalized treatment based on likely progression pattern

USE CASE 3: Clinical Trial Enrichment
──────────────────────────────────────────────────────────────────────
Trial:      Testing CNS-penetrant drug for brain met prevention
Question:   Who should we enroll?

Model Use:
- Predict π_brain(t) for all patients
- Enrich trial with high-risk patients (π_brain > threshold)
- Smaller sample size needed
- Higher event rate → faster trial

Value:      More efficient trials, faster drug development

USE CASE 4: Patient Counseling
──────────────────────────────────────────────────────────────────────
Patient:    35F, newly diagnosed Stage IIB
Question:   What's my prognosis?

Model Use:
- Generate personalized risk trajectory
- Show most likely sites with timeline
- Uncertainty bounds for informed decision-making

Value:      Informed consent, treatment planning, psychological preparation
"""

# ============================================================================
# NEXT STEPS FOR YOUR DATA
# ============================================================================

"""
╔═══════════════════════════════════════════════════════════════════════╗
║              IMMEDIATE NEXT STEPS                                      ║
╚═══════════════════════════════════════════════════════════════════════╝

1. DATA AUDIT
   ──────────────────────────────────────────────────────────────────
   □ Check data completeness for 611 patients
   □ Verify site annotations (7 sites)
   □ Confirm censoring times (84 months max)
   □ Identify missing data patterns
   □ Check for time-varying covariates

2. BASELINE ANALYSIS
   ──────────────────────────────────────────────────────────────────
   □ Kaplan-Meier curves per site
   □ Cumulative incidence accounting for death
   □ Site-specific event rates over time
   □ Stratify by stage, BRAF status if available

3. FEATURE ENGINEERING
   ──────────────────────────────────────────────────────────────────
   □ Create molecular feature matrix
   □ Code mutations as binary indicators
   □ Normalize continuous features
   □ Handle missing molecular data (imputation?)

4. MODEL FITTING
   ──────────────────────────────────────────────────────────────────
   □ Start with simple model (no interactions)
   □ Fit site-specific baseline risks
   □ Add one covariate at a time
   □ Check convergence diagnostics

5. VALIDATION
   ──────────────────────────────────────────────────────────────────
   □ Split data (training/test)
   □ Cross-validation
   □ Calibration plots
   □ Discrimination (C-index per site)
   □ Compare with Cox model baseline

6. VISUALIZATION
   ──────────────────────────────────────────────────────────────────
   □ Risk trajectories by patient subgroup
   □ Site interaction network
   □ Molecular effect sizes
   □ Uncertainty intervals

7. MANUSCRIPT/PRESENTATION
   ──────────────────────────────────────────────────────────────────
   □ Methods description
   □ Comparison with Markov chain results from slides
   □ Clinical implications
   □ Limitations and future directions
"""

if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*80)
    print("This comparison document explains the adaptation of BPTM to metastasis modeling")
    print("See bayesian_metastasis_transition_model.py for implementation")
    print("See integrate_metastasis_pipeline.py for data integration")
    print("="*80)
