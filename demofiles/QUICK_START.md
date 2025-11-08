# 🚀 Quick Start Guide: Running the Metastasis Model

## Option 1: Jupyter Notebook (Interactive) ⭐ RECOMMENDED

### Local Setup
```bash
# 1. Download the notebook
# demo_bmtm_notebook.ipynb

# 2. Make sure you have the model file in the same directory
# bayesian_metastasis_transition_model.py

# 3. Launch Jupyter
jupyter notebook

# 4. Open demo_bmtm_notebook.ipynb

# 5. Run each cell sequentially (Shift+Enter)
```

### Google Colab (No setup needed!)
```bash
# 1. Upload both files to Colab:
#    - demo_bmtm_notebook.ipynb
#    - bayesian_metastasis_transition_model.py

# 2. Run cells one by one

# That's it!
```

## Option 2: Python Script (All at once)

```bash
# Run the complete demo
python demo_bmtm_full_pipeline.py

# This will:
# - Generate synthetic data
# - Fit the model
# - Create visualizations
# - Save 3 PNG files
```

## What You'll Get

### The Notebook Gives You:
✅ **Interactive exploration** - run cells individually
✅ **See outputs inline** - tables, plots, predictions
✅ **Experiment easily** - change parameters and re-run
✅ **Learn step-by-step** - detailed explanations
✅ **Modify on the fly** - tweak data generation, model settings

### The Script Gives You:
✅ **Quick results** - everything runs automatically
✅ **Batch processing** - good for testing different configs
✅ **Production ready** - clean, organized output

## 📁 Files You Need

**Required:**
- `bayesian_metastasis_transition_model.py` - The model class
- `demo_bmtm_notebook.ipynb` OR `demo_bmtm_full_pipeline.py`

**Optional (for reference):**
- `integrate_metastasis_pipeline.py` - Integration with your data
- `BPTM_to_BMTM_comparison.py` - Framework comparison
- `README.md` - Full documentation

## 🎯 What Each Cell Does

### Notebook Structure:
1. **Imports** - Load libraries
2. **Data Generation** - Create 611 synthetic patients
3. **Molecular Features** - BRAF, NRAS, TMB, etc.
4. **Metastasis Events** - Time-to-met for 7 sites
5. **Model Fitting** - 🔥 THE LEARNING HAPPENS HERE
6. **Predictions** - Compare BRAF+ vs BRAF-
7. **Visualizations** - Beautiful plots!

## 💡 Tips for Notebook Use

### Start Simple:
```python
# Just run cells 1-4 first to see data generation
# Then add model fitting (cell 5)
# Then predictions and plots
```

### Experiment:
```python
# Change BRAF prevalence
patients['BRAF_V600E'] = np.random.binomial(1, 0.30, n_patients)

# Re-run model fitting cell
# See how predictions change!
```

### Debug:
```python
# Check data
print(patients.head())
print(met_data[met_data['site'] == 'Brain'].head())

# Check model parameters
print(f"Brain baseline risk: {model.alpha['Brain']}")
print(f"Brain time trend: {model.beta['Brain']}")
```

## 🐛 Troubleshooting

**Problem:** "ModuleNotFoundError: No module named 'bayesian_metastasis_transition_model'"

**Solution:** Make sure both files are in the same directory:
```bash
ls
# Should see:
# bayesian_metastasis_transition_model.py
# demo_bmtm_notebook.ipynb
```

**Problem:** Plots don't show

**Solution:** 
```python
# Add to first cell
%matplotlib inline
import matplotlib.pyplot as plt
```

**Problem:** Model fitting is slow

**Solution:** This is normal! It's learning from 611×7 = 4277 observations
- Takes ~30 seconds on laptop
- If too slow, reduce n_patients to 100 for testing

## 🎓 Learning the Model

### Watch These Variables:
```python
# Before fitting
print(model.alpha)  # Will be empty {}

# After fitting
print(model.alpha)  # Now has learned values!
# {'LN': 1.125, 'Lung': 1.030, 'Brain': 0.013, ...}
```

### See Learning in Action:
```python
# Predict BEFORE fitting (random predictions)
random_risk = model.predict_site_risk('Brain', time=12)

# Fit model
model.fit_maximum_likelihood(met_data, molecular_features)

# Predict AFTER fitting (learned predictions)
learned_risk = model.predict_site_risk('Brain', time=12)

# Compare!
print(f"Random: {random_risk:.4f} vs Learned: {learned_risk:.4f}")
```

## 🚀 Ready to Use Your Real Data?

Once you're comfortable with the notebook, replace synthetic data:

```python
# Instead of generating synthetic data:
# patients = generate_synthetic_data()

# Load your DFCI data:
patients = pd.read_csv('your_dfci_melanoma_patients.csv')
met_data = pd.read_csv('your_metastasis_events.csv')
molecular_features = load_your_mutation_data()

# Then run the rest of the notebook unchanged!
```

## 📊 Expected Output

You should see:
- ✅ Data summary tables (event rates, demographics)
- ✅ 3 main plots:
  1. Risk trajectories (BRAF+ vs BRAF- over 60 months)
  2. Baseline risks by site (bar charts)
  3. Risk distributions at 12 months (histograms)
- ✅ Prediction tables showing site-specific risks
- ✅ Learned parameters (α, β for each site)

## 🎉 Success Looks Like:

```
✅ Data generation complete!
✅ Model fitted successfully!
✅ Predictions:
   Brain (BRAF+): 0.0064
   Brain (BRAF-): 0.0100
   BRAF+ has 2.5x higher brain met risk
✅ 3 visualizations created
```

---

**Questions? The notebook has detailed comments in each cell!**

**Want more? Check out README.md for full documentation**
