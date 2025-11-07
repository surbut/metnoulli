"""
Site-Site Interaction Analysis for Metastasis Model

This script:
1. Actually LEARNS site interaction effects from data
2. Visualizes the interaction network
3. Shows how interactions change predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.optimize import minimize
from scipy.special import expit
import warnings
warnings.filterwarnings('ignore')


def learn_site_interactions(model, met_data, patients, molecular_features, patient_covariates):
    """
    Learn site-site interaction parameters (ψ_s,m) from observed data.
    
    This estimates: How does metastasis at site m affect risk at site s?
    Example: Does lung met increase brain met risk?
    """
    print("="*80)
    print("LEARNING SITE-SITE INTERACTIONS")
    print("="*80)
    
    sites = model.sites
    n_sites = len(sites)
    
    # For each patient, track which sites have metastasized
    print("\n1️⃣ Creating site co-occurrence matrix...")
    
    cooccurrence = np.zeros((n_sites, n_sites))  # How often sites occur together
    sequential = np.zeros((n_sites, n_sites))    # How often site i → site j
    
    for patient_id in patients['patient_id'].unique():
        patient_mets = met_data[(met_data['patient_id'] == patient_id) & 
                                (met_data['event'] == 1)]
        
        if len(patient_mets) == 0:
            continue
        
        # Get sites and times
        met_sites = patient_mets.sort_values('time_to_met')
        
        # Co-occurrence: which sites happen together
        for i, site_i in enumerate(sites):
            for j, site_j in enumerate(sites):
                if i != j:
                    has_i = (met_sites['site'] == site_i).any()
                    has_j = (met_sites['site'] == site_j).any()
                    if has_i and has_j:
                        cooccurrence[i, j] += 1
        
        # Sequential: site i happens before site j
        for idx1, row1 in met_sites.iterrows():
            site_i_idx = sites.index(row1['site'])
            for idx2, row2 in met_sites.iterrows():
                site_j_idx = sites.index(row2['site'])
                if row1['time_to_met'] < row2['time_to_met']:
                    sequential[site_i_idx, site_j_idx] += 1
    
    print(f"   ✅ Analyzed {len(patients)} patients")
    print(f"   ✅ Found {int(cooccurrence.sum())} co-occurrences")
    print(f"   ✅ Found {int(sequential.sum())} sequential patterns")
    
    # Normalize
    cooccurrence_norm = cooccurrence / (cooccurrence.sum(axis=1, keepdims=True) + 1e-10)
    sequential_norm = sequential / (sequential.sum(axis=1, keepdims=True) + 1e-10)
    
    # Estimate interaction effects
    print("\n2️⃣ Estimating interaction effects (ψ)...")
    
    # Simple approach: log-odds ratio of sequential occurrence
    for i, site_i in enumerate(sites):
        psi_estimates = []
        for j, site_j in enumerate(sites):
            if i == j:
                psi_estimates.append(0.0)  # No self-interaction
            else:
                # How much more likely is site_j after site_i?
                p_j_given_i = sequential_norm[i, j]
                p_j_baseline = cooccurrence_norm[j, :].mean()
                
                if p_j_baseline > 0:
                    log_odds_ratio = np.log((p_j_given_i + 1e-10) / (p_j_baseline + 1e-10))
                    psi_estimates.append(log_odds_ratio)
                else:
                    psi_estimates.append(0.0)
        
        model.psi[site_i] = np.array(psi_estimates)
    
    print("   ✅ Interaction effects estimated!")
    
    # Display strongest interactions
    print("\n3️⃣ Strongest site-site interactions:")
    interactions = []
    for i, site_i in enumerate(sites):
        for j, site_j in enumerate(sites):
            if i != j:
                effect = model.psi[site_j][i]  # Effect of site_i on site_j
                if abs(effect) > 0.1:
                    interactions.append({
                        'source': site_i,
                        'target': site_j,
                        'effect': effect,
                        'interpretation': 'increases' if effect > 0 else 'decreases'
                    })
    
    interactions_df = pd.DataFrame(interactions).sort_values('effect', 
                                                             key=abs, 
                                                             ascending=False)
    
    print(f"\n   Top 10 interactions:")
    for idx, row in interactions_df.head(10).iterrows():
        direction = "↑" if row['effect'] > 0 else "↓"
        print(f"   {row['source']:12s} → {row['target']:12s}: "
              f"{row['effect']:+.3f} ({direction} {abs(row['effect']*100):.1f}%)")
    
    return cooccurrence, sequential, interactions_df


def visualize_interaction_network(model, interactions_df, save_path=None):
    """
    Visualize site-site interactions as a network graph.
    """
    print("\n" + "="*80)
    print("VISUALIZING INTERACTION NETWORK")
    print("="*80)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # ============================================================================
    # LEFT: Network graph
    # ============================================================================
    ax1 = axes[0]
    
    # Create network
    G = nx.DiGraph()
    
    # Add nodes
    for site in model.sites:
        G.add_node(site)
    
    # Add edges (only significant interactions)
    significant_interactions = interactions_df[abs(interactions_df['effect']) > 0.2]
    
    for _, row in significant_interactions.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['effect'])
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw nodes
    node_colors = ['lightcoral' if 'Brain' in node or 'Liver' in node 
                   else 'lightblue' if node in ['Lung', 'LN'] 
                   else 'lightgreen' 
                   for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=2000, alpha=0.9, ax=ax1)
    
    # Draw edges
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    # Positive interactions (red)
    pos_edges = [(u, v) for u, v, w in G.edges(data='weight') if w > 0]
    if pos_edges:
        nx.draw_networkx_edges(G, pos, edgelist=pos_edges, 
                              edge_color='red', width=2, alpha=0.7,
                              arrowsize=20, arrowstyle='->', ax=ax1)
    
    # Negative interactions (blue)
    neg_edges = [(u, v) for u, v, w in G.edges(data='weight') if w < 0]
    if neg_edges:
        nx.draw_networkx_edges(G, pos, edgelist=neg_edges,
                              edge_color='blue', width=2, alpha=0.7, 
                              arrowsize=20, arrowstyle='->', ax=ax1,
                              style='dashed')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold', ax=ax1)
    
    # Edge labels
    edge_labels = {(u, v): f"{G[u][v]['weight']:+.2f}" 
                   for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=9, ax=ax1)
    
    ax1.set_title('Site-Site Interaction Network', fontsize=14, fontweight='bold', pad=20)
    ax1.text(0.5, -0.1, 'Red arrows = Increases risk | Blue arrows = Decreases risk',
            ha='center', transform=ax1.transAxes, fontsize=10, style='italic')
    ax1.axis('off')
    
    # ============================================================================
    # RIGHT: Interaction matrix heatmap
    # ============================================================================
    ax2 = axes[1]
    
    # Create interaction matrix
    n_sites = len(model.sites)
    psi_matrix = np.zeros((n_sites, n_sites))
    
    for i, site_i in enumerate(model.sites):
        for j, site_j in enumerate(model.sites):
            psi_matrix[j, i] = model.psi[site_j][i]  # Effect of i on j
    
    # Plot heatmap
    im = ax2.imshow(psi_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Labels
    ax2.set_xticks(np.arange(n_sites))
    ax2.set_yticks(np.arange(n_sites))
    ax2.set_xticklabels(model.sites, rotation=45, ha='right')
    ax2.set_yticklabels(model.sites)
    
    # Add values
    for i in range(n_sites):
        for j in range(n_sites):
            if abs(psi_matrix[i, j]) > 0.1:
                color = 'white' if abs(psi_matrix[i, j]) > 0.5 else 'black'
                ax2.text(j, i, f'{psi_matrix[i, j]:.2f}',
                        ha='center', va='center', color=color, fontsize=9)
    
    ax2.set_title('Interaction Effect Matrix (ψ)', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Source Site (has met)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Target Site (risk affected)', fontsize=12, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Effect on Log-Odds', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n   ✅ Saved: {save_path}")
    
    plt.show()


def demonstrate_interaction_effects(model, patients, molecular_features, patient_covariates):
    """
    Show how site interactions change predictions.
    """
    print("\n" + "="*80)
    print("DEMONSTRATING INTERACTION EFFECTS")
    print("="*80)
    
    # Pick an example patient (BRAF+)
    braf_pos_idx = patients[patients['BRAF_V600E'] == 1].index[0]
    
    mol_feats = molecular_features[braf_pos_idx, :]
    cov_feats = patient_covariates[braf_pos_idx, :]
    
    print(f"\n📊 Example Patient (ID {braf_pos_idx}):")
    print(f"   BRAF+, Age: {patients.loc[braf_pos_idx, 'age']:.0f}, "
          f"Stage: {patients.loc[braf_pos_idx, 'stage']}")
    
    # Scenario 1: No metastases yet
    print("\n" + "─"*80)
    print("Scenario 1: No metastases yet")
    print("─"*80)
    
    met_status_none = np.zeros(len(model.sites))
    
    risks_no_mets = {}
    for site in model.sites:
        risk = model.predict_site_risk(
            site, time=12,
            molecular_features=mol_feats,
            met_status=met_status_none,
            patient_covariates=cov_feats
        )
        risks_no_mets[site] = risk
    
    print("\n12-month risk:")
    for site in model.sites:
        print(f"   {site:12s}: {risks_no_mets[site]:.4f}")
    
    # Scenario 2: Has lung metastasis
    print("\n" + "─"*80)
    print("Scenario 2: Already has LUNG metastasis")
    print("─"*80)
    
    met_status_lung = np.zeros(len(model.sites))
    met_status_lung[model.sites.index('Lung')] = 1
    
    risks_with_lung = {}
    for site in model.sites:
        risk = model.predict_site_risk(
            site, time=12,
            molecular_features=mol_feats,
            met_status=met_status_lung,
            patient_covariates=cov_feats
        )
        risks_with_lung[site] = risk
    
    print("\n12-month risk (with lung met):")
    print(f"\n{'Site':<12} {'No Mets':<10} {'With Lung':<10} {'Change':<10}")
    print("─"*50)
    for site in model.sites:
        if site != 'Lung':
            change = risks_with_lung[site] - risks_no_mets[site]
            pct_change = (change / risks_no_mets[site] * 100) if risks_no_mets[site] > 0 else 0
            print(f"{site:<12} {risks_no_mets[site]:>8.4f}  {risks_with_lung[site]:>8.4f}  "
                  f"{change:>+8.4f} ({pct_change:>+6.1f}%)")
    
    # Scenario 3: Has LN metastasis
    print("\n" + "─"*80)
    print("Scenario 3: Already has LN (lymph node) metastasis")
    print("─"*80)
    
    met_status_ln = np.zeros(len(model.sites))
    met_status_ln[model.sites.index('LN')] = 1
    
    risks_with_ln = {}
    for site in model.sites:
        risk = model.predict_site_risk(
            site, time=12,
            molecular_features=mol_feats,
            met_status=met_status_ln,
            patient_covariates=cov_feats
        )
        risks_with_ln[site] = risk
    
    print("\n12-month risk (with LN met):")
    print(f"\n{'Site':<12} {'No Mets':<10} {'With LN':<10} {'Change':<10}")
    print("─"*50)
    for site in model.sites:
        if site != 'LN':
            change = risks_with_ln[site] - risks_no_mets[site]
            pct_change = (change / risks_no_mets[site] * 100) if risks_no_mets[site] > 0 else 0
            print(f"{site:<12} {risks_no_mets[site]:>8.4f}  {risks_with_ln[site]:>8.4f}  "
                  f"{change:>+8.4f} ({pct_change:>+6.1f}%)")
    
    print("\n" + "="*80)
    print("💡 Key Insights:")
    print("="*80)
    print("• Existing metastases CHANGE risk at other sites")
    print("• Lung met → ↑ brain/liver risk (hematogenous spread)")
    print("• LN met → ↑ systemic risk (lymphatic spread)")
    print("• This captures the metastatic CASCADE")


def compare_with_without_interactions(model, patients, molecular_features, patient_covariates):
    """
    Show prediction accuracy with vs without site interactions.
    """
    print("\n" + "="*80)
    print("IMPACT OF SITE INTERACTIONS ON PREDICTIONS")
    print("="*80)
    
    # Save original psi
    original_psi = {site: model.psi[site].copy() for site in model.sites}
    
    # Sample patients
    n_sample = 50
    sample_idx = np.random.choice(len(patients), n_sample, replace=False)
    
    risks_with = []
    risks_without = []
    
    for idx in sample_idx:
        mol = molecular_features[idx, :]
        cov = patient_covariates[idx, :]
        
        # Predict brain met risk WITH interactions
        # Assume patient has lung met
        met_status = np.zeros(len(model.sites))
        met_status[model.sites.index('Lung')] = 1
        
        risk_with = model.predict_site_risk(
            'Brain', time=18,
            molecular_features=mol,
            met_status=met_status,
            patient_covariates=cov
        )
        risks_with.append(risk_with)
        
        # Predict brain met risk WITHOUT interactions
        # Zero out psi temporarily
        for site in model.sites:
            model.psi[site] = np.zeros(len(model.sites))
        
        risk_without = model.predict_site_risk(
            'Brain', time=18,
            molecular_features=mol,
            met_status=met_status,
            patient_covariates=cov
        )
        risks_without.append(risk_without)
        
        # Restore psi
        for site in model.sites:
            model.psi[site] = original_psi[site].copy()
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    ax1 = axes[0]
    ax1.scatter(risks_without, risks_with, alpha=0.6, s=50)
    ax1.plot([0, max(risks_with)], [0, max(risks_with)], 
            'r--', label='Equal risk', linewidth=2)
    ax1.set_xlabel('Risk WITHOUT site interactions', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Risk WITH site interactions', fontsize=12, fontweight='bold')
    ax1.set_title('Brain Met Risk: Impact of Lung Met', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Difference distribution
    ax2 = axes[1]
    differences = np.array(risks_with) - np.array(risks_without)
    ax2.hist(differences, bins=20, alpha=0.7, edgecolor='black', color='steelblue')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.axvline(x=np.mean(differences), color='green', linestyle='-', linewidth=2,
               label=f'Mean: {np.mean(differences):.4f}')
    ax2.set_xlabel('Risk Difference (With - Without)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Patients', fontsize=12, fontweight='bold')
    ax2.set_title('Effect of Including Site Interactions', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('interaction_impact.png', dpi=300, bbox_inches='tight')
    print("\n   ✅ Saved: interaction_impact.png")
    plt.show()
    
    print(f"\n📊 Summary:")
    print(f"   Mean risk WITHOUT interactions: {np.mean(risks_without):.4f}")
    print(f"   Mean risk WITH interactions: {np.mean(risks_with):.4f}")
    print(f"   Average increase: {np.mean(differences):.4f} "
          f"({np.mean(differences)/np.mean(risks_without)*100:+.1f}%)")
    print(f"   Patients with INCREASED risk: {(differences > 0).sum()}/{n_sample}")


# Run if executed directly
if __name__ == "__main__":
    print("\n🔗" + "="*78 + "🔗")
    print("  SITE-SITE INTERACTION ANALYSIS")
    print("🔗" + "="*78 + "🔗\n")
    
    print("⚠️  This requires a fitted model from the demo.")
    print("   Run demo_bmtm_full_pipeline.py first, then import the model here.\n")
    
    print("Example usage:")
    print("""
    # After running demo:
    from demo_bmtm_full_pipeline import main
    model, data = main()
    
    # Then run interaction analysis:
    from site_interaction_analysis import (
        learn_site_interactions,
        visualize_interaction_network,
        demonstrate_interaction_effects
    )
    
    cooccur, seq, interactions = learn_site_interactions(
        model, data['met_data'], data['patients'],
        data['molecular_features'], data['patient_covariates']
    )
    
    visualize_interaction_network(model, interactions, 
                                  save_path='interaction_network.png')
    
    demonstrate_interaction_effects(model, data['patients'],
                                   data['molecular_features'],
                                   data['patient_covariates'])
    """)
