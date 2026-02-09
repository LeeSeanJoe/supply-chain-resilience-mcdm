"""
Sensitivity Analysis Module

Two analyses:
1. Probability Sensitivity - Vary cyber disruption probability (1-20%)
2. Weight Sensitivity - Vary stakeholder weight on cost criterion

Outputs: Plots showing threshold points where optimal configuration changes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from mcdm import wsm, wpm, topsis, promethee_ii, CriterionType
from criteria import Scenario, build_decision_matrix, build_expected_matrix
from stakeholders import get_stakeholder


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIGS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]

CRITERIA_TYPES = [
    CriterionType.COST,      # Cost - lower is better
    CriterionType.BENEFIT,   # Reliability - higher is better
    CriterionType.COST,      # Recovery - lower is better
    CriterionType.COST,      # FinExposure - lower is better
    CriterionType.BENEFIT    # AssetEfficiency - higher is better
]


# =============================================================================
# PROBABILITY SENSITIVITY
# =============================================================================

def run_probability_sensitivity(
    stakeholder_name: str = "COO",
    cyber_probs: np.ndarray = None,
    method: str = "TOPSIS"
) -> pd.DataFrame:
    """
    Vary cyber disruption probability and track ranking changes.
    
    Args:
        stakeholder_name: Which stakeholder weights to use
        cyber_probs: Array of cyber probabilities to test (default: 0.01 to 0.20)
        method: MCDM method to use
        
    Returns:
        DataFrame with columns: cyber_prob, best_config, and scores for each config
    """
    if cyber_probs is None:
        cyber_probs = np.arange(0.01, 0.21, 0.01)
    
    weights = get_stakeholder(stakeholder_name).weights
    results = []
    
    for cyber_p in cyber_probs:
        # Adjust probabilities (keep ratios for other disruptions)
        # Normal absorbs the change
        other_disruption_total = 0.05 + 0.04 + 0.01  # supplier + demand + compound
        normal_p = max(0.0, 1.0 - cyber_p - other_disruption_total)
        
        scenario_probs = {
            Scenario.NORMAL: normal_p,
            Scenario.CYBERATTACK: cyber_p,
            Scenario.SUPPLIER_FAILURE: 0.05,
            Scenario.DEMAND_SURGE: 0.04,
            Scenario.COMPOUND: 0.01
        }
        
        # Build expected matrix
        matrix = build_expected_matrix(CONFIGS, scenario_probs)
        
        # Run MCDM method
        if method == "TOPSIS":
            result = topsis(matrix, weights, CRITERIA_TYPES)
        elif method == "WSM":
            result = wsm(matrix, weights, CRITERIA_TYPES)
        elif method == "WPM":
            result = wpm(matrix, weights, CRITERIA_TYPES)
        elif method == "PROMETHEE":
            result = promethee_ii(matrix, weights, CRITERIA_TYPES)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Find best config
        best_idx = np.argmin(result.rankings)
        best_config = CONFIGS[best_idx]
        
        row = {
            'cyber_prob': cyber_p,
            'best_config': best_config,
        }
        # Add scores for each config
        for i, config in enumerate(CONFIGS):
            row[f'score_{config}'] = result.scores[i]
            row[f'rank_{config}'] = result.rankings[i]
        
        results.append(row)
    
    return pd.DataFrame(results)


def plot_probability_sensitivity(
    df: pd.DataFrame, 
    output_path: str = "probability_sensitivity.png",
    stakeholder_name: str = "COO",
    method: str = "TOPSIS"
):
    """
    Plot how configuration scores change with cyber probability.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Scores over probability
    ax1 = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    
    for i, config in enumerate(CONFIGS):
        ax1.plot(
            df['cyber_prob'] * 100, 
            df[f'score_{config}'], 
            label=config,
            color=colors[i],
            linewidth=2,
            marker='o',
            markersize=4
        )
    
    ax1.set_xlabel('Cyber Disruption Probability (%)', fontsize=12)
    ax1.set_ylabel(f'{method} Score', fontsize=12)
    ax1.set_title(f'{stakeholder_name}: Configuration Scores vs Cyber Probability', fontsize=12)
    ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 21)
    
    # Right plot: Best configuration regions
    ax2 = axes[1]
    
    # Find transition points
    transitions = []
    prev_best = df.iloc[0]['best_config']
    for idx, row in df.iterrows():
        if row['best_config'] != prev_best:
            transitions.append({
                'prob': row['cyber_prob'],
                'from': prev_best,
                'to': row['best_config']
            })
            prev_best = row['best_config']
    
    # Create regions
    config_colors = {f'C{i}': colors[i-1] for i in range(1, 9)}
    
    start_prob = df['cyber_prob'].min()
    for trans in transitions:
        ax2.axvspan(
            start_prob * 100, 
            trans['prob'] * 100, 
            alpha=0.3, 
            color=config_colors[trans['from']],
            label=trans['from'] if start_prob == df['cyber_prob'].min() else None
        )
        start_prob = trans['prob']
    
    # Final region
    final_config = df.iloc[-1]['best_config']
    ax2.axvspan(
        start_prob * 100, 
        df['cyber_prob'].max() * 100, 
        alpha=0.3, 
        color=config_colors[final_config]
    )
    
    # Mark transitions
    for trans in transitions:
        ax2.axvline(x=trans['prob'] * 100, color='red', linestyle='--', linewidth=2)
        ax2.annotate(
            f"{trans['from']}→{trans['to']}", 
            xy=(trans['prob'] * 100, 0.5),
            xytext=(trans['prob'] * 100 + 1, 0.7),
            fontsize=10,
            fontweight='bold',
            color='red'
        )
    
    ax2.set_xlabel('Cyber Disruption Probability (%)', fontsize=12)
    ax2.set_ylabel('Optimal Configuration', fontsize=12)
    ax2.set_title(f'{stakeholder_name}: Optimal Config by Cyber Probability ({method})', fontsize=12)
    ax2.set_xlim(0, 21)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    
    # Add legend for configs shown
    shown_configs = df['best_config'].unique()
    handles = [plt.Rectangle((0,0),1,1, color=config_colors[c], alpha=0.3) for c in shown_configs]
    ax2.legend(handles, shown_configs, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    return transitions


# =============================================================================
# WEIGHT SENSITIVITY
# =============================================================================

def run_weight_sensitivity(
    stakeholder_name: str = "COO",
    criterion_idx: int = 0,  # 0 = Cost
    criterion_name: str = "Cost",
    weight_range: np.ndarray = None,
    condition: str = "disrupted",
    method: str = "TOPSIS"
) -> pd.DataFrame:
    """
    Vary weight on one criterion and track ranking changes.
    Other weights are scaled proportionally to maintain sum = 1.
    
    Args:
        stakeholder_name: Base stakeholder profile
        criterion_idx: Which criterion to vary (0=Cost, 1=Reliability, etc.)
        criterion_name: Name for labeling
        weight_range: Array of weights to test
        condition: "normal" or "disrupted"
        method: MCDM method to use
        
    Returns:
        DataFrame with results
    """
    if weight_range is None:
        weight_range = np.arange(0.0, 0.65, 0.05)
    
    base_weights = get_stakeholder(stakeholder_name).weights.copy()
    results = []
    
    # Build matrix once
    if condition == "normal":
        matrix = build_decision_matrix(CONFIGS, Scenario.NORMAL)
    else:
        scenario_probs = {
            Scenario.NORMAL: 0.85,
            Scenario.CYBERATTACK: 0.05,
            Scenario.SUPPLIER_FAILURE: 0.05,
            Scenario.DEMAND_SURGE: 0.04,
            Scenario.COMPOUND: 0.01
        }
        matrix = build_expected_matrix(CONFIGS, scenario_probs)
    
    for new_weight in weight_range:
        # Adjust weights: set criterion to new_weight, scale others proportionally
        weights = base_weights.copy()
        old_weight = weights[criterion_idx]
        remaining = 1.0 - new_weight
        
        if old_weight < 1.0:
            scale_factor = remaining / (1.0 - old_weight)
        else:
            scale_factor = 0
        
        for i in range(len(weights)):
            if i == criterion_idx:
                weights[i] = new_weight
            else:
                weights[i] = weights[i] * scale_factor
        
        # Normalize to ensure sum = 1 (handle floating point)
        weights = weights / weights.sum()
        
        # Run MCDM method
        if method == "TOPSIS":
            result = topsis(matrix, weights, CRITERIA_TYPES)
        elif method == "WSM":
            result = wsm(matrix, weights, CRITERIA_TYPES)
        elif method == "WPM":
            result = wpm(matrix, weights, CRITERIA_TYPES)
        elif method == "PROMETHEE":
            result = promethee_ii(matrix, weights, CRITERIA_TYPES)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Find best config
        best_idx = np.argmin(result.rankings)
        best_config = CONFIGS[best_idx]
        
        row = {
            f'{criterion_name.lower()}_weight': new_weight,
            'best_config': best_config,
        }
        for i, config in enumerate(CONFIGS):
            row[f'score_{config}'] = result.scores[i]
            row[f'rank_{config}'] = result.rankings[i]
        
        results.append(row)
    
    return pd.DataFrame(results)


def plot_weight_sensitivity(
    df: pd.DataFrame,
    criterion_name: str = "Cost",
    output_path: str = "weight_sensitivity.png",
    stakeholder_name: str = "COO",
    method: str = "TOPSIS"
):
    """
    Plot how configuration scores change with criterion weight.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    weight_col = f'{criterion_name.lower()}_weight'
    
    # Left plot: Scores
    ax1 = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    
    for i, config in enumerate(CONFIGS):
        ax1.plot(
            df[weight_col] * 100,
            df[f'score_{config}'],
            label=config,
            color=colors[i],
            linewidth=2,
            marker='o',
            markersize=4
        )
    
    ax1.set_xlabel(f'{criterion_name} Weight (%)', fontsize=12)
    ax1.set_ylabel(f'{method} Score', fontsize=12)
    ax1.set_title(f'{stakeholder_name}: Scores vs {criterion_name} Weight (Disrupted)', fontsize=12)
    ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Best config regions
    ax2 = axes[1]
    
    # Find transitions
    transitions = []
    prev_best = df.iloc[0]['best_config']
    for idx, row in df.iterrows():
        if row['best_config'] != prev_best:
            transitions.append({
                'weight': row[weight_col],
                'from': prev_best,
                'to': row['best_config']
            })
            prev_best = row['best_config']
    
    config_colors = {f'C{i}': colors[i-1] for i in range(1, 9)}
    
    start_weight = df[weight_col].min()
    for trans in transitions:
        ax2.axvspan(
            start_weight * 100,
            trans['weight'] * 100,
            alpha=0.3,
            color=config_colors[trans['from']]
        )
        start_weight = trans['weight']
    
    final_config = df.iloc[-1]['best_config']
    ax2.axvspan(
        start_weight * 100,
        df[weight_col].max() * 100,
        alpha=0.3,
        color=config_colors[final_config]
    )
    
    for trans in transitions:
        ax2.axvline(x=trans['weight'] * 100, color='red', linestyle='--', linewidth=2)
        ax2.annotate(
            f"{trans['from']}→{trans['to']}",
            xy=(trans['weight'] * 100, 0.5),
            xytext=(trans['weight'] * 100 + 2, 0.7),
            fontsize=10,
            fontweight='bold',
            color='red'
        )
    
    ax2.set_xlabel(f'{criterion_name} Weight (%)', fontsize=12)
    ax2.set_title(f'{stakeholder_name}: Optimal Config by {criterion_name} Weight', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    
    shown_configs = df['best_config'].unique()
    handles = [plt.Rectangle((0,0),1,1, color=config_colors[c], alpha=0.3) for c in shown_configs]
    ax2.legend(handles, shown_configs, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    return transitions


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def generate_sensitivity_report(
    prob_transitions: List[Dict],
    weight_transitions: List[Dict],
    output_path: str = "sensitivity_report.txt"
):
    """
    Generate text summary of sensitivity findings.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("SENSITIVITY ANALYSIS REPORT")
    lines.append("=" * 70)
    
    lines.append("\n1. PROBABILITY SENSITIVITY (COO, TOPSIS)")
    lines.append("-" * 50)
    
    if prob_transitions:
        for trans in prob_transitions:
            lines.append(
                f"   At {trans['prob']*100:.0f}% cyber probability: "
                f"optimal shifts from {trans['from']} to {trans['to']}"
            )
        lines.append(f"\n   Finding: The recommendation is sensitive to cyber risk assumptions.")
        lines.append(f"   Below {prob_transitions[0]['prob']*100:.0f}%, {prob_transitions[0]['from']} is optimal.")
        lines.append(f"   Above {prob_transitions[-1]['prob']*100:.0f}%, {prob_transitions[-1]['to']} is optimal.")
    else:
        lines.append("   No transitions found - ranking is stable across all probabilities tested.")
        lines.append("   Finding: The recommendation is robust to cyber probability assumptions.")
    
    lines.append("\n2. WEIGHT SENSITIVITY (COO, Cost weight, TOPSIS)")
    lines.append("-" * 50)
    
    if weight_transitions:
        for trans in weight_transitions:
            lines.append(
                f"   At {trans['weight']*100:.0f}% cost weight: "
                f"optimal shifts from {trans['from']} to {trans['to']}"
            )
        lines.append(f"\n   Finding: The recommendation depends on cost prioritization.")
    else:
        lines.append("   No transitions found - ranking is stable across all weights tested.")
        lines.append("   Finding: The recommendation is robust to cost weight assumptions.")
    
    lines.append("\n" + "=" * 70)
    
    report = "\n".join(lines)
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\nSaved: {output_path}")
    
    return report


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    # 1. Probability sensitivity (Sean + Yuantong's task)
    print("\n1. Running probability sensitivity (COO, TOPSIS)...")
    prob_df = run_probability_sensitivity(
        stakeholder_name="COO",
        method="TOPSIS"
    )
    prob_transitions = plot_probability_sensitivity(
        prob_df,
        output_path="probability_sensitivity.png",
        stakeholder_name="COO",
        method="TOPSIS"
    )
    
    # Also run for CFO
    print("\n   Running probability sensitivity (CFO, TOPSIS)...")
    prob_df_cfo = run_probability_sensitivity(
        stakeholder_name="CFO",
        method="TOPSIS"
    )
    prob_transitions_cfo = plot_probability_sensitivity(
        prob_df_cfo,
        output_path="probability_sensitivity_cfo.png",
        stakeholder_name="CFO",
        method="TOPSIS"
    )
    
    # 2. Weight sensitivity (Alice's task)
    print("\n2. Running weight sensitivity (COO, Cost weight, TOPSIS)...")
    weight_df = run_weight_sensitivity(
        stakeholder_name="COO",
        criterion_idx=0,
        criterion_name="Cost",
        condition="disrupted",
        method="TOPSIS"
    )
    weight_transitions = plot_weight_sensitivity(
        weight_df,
        criterion_name="Cost",
        output_path="weight_sensitivity_cost.png",
        stakeholder_name="COO",
        method="TOPSIS"
    )
    
    # Also vary Reliability weight
    print("\n   Running weight sensitivity (COO, Reliability weight, TOPSIS)...")
    weight_df_rel = run_weight_sensitivity(
        stakeholder_name="COO",
        criterion_idx=1,
        criterion_name="Reliability",
        condition="disrupted",
        method="TOPSIS"
    )
    weight_transitions_rel = plot_weight_sensitivity(
        weight_df_rel,
        criterion_name="Reliability",
        output_path="weight_sensitivity_reliability.png",
        stakeholder_name="COO",
        method="TOPSIS"
    )
    
    # 3. Generate report
    print("\n3. Generating sensitivity report...")
    generate_sensitivity_report(
        prob_transitions,
        weight_transitions,
        output_path="sensitivity_report.txt"
    )
    
    print("\n" + "=" * 70)
    print("Sensitivity analysis complete!")
