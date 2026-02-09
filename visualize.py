"""
Visualization Module for MCDM Results

Generates:
1. Heatmap of rankings by method × stakeholder × condition
2. Criteria scores comparison across configurations
3. Ranking shift visualization (normal → disrupted)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

from mcdm import wsm, wpm, topsis, promethee_ii, CriterionType
from criteria import Scenario, build_decision_matrix, build_expected_matrix
from stakeholders import STAKEHOLDERS, CRITERIA_NAMES, get_stakeholder


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIGS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
METHODS = ["WSM", "WPM", "TOPSIS", "PROMETHEE"]
STAKEHOLDER_NAMES = ["CFO", "COO", "CMO"]

# Criteria directions: True = benefit (higher is better)
CRITERIA_TYPES = [
    CriterionType.COST,      # Cost - lower is better
    CriterionType.BENEFIT,   # Reliability - higher is better
    CriterionType.COST,      # Recovery - lower is better
    CriterionType.COST,      # FinExposure - lower is better
    CriterionType.BENEFIT    # AssetEfficiency - higher is better
]


# =============================================================================
# DATA GENERATION
# =============================================================================

def get_rankings(matrix: np.ndarray, weights: np.ndarray) -> Dict[str, List[int]]:
    """Get rankings from all four MCDM methods."""
    results = {}
    
    # WSM
    wsm_result = wsm(matrix, weights, CRITERIA_TYPES)
    results["WSM"] = wsm_result.rankings
    
    # WPM
    wpm_result = wpm(matrix, weights, CRITERIA_TYPES)
    results["WPM"] = wpm_result.rankings
    
    # TOPSIS
    topsis_result = topsis(matrix, weights, CRITERIA_TYPES)
    results["TOPSIS"] = topsis_result.rankings
    
    # PROMETHEE II
    prom_result = promethee_ii(matrix, weights, CRITERIA_TYPES)
    results["PROMETHEE"] = prom_result.rankings
    
    return results


def build_ranking_dataframe() -> pd.DataFrame:
    """Build complete ranking dataframe for all combinations."""
    rows = []
    
    # Disruption probabilities for expected matrix
    scenario_probs = {
        Scenario.NORMAL: 0.85,
        Scenario.CYBERATTACK: 0.05,
        Scenario.SUPPLIER_FAILURE: 0.05,
        Scenario.DEMAND_SURGE: 0.04,
        Scenario.COMPOUND: 0.01
    }
    
    for stakeholder_name in STAKEHOLDER_NAMES:
        weights = get_stakeholder(stakeholder_name).weights
        
        for condition in ["normal", "disrupted"]:
            if condition == "normal":
                matrix = build_decision_matrix(CONFIGS, Scenario.NORMAL)
            else:
                matrix = build_expected_matrix(CONFIGS, scenario_probs)
            
            rankings = get_rankings(matrix, weights)
            
            for method in METHODS:
                for i, config in enumerate(CONFIGS):
                    rows.append({
                        "Stakeholder": stakeholder_name,
                        "Condition": condition,
                        "Method": method,
                        "Config": config,
                        "Rank": rankings[method][i]
                    })
    
    return pd.DataFrame(rows)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_ranking_heatmap(df: pd.DataFrame, output_path: str = "ranking_heatmap.png"):
    """
    Create heatmap showing rankings for each config across all conditions.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Configuration Rankings by Stakeholder and Condition", fontsize=14, fontweight='bold')
    
    conditions = ["normal", "disrupted"]
    
    for col, stakeholder in enumerate(STAKEHOLDER_NAMES):
        for row, condition in enumerate(conditions):
            ax = axes[row, col]
            
            # Filter data
            subset = df[(df["Stakeholder"] == stakeholder) & (df["Condition"] == condition)]
            pivot = subset.pivot(index="Config", columns="Method", values="Rank")
            pivot = pivot[METHODS]  # Ensure column order
            
            # Heatmap
            sns.heatmap(
                pivot, 
                annot=True, 
                fmt="d", 
                cmap="RdYlGn_r",  # Red=bad (high rank), Green=good (low rank)
                vmin=1, 
                vmax=8,
                ax=ax,
                cbar=col == 2  # Only show colorbar on right
            )
            
            ax.set_title(f"{stakeholder} - {condition.capitalize()}")
            ax.set_xlabel("Method" if row == 1 else "")
            ax.set_ylabel("Config" if col == 0 else "")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_criteria_comparison(output_path: str = "criteria_comparison.png"):
    """
    Bar chart comparing criteria scores across configurations.
    """
    matrix_normal = build_decision_matrix(CONFIGS, Scenario.NORMAL)
    
    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    fig.suptitle("Criteria Scores by Configuration (Normal Condition)", fontsize=14, fontweight='bold')
    
    criteria_labels = ["Cost ($)", "Reliability (%)", "Recovery (days)", "Fin. Exposure ($)", "Asset Efficiency"]
    colors = ['#e74c3c', '#2ecc71', '#e74c3c', '#e74c3c', '#2ecc71']  # Red for cost, green for benefit
    
    for i, (ax, label, color) in enumerate(zip(axes, criteria_labels, colors)):
        values = matrix_normal[:, i]
        
        # Convert reliability to percentage
        if i == 1:
            values = values * 100
        
        bars = ax.bar(CONFIGS, values, color=color, alpha=0.7, edgecolor='black')
        ax.set_title(label)
        ax.set_xlabel("Configuration")
        
        # Highlight best
        if color == '#2ecc71':  # Benefit - max is best
            best_idx = np.argmax(values)
        else:  # Cost - min is best
            best_idx = np.argmin(values)
        bars[best_idx].set_alpha(1.0)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
        
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_ranking_shift(df: pd.DataFrame, output_path: str = "ranking_shift.png"):
    """
    Visualize how rankings shift from normal to disrupted condition.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle("Ranking Shifts: Normal → Disrupted", fontsize=14, fontweight='bold')
    
    for ax, stakeholder in zip(axes, STAKEHOLDER_NAMES):
        # Get average rank across methods for each config
        normal = df[(df["Stakeholder"] == stakeholder) & (df["Condition"] == "normal")]
        disrupted = df[(df["Stakeholder"] == stakeholder) & (df["Condition"] == "disrupted")]
        
        normal_avg = normal.groupby("Config")["Rank"].mean()
        disrupted_avg = disrupted.groupby("Config")["Rank"].mean()
        
        # Plot
        x = np.arange(len(CONFIGS))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, normal_avg[CONFIGS], width, label='Normal', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, disrupted_avg[CONFIGS], width, label='Disrupted', color='#e74c3c', alpha=0.8)
        
        ax.set_xlabel("Configuration")
        ax.set_ylabel("Average Rank (lower is better)")
        ax.set_title(stakeholder)
        ax.set_xticks(x)
        ax.set_xticklabels(CONFIGS)
        ax.legend()
        ax.set_ylim(0, 9)
        ax.invert_yaxis()  # Lower rank at top
        
        # Add shift arrows for significant changes
        for i, config in enumerate(CONFIGS):
            shift = disrupted_avg[config] - normal_avg[config]
            if abs(shift) > 1:
                color = 'green' if shift < 0 else 'red'
                ax.annotate(
                    '', 
                    xy=(i + width/2, disrupted_avg[config]),
                    xytext=(i - width/2, normal_avg[config]),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2)
                )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_best_config_summary(df: pd.DataFrame, output_path: str = "best_config_summary.png"):
    """
    Summary table showing best config for each combination.
    """
    # Pivot to get best config
    best_configs = df[df["Rank"] == 1].pivot_table(
        index=["Stakeholder", "Condition"],
        columns="Method",
        values="Config",
        aggfunc="first"
    )
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    # Create table
    table_data = []
    for stakeholder in STAKEHOLDER_NAMES:
        for condition in ["normal", "disrupted"]:
            row = [f"{stakeholder} ({condition})"]
            for method in METHODS:
                try:
                    config = best_configs.loc[(stakeholder, condition), method]
                    row.append(config)
                except:
                    row.append("-")
            table_data.append(row)
    
    table = ax.table(
        cellText=table_data,
        colLabels=["Scenario"] + METHODS,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Color cells
    for i in range(len(table_data)):
        for j in range(1, 5):
            cell = table[(i + 1, j)]
            config = table_data[i][j]
            if config == "C3":
                cell.set_facecolor('#aed6f1')
            elif config == "C7":
                cell.set_facecolor('#abebc6')
            elif config == "C6":
                cell.set_facecolor('#f9e79f')
    
    ax.set_title("Best Configuration by Stakeholder × Condition × Method", 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Generating visualizations...")
    print("=" * 60)
    
    # Build data
    df = build_ranking_dataframe()
    
    # Generate plots
    plot_ranking_heatmap(df, "ranking_heatmap.png")
    plot_criteria_comparison("criteria_comparison.png")
    plot_ranking_shift(df, "ranking_shift.png")
    plot_best_config_summary(df, "best_config_summary.png")
    
    print("=" * 60)
    print("All visualizations generated!")
