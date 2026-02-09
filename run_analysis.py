"""
Full Analysis Pipeline

Runs all MCDM methods across all stakeholders and conditions,
producing the 24 rankings (4 methods × 3 stakeholders × 2 conditions).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass

from mcdm import (
    CriterionType, MCDMResult, 
    wsm, wpm, topsis, promethee_ii,
    run_all_methods, compare_rankings
)
from criteria import (
    Scenario, build_decision_matrix, build_expected_matrix
)
from configurations import CONFIG_SPECS
from stakeholders import STAKEHOLDERS, CRITERIA_NAMES


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIGS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]

CRITERION_TYPES = [
    CriterionType.COST,     # Cost: lower is better
    CriterionType.BENEFIT,  # Reliability: higher is better
    CriterionType.COST,     # Recovery Time: lower is better
    CriterionType.COST,     # Financial Exposure: lower is better
    CriterionType.BENEFIT   # Asset Efficiency: higher is better
]

# Default disruption probabilities (from Allianz risk reports)
DEFAULT_SCENARIO_PROBS = {
    Scenario.NORMAL: 0.85,
    Scenario.CYBERATTACK: 0.05,
    Scenario.SUPPLIER_FAILURE: 0.05,
    Scenario.DEMAND_SURGE: 0.04,
    Scenario.COMPOUND: 0.01
}

METHODS = ["WSM", "WPM", "TOPSIS", "PROMETHEE"]


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

@dataclass
class AnalysisResult:
    """Container for a single analysis run."""
    stakeholder: str
    condition: str  # "normal" or "disrupted"
    method: str
    scores: np.ndarray
    rankings: np.ndarray
    best_config: str
    best_score: float


def run_single_analysis(
    matrix: np.ndarray,
    weights: np.ndarray,
    method: str,
    stakeholder: str,
    condition: str
) -> AnalysisResult:
    """Run a single MCDM analysis."""
    if method == "WSM":
        result = wsm(matrix, weights, CRITERION_TYPES)
    elif method == "WPM":
        result = wpm(matrix, weights, CRITERION_TYPES)
    elif method == "TOPSIS":
        result = topsis(matrix, weights, CRITERION_TYPES)
    else:  # PROMETHEE
        result = promethee_ii(matrix, weights, CRITERION_TYPES)
    
    best_idx = result.get_best()
    
    return AnalysisResult(
        stakeholder=stakeholder,
        condition=condition,
        method=method,
        scores=result.scores,
        rankings=result.rankings,
        best_config=CONFIGS[best_idx],
        best_score=result.scores[best_idx]
    )


def run_full_analysis(
    scenario_probs: Dict[Scenario, float] = None,
    stakeholders: List[str] = None
) -> List[AnalysisResult]:
    """
    Run complete analysis: all methods × stakeholders × conditions.
    
    Args:
        scenario_probs: Disruption probabilities (default: DEFAULT_SCENARIO_PROBS)
        stakeholders: Which stakeholders to include (default: CFO, COO, CMO)
        
    Returns:
        List of 24 AnalysisResult objects (4 methods × 3 stakeholders × 2 conditions)
    """
    if scenario_probs is None:
        scenario_probs = DEFAULT_SCENARIO_PROBS
    
    if stakeholders is None:
        stakeholders = ["CFO", "COO", "CMO"]
    
    # Build decision matrices
    normal_matrix = build_decision_matrix(CONFIGS, Scenario.NORMAL)
    disrupted_matrix = build_expected_matrix(CONFIGS, scenario_probs)
    
    results = []
    
    for stakeholder in stakeholders:
        weights = STAKEHOLDERS[stakeholder].weights
        
        for condition, matrix in [("normal", normal_matrix), ("disrupted", disrupted_matrix)]:
            for method in METHODS:
                result = run_single_analysis(
                    matrix, weights, method, stakeholder, condition
                )
                results.append(result)
    
    return results


def results_to_dataframe(results: List[AnalysisResult]) -> pd.DataFrame:
    """Convert results to a pandas DataFrame."""
    data = []
    for r in results:
        row = {
            "Stakeholder": r.stakeholder,
            "Condition": r.condition,
            "Method": r.method,
            "Best Config": r.best_config,
            "Best Score": r.best_score
        }
        # Add rankings for each config
        for i, config in enumerate(CONFIGS):
            row[f"Rank_{config}"] = int(r.rankings[i])
        data.append(row)
    
    return pd.DataFrame(data)


def print_summary_table(results: List[AnalysisResult]) -> None:
    """Print summary showing best config for each combination."""
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION BY METHOD × STAKEHOLDER × CONDITION")
    print("=" * 80)
    
    # Build lookup
    lookup = {}
    for r in results:
        key = (r.stakeholder, r.condition, r.method)
        lookup[key] = r.best_config
    
    stakeholders = sorted(set(r.stakeholder for r in results))
    conditions = ["normal", "disrupted"]
    
    for condition in conditions:
        print(f"\n{condition.upper()} CONDITION:")
        print("-" * 60)
        header = f"{'Stakeholder':<15}" + "".join(f"{m:>12}" for m in METHODS)
        print(header)
        print("-" * 60)
        
        for stakeholder in stakeholders:
            row = f"{stakeholder:<15}"
            for method in METHODS:
                config = lookup[(stakeholder, condition, method)]
                row += f"{config:>12}"
            print(row)


def print_ranking_details(results: List[AnalysisResult], stakeholder: str) -> None:
    """Print detailed rankings for a specific stakeholder."""
    print(f"\n{'=' * 80}")
    print(f"DETAILED RANKINGS: {stakeholder}")
    print("=" * 80)
    
    subset = [r for r in results if r.stakeholder == stakeholder]
    
    for condition in ["normal", "disrupted"]:
        print(f"\n{condition.upper()}:")
        print("-" * 70)
        
        header = f"{'Config':<10}" + "".join(f"{m:>12}" for m in METHODS)
        print(header)
        print("-" * 70)
        
        condition_results = {r.method: r for r in subset if r.condition == condition}
        
        for i, config in enumerate(CONFIGS):
            row = f"{config:<10}"
            for method in METHODS:
                rank = int(condition_results[method].rankings[i])
                row += f"{rank:>12}"
            print(row)


def analyze_method_agreement(results: List[AnalysisResult]) -> None:
    """Analyze where methods agree and disagree."""
    print("\n" + "=" * 80)
    print("METHOD AGREEMENT ANALYSIS")
    print("=" * 80)
    
    stakeholders = sorted(set(r.stakeholder for r in results))
    conditions = ["normal", "disrupted"]
    
    for stakeholder in stakeholders:
        for condition in conditions:
            subset = [r for r in results 
                     if r.stakeholder == stakeholder and r.condition == condition]
            
            best_configs = [r.best_config for r in subset]
            unique_bests = set(best_configs)
            
            if len(unique_bests) == 1:
                agreement = "FULL AGREEMENT"
            elif len(unique_bests) == 2:
                agreement = "PARTIAL AGREEMENT"
            else:
                agreement = "DISAGREEMENT"
            
            print(f"\n{stakeholder} ({condition}): {agreement}")
            for r in subset:
                print(f"  {r.method}: {r.best_config}")


def compare_normal_vs_disrupted(results: List[AnalysisResult]) -> None:
    """Show how rankings shift between normal and disrupted conditions."""
    print("\n" + "=" * 80)
    print("RANKING SHIFTS: NORMAL → DISRUPTED")
    print("=" * 80)
    
    stakeholders = sorted(set(r.stakeholder for r in results))
    
    for stakeholder in stakeholders:
        print(f"\n{stakeholder}:")
        print("-" * 50)
        
        for method in METHODS:
            normal = [r for r in results 
                     if r.stakeholder == stakeholder 
                     and r.condition == "normal" 
                     and r.method == method][0]
            disrupted = [r for r in results 
                        if r.stakeholder == stakeholder 
                        and r.condition == "disrupted" 
                        and r.method == method][0]
            
            normal_order = [CONFIGS[i] for i in normal.rankings.argsort()]
            disrupted_order = [CONFIGS[i] for i in disrupted.rankings.argsort()]
            
            if normal_order == disrupted_order:
                status = "(unchanged)"
            elif normal.best_config == disrupted.best_config:
                status = "(best unchanged)"
            else:
                status = f"(SHIFT: {normal.best_config} → {disrupted.best_config})"
            
            print(f"  {method}: {status}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run full analysis and print all results."""
    print("MULTI-CRITERIA SUPPLY CHAIN RESILIENCE ANALYSIS")
    print("=" * 80)
    print("Running 4 MCDM methods × 3 stakeholders × 2 conditions = 24 analyses")
    
    # Run analysis
    results = run_full_analysis()
    
    # Print summary
    print_summary_table(results)
    
    # Method agreement
    analyze_method_agreement(results)
    
    # Normal vs disrupted shifts
    compare_normal_vs_disrupted(results)
    
    # Detailed rankings for each stakeholder
    for stakeholder in ["CFO", "COO", "CMO"]:
        print_ranking_details(results, stakeholder)
    
    # Export to CSV
    df = results_to_dataframe(results)
    print("\n\nRESULTS DATAFRAME:")
    print(df[["Stakeholder", "Condition", "Method", "Best Config"]].to_string())
    
    return results


if __name__ == "__main__":
    results = main()
