"""
Multi-Criteria Decision Making (MCDM) Methods

This module implements four MCDM methods for ranking supply chain configurations:
- WSM (Weighted Sum Model)
- WPM (Weighted Product Model)
- TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
- PROMETHEE II (Preference Ranking Organization Method for Enrichment Evaluation)

Each method takes a decision matrix (alternatives × criteria) and weights,
and returns scores/rankings for all alternatives.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class CriterionType(Enum):
    """Criterion optimization direction."""
    BENEFIT = "benefit"  # Higher is better (e.g., reliability, service level)
    COST = "cost"        # Lower is better (e.g., cost, recovery time)


@dataclass
class MCDMResult:
    """Container for MCDM method results."""
    method: str
    scores: np.ndarray
    rankings: np.ndarray  # 1 = best, n = worst
    
    def get_best(self) -> int:
        """Return index of best alternative."""
        return int(np.argmin(self.rankings))
    
    def get_ranking_order(self) -> List[int]:
        """Return alternative indices in rank order (best first)."""
        return list(np.argsort(self.rankings))


# =============================================================================
# NORMALIZATION FUNCTIONS
# =============================================================================

def normalize_minmax(matrix: np.ndarray, criterion_types: List[CriterionType]) -> np.ndarray:
    """
    Min-max normalization to [0, 1] scale.
    
    For benefit criteria: (x - min) / (max - min)
    For cost criteria: (max - x) / (max - min)
    
    Args:
        matrix: Decision matrix (m alternatives × n criteria)
        criterion_types: List of CriterionType for each criterion
        
    Returns:
        Normalized matrix with all values in [0, 1], higher = better
    """
    m, n = matrix.shape
    normalized = np.zeros_like(matrix, dtype=float)
    
    for j in range(n):
        col = matrix[:, j]
        col_min, col_max = col.min(), col.max()
        
        # Avoid division by zero if all values are the same
        if col_max - col_min < 1e-10:
            normalized[:, j] = 1.0  # All alternatives equal on this criterion
        elif criterion_types[j] == CriterionType.BENEFIT:
            normalized[:, j] = (col - col_min) / (col_max - col_min)
        else:  # COST
            normalized[:, j] = (col_max - col) / (col_max - col_min)
    
    return normalized


def normalize_vector(matrix: np.ndarray) -> np.ndarray:
    """
    Vector normalization (used by TOPSIS).
    
    x_normalized = x / sqrt(sum(x^2))
    
    Args:
        matrix: Decision matrix (m alternatives × n criteria)
        
    Returns:
        Vector-normalized matrix
    """
    norms = np.sqrt((matrix ** 2).sum(axis=0))
    # Avoid division by zero
    norms[norms < 1e-10] = 1.0
    return matrix / norms


# =============================================================================
# WSM - WEIGHTED SUM MODEL
# =============================================================================

def wsm(
    matrix: np.ndarray,
    weights: np.ndarray,
    criterion_types: List[CriterionType]
) -> MCDMResult:
    """
    Weighted Sum Model (WSM).
    
    The simplest MCDM method. Normalizes criteria to [0,1], multiplies by weights,
    and sums across criteria. Fully compensatory: a high score on one criterion
    can fully offset a low score on another.
    
    Score_i = Σ(w_j × normalized_ij)
    
    Args:
        matrix: Decision matrix (m alternatives × n criteria), raw values
        weights: Criterion weights (must sum to 1)
        criterion_types: List indicating if each criterion is BENEFIT or COST
        
    Returns:
        MCDMResult with scores (higher = better) and rankings
        
    Example:
        >>> matrix = np.array([[100, 0.95, 5], [150, 0.98, 3], [120, 0.90, 7]])
        >>> weights = np.array([0.4, 0.35, 0.25])
        >>> types = [CriterionType.COST, CriterionType.BENEFIT, CriterionType.COST]
        >>> result = wsm(matrix, weights, types)
    """
    weights = np.array(weights)
    if not np.isclose(weights.sum(), 1.0):
        weights = weights / weights.sum()  # Normalize weights
    
    # Normalize matrix (higher = better after normalization)
    normalized = normalize_minmax(matrix, criterion_types)
    
    # Compute weighted sum
    scores = (normalized * weights).sum(axis=1)
    
    # Rankings (1 = best)
    rankings = scores.argsort()[::-1].argsort() + 1
    
    return MCDMResult(method="WSM", scores=scores, rankings=rankings)


# =============================================================================
# WPM - WEIGHTED PRODUCT MODEL
# =============================================================================

def wpm(
    matrix: np.ndarray,
    weights: np.ndarray,
    criterion_types: List[CriterionType],
    epsilon: float = 1e-6
) -> MCDMResult:
    """
    Weighted Product Model (WPM).
    
    Multiplicative aggregation: each criterion value is raised to the power of
    its weight, then all values are multiplied together. More punishing of
    weak performance on any criterion compared to WSM.
    
    Score_i = Π(normalized_ij ^ w_j)
    
    Key difference from WSM: In WSM, a score of 0.95 on cost and 0.05 on recovery
    can still rank well if cost weight is high. In WPM, that 0.05 raised to a power
    drags the entire product down.
    
    Args:
        matrix: Decision matrix (m alternatives × n criteria)
        weights: Criterion weights (must sum to 1)
        criterion_types: List indicating if each criterion is BENEFIT or COST
        epsilon: Small value added to avoid zero values (which would zero the product)
        
    Returns:
        MCDMResult with scores (higher = better) and rankings
    """
    weights = np.array(weights)
    if not np.isclose(weights.sum(), 1.0):
        weights = weights / weights.sum()
    
    # Normalize matrix
    normalized = normalize_minmax(matrix, criterion_types)
    
    # Add epsilon to avoid zeros (a zero would make the entire product zero)
    normalized = normalized + epsilon
    
    # Compute weighted product: product of (value ^ weight) across criteria
    scores = np.prod(normalized ** weights, axis=1)
    
    # Rankings (1 = best)
    rankings = scores.argsort()[::-1].argsort() + 1
    
    return MCDMResult(method="WPM", scores=scores, rankings=rankings)


# =============================================================================
# TOPSIS - TECHNIQUE FOR ORDER OF PREFERENCE BY SIMILARITY TO IDEAL SOLUTION
# =============================================================================

def topsis(
    matrix: np.ndarray,
    weights: np.ndarray,
    criterion_types: List[CriterionType]
) -> MCDMResult:
    """
    TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution).
    
    Measures how close each alternative is to the ideal solution (best on every
    criterion) and how far it is from the anti-ideal (worst on everything).
    The alternative closest to ideal and farthest from anti-ideal wins.
    
    Steps:
    1. Vector-normalize the decision matrix
    2. Apply weights to get weighted normalized matrix
    3. Identify ideal (A+) and anti-ideal (A-) solutions
    4. Compute Euclidean distance to A+ and A- for each alternative
    5. Compute closeness coefficient: CC = d- / (d+ + d-)
    
    Args:
        matrix: Decision matrix (m alternatives × n criteria)
        weights: Criterion weights (must sum to 1)
        criterion_types: List indicating if each criterion is BENEFIT or COST
        
    Returns:
        MCDMResult with closeness coefficients (higher = better) and rankings
    """
    weights = np.array(weights)
    if not np.isclose(weights.sum(), 1.0):
        weights = weights / weights.sum()
    
    m, n = matrix.shape
    
    # Step 1: Vector normalization
    normalized = normalize_vector(matrix)
    
    # Step 2: Apply weights
    weighted = normalized * weights
    
    # Step 3: Identify ideal (A+) and anti-ideal (A-)
    ideal = np.zeros(n)
    anti_ideal = np.zeros(n)
    
    for j in range(n):
        if criterion_types[j] == CriterionType.BENEFIT:
            ideal[j] = weighted[:, j].max()
            anti_ideal[j] = weighted[:, j].min()
        else:  # COST
            ideal[j] = weighted[:, j].min()
            anti_ideal[j] = weighted[:, j].max()
    
    # Step 4: Compute distances
    dist_to_ideal = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
    dist_to_anti = np.sqrt(((weighted - anti_ideal) ** 2).sum(axis=1))
    
    # Step 5: Closeness coefficient
    # Handle edge case where both distances are zero
    denominator = dist_to_ideal + dist_to_anti
    denominator[denominator < 1e-10] = 1e-10
    
    scores = dist_to_anti / denominator
    
    # Rankings (1 = best, highest CC)
    rankings = scores.argsort()[::-1].argsort() + 1
    
    return MCDMResult(method="TOPSIS", scores=scores, rankings=rankings)


# =============================================================================
# PROMETHEE II - PREFERENCE RANKING ORGANIZATION METHOD FOR ENRICHMENT EVALUATION
# =============================================================================

def promethee_ii(
    matrix: np.ndarray,
    weights: np.ndarray,
    criterion_types: List[CriterionType],
    q_thresholds: Optional[np.ndarray] = None,
    p_thresholds: Optional[np.ndarray] = None
) -> MCDMResult:
    """
    PROMETHEE II (Preference Ranking Organization Method for Enrichment Evaluation).
    
    Pairwise comparison method. For each pair of alternatives, computes how much
    A outperforms B on each criterion using preference functions. Aggregates into
    positive flow (how much A outranks others) and negative flow (how much others
    outrank A). Net flow = positive - negative.
    
    Uses V-shape (linear) preference function with indifference (q) and preference (p)
    thresholds:
    - Difference < q: preference = 0 (indifference)
    - Difference > p: preference = 1 (strict preference)
    - q <= difference <= p: preference = (difference - q) / (p - q) (linear)
    
    Why PROMETHEE can disagree with TOPSIS:
    - TOPSIS measures distance to ideal point (how close to perfection)
    - PROMETHEE measures pairwise consistency (how reliably you beat others)
    - A configuration that narrowly beats every competitor on most criteria ranks
      highly in PROMETHEE even if far from the theoretical ideal in TOPSIS
    
    Args:
        matrix: Decision matrix (m alternatives × n criteria)
        weights: Criterion weights (must sum to 1)
        criterion_types: List indicating if each criterion is BENEFIT or COST
        q_thresholds: Indifference thresholds (differences below this = 0 preference)
                      If None, defaults to 0.05 * range for each criterion
        p_thresholds: Preference thresholds (differences above this = 1 preference)
                      If None, defaults to 0.3 * range for each criterion
        
    Returns:
        MCDMResult with net flows (higher = better) and rankings
    """
    weights = np.array(weights)
    if not np.isclose(weights.sum(), 1.0):
        weights = weights / weights.sum()
    
    m, n = matrix.shape
    
    # Set default thresholds based on criterion ranges
    if q_thresholds is None or p_thresholds is None:
        ranges = matrix.max(axis=0) - matrix.min(axis=0)
        ranges[ranges < 1e-10] = 1.0  # Avoid division by zero
        
        if q_thresholds is None:
            q_thresholds = 0.05 * ranges  # 5% of range = indifference
        if p_thresholds is None:
            p_thresholds = 0.30 * ranges  # 30% of range = strict preference
    
    q_thresholds = np.array(q_thresholds)
    p_thresholds = np.array(p_thresholds)
    
    # Ensure p > q
    p_thresholds = np.maximum(p_thresholds, q_thresholds + 1e-10)
    
    def preference_function(diff: float, q: float, p: float) -> float:
        """V-shape (linear) preference function."""
        abs_diff = abs(diff)
        if abs_diff <= q:
            return 0.0
        elif abs_diff >= p:
            return 1.0
        else:
            return (abs_diff - q) / (p - q)
    
    # Compute pairwise preference matrix
    # pi[a, b] = aggregated preference of a over b
    pi = np.zeros((m, m))
    
    for a in range(m):
        for b in range(m):
            if a == b:
                continue
            
            aggregated_pref = 0.0
            for j in range(n):
                # Compute difference based on criterion type
                if criterion_types[j] == CriterionType.BENEFIT:
                    diff = matrix[a, j] - matrix[b, j]
                else:  # COST: lower is better, so flip
                    diff = matrix[b, j] - matrix[a, j]
                
                # Only count if a is better than b (diff > 0)
                if diff > 0:
                    pref = preference_function(diff, q_thresholds[j], p_thresholds[j])
                    aggregated_pref += weights[j] * pref
            
            pi[a, b] = aggregated_pref
    
    # Compute flows
    # Positive flow: how much a outranks others (average of pi[a, :])
    # Negative flow: how much others outrank a (average of pi[:, a])
    positive_flow = pi.sum(axis=1) / (m - 1)
    negative_flow = pi.sum(axis=0) / (m - 1)
    
    # Net flow
    net_flow = positive_flow - negative_flow
    
    # Rankings (1 = best, highest net flow)
    rankings = net_flow.argsort()[::-1].argsort() + 1
    
    return MCDMResult(method="PROMETHEE II", scores=net_flow, rankings=rankings)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def run_all_methods(
    matrix: np.ndarray,
    weights: np.ndarray,
    criterion_types: List[CriterionType],
    alternative_names: Optional[List[str]] = None
) -> Dict[str, MCDMResult]:
    """
    Run all four MCDM methods on the same decision matrix.
    
    Args:
        matrix: Decision matrix (m alternatives × n criteria)
        weights: Criterion weights
        criterion_types: List of CriterionType for each criterion
        alternative_names: Optional names for alternatives (for display)
        
    Returns:
        Dictionary mapping method name to MCDMResult
    """
    results = {
        "WSM": wsm(matrix, weights, criterion_types),
        "WPM": wpm(matrix, weights, criterion_types),
        "TOPSIS": topsis(matrix, weights, criterion_types),
        "PROMETHEE": promethee_ii(matrix, weights, criterion_types)
    }
    return results


def compare_rankings(results: Dict[str, MCDMResult], alternative_names: List[str]) -> None:
    """Print a comparison table of rankings across methods."""
    print("\n" + "=" * 60)
    print("MCDM RANKING COMPARISON")
    print("=" * 60)
    
    header = f"{'Alternative':<15}"
    for method in results:
        header += f"{method:>12}"
    print(header)
    print("-" * 60)
    
    n_alternatives = len(alternative_names)
    for i in range(n_alternatives):
        row = f"{alternative_names[i]:<15}"
        for method, result in results.items():
            row += f"{int(result.rankings[i]):>12}"
        print(row)
    
    print("-" * 60)
    print("Best alternative by method:")
    for method, result in results.items():
        best_idx = result.get_best()
        print(f"  {method}: {alternative_names[best_idx]} (score: {result.scores[best_idx]:.4f})")


def check_rank_reversal(
    matrix: np.ndarray,
    weights: np.ndarray,
    criterion_types: List[CriterionType],
    alternative_names: List[str],
    method: str = "TOPSIS"
) -> Dict[str, any]:
    """
    Test for rank reversal by removing each alternative and re-running MCDM.
    
    Rank reversal occurs when adding or removing an alternative changes the
    ranking of the remaining alternatives. TOPSIS is particularly susceptible
    because its ideal/anti-ideal points are derived from the alternatives.
    
    Args:
        matrix: Decision matrix
        weights: Criterion weights
        criterion_types: Criterion types
        alternative_names: Names of alternatives
        method: Which MCDM method to test
        
    Returns:
        Dictionary with rank reversal analysis results
    """
    m = matrix.shape[0]
    
    # Get baseline ranking
    if method == "TOPSIS":
        baseline = topsis(matrix, weights, criterion_types)
    elif method == "WSM":
        baseline = wsm(matrix, weights, criterion_types)
    elif method == "WPM":
        baseline = wpm(matrix, weights, criterion_types)
    else:
        baseline = promethee_ii(matrix, weights, criterion_types)
    
    reversals = []
    
    for remove_idx in range(m):
        # Create reduced matrix
        mask = np.ones(m, dtype=bool)
        mask[remove_idx] = False
        reduced_matrix = matrix[mask]
        reduced_names = [n for i, n in enumerate(alternative_names) if i != remove_idx]
        
        # Re-run MCDM
        if method == "TOPSIS":
            reduced_result = topsis(reduced_matrix, weights, criterion_types)
        elif method == "WSM":
            reduced_result = wsm(reduced_matrix, weights, criterion_types)
        elif method == "WPM":
            reduced_result = wpm(reduced_matrix, weights, criterion_types)
        else:
            reduced_result = promethee_ii(reduced_matrix, weights, criterion_types)
        
        # Check if relative rankings changed
        # Map original indices to reduced indices
        original_order = baseline.get_ranking_order()
        original_order_reduced = [i for i in original_order if i != remove_idx]
        
        reduced_order = reduced_result.get_ranking_order()
        # Map back to original indices
        reduced_order_original = [
            list(np.where(mask)[0])[i] for i in reduced_order
        ]
        
        if original_order_reduced != reduced_order_original:
            reversals.append({
                "removed": alternative_names[remove_idx],
                "original_order": [alternative_names[i] for i in original_order_reduced],
                "new_order": [alternative_names[i] for i in reduced_order_original]
            })
    
    return {
        "method": method,
        "baseline_ranking": [alternative_names[i] for i in baseline.get_ranking_order()],
        "reversals_found": len(reversals),
        "reversals": reversals
    }


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    # Demo with dummy data representing 8 supply chain configurations
    # Criteria: Cost, Reliability, Recovery Time, Financial Exposure, Asset Efficiency
    # (based on SCOR model)
    
    print("MCDM Module Demo")
    print("=" * 60)
    
    # Decision matrix: 8 configurations × 5 criteria
    # Columns: Cost (lower=better), Reliability (higher=better), 
    #          Recovery Time (lower=better), Financial Exposure (lower=better),
    #          Asset Efficiency (higher=better)
    matrix = np.array([
        [100, 0.92, 15, 500, 0.08],   # C1: Single/Lean/None - cheapest but fragile
        [120, 0.94, 12, 400, 0.09],   # C2: Single/Buffered/None
        [140, 0.95, 10, 350, 0.10],   # C3: Dual/Lean/None
        [160, 0.96, 8, 300, 0.11],    # C4: Dual/Buffered/None
        [150, 0.93, 6, 320, 0.10],    # C5: Single/Lean/Full IT
        [200, 0.98, 3, 150, 0.12],    # C6: Dual/Buffered/Full - max resilience
        [170, 0.95, 7, 280, 0.11],    # C7: Multi/Lean/Partial
        [180, 0.96, 5, 220, 0.12],    # C8: Multi/Buffered/Partial - balanced
    ])
    
    config_names = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
    criteria_names = ["Cost", "Reliability", "Recovery", "FinExposure", "AssetEff"]
    
    criterion_types = [
        CriterionType.COST,     # Cost: lower is better
        CriterionType.BENEFIT,  # Reliability: higher is better
        CriterionType.COST,     # Recovery Time: lower is better
        CriterionType.COST,     # Financial Exposure: lower is better
        CriterionType.BENEFIT   # Asset Efficiency: higher is better
    ]
    
    # Three stakeholder profiles
    stakeholder_weights = {
        "CFO": np.array([0.40, 0.15, 0.10, 0.25, 0.10]),  # Cost-focused
        "COO": np.array([0.15, 0.30, 0.25, 0.15, 0.15]),  # Operations-focused
        "CMO": np.array([0.10, 0.35, 0.15, 0.15, 0.25])   # Customer-focused
    }
    
    print("\nDecision Matrix:")
    print("-" * 60)
    header = f"{'Config':<8}" + "".join(f"{c:>12}" for c in criteria_names)
    print(header)
    for i, name in enumerate(config_names):
        row = f"{name:<8}" + "".join(f"{matrix[i,j]:>12.2f}" for j in range(5))
        print(row)
    
    # Run analysis for each stakeholder
    for stakeholder, weights in stakeholder_weights.items():
        print(f"\n\n{'=' * 60}")
        print(f"STAKEHOLDER: {stakeholder}")
        print(f"Weights: {dict(zip(criteria_names, weights))}")
        print("=" * 60)
        
        results = run_all_methods(matrix, weights, criterion_types)
        compare_rankings(results, config_names)
    
    # Rank reversal test
    print("\n\n" + "=" * 60)
    print("RANK REVERSAL TEST (TOPSIS, CFO weights)")
    print("=" * 60)
    reversal_results = check_rank_reversal(
        matrix, 
        stakeholder_weights["CFO"], 
        criterion_types, 
        config_names,
        method="TOPSIS"
    )
    print(f"Baseline ranking: {reversal_results['baseline_ranking']}")
    print(f"Reversals found: {reversal_results['reversals_found']}")
    for rev in reversal_results['reversals']:
        print(f"\n  Removed {rev['removed']}:")
        print(f"    Original order: {rev['original_order']}")
        print(f"    New order:      {rev['new_order']}")
