"""
SCOR-Based Evaluation Criteria - Integrated with Flow Solver

Computes five SCOR criteria from actual network simulation:
1. Cost - Total Cost to Serve (from flow solver)
2. Reliability - Fill Rate (from flow solver)
3. Recovery Time - Days to 95% service (from disruptions.py)
4. Financial Exposure - Expected loss from shortfall
5. Asset Efficiency - Throughput per unit cost

All values derived from LP flow solver output, not heuristics.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from configurations import build_configuration, get_config_properties, CONFIG_SPECS
from disruptions import (
    DisruptionType, 
    apply_disruption, 
    estimate_recovery_time as get_recovery_time
)
from flow_solver import solve_min_cost_flow


class Scenario(Enum):
    """Disruption scenarios (maps to DisruptionType)."""
    NORMAL = "normal"
    CYBERATTACK = "cyberattack"
    SUPPLIER_FAILURE = "supplier_failure"
    DEMAND_SURGE = "demand_surge"
    COMPOUND = "compound"


# Mapping between Scenario and DisruptionType
SCENARIO_TO_DISRUPTION = {
    Scenario.NORMAL: DisruptionType.NORMAL,
    Scenario.CYBERATTACK: DisruptionType.CYBERATTACK,
    Scenario.SUPPLIER_FAILURE: DisruptionType.SUPPLIER_FAILURE,
    Scenario.DEMAND_SURGE: DisruptionType.DEMAND_SURGE,
    Scenario.COMPOUND: DisruptionType.COMPOUND,
}


@dataclass
class CriteriaScores:
    """Container for all five criteria scores."""
    cost: float                 # Lower is better ($)
    reliability: float          # Higher is better (0-1)
    recovery_time: float        # Lower is better (days)
    financial_exposure: float   # Lower is better ($)
    asset_efficiency: float     # Higher is better (ratio)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array in standard order."""
        return np.array([
            self.cost,
            self.reliability,
            self.recovery_time,
            self.financial_exposure,
            self.asset_efficiency
        ])
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "Cost": self.cost,
            "Reliability": self.reliability,
            "Recovery": self.recovery_time,
            "FinExposure": self.financial_exposure,
            "AssetEfficiency": self.asset_efficiency
        }


# =============================================================================
# FLOW SOLVER INTEGRATION
# =============================================================================

def run_flow_simulation(config_id: str, scenario: Scenario) -> Dict:
    """
    Run flow solver for a configuration under a scenario.
    
    Returns dict with:
    - total_cost: Flow cost from LP
    - fill_rate: Satisfied demand / total demand
    - satisfied_demand: Units delivered
    - total_demand: Units requested
    """
    # Build network
    net = build_configuration(config_id)
    
    # Apply disruption
    disruption_type = SCENARIO_TO_DISRUPTION[scenario]
    disrupted_net = apply_disruption(net, disruption_type)
    
    # Solve
    result = solve_min_cost_flow(disrupted_net)
    
    # Get total demand for fill rate calculation
    total_demand = sum(n.demand for n in disrupted_net.nodes.values())
    
    return {
        'total_cost': result.get('total_cost', 0),
        'fill_rate': result.get('fill_rate', 0),
        'satisfied_demand': result.get('satisfied_demand', 0),
        'total_demand': total_demand,
        'success': result.get('success', False)
    }


# =============================================================================
# CRITERION FUNCTIONS
# =============================================================================

def compute_cost(config_id: str, scenario: Scenario, flow_result: Optional[Dict] = None) -> float:
    """
    Compute Total Cost to Serve.
    
    Components:
    - Flow cost (from LP solver)
    - Fixed infrastructure cost (IT redundancy, buffer capacity)
    
    Args:
        config_id: Configuration identifier (C1-C8)
        scenario: Disruption scenario
        flow_result: Pre-computed flow result (optional, will compute if None)
        
    Returns:
        Total cost (lower is better)
    """
    if flow_result is None:
        flow_result = run_flow_simulation(config_id, scenario)
    
    # Flow cost from LP
    flow_cost = flow_result['total_cost']
    
    # Fixed infrastructure costs
    props = get_config_properties(config_id)
    
    # IT infrastructure cost
    it_fixed_cost = {
        "none": 0,
        "partial": 100,
        "full": 250
    }[props.it_redundancy]
    
    # Buffer holding cost (if buffered inventory)
    buffer_cost = 50 if props.inventory == "buffered" else 0
    
    # Multi-sourcing overhead
    sourcing_overhead = {
        "single": 0,
        "dual": 80,
        "multi": 150
    }[props.sourcing]
    
    total_cost = flow_cost + it_fixed_cost + buffer_cost + sourcing_overhead
    
    return total_cost


def compute_reliability(config_id: str, scenario: Scenario, flow_result: Optional[Dict] = None) -> float:
    """
    Compute Perfect Order Fulfillment Rate.
    
    Directly from flow solver: satisfied_demand / total_demand
    
    Args:
        config_id: Configuration identifier (C1-C8)
        scenario: Disruption scenario
        flow_result: Pre-computed flow result (optional)
        
    Returns:
        Fill rate (0-1, higher is better)
    """
    if flow_result is None:
        flow_result = run_flow_simulation(config_id, scenario)
    
    return flow_result['fill_rate']


def compute_recovery_time(config_id: str, scenario: Scenario) -> float:
    """
    Compute Recovery Time (days to 95% service level).
    
    Uses estimate_recovery_time from disruptions.py.
    
    Args:
        config_id: Configuration identifier (C1-C8)
        scenario: Disruption scenario
        
    Returns:
        Recovery time in days (lower is better)
    """
    disruption_type = SCENARIO_TO_DISRUPTION[scenario]
    return get_recovery_time(config_id, disruption_type)


def compute_financial_exposure(
    config_id: str, 
    scenario: Scenario, 
    flow_result: Optional[Dict] = None,
    revenue_per_unit: float = 10.0,
    penalty_per_unit: float = 5.0
) -> float:
    """
    Compute Expected Financial Exposure.
    
    Components:
    - Lost revenue from unfulfilled demand
    - Contractual penalties for shortfall
    - Recovery costs (proportional to recovery time)
    
    Args:
        config_id: Configuration identifier (C1-C8)
        scenario: Disruption scenario
        flow_result: Pre-computed flow result (optional)
        revenue_per_unit: Revenue per unit of demand
        penalty_per_unit: Penalty per unit of unfulfilled demand
        
    Returns:
        Financial exposure in $ (lower is better)
    """
    if flow_result is None:
        flow_result = run_flow_simulation(config_id, scenario)
    
    # Shortfall
    shortfall = flow_result['total_demand'] - flow_result['satisfied_demand']
    
    # Lost revenue
    lost_revenue = shortfall * revenue_per_unit
    
    # Penalties
    penalties = shortfall * penalty_per_unit
    
    # Recovery costs (proportional to time and shortfall magnitude)
    recovery_days = compute_recovery_time(config_id, scenario)
    recovery_cost = recovery_days * shortfall * 0.5  # $0.50 per unit per day
    
    total_exposure = lost_revenue + penalties + recovery_cost
    
    return total_exposure


def compute_asset_efficiency(
    config_id: str, 
    scenario: Scenario, 
    flow_result: Optional[Dict] = None
) -> float:
    """
    Compute Asset Efficiency (throughput / cost).
    
    Measures how efficiently the configuration converts investment into output.
    
    Args:
        config_id: Configuration identifier (C1-C8)
        scenario: Disruption scenario
        flow_result: Pre-computed flow result (optional)
        
    Returns:
        Asset efficiency ratio (higher is better)
    """
    if flow_result is None:
        flow_result = run_flow_simulation(config_id, scenario)
    
    throughput = flow_result['satisfied_demand']
    cost = compute_cost(config_id, scenario, flow_result)
    
    # Avoid division by zero
    if cost <= 0:
        return 0.0
    
    efficiency = throughput / cost
    
    return efficiency


# =============================================================================
# AGGREGATE SCORING
# =============================================================================

def evaluate_configuration(
    config_id: str, 
    scenario: Scenario = Scenario.NORMAL
) -> CriteriaScores:
    """
    Evaluate a configuration on all five criteria.
    
    Runs flow simulation once and reuses result for all criteria.
    
    Args:
        config_id: Configuration identifier (C1-C8)
        scenario: Disruption scenario
        
    Returns:
        CriteriaScores object with all five criteria values
    """
    # Run simulation once
    flow_result = run_flow_simulation(config_id, scenario)
    
    return CriteriaScores(
        cost=compute_cost(config_id, scenario, flow_result),
        reliability=compute_reliability(config_id, scenario, flow_result),
        recovery_time=compute_recovery_time(config_id, scenario),
        financial_exposure=compute_financial_exposure(config_id, scenario, flow_result),
        asset_efficiency=compute_asset_efficiency(config_id, scenario, flow_result)
    )


def build_decision_matrix(
    config_ids: List[str],
    scenario: Scenario = Scenario.NORMAL
) -> np.ndarray:
    """
    Build decision matrix for all configurations under a scenario.
    
    Args:
        config_ids: List of configuration identifiers
        scenario: Disruption scenario
        
    Returns:
        Decision matrix (m configurations × 5 criteria)
    """
    scores = [evaluate_configuration(c, scenario).to_array() for c in config_ids]
    return np.array(scores)


def build_expected_matrix(
    config_ids: List[str],
    scenario_probs: Dict[Scenario, float]
) -> np.ndarray:
    """
    Build expected decision matrix weighted by scenario probabilities.
    
    Args:
        config_ids: List of configuration identifiers
        scenario_probs: Dictionary mapping scenarios to probabilities
        
    Returns:
        Expected decision matrix (probability-weighted average across scenarios)
    """
    matrices = []
    weights = []
    
    for scenario, prob in scenario_probs.items():
        matrices.append(build_decision_matrix(config_ids, scenario))
        weights.append(prob)
    
    # Weighted average
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize
    
    expected = sum(w * m for w, m in zip(weights, matrices))
    return expected


# =============================================================================
# CRITERIA METADATA
# =============================================================================

CRITERIA_INFO = {
    "Cost": {
        "index": 0,
        "direction": "min",  # Lower is better
        "unit": "$",
        "description": "Total cost to serve including flow, infrastructure, and overhead"
    },
    "Reliability": {
        "index": 1,
        "direction": "max",  # Higher is better
        "unit": "%",
        "description": "Perfect order fulfillment rate (fill rate)"
    },
    "Recovery": {
        "index": 2,
        "direction": "min",  # Lower is better
        "unit": "days",
        "description": "Days to recover to 95% service level"
    },
    "FinExposure": {
        "index": 3,
        "direction": "min",  # Lower is better
        "unit": "$",
        "description": "Expected financial loss from disruption"
    },
    "AssetEfficiency": {
        "index": 4,
        "direction": "max",  # Higher is better
        "unit": "ratio",
        "description": "Throughput per unit cost"
    }
}

# For MCDM methods: True = benefit (higher is better), False = cost (lower is better)
CRITERIA_DIRECTIONS = [False, True, False, False, True]  # [Cost, Rel, Rec, FinExp, AssetEff]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    configs = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
    
    print("CRITERIA EVALUATION - FLOW SOLVER INTEGRATED")
    print("=" * 90)
    
    # Normal operations
    print("\nNORMAL OPERATIONS:")
    print("-" * 90)
    header = f"{'Config':<8}{'Cost':>12}{'Reliability':>12}{'Recovery':>12}{'FinExposure':>14}{'AssetEff':>12}"
    print(header)
    print("-" * 90)
    
    for c in configs:
        scores = evaluate_configuration(c, Scenario.NORMAL)
        print(f"{c:<8}{scores.cost:>12.1f}{scores.reliability:>11.1%}{scores.recovery_time:>12.1f}"
              f"{scores.financial_exposure:>14.1f}{scores.asset_efficiency:>12.4f}")
    
    # Cyberattack scenario
    print("\n\nCYBERATTACK SCENARIO:")
    print("-" * 90)
    print(header)
    print("-" * 90)
    
    for c in configs:
        scores = evaluate_configuration(c, Scenario.CYBERATTACK)
        print(f"{c:<8}{scores.cost:>12.1f}{scores.reliability:>11.1%}{scores.recovery_time:>12.1f}"
              f"{scores.financial_exposure:>14.1f}{scores.asset_efficiency:>12.4f}")
    
    # Compound scenario (worst case)
    print("\n\nCOMPOUND SCENARIO (Cyber + Supplier Failure):")
    print("-" * 90)
    print(header)
    print("-" * 90)
    
    for c in configs:
        scores = evaluate_configuration(c, Scenario.COMPOUND)
        print(f"{c:<8}{scores.cost:>12.1f}{scores.reliability:>11.1%}{scores.recovery_time:>12.1f}"
              f"{scores.financial_exposure:>14.1f}{scores.asset_efficiency:>12.4f}")
    
    # Expected matrix with disruption probabilities
    print("\n\nEXPECTED VALUES (Probability-Weighted):")
    print("-" * 90)
    print("Scenario probabilities: Normal=85%, Cyber=5%, Supplier=5%, Demand=4%, Compound=1%")
    print("-" * 90)
    
    scenario_probs = {
        Scenario.NORMAL: 0.85,
        Scenario.CYBERATTACK: 0.05,
        Scenario.SUPPLIER_FAILURE: 0.05,
        Scenario.DEMAND_SURGE: 0.04,
        Scenario.COMPOUND: 0.01
    }
    
    expected_matrix = build_expected_matrix(configs, scenario_probs)
    
    print(header)
    print("-" * 90)
    for i, c in enumerate(configs):
        row = expected_matrix[i]
        print(f"{c:<8}{row[0]:>12.1f}{row[1]:>11.1%}{row[2]:>12.1f}"
              f"{row[3]:>14.1f}{row[4]:>12.4f}")
