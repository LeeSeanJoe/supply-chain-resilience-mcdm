# disruptions.py
"""
Disruption Scenarios for Supply Chain Network

Four disruption types based on real-world incidents:
1. Cyberattack - Warehouse visibility/processing loss (Asahi 2025 style)
2. Supplier Failure - Primary supplier goes offline (Toyota/Kojima 2022 style)
3. Demand Surge - Bullwhip effect spike (30-100% increase)
4. Compound - Cyber + supplier failure simultaneously

Each function takes a Network and returns a modified copy.
IT redundancy level affects cyber disruption severity.
"""

from models import Network, Node, Edge
from configurations import clone_network, get_config_properties
from typing import Dict, Optional
from enum import Enum


class DisruptionType(Enum):
    """Disruption scenario types."""
    NORMAL = "normal"
    CYBERATTACK = "cyberattack"
    SUPPLIER_FAILURE = "supplier_failure"
    DEMAND_SURGE = "demand_surge"
    COMPOUND = "compound"


# =============================================================================
# DISRUPTION FUNCTIONS
# =============================================================================

def apply_cyberattack(
    net: Network, 
    severity: float = 1.0,
    affected_warehouses: Optional[list] = None
) -> Network:
    """
    Simulate cyberattack on warehouse operations.
    
    Based on Asahi 2025 incident: ransomware paralyzed ordering/shipment
    systems, forcing manual operations with ~10% throughput.
    
    Effects:
    - Warehouse outbound capacity reduced (severity dependent)
    - IT redundancy mitigates: none=full impact, partial=50%, full=10%
    
    Args:
        net: Network to disrupt
        severity: Disruption severity (0-1, default 1.0)
        affected_warehouses: Which warehouses affected (default: all)
        
    Returns:
        Modified network copy
    """
    disrupted = clone_network(net)
    
    if affected_warehouses is None:
        affected_warehouses = ["W1", "W2", "W3"]
    
    # IT redundancy mitigation factor
    it_redundancy = getattr(disrupted, 'it_redundancy', 'none')
    mitigation = {
        "none": 0.0,      # No protection: full impact
        "partial": 0.5,   # 50% protection
        "full": 0.9       # 90% protection (still some impact)
    }.get(it_redundancy, 0.0)
    
    # Effective severity after IT mitigation
    effective_severity = severity * (1 - mitigation)
    
    # Reduce warehouse outbound capacity
    # At full severity with no IT: capacity drops to 10% (Asahi-level)
    capacity_factor = 1.0 - (effective_severity * 0.9)
    
    for edge in disrupted.edges:
        if edge.from_node.node_id in affected_warehouses:
            edge.max_flow = edge.max_flow * capacity_factor
    
    return disrupted


def apply_supplier_failure(
    net: Network, 
    failed_supplier: str = "S1",
    partial: bool = False
) -> Network:
    """
    Simulate supplier failure/disruption.
    
    Based on Toyota/Kojima 2022: cyberattack on parts supplier forced
    production halt across 14 plants.
    
    Effects:
    - Failed supplier's supply set to 0
    - All edges from that supplier have max_flow = 0
    
    Args:
        net: Network to disrupt
        failed_supplier: Which supplier fails (default: S1, the primary)
        partial: If True, reduce to 20% instead of 0
        
    Returns:
        Modified network copy
    """
    disrupted = clone_network(net)
    
    # Reduce supplier capacity
    supplier_node = disrupted.nodes.get(failed_supplier)
    if supplier_node:
        if partial:
            supplier_node.supply = int(supplier_node.supply * 0.2)
        else:
            supplier_node.supply = 0
    
    # Reduce/eliminate edges from failed supplier
    for edge in disrupted.edges:
        if edge.from_node.node_id == failed_supplier:
            if partial:
                edge.max_flow = edge.max_flow * 0.2
            else:
                edge.max_flow = 0
    
    return disrupted


def apply_demand_surge(
    net: Network, 
    multiplier: float = 1.5,
    affected_retailers: Optional[list] = None
) -> Network:
    """
    Simulate demand surge (bullwhip effect).
    
    Effects:
    - Retailer demand increased by multiplier
    - Buffered inventory helps absorb (handled in flow solver feasibility)
    
    Args:
        net: Network to disrupt
        multiplier: Demand multiplier (1.5 = 50% increase, 2.0 = 100%)
        affected_retailers: Which retailers affected (default: all)
        
    Returns:
        Modified network copy
    """
    disrupted = clone_network(net)
    
    if affected_retailers is None:
        affected_retailers = ["R1", "R2", "R3", "R4", "R5"]
    
    for r_id in affected_retailers:
        retailer = disrupted.nodes.get(r_id)
        if retailer:
            retailer.demand = int(retailer.demand * multiplier)
    
    return disrupted


def apply_compound(
    net: Network,
    cyber_severity: float = 0.8,
    failed_supplier: str = "S1"
) -> Network:
    """
    Simulate compound disruption: cyber + supplier failure.
    
    Worst-case scenario where multiple disruptions hit simultaneously.
    
    Args:
        net: Network to disrupt
        cyber_severity: Cyberattack severity
        failed_supplier: Which supplier fails
        
    Returns:
        Modified network copy with both disruptions applied
    """
    # Apply cyber first, then supplier failure
    disrupted = apply_cyberattack(net, severity=cyber_severity)
    disrupted = apply_supplier_failure(disrupted, failed_supplier=failed_supplier)
    
    return disrupted


# =============================================================================
# UNIFIED DISRUPTION INTERFACE
# =============================================================================

def apply_disruption(
    net: Network, 
    disruption_type: DisruptionType,
    **kwargs
) -> Network:
    """
    Apply a disruption scenario to a network.
    
    Args:
        net: Network to disrupt
        disruption_type: Type of disruption
        **kwargs: Disruption-specific parameters
        
    Returns:
        Modified network copy (original unchanged)
    """
    if disruption_type == DisruptionType.NORMAL:
        return clone_network(net)  # No disruption, just return copy
    
    elif disruption_type == DisruptionType.CYBERATTACK:
        severity = kwargs.get('severity', 1.0)
        return apply_cyberattack(net, severity=severity)
    
    elif disruption_type == DisruptionType.SUPPLIER_FAILURE:
        failed_supplier = kwargs.get('failed_supplier', 'S1')
        return apply_supplier_failure(net, failed_supplier=failed_supplier)
    
    elif disruption_type == DisruptionType.DEMAND_SURGE:
        multiplier = kwargs.get('multiplier', 1.5)
        return apply_demand_surge(net, multiplier=multiplier)
    
    elif disruption_type == DisruptionType.COMPOUND:
        return apply_compound(net, **kwargs)
    
    else:
        raise ValueError(f"Unknown disruption type: {disruption_type}")


# =============================================================================
# RECOVERY TIME ESTIMATION
# =============================================================================

def estimate_recovery_time(
    config_id: str,
    disruption_type: DisruptionType
) -> float:
    """
    Estimate days to recover to 95% service level.
    
    Based on real-world incident recovery times:
    - Asahi (cyber): ~14-21 days to restore operations
    - Toyota/Kojima (supplier): ~1-3 days (just-in-time exposure)
    - Demand surge: ~7 days to stabilize
    
    IT redundancy and multi-sourcing reduce recovery time.
    
    Args:
        config_id: Configuration identifier
        disruption_type: Type of disruption
        
    Returns:
        Estimated recovery time in days
    """
    if disruption_type == DisruptionType.NORMAL:
        return 0.0
    
    from configurations import get_config_properties
    props = get_config_properties(config_id)
    
    # Base recovery times (days)
    base_times = {
        DisruptionType.CYBERATTACK: 14.0,
        DisruptionType.SUPPLIER_FAILURE: 21.0,
        DisruptionType.DEMAND_SURGE: 7.0,
        DisruptionType.COMPOUND: 28.0
    }
    
    base = base_times.get(disruption_type, 14.0)
    
    # Mitigation factors
    if disruption_type == DisruptionType.CYBERATTACK:
        # IT redundancy is primary mitigation
        it_factor = {"none": 1.0, "partial": 0.6, "full": 0.25}[props.it_redundancy]
        return base * it_factor
    
    elif disruption_type == DisruptionType.SUPPLIER_FAILURE:
        # Multi-sourcing is primary mitigation
        source_factor = {"single": 1.0, "dual": 0.5, "multi": 0.3}[props.sourcing]
        return base * source_factor
    
    elif disruption_type == DisruptionType.DEMAND_SURGE:
        # Buffer inventory is primary mitigation
        inv_factor = {"lean": 1.0, "buffered": 0.4}[props.inventory]
        return base * inv_factor
    
    elif disruption_type == DisruptionType.COMPOUND:
        # Slowest bottleneck determines recovery
        it_factor = {"none": 1.0, "partial": 0.6, "full": 0.25}[props.it_redundancy]
        source_factor = {"single": 1.0, "dual": 0.5, "multi": 0.3}[props.sourcing]
        return base * max(it_factor, source_factor)
    
    return base


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    from configurations import build_configuration, CONFIG_SPECS
    from flow_solver import solve_min_cost_flow
    
    print("DISRUPTION SCENARIO DEMO")
    print("=" * 80)
    
    # Test on C1 (most vulnerable) and C6 (most resilient)
    test_configs = ["C1", "C6"]
    
    for config_id in test_configs:
        net = build_configuration(config_id)
        props = get_config_properties(config_id)
        
        print(f"\n{'=' * 80}")
        print(f"CONFIG {config_id}: {props.description}")
        print(f"  Sourcing: {props.sourcing}, Inventory: {props.inventory}, IT: {props.it_redundancy}")
        print("=" * 80)
        
        for disruption_type in DisruptionType:
            disrupted = apply_disruption(net, disruption_type)
            result = solve_min_cost_flow(disrupted)
            recovery = estimate_recovery_time(config_id, disruption_type)
            
            total_demand = disrupted.get_total_demand()
            
            if result['success']:
                fill_rate = result['satisfied_demand'] / total_demand if total_demand > 0 else 0
                print(f"\n  {disruption_type.value:20s} | Cost: {result['total_cost']:8.1f} | "
                      f"Fill: {fill_rate:5.1%} | Recovery: {recovery:5.1f} days")
            else:
                print(f"\n  {disruption_type.value:20s} | INFEASIBLE: {result['message']}")
                print(f"      Max satisfiable: {result['satisfied_demand']} / {total_demand}")
                print(f"      Recovery: {recovery:.1f} days")
