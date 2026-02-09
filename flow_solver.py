# flow_solver.py
"""
Min-Cost Flow Solver for Supply Chain Networks

Solves the minimum cost flow problem using linear programming.
Handles partial demand satisfaction when supply < demand.
"""

import numpy as np
from scipy.optimize import linprog
from models import Network


def solve_min_cost_flow(network: Network, verbose: bool = False):
    """
    Solve min-cost flow problem, allowing partial demand satisfaction.
    
    Uses a two-phase approach:
    1. Try exact demand satisfaction
    2. If infeasible, maximize flow while minimizing cost
    
    Args:
        network: Network with nodes and edges
        verbose: Print debug information
        
    Returns:
        Dictionary with success, total_cost, flows, satisfied_demand, fill_rate
    """
    edges = network.edges
    nodes = list(network.nodes.values())
    n_edges = len(edges)

    total_supply = sum(n.supply for n in nodes)
    total_demand = sum(n.demand for n in nodes)
    
    if verbose:
        print(f"Debug: Total supply={total_supply}, Total demand={total_demand}")

    if n_edges == 0:
        return {
            'success': False,
            'total_cost': 0,
            'flows': {},
            'satisfied_demand': 0,
            'fill_rate': 0,
            'message': "No edges in network."
        }

    # Try exact solution first if supply >= demand
    if total_supply >= total_demand:
        result = _solve_exact(network, edges, nodes, n_edges, total_demand)
        if result['success']:
            return result
        # Exact failed, fall through to partial
    
    # Fall back to max-flow formulation (partial satisfaction)
    result = _solve_max_flow(network, edges, nodes, n_edges, total_demand)
    return result


def _solve_exact(network, edges, nodes, n_edges, total_demand):
    """Try to solve with exact demand satisfaction, allowing unused supply."""
    
    # Add dummy sink for excess supply
    suppliers = [n for n in nodes if n.node_type == "supplier" and n.supply > 0]
    n_sink_edges = len(suppliers)
    n_vars = n_edges + n_sink_edges
    
    # Objective: min sum(cost * flow) + 0 * sink_flow
    c = np.zeros(n_vars)
    for i, edge in enumerate(edges):
        c[i] = float(edge.cost_per_unit)
    # Sink edges have zero cost
    
    # Bounds: 0 <= flow <= max_flow for edges, 0 <= sink <= supply for sink edges
    bounds = [(0.0, float(edge.max_flow)) for edge in edges]
    for supplier in suppliers:
        bounds.append((0.0, float(supplier.supply)))
    
    # Flow conservation with sink at suppliers
    A_eq = []
    b_eq = []
    
    supplier_idx = {s.node_id: i for i, s in enumerate(suppliers)}

    for node in nodes:
        row = [0.0] * n_vars
        rhs = float(node.demand - node.supply)

        for j, edge in enumerate(edges):
            if edge.to_node.node_id == node.node_id:
                row[j] += 1.0
            elif edge.from_node.node_id == node.node_id:
                row[j] -= 1.0
        
        # Add sink variable for suppliers (unused supply goes to sink)
        if node.node_id in supplier_idx:
            sink_idx = n_edges + supplier_idx[node.node_id]
            row[sink_idx] = -1.0  # sink is outflow

        A_eq.append(row)
        b_eq.append(rhs)

    A_eq = np.array(A_eq, dtype=float)
    b_eq = np.array(b_eq, dtype=float)

    res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if res.success:
        flows = {edge.edge_id: float(res.x[i]) for i, edge in enumerate(edges)}
        satisfied_demand = sum(
            res.x[i] for i, edge in enumerate(edges) 
            if edge.to_node.node_type == "retailer"
        )
        fill_rate = satisfied_demand / total_demand if total_demand > 0 else 1.0
        
        # Calculate real cost (only edge flows, not sinks)
        real_cost = sum(res.x[i] * edges[i].cost_per_unit for i in range(n_edges))
        
        return {
            'success': True,
            'total_cost': float(real_cost),
            'flows': flows,
            'satisfied_demand': float(satisfied_demand),
            'fill_rate': float(fill_rate),
            'message': "success"
        }
    else:
        return {'success': False, 'message': res.message}


def _solve_max_flow(network, edges, nodes, n_edges, total_demand):
    """
    Solve with partial demand satisfaction using slack variables.
    
    Adds slack variables at each retailer to absorb unmet demand,
    and sink variables at suppliers for unused supply.
    """
    retailers = [n for n in nodes if n.node_type == "retailer"]
    suppliers = [n for n in nodes if n.node_type == "supplier" and n.supply > 0]
    n_slack = len(retailers)
    n_sink = len(suppliers)
    n_vars = n_edges + n_slack + n_sink
    
    # Cost vector: [edge costs..., slack penalties..., sink costs (zero)...]
    max_cost = max(e.cost_per_unit for e in edges) if edges else 1
    slack_penalty = max_cost * 100
    
    c = np.zeros(n_vars)
    for i, edge in enumerate(edges):
        c[i] = float(edge.cost_per_unit)
    for i in range(n_slack):
        c[n_edges + i] = slack_penalty
    # Sink edges at end have zero cost
    
    # Bounds
    bounds = [(0.0, float(edge.max_flow)) for edge in edges]
    for retailer in retailers:
        bounds.append((0.0, float(retailer.demand)))
    for supplier in suppliers:
        bounds.append((0.0, float(supplier.supply)))
    
    # Flow conservation
    A_eq = []
    b_eq = []
    
    retailer_idx = {r.node_id: i for i, r in enumerate(retailers)}
    supplier_idx = {s.node_id: i for i, s in enumerate(suppliers)}
    
    for node in nodes:
        row = [0.0] * n_vars
        rhs = float(node.demand - node.supply)

        for j, edge in enumerate(edges):
            if edge.to_node.node_id == node.node_id:
                row[j] += 1.0
            elif edge.from_node.node_id == node.node_id:
                row[j] -= 1.0
        
        # Slack at retailers
        if node.node_id in retailer_idx:
            slack_idx = n_edges + retailer_idx[node.node_id]
            row[slack_idx] = 1.0
        
        # Sink at suppliers
        if node.node_id in supplier_idx:
            sink_idx = n_edges + n_slack + supplier_idx[node.node_id]
            row[sink_idx] = -1.0

        A_eq.append(row)
        b_eq.append(rhs)

    A_eq = np.array(A_eq, dtype=float)
    b_eq = np.array(b_eq, dtype=float)

    res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if res.success:
        flows = {edge.edge_id: float(res.x[i]) for i, edge in enumerate(edges)}
        
        total_slack = sum(res.x[n_edges + i] for i in range(n_slack))
        satisfied_demand = total_demand - total_slack
        
        real_cost = sum(res.x[i] * edges[i].cost_per_unit for i in range(n_edges))
        
        fill_rate = satisfied_demand / total_demand if total_demand > 0 else 1.0
        
        return {
            'success': True,
            'total_cost': float(real_cost),
            'flows': flows,
            'satisfied_demand': float(satisfied_demand),
            'fill_rate': float(fill_rate),
            'message': f"Partial satisfaction: {fill_rate:.1%}"
        }
    else:
        return {
            'success': False,
            'total_cost': 0,
            'flows': {},
            'satisfied_demand': 0,
            'fill_rate': 0,
            'message': f"Failed: {res.message}"
        }
