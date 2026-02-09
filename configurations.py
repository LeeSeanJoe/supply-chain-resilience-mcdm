# configurations.py
"""
Supply Chain Configurations C1-C8

Strategic archetypes based on three decision dimensions:
1. Sourcing: single (S1 only), dual (S1+S2), multi (S1+S2+S3)
2. Inventory: lean (no buffer) vs buffered (safety stock at warehouses)
3. IT Redundancy: none, partial, full (affects cyber disruption response)

Network structure:
- 3 Suppliers (S1, S2, S3) with varying supply capacities
- 2 Factories (F1, F2)
- 3 Warehouses (W1, W2, W3) with optional buffer stock
- 5 Retailers (R1-R5) with fixed demand totaling 200 units

| Config | Sourcing | Inventory | IT Redundancy | Strategic Posture |
|--------|----------|-----------|---------------|-------------------|
| C1     | Single   | Lean      | None          | Cost-minimized    |
| C2     | Single   | Buffered  | None          | Inventory hedge   |
| C3     | Dual     | Lean      | None          | Supplier hedge    |
| C4     | Dual     | Buffered  | None          | Hedged, no cyber  |
| C5     | Single   | Lean      | Full          | Cyber-resilient   |
| C6     | Dual     | Buffered  | Full          | Max resilience    |
| C7     | Multi    | Lean      | Partial       | Diversified       |
| C8     | Multi    | Buffered  | Partial       | Balanced          |
"""

from models import Network, Node, Edge
from dataclasses import dataclass
from typing import Dict, List, Tuple
from copy import deepcopy


@dataclass
class ConfigProperties:
    """Properties defining a configuration's strategic choices."""
    config_id: str
    sourcing: str        # "single", "dual", "multi"
    inventory: str       # "lean", "buffered"
    it_redundancy: str   # "none", "partial", "full"
    description: str


# Configuration definitions
CONFIG_SPECS: Dict[str, ConfigProperties] = {
    "C1": ConfigProperties("C1", "single", "lean", "none", "Cost-minimized: single source, no buffers, no IT backup"),
    "C2": ConfigProperties("C2", "single", "buffered", "none", "Inventory hedge: single source with safety stock"),
    "C3": ConfigProperties("C3", "dual", "lean", "none", "Supplier hedge: dual source, lean operations"),
    "C4": ConfigProperties("C4", "dual", "buffered", "none", "Hedged operations: dual source with buffers"),
    "C5": ConfigProperties("C5", "single", "lean", "full", "Cyber-resilient: single source with full IT backup"),
    "C6": ConfigProperties("C6", "dual", "buffered", "full", "Maximum resilience: all hedges active"),
    "C7": ConfigProperties("C7", "multi", "lean", "partial", "Diversified: multi-source, lean, partial IT"),
    "C8": ConfigProperties("C8", "multi", "buffered", "partial", "Balanced: multi-source with buffers and partial IT"),
}


# =============================================================================
# BASE NETWORK CONSTRUCTION
# =============================================================================

def create_base_nodes() -> Dict[str, Node]:
    """Create all nodes with base supply/demand values."""
    nodes = {}
    
    # Suppliers (capacity varies by sourcing strategy)
    nodes["S1"] = Node("S1", "supplier", supply=120, demand=0)  # Primary supplier
    nodes["S2"] = Node("S2", "supplier", supply=100, demand=0)  # Secondary supplier
    nodes["S3"] = Node("S3", "supplier", supply=60, demand=0)   # Tertiary supplier
    
    # Factories (pass-through, no supply/demand)
    nodes["F1"] = Node("F1", "factory", supply=0, demand=0, capacity=150)
    nodes["F2"] = Node("F2", "factory", supply=0, demand=0, capacity=150)
    
    # Warehouses (pass-through, capacity varies with inventory strategy)
    nodes["W1"] = Node("W1", "warehouse", supply=0, demand=0, capacity=100)
    nodes["W2"] = Node("W2", "warehouse", supply=0, demand=0, capacity=100)
    nodes["W3"] = Node("W3", "warehouse", supply=0, demand=0, capacity=100)
    
    # Retailers (fixed demand, total = 200)
    nodes["R1"] = Node("R1", "retailer", supply=0, demand=40)
    nodes["R2"] = Node("R2", "retailer", supply=0, demand=50)
    nodes["R3"] = Node("R3", "retailer", supply=0, demand=30)
    nodes["R4"] = Node("R4", "retailer", supply=0, demand=50)
    nodes["R5"] = Node("R5", "retailer", supply=0, demand=30)
    
    return nodes


def build_configuration(config_id: str) -> Network:
    """
    Build a network for the specified configuration.
    
    Args:
        config_id: One of "C1" through "C8"
        
    Returns:
        Network object with appropriate nodes and edges
    """
    if config_id not in CONFIG_SPECS:
        raise ValueError(f"Unknown config: {config_id}. Valid: {list(CONFIG_SPECS.keys())}")
    
    spec = CONFIG_SPECS[config_id]
    net = Network(name=config_id)
    
    # Create nodes
    nodes = create_base_nodes()
    
    # Adjust supplier availability based on sourcing strategy
    if spec.sourcing == "single":
        # Only S1 active
        nodes["S2"].supply = 0
        nodes["S3"].supply = 0
    elif spec.sourcing == "dual":
        # S1 and S2 active
        nodes["S3"].supply = 0
    # else "multi": all suppliers active
    
    # Add buffer capacity for buffered inventory
    if spec.inventory == "buffered":
        # Buffer adds effective capacity (modeled as increased throughput)
        for w_id in ["W1", "W2", "W3"]:
            nodes[w_id].capacity = 120  # 20% more capacity with buffer
    
    # Store IT redundancy level as network attribute
    net.it_redundancy = spec.it_redundancy
    
    # Add all nodes to network
    for node in nodes.values():
        net.add_node(node)
    
    # Build edges based on sourcing strategy
    _add_edges(net, spec)
    
    return net


def _add_edges(net: Network, spec: ConfigProperties) -> None:
    """Add edges based on configuration properties."""
    
    # Define base costs
    SUPPLIER_TO_FACTORY_COST = 2.0
    FACTORY_TO_WAREHOUSE_COST = 3.0
    WAREHOUSE_TO_RETAILER_COST = 1.0
    
    # Buffered inventory adds holding cost
    inventory_cost_multiplier = 1.0 if spec.inventory == "lean" else 1.15
    
    # IT redundancy adds overhead cost
    it_cost_multiplier = {
        "none": 1.0,
        "partial": 1.10,
        "full": 1.20
    }[spec.it_redundancy]
    
    # Determine active suppliers
    if spec.sourcing == "single":
        active_suppliers = ["S1"]
    elif spec.sourcing == "dual":
        active_suppliers = ["S1", "S2"]
    else:  # multi
        active_suppliers = ["S1", "S2", "S3"]
    
    # Supplier -> Factory edges
    for s_id in active_suppliers:
        for f_id in ["F1", "F2"]:
            cost = SUPPLIER_TO_FACTORY_COST * it_cost_multiplier
            max_flow = 80 if s_id != "S3" else 50  # S3 is smaller
            net.add_edge(Edge(
                net.get_node(s_id), 
                net.get_node(f_id), 
                cost_per_unit=cost, 
                max_flow=max_flow
            ))
    
    # Factory -> Warehouse edges
    for f_id in ["F1", "F2"]:
        for w_id in ["W1", "W2", "W3"]:
            cost = FACTORY_TO_WAREHOUSE_COST * inventory_cost_multiplier * it_cost_multiplier
            max_flow = 70
            net.add_edge(Edge(
                net.get_node(f_id), 
                net.get_node(w_id), 
                cost_per_unit=cost, 
                max_flow=max_flow
            ))
    
    # Warehouse -> Retailer edges
    for w_id in ["W1", "W2", "W3"]:
        for r_id in ["R1", "R2", "R3", "R4", "R5"]:
            cost = WAREHOUSE_TO_RETAILER_COST * inventory_cost_multiplier
            max_flow = 50
            net.add_edge(Edge(
                net.get_node(w_id), 
                net.get_node(r_id), 
                cost_per_unit=cost, 
                max_flow=max_flow
            ))


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_all_configurations() -> List[Network]:
    """Return list of all 8 configuration networks."""
    return [build_configuration(f"C{i}") for i in range(1, 9)]


def get_configuration(config_id: str) -> Network:
    """Get a specific configuration by ID."""
    return build_configuration(config_id)


def get_config_properties(config_id: str) -> ConfigProperties:
    """Get the properties for a configuration."""
    return CONFIG_SPECS[config_id]


def print_configuration_summary():
    """Print summary of all configurations."""
    print("\nSUPPLY CHAIN CONFIGURATIONS")
    print("=" * 80)
    print(f"{'Config':<8}{'Sourcing':<12}{'Inventory':<12}{'IT Redund.':<12}{'Description':<40}")
    print("-" * 80)
    
    for config_id, spec in CONFIG_SPECS.items():
        print(f"{spec.config_id:<8}{spec.sourcing:<12}{spec.inventory:<12}"
              f"{spec.it_redundancy:<12}{spec.description:<40}")


def clone_network(net: Network) -> Network:
    """Create a deep copy of a network (for disruption testing)."""
    return deepcopy(net)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print_configuration_summary()
    
    print("\n\nNETWORK DETAILS:")
    print("=" * 80)
    
    for config_id in CONFIG_SPECS:
        net = build_configuration(config_id)
        total_supply = net.get_total_supply()
        total_demand = net.get_total_demand()
        
        print(f"\n{config_id}: {CONFIG_SPECS[config_id].description}")
        print(f"  Nodes: {len(net.nodes)}, Edges: {len(net.edges)}")
        print(f"  Total Supply: {total_supply}, Total Demand: {total_demand}")
        print(f"  IT Redundancy: {net.it_redundancy}")
        
        # Show active suppliers
        active = [n.node_id for n in net.nodes.values() 
                  if n.node_type == "supplier" and n.supply > 0]
        print(f"  Active Suppliers: {active}")
