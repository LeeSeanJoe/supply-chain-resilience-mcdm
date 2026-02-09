"""
Network Visualization Module

Creates diagrams showing:
1. Supply chain topology (nodes and edges)
2. Configuration comparison (which suppliers active)
3. Disruption impact (before/after)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, List, Optional

from configurations import build_configuration, CONFIG_SPECS, get_config_properties
from disruptions import apply_disruption, DisruptionType


# =============================================================================
# LAYOUT CONFIGURATION
# =============================================================================

# Fixed positions for nodes (left to right flow)
NODE_POSITIONS = {
    # Suppliers (left)
    'S1': (0, 2),
    'S2': (0, 1),
    'S3': (0, 0),
    # Factories
    'F1': (1.5, 1.5),
    'F2': (1.5, 0.5),
    # Warehouses
    'W1': (3, 2),
    'W2': (3, 1),
    'W3': (3, 0),
    # Retailers (right)
    'R1': (4.5, 2.5),
    'R2': (4.5, 1.75),
    'R3': (4.5, 1),
    'R4': (4.5, 0.25),
    'R5': (4.5, -0.5),
}

NODE_COLORS = {
    'supplier': '#3498db',   # Blue
    'factory': '#9b59b6',    # Purple
    'warehouse': '#2ecc71',  # Green
    'retailer': '#e74c3c',   # Red
}

NODE_LABELS = {
    'supplier': 'Supplier',
    'factory': 'Factory',
    'warehouse': 'Warehouse',
    'retailer': 'Retailer',
}


# =============================================================================
# DRAWING FUNCTIONS
# =============================================================================

def draw_network(
    ax,
    config_id: str = "C3",
    disruption: Optional[DisruptionType] = None,
    title: str = None,
    show_inactive: bool = True
):
    """
    Draw a supply chain network on the given axes.
    
    Args:
        ax: Matplotlib axes
        config_id: Configuration to draw
        disruption: Optional disruption to apply
        title: Plot title
        show_inactive: Whether to show inactive edges (grey dashed)
    """
    # Build network
    net = build_configuration(config_id)
    if disruption:
        net = apply_disruption(net, disruption)
    
    # Get active suppliers
    active_suppliers = set()
    for node in net.nodes.values():
        if node.node_type == 'supplier' and node.supply > 0:
            active_suppliers.add(node.node_id)
    
    # Draw edges first (so nodes are on top)
    for edge in net.edges:
        from_pos = NODE_POSITIONS[edge.from_node.node_id]
        to_pos = NODE_POSITIONS[edge.to_node.node_id]
        
        # Determine edge style
        is_active = edge.max_flow > 0
        
        if is_active:
            # Active edge - solid, thickness based on capacity
            alpha = min(1.0, edge.max_flow / 80)
            linewidth = 1 + edge.max_flow / 40
            color = '#2c3e50'
            linestyle = '-'
        elif show_inactive:
            # Inactive edge - dashed grey
            alpha = 0.3
            linewidth = 1
            color = '#bdc3c7'
            linestyle = '--'
        else:
            continue
        
        ax.annotate(
            '',
            xy=to_pos,
            xytext=from_pos,
            arrowprops=dict(
                arrowstyle='->',
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                linestyle=linestyle,
                connectionstyle='arc3,rad=0.1'
            )
        )
    
    # Draw nodes
    for node_id, pos in NODE_POSITIONS.items():
        node = net.nodes.get(node_id)
        if node is None:
            continue
        
        # Determine node appearance
        color = NODE_COLORS[node.node_type]
        
        # Check if node is affected by disruption
        is_affected = False
        if disruption == DisruptionType.CYBERATTACK and node.node_type == 'warehouse':
            is_affected = True
        elif disruption == DisruptionType.SUPPLIER_FAILURE and node_id == 'S1':
            is_affected = True
        
        # Inactive suppliers (supply = 0)
        is_inactive = node.node_type == 'supplier' and node.supply == 0
        
        if is_affected:
            edgecolor = 'red'
            linewidth = 3
            alpha = 0.5
        elif is_inactive:
            edgecolor = 'grey'
            linewidth = 1
            alpha = 0.3
        else:
            edgecolor = 'black'
            linewidth = 1
            alpha = 1.0
        
        # Draw node
        circle = plt.Circle(
            pos, 
            0.15, 
            color=color, 
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=10
        )
        ax.add_patch(circle)
        
        # Label
        ax.text(
            pos[0], pos[1], 
            node_id, 
            ha='center', va='center',
            fontsize=8, fontweight='bold',
            color='white' if alpha > 0.5 else 'grey',
            zorder=11
        )
    
    # Configure axes
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-1, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')


def plot_topology_diagram(output_path: str = "network_topology.png"):
    """
    Create a clean topology diagram showing the supply chain structure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    draw_network(ax, config_id="C7", title="Supply Chain Network Topology (C7: Multi-source)")
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color=NODE_COLORS['supplier'], label='Suppliers (S1-S3)'),
        mpatches.Patch(color=NODE_COLORS['factory'], label='Factories (F1-F2)'),
        mpatches.Patch(color=NODE_COLORS['warehouse'], label='Warehouses (W1-W3)'),
        mpatches.Patch(color=NODE_COLORS['retailer'], label='Retailers (R1-R5)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)
    
    # Add flow direction arrow
    ax.annotate(
        'Material Flow →',
        xy=(2.5, -0.7),
        fontsize=10,
        ha='center'
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_configuration_comparison(output_path: str = "configuration_comparison.png"):
    """
    Compare single, dual, and multi-source configurations side by side.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    configs = [
        ("C1", "Single Source (C1)"),
        ("C3", "Dual Source (C3)"),
        ("C7", "Multi Source (C7)"),
    ]
    
    for ax, (config_id, title) in zip(axes, configs):
        draw_network(ax, config_id=config_id, title=title, show_inactive=True)
    
    plt.suptitle("Sourcing Strategy Comparison", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_disruption_impact(output_path: str = "disruption_impact.png"):
    """
    Show before/after for each disruption type.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Top row: C3 (Dual source, no IT protection)
    config = "C3"
    
    axes[0, 0].set_title("C3 - Normal", fontsize=10)
    draw_network(axes[0, 0], config_id=config, disruption=None)
    
    axes[0, 1].set_title("C3 - Cyberattack", fontsize=10)
    draw_network(axes[0, 1], config_id=config, disruption=DisruptionType.CYBERATTACK)
    axes[0, 1].text(3, -0.8, "⚠ Warehouse capacity reduced", ha='center', fontsize=9, color='red')
    
    axes[0, 2].set_title("C3 - Supplier Failure", fontsize=10)
    draw_network(axes[0, 2], config_id=config, disruption=DisruptionType.SUPPLIER_FAILURE)
    axes[0, 2].text(0, -0.8, "⚠ S1 offline", ha='center', fontsize=9, color='red')
    
    # Bottom row: C6 (Dual source, full IT protection)
    config = "C6"
    
    axes[1, 0].set_title("C6 - Normal", fontsize=10)
    draw_network(axes[1, 0], config_id=config, disruption=None)
    
    axes[1, 1].set_title("C6 - Cyberattack (IT Protected)", fontsize=10)
    draw_network(axes[1, 1], config_id=config, disruption=DisruptionType.CYBERATTACK)
    axes[1, 1].text(3, -0.8, "✓ IT redundancy mitigates impact", ha='center', fontsize=9, color='green')
    
    axes[1, 2].set_title("C6 - Supplier Failure", fontsize=10)
    draw_network(axes[1, 2], config_id=config, disruption=DisruptionType.SUPPLIER_FAILURE)
    axes[1, 2].text(0, -0.8, "✓ S2 maintains supply", ha='center', fontsize=9, color='green')
    
    plt.suptitle("Disruption Impact: C3 (No Protection) vs C6 (Full Protection)", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_all_configurations(output_path: str = "all_configurations.png"):
    """
    Show all 8 configurations in a grid.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, config_id in enumerate([f"C{j}" for j in range(1, 9)]):
        props = get_config_properties(config_id)
        title = f"{config_id}: {props.sourcing}/{props.inventory}/{props.it_redundancy}"
        draw_network(axes[i], config_id=config_id, title=title, show_inactive=True)
    
    plt.suptitle("All 8 Supply Chain Configurations", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("NETWORK VISUALIZATION")
    print("=" * 50)
    
    print("\n1. Creating topology diagram...")
    plot_topology_diagram("network_topology.png")
    
    print("\n2. Creating configuration comparison...")
    plot_configuration_comparison("configuration_comparison.png")
    
    print("\n3. Creating disruption impact diagram...")
    plot_disruption_impact("disruption_impact.png")
    
    print("\n4. Creating all configurations grid...")
    plot_all_configurations("all_configurations.png")
    
    print("\n" + "=" * 50)
    print("Network visualization complete!")
