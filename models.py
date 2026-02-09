# models.py

class Node:
    def __init__(self, node_id, node_type, supply=0, demand=0, capacity=None):
        """
        节点类
        :param node_id: 节点唯一标识，如 "S1", "F2"
        :param node_type: 类型 ("supplier", "factory", "warehouse", "retailer")
        :param supply: 最大供应量（仅供应商 > 0）
        :param demand: 固定需求量（仅零售商 > 0）
        :param capacity: 处理能力上限（工厂/仓库可选）
        """
        self.node_id = node_id
        self.node_type = node_type
        self.supply = supply      # 供应商能提供的最大量
        self.demand = demand      # 零售商必须满足的需求
        self.capacity = capacity  # 工厂/仓库的最大吞吐量（可选）

    def __repr__(self):
        return f"Node({self.node_id}, type={self.node_type}, supply={self.supply}, demand={self.demand})"


class Edge:
    def __init__(self, from_node, to_node, cost_per_unit, max_flow=float('inf')):
        """
        边类（运输路径）
        :param from_node: 起点（Node 对象）
        :param to_node: 终点（Node 对象）
        :param cost_per_unit: 每单位货物的运输成本
        :param max_flow: 最大允许流量（默认无限）
        """
        self.edge_id = f"{from_node.node_id}_to_{to_node.node_id}"
        self.from_node = from_node
        self.to_node = to_node
        self.cost_per_unit = cost_per_unit
        self.max_flow = max_flow

    def __repr__(self):
        return f"Edge({self.edge_id}, cost={self.cost_per_unit}, max_flow={self.max_flow})"


class Network:
    def __init__(self, name=""):
        self.name = name
        self.nodes = {}   # key: node_id, value: Node
        self.edges = []   # list of Edge

    def add_node(self, node):
        self.nodes[node.node_id] = node

    def add_edge(self, edge):
        self.edges.append(edge)

    def get_node(self, node_id):
        return self.nodes[node_id]

    def get_total_demand(self):
        return sum(node.demand for node in self.nodes.values())

    def get_total_supply(self):
        return sum(node.supply for node in self.nodes.values())