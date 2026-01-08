from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd


@dataclass
class Node:
    node_id: str
    location_x: float
    location_y: float
    is_terminal: bool


@dataclass
class Edge:
    edge_id: str
    start_node: str
    end_node: str
    base_travel_time_min: float
    capacity_per_hour: int
    bidirectional: bool = True


@dataclass
class Asset:
    asset_id: str
    edge_id: str
    asset_type: str
    condition_initial: float
    condition_transition_rate: float          # time-based degradation per day
    maintenance_cost_per_unit: float
    maintenance_duration_hours: float
    usage_degradation_per_passage: float = 0.0  # NEW: degradation per traversal (tunable)


@dataclass
class PassengerFlow:
    flow_id: str
    origin: str
    destination: str
    start_time_hour: float
    end_time_hour: float
    rate_per_hour: float


@dataclass
class SimulationResult:
    avg_condition: float
    avg_travel_time: float
    served_trips: int
    generated_trips: int
    failed_trips: int
    total_cost: float
    schedule: Dict[str, float]


def load_nodes(path: str) -> Dict[str, Node]:
    df = pd.read_csv(path)
    nodes: Dict[str, Node] = {}
    for _, row in df.iterrows():
        nodes[str(row["node_id"])] = Node(
            node_id=str(row["node_id"]),
            location_x=float(row["location_x"]),
            location_y=float(row["location_y"]),
            is_terminal=bool(row["is_terminal"]),
        )
    return nodes


def load_edges(path: str) -> Dict[str, Edge]:
    df = pd.read_csv(path)
    edges: Dict[str, Edge] = {}
    for _, row in df.iterrows():
        bidirectional = bool(row.get("bidirectional", True))
        edge = Edge(
            edge_id=str(row["edge_id"]),
            start_node=str(row["start_node"]),
            end_node=str(row["end_node"]),
            base_travel_time_min=float(row["base_travel_time_min"]),
            capacity_per_hour=int(row["capacity_per_hour"]),
            bidirectional=bidirectional,
        )
        edges[edge.edge_id] = edge
    return edges


def load_assets(path: str) -> Dict[str, Asset]:
    df = pd.read_csv(path)
    assets: Dict[str, Asset] = {}
    for _, row in df.iterrows():
        assets[str(row["asset_id"])] = Asset(
            asset_id=str(row["asset_id"]),
            edge_id=str(row["edge_id"]),
            asset_type=str(row["asset_type"]),
            condition_initial=float(row["condition_initial"]),
            condition_transition_rate=float(row["condition_transition_rate"]),
            maintenance_cost_per_unit=float(row["maintenance_cost_per_unit"]),
            maintenance_duration_hours=float(row["maintenance_duration_hours"]),
            usage_degradation_per_passage=float(row.get("usage_degradation_per_passage", 0.0)),
        )
    return assets


def load_passenger_flows(path: str) -> List[PassengerFlow]:
    df = pd.read_csv(path)
    flows: List[PassengerFlow] = []
    for _, row in df.iterrows():
        flows.append(
            PassengerFlow(
                flow_id=str(row["flow_id"]),
                origin=str(row["origin"]),
                destination=str(row["destination"]),
                start_time_hour=float(row["start_time_hour"]),
                end_time_hour=float(row["end_time_hour"]),
                rate_per_hour=float(row["rate_per_hour"]),
            )
        )
    return flows
