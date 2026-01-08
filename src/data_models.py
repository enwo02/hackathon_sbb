from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import pandas as pd


@dataclass
class Node:
    node_id: str
    location_x: float
    location_y: float


@dataclass
class Edge:
    edge_id: str
    start_node: str
    end_node: str
    base_travel_time_min: int
    mode: str
    capacity_at_day: int
    capacity_at_night: int


@dataclass
class Asset:
    asset_id: str
    edge_id: str
    asset_type: str
    condition_initial: float
    condition_transition_rate: float
    maintenance_cost_per_unit: float
    maintenance_duration_half_days: float
    location_x: float
    location_y: float


@dataclass
class PassengerFlow:
    flow_id: str
    origin: str
    destination: str
    passengers: int


@dataclass
class SimulationResult:
    avg_condition: float
    avg_travel_time: float
    passenger_count: int
    total_cost: float
    schedule: Dict[str, float]


def parse_asset_path(path_str: str) -> List[str]:
    if not isinstance(path_str, str):
        return []
    return [part.strip() for part in path_str.split(">") if part.strip()]


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
            mode=str(row["mode"]),
            base_travel_time_min=float(row["base_travel_time_min"]),
            capacity_at_day=int(row["capacity_at_day"]),
            capacity_at_night=int(row["capacity_at_night"]),
        )
        edges[edge.edge_id] = edge
    return edges


def load_assets(path: str) -> Dict[str, Asset]:
    df = pd.read_csv(path)
    assets: Dict[str, Asset] = {}
    for _, row in df.iterrows():
        asset = Asset(
            asset_id=str(row["asset_id"]),
            edge_id=str(row["edge_id"]),
            asset_type=str(row["asset_type"]),
            condition_initial=float(row["condition_initial"]),
            condition_transition_rate=float(row["condition_transition_rate"]),
            maintenance_cost_per_unit=float(row["maintenance_cost_per_unit"]),
            maintenance_duration_hours=float(row["maintenance_duration_hours"]),
        )
        assets[asset.asset_id] = asset
    return assets


def load_passenger_flows(path: str) -> List[PassengerFlow]:
    df = pd.read_csv(path)
    flows: List[PassengerFlow] = []
    for _, row in df.iterrows():
        flow = PassengerFlow(
            flow_id=str(row["flow_id"]),
            origin=str(row["origin"]),
            destination=str(row["destination"]),
            start_time_hour=float(row["start_time_hour"]),
            end_time_hour=float(row["end_time_hour"]),
            rate_per_hour=float(row["rate_per_hour"]),
        )
        flows.append(flow)
    return flows
