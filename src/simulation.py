from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Set, Tuple
import numpy as np
import simpy

from data_models import Asset, Edge, Node, PassengerFlow, SimulationResult


@dataclass
class MaintenanceWindow:
    start_hour: float
    end_hour: float


@dataclass
class NetworkGraph:
    """Graph representation of the network for pathfinding using Edges"""
    nodes: Dict[str, Node]
    edges: Dict[str, Edge]
    assets: Dict[str, Asset]
    adjacency: Dict[str, List[Tuple[str, str]]]  # node_id -> [(neighbor_id, edge_id)]

    @staticmethod
    def from_data(nodes: Dict[str, Node], edges: Dict[str, Edge], assets: Dict[str, Asset]) -> NetworkGraph:
        adjacency: Dict[str, List[Tuple[str, str]]] = {node_id: [] for node_id in nodes}
        for edge_id, edge in edges.items():
            adjacency[edge.start_node].append((edge.end_node, edge_id))
            # always bidirectional
            adjacency[edge.end_node].append((edge.start_node, edge_id))
        return NetworkGraph(nodes=nodes, edges=edges, assets=assets, adjacency=adjacency)

    def find_path(
        self, origin: str, destination: str, blocked_edges: Set[str], current_time: float
    ) -> Optional[List[str]]:
        """Dijkstra shortest path finding, returns list of edge_ids or None"""
        if origin not in self.nodes or destination not in self.nodes:
            return None
        if origin == destination:
            return []

        distances: Dict[str, float] = {node_id: float("inf") for node_id in self.nodes}
        distances[origin] = 0.0
        previous: Dict[str, Optional[Tuple[str, str]]] = {node_id: None for node_id in self.nodes}
        pq: List[Tuple[float, str]] = [(0.0, origin)]
        visited: Set[str] = set()

        while pq:
            current_dist, current_node = heapq.heappop(pq)
            if current_node in visited:
                continue
            visited.add(current_node)
            if current_node == destination:
                break

            for neighbor, edge_id in self.adjacency[current_node]:
                if edge_id in blocked_edges:
                    continue
                edge = self.edges[edge_id]
                travel_time = edge.base_travel_time_min / 60.0
                new_dist = current_dist + travel_time
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = (current_node, edge_id)
                    heapq.heappush(pq, (new_dist, neighbor))

        if distances[destination] == float("inf"):
            return None

        path_edges: List[str] = []
        current = destination
        while previous[current] is not None:
            prev_node, edge_id = previous[current]
            path_edges.append(edge_id)
            current = prev_node
        path_edges.reverse()
        return path_edges


class NetworkSimulator:
    def __init__(
        self,
        nodes: Dict[str, Node],
        edges: Dict[str, Edge],
        assets: Dict[str, Asset],
        flows: List[PassengerFlow],
        maintenance_plan: Dict[str, float],
        horizon_hours: float,
        rng: np.random.Generator,
    ) -> None:
        self.nodes = nodes
        self.edges = edges
        self.assets = assets
        self.flows = flows
        self.horizon_hours = horizon_hours
        self.rng = rng
        self.env = simpy.Environment()
        self.graph = NetworkGraph.from_data(nodes, edges, assets)
        self.maintenance_windows: Dict[str, MaintenanceWindow] = {}
        self.asset_last_maintenance: Dict[str, float] = {k: 0.0 for k in assets}
        self.asset_condition_after_maintenance: Dict[str, float] = {
            k: v.condition_initial for k, v in assets.items()
        }
        # Map assets to edges for edge blockage
        self.edge_assets: Dict[str, List[str]] = {edge_id: [] for edge_id in edges}
        for asset_id, asset in assets.items():
            self.edge_assets[asset.edge_id].append(asset_id)
        
        for asset_id, start in maintenance_plan.items():
            asset = assets[asset_id]
            self.maintenance_windows[asset_id] = MaintenanceWindow(
                start_hour=start,
                end_hour=start + asset.maintenance_duration_half_days,
            )
        self.total_travel_time = 0.0
        self.total_passengers = 0
        self.failed_trips = 0
        self.condition_samples = 0
        self.condition_sum = 0.0
        self.total_cost = self._compute_cost(maintenance_plan)

    def _compute_cost(self, plan: Dict[str, float]) -> float:
        cost = 0.0
        for asset_id in plan:
            asset = self.assets[asset_id]
            cost += asset.maintenance_cost_per_unit * asset.maintenance_duration_half_days
        return cost

    def _asset_condition(self, asset_id: str, current_time: float) -> float:
        asset = self.assets[asset_id]
        last_maintenance = self.asset_last_maintenance[asset_id]
        elapsed_days = max(0.0, (current_time - last_maintenance) / 24.0)
        base_condition = self.asset_condition_after_maintenance[asset_id]
        degraded = asset.condition_transition_rate * elapsed_days
        condition = max(0.1, min(1.0, base_condition - degraded))
        if asset_id in self.maintenance_windows:
            window = self.maintenance_windows[asset_id]
            if window.start_hour <= current_time < window.end_hour:
                condition = 0.2
            elif current_time >= window.end_hour and last_maintenance < window.end_hour:
                self.asset_last_maintenance[asset_id] = window.end_hour
                self.asset_condition_after_maintenance[asset_id] = 1.0
                condition = 1.0
        return condition

    def _get_blocked_assets(self, current_time: float) -> Set[str]:
        """Return set of assets currently under maintenance"""
        blocked: Set[str] = set()
        for asset_id, window in self.maintenance_windows.items():
            if window.start_hour <= current_time < window.end_hour:
                blocked.add(asset_id)
        return blocked

    def _get_blocked_edges(self, current_time: float) -> Set[str]:
        """Return set of edges that have assets under maintenance"""
        blocked_assets = self._get_blocked_assets(current_time)
        blocked_edges: Set[str] = set()
        for asset_id in blocked_assets:
            asset = self.assets[asset_id]
            blocked_edges.add(asset.edge_id)
        return blocked_edges

    def _sample_conditions(self) -> Generator[simpy.events.Timeout, None, None]:
        while self.env.now <= self.horizon_hours:
            for asset_id in self.assets:
                condition = self._asset_condition(asset_id, self.env.now)
                self.condition_sum += condition
                self.condition_samples += 1
            yield self.env.timeout(1.0)

    def _passenger_trip(self, flow: PassengerFlow, start_time: float) -> Generator[simpy.events.Timeout, None, None]:
        travel_start = start_time
        yield self.env.timeout(max(0.0, travel_start - self.env.now))
        current_time = self.env.now

        # Find path dynamically, avoiding blocked edges
        blocked_edges = self._get_blocked_edges(current_time)
        edge_path = self.graph.find_path(flow.origin, flow.destination, blocked_edges, current_time)

        if edge_path is None:
            # No route available - trip fails
            self.failed_trips += 1
            return

        # Travel along path, updating conditions as we go
        for edge_id in edge_path:
            edge = self.edges[edge_id]
            
            # Check if edge is blocked during travel
            blocked_assets = self._get_blocked_assets(current_time)
            edge_blocked_assets = [a for a in self.edge_assets[edge_id] if a in blocked_assets]
            
            if edge_blocked_assets:
                # Wait for all maintenance on this edge to finish
                earliest_end = min(
                    self.maintenance_windows[a].end_hour 
                    for a in edge_blocked_assets
                )
                yield self.env.timeout(max(0.0, earliest_end - current_time))
                current_time = self.env.now
                # Update maintenance flags
                for asset_id in edge_blocked_assets:
                    self.asset_last_maintenance[asset_id] = current_time
                    self.asset_condition_after_maintenance[asset_id] = 1.0

            # Calculate edge condition (worst asset condition on edge)
            edge_condition = 1.0
            for asset_id in self.edge_assets[edge_id]:
                asset_condition = self._asset_condition(asset_id, current_time)
                edge_condition = min(edge_condition, asset_condition)
            
            # Travel time increases with degraded condition
            slowdown = 1.0 + (1.0 - edge_condition)
            travel_time = (edge.capacity_at_day / 60.0) * slowdown # TODO Night also
            yield self.env.timeout(travel_time)
            current_time = self.env.now

        total_time = self.env.now - travel_start
        self.total_travel_time += total_time
        self.total_passengers += 1

    def _passenger_source(self, flow: PassengerFlow) -> Generator[simpy.events.Timeout, None, None]:
        current_time = flow.start_time_hour
        while current_time < flow.end_time_hour and current_time < self.horizon_hours:
            self.env.process(self._passenger_trip(flow, current_time))
            interarrival = self.rng.exponential(1.0 / max(flow.rate_per_hour, 1e-6))
            current_time += interarrival
            yield self.env.timeout(max(0.0, current_time - self.env.now))

    def run(self) -> SimulationResult:
        for flow in self.flows:
            self.env.process(self._passenger_source(flow))
        self.env.process(self._sample_conditions())
        self.env.run(until=self.horizon_hours)
        avg_condition = self.condition_sum / max(self.condition_samples, 1)
        avg_travel_time = self.total_travel_time / max(self.total_passengers, 1)
        return SimulationResult(
            avg_condition=avg_condition,
            avg_travel_time=avg_travel_time,
            passenger_count=self.total_passengers,
            total_cost=self.total_cost,
            schedule={k: v.start_hour if isinstance(v, MaintenanceWindow) else v for k, v in self.maintenance_windows.items()},
        )


def simulate_schedule(
    nodes: Dict[str, Node],
    edges: Dict[str, Edge],
    assets: Dict[str, Asset],
    flows: List[PassengerFlow],
    schedule: Dict[str, float],
    horizon_hours: float,
    rng: np.random.Generator,
) -> SimulationResult:
    simulator = NetworkSimulator(nodes, edges, assets, flows, schedule, horizon_hours, rng)
    return simulator.run()
