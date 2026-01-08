from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from data_models import load_assets, load_edges, load_nodes, load_passenger_flows
from ga import GAConfig, NSGA2Optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NSGA-II maintenance scheduler")
    parser.add_argument("--nodes", default="data/nodes_template.csv", help="Path to nodes csv")
    parser.add_argument("--edges", default="data/edges_template.csv", help="Path to edges csv")
    parser.add_argument("--assets", default="data/assets_template.csv", help="Path to assets csv")
    parser.add_argument("--flowsday", default="data/passenger_flows_day.csv", help="Path to passenger flows at day csv")
    parser.add_argument("--flowsnight", default="data/passenger_flows_night.csv", help="Path to passenger flows at night csv")
    parser.add_argument("--population", type=int, default=20, help="NSGA-II population size")
    parser.add_argument("--generations", type=int, default=10, help="NSGA-II generations")
    parser.add_argument("--cx", type=float, default=0.3, help="Crossover probability")
    parser.add_argument("--mut", type=float, default=0.1, help="Mutation probability")
    parser.add_argument(
        "--weights",
        type=float,
        nargs=3,
        default=(0.3, 0.4, 0.3),
        help="Weights for condition, travel time, cost (sum=1)",
    )
    parser.add_argument("--output", default="output/best_schedule.csv", help="Output csv for best schedule")
    return parser.parse_args()


def save_schedule(path: str, schedule: dict) -> None:
    df = pd.DataFrame(
        [{"asset_id": asset_id, "start_hour": start_hour} for asset_id, start_hour in schedule.items()]
    ).sort_values("asset_id")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    nodes = load_nodes(args.nodes)
    edges = load_edges(args.edges)
    assets = load_assets(args.assets)
    flows_day = load_passenger_flows(args.flowsday)
    flows_night = load_passenger_flows(args.flowsnight)
    config = GAConfig(
        population_size=args.population,
        generations=args.generations,
        crossover_prob=args.cx,
        mutation_prob=args.mut,
        weights=tuple(args.weights),
    )
    optimizer = NSGA2Optimizer(nodes, edges, assets, flows_day, flows_night, config)
    best_schedule, best_metrics, pareto = optimizer.run()
    save_schedule(args.output, best_schedule)
    summary = {
        "best_schedule": best_schedule,
        "weighted_objectives": {
            "condition_penalty": best_metrics[0],
            "travel_penalty": best_metrics[1],
            "cost_penalty": best_metrics[2],
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
