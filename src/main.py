from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from data_models import load_assets, load_edges, load_nodes, load_passenger_flows
from ga import GAConfig, NSGA2Optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NSGA-II maintenance scheduler (multi-objective)")
    parser.add_argument("--nodes", default="data/nodes_template.csv", help="Path to nodes csv")
    parser.add_argument("--edges", default="data/edges_template.csv", help="Path to edges csv")
    parser.add_argument("--assets", default="data/assets_template.csv", help="Path to assets csv")
    parser.add_argument("--flows", default="data/passenger_flows_template.csv", help="Path to passenger flows csv")
    parser.add_argument("--horizon", type=float, default=168.0, help="Simulation horizon in hours")
    parser.add_argument("--population", type=int, default=40, help="NSGA-II population size")
    parser.add_argument("--generations", type=int, default=30, help="NSGA-II generations")
    parser.add_argument("--cx", type=float, default=0.9, help="Crossover probability")
    parser.add_argument("--mut", type=float, default=0.2, help="Mutation probability")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="output/best_schedule.csv", help="Output csv for best schedule")
    parser.add_argument("--pareto_out", default="output/pareto_front.csv", help="Output csv for Pareto front")
    return parser.parse_args()


def save_schedule(path: str, schedule: dict) -> None:
    df = pd.DataFrame([{"asset_id": aid, "start_hour": sh} for aid, sh in schedule.items()]).sort_values("asset_id")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_pareto(path: str, pareto) -> None:
    rows = []
    for ind in pareto:
        f = ind.fitness.values
        rows.append(
            {
                "served_adjusted": f[0],
                "avg_condition": f[1],
                "avg_travel_time": f[2],
                "total_cost": f[3],
                "genes": json.dumps(list(map(float, ind))),
            }
        )
    df = pd.DataFrame(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()

    nodes = load_nodes(args.nodes)
    edges = load_edges(args.edges)
    assets = load_assets(args.assets)
    flows = load_passenger_flows(args.flows)

    config = GAConfig(
        population_size=args.population,
        generations=args.generations,
        crossover_prob=args.cx,
        mutation_prob=args.mut,
        horizon_hours=args.horizon,
        seed=args.seed,
    )

    optimizer = NSGA2Optimizer(nodes, edges, assets, flows, config)
    best_schedule, best_metrics, pareto = optimizer.run()

    save_schedule(args.output, best_schedule)
    save_pareto(args.pareto_out, pareto)

    summary = {
        "best_schedule": best_schedule,
        "objectives": {
            "served_adjusted": best_metrics[0],
            "avg_condition": best_metrics[1],
            "avg_travel_time": best_metrics[2],
            "total_cost": best_metrics[3],
        },
        "notes": [
            "Objectives are true multi-objective (NSGA-II): maximize served, condition; minimize travel time, cost.",
            "Edge capacity modeled via simpy.Resource; maintenance reduces capacity & increases travel time.",
            "Asset condition degrades with time and usage (usage_degradation_per_passage).",
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
