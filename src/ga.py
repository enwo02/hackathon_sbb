from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional

import numpy as np
from deap import base, creator, tools

from data_models import Asset, Edge, Node, PassengerFlow, SimulationResult
from simulation import simulate_schedule

import logging 
import time
from pathlib import Path
import json


@dataclass
class GAConfig:
    population_size: int = 20
    generations: int = 15
    crossover_prob: float = 0.3
    mutation_prob: float = 0.1
    horizon_hours: float = 2*365
    seed: int = 42


class NSGA2Optimizer:
    """
    True multi-objective NSGA-II:
      Maximize: served_trips, avg_condition
      Minimize: avg_travel_time, total_cost
    """

    def __init__(
        self,
        nodes: Dict[str, Node],
        edges: Dict[str, Edge],
        assets: Dict[str, Asset],
        flows_day: List[PassengerFlow],
        flows_night: List[PassengerFlow],
        config: GAConfig,
    ) -> None:
        self.nodes = nodes
        self.edges = edges
        self.assets = assets
        self.flows_day = flows_day
        self.flows_night = flows_night
        self.config = config
        self.asset_ids = list(assets.keys())
        self.logger = logging.getLogger("ga")

        # Reproducibility
        random.seed(config.seed)
        self.base_seed = int(config.seed)
        self.eval_counter = 0

        self.toolbox = base.Toolbox()
        self._setup_deap()

        self._progress_cb: Optional[Callable[[int, Dict[str, float], tuple, tools.ParetoFront], None]] = None

    def _setup_deap(self) -> None:
        # (served_trips, avg_condition, avg_travel_time, total_cost)
        # we want: max, max, min, min
        weights = (1.0, 1.0, -1.0, -1.0)

        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=weights)
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMulti)

        self.toolbox.register("attr_time", self._random_start)
        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            self.toolbox.attr_time,
            n=len(self.asset_ids),
        )
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Better operators for bounded continuous genes:
        # - SBX crossover
        # - Polynomial bounded mutation
        self.toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0.0, up=self.config.horizon_hours, eta=15.0)
        self.toolbox.register(
            "mutate",
            tools.mutPolynomialBounded,
            low=0.0,
            up=self.config.horizon_hours,
            eta=20.0,
            indpb=0.3,
        )
        self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.register("evaluate", self._evaluate)

    def _random_start(self) -> float:
        return random.uniform(0.0, self.config.horizon_hours)

    def _max_start(self, idx: int) -> float:
        asset_id = self.asset_ids[idx % len(self.asset_ids)]
        asset = self.assets[asset_id]
        return max(0.0, self.config.horizon_hours - asset.maintenance_duration_half_days)

    def _bounds_correction(self, individual: List[float]) -> None:
        for i in range(len(individual)):
            upper = self._max_start(i)
            individual[i] = max(0.0, min(upper, float(individual[i])))

    def _evaluate(self, individual: List[float]):
        t0 = time.perf_counter()

        self._bounds_correction(individual)

        schedule = {asset_id: float(individual[i]) for i, asset_id in enumerate(self.asset_ids)}

        # Common Random Numbers: deterministic seed per evaluation to make comparisons fairer
        self.eval_counter += 1
        rng = np.random.default_rng(self.base_seed + self.eval_counter)

        result: SimulationResult = simulate_schedule(
            self.nodes,
            self.edges,
            self.assets,
            self.flows_day,
            self.flows_night,
            schedule,
            rng,
            self.config.horizon_hours,
        )

        # Objectives:
        # - maximize served trips
        # - maximize avg condition
        # - minimize avg travel time
        # - minimize total cost
        served = float(result.served_trips)

        # Optional: penalize failures explicitly by reducing served (soft)
        # If you prefer hard service-level constraints, do that after Pareto filtering.
        served_adjusted = served #- 2.0 * float(result.failed_trips)

        dt = time.perf_counter() - t0
        if self.eval_counter % 50 == 0:  # alle 50 Evaluations
            self.logger.info(
                "Eval %d | served=%d failed=%d avg_time=%.3f avg_cond=%.3f cost=%.1f | sim=%.2fs",
                self.eval_counter,
                result.served_trips,
                result.failed_trips,
                result.avg_travel_time,
                result.avg_condition,
                result.total_cost,
                dt,
            )       


        return (
            served_adjusted,
            float(result.avg_condition),
            float(result.avg_travel_time),
            float(result.total_cost),
        )

    @staticmethod
    def _choose_solution_from_pareto(pareto: List, eps: float = 1e-9):
        """
        Choose a single solution from Pareto set using normalized distance to ideal point:
        ideal = (max served, max cond, min time, min cost)
        """
        fits = np.array([ind.fitness.values for ind in pareto], dtype=float)
        ideal = np.array([fits[:, 0].max(), fits[:, 1].max(), fits[:, 2].min(), fits[:, 3].min()], dtype=float)

        # normalize each objective to [0,1] range to avoid scale dominance
        mins = fits.min(axis=0)
        maxs = fits.max(axis=0)
        denom = np.maximum(maxs - mins, eps)
        norm = (fits - mins) / denom

        ideal_norm = (ideal - mins) / denom
        d = np.linalg.norm(norm - ideal_norm, axis=1)
        return pareto[int(np.argmin(d))]

    def run(self, progress_cb: Optional[Callable[[int, Dict[str, float], tuple, tools.ParetoFront], None]] = None):
        self._progress_cb = progress_cb

        pop = self.toolbox.population(n=self.config.population_size)

        # Evaluate initial population
        invalid = [ind for ind in pop if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = self.toolbox.evaluate(ind)

        # NSGA-II requires initial sorting
        pop = self.toolbox.select(pop, len(pop))
        hall = tools.ParetoFront()
        hall.update(pop)

        # Generation 0 snapshot (after initial evaluation)
        best0 = self._choose_solution_from_pareto(list(hall))
        best_schedule0 = {asset_id: float(best0[i]) for i, asset_id in enumerate(self.asset_ids)}
        best_metrics0 = best0.fitness.values
        if self._progress_cb is not None:
            self._progress_cb(0, best_schedule0, best_metrics0, hall)

        for _gen in range(1, self.config.generations + 1):
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [self.toolbox.clone(ind) for ind in offspring]

            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= self.config.crossover_prob:
                    self.toolbox.mate(c1, c2)
                    del c1.fitness.values, c2.fitness.values

            for mutant in offspring:
                if random.random() <= self.config.mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid:
                ind.fitness.values = self.toolbox.evaluate(ind)

            pop = self.toolbox.select(pop + offspring, self.config.population_size)
            hall.update(pop)
            # Write progress file so frontends can monitor progress
            try:
                outdir = Path("output")
                outdir.mkdir(parents=True, exist_ok=True)
                prog = {"current_generation": int(_gen), "total_generations": int(self.config.generations)}
                with open(outdir / "progress.json", "w", encoding="utf-8") as pf:
                    json.dump(prog, pf)
            except Exception as e:
                print(f"[DEBUG ga] failed to write progress.json: {e}", flush=True)

            # Choose representative “balanced” schedule from current Pareto
            best_gen = self._choose_solution_from_pareto(list(hall))
            best_schedule_gen = {asset_id: float(best_gen[i]) for i, asset_id in enumerate(self.asset_ids)}
            best_metrics_gen = best_gen.fitness.values
            if self._progress_cb is not None:
                self._progress_cb(_gen, best_schedule_gen, best_metrics_gen, hall)

        # Final best (already selected in last iteration above, but recompute for clarity)
        best = self._choose_solution_from_pareto(list(hall))

        best_schedule = {asset_id: float(best[i]) for i, asset_id in enumerate(self.asset_ids)}
        best_metrics = best.fitness.values  # already evaluated

        if _gen % 1 == 0:
            self.logger.info(
                "Gen %d/%d | Pareto size=%d | pop[0] fitness=%s",
                _gen,
                self.config.generations,
                len(hall),
                pop[0].fitness.values,
            )

        return best_schedule, best_metrics, hall
