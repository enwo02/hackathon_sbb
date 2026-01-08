from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from deap import algorithms, base, creator, tools

from data_models import Asset, Edge, Node, PassengerFlow, SimulationResult
from simulation import simulate_schedule


@dataclass
class GAConfig:
    population_size: int = 20
    generations: int = 10
    crossover_prob: float = 0.3
    mutation_prob: float = 0.1
    horizon_hours: float = 168.0
    weights: Tuple[float, float, float] = (0.3, 0.4, 0.3)


class NSGA2Optimizer:
    def __init__(self, nodes: Dict[str, Node], edges: Dict[str, Edge], assets: Dict[str, Asset], flows: List[PassengerFlow], config: GAConfig) -> None:
        self.nodes = nodes
        self.edges = edges
        self.assets = assets
        self.flows = flows
        self.config = config
        self.asset_ids = list(assets.keys())
        self.rng = np.random.default_rng(42)
        self.toolbox = base.Toolbox()
        self._setup_deap()

    def _setup_deap(self) -> None:
        weights = (-1.0, -1.0, -1.0)
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
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=6.0, indpb=0.4)
        self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.register("evaluate", self._evaluate)

    def _random_start(self) -> float:
        return random.uniform(0.0, self.config.horizon_hours)

    def _max_start(self, idx: int) -> float:
        asset_id = self.asset_ids[idx % len(self.asset_ids)]
        asset = self.assets[asset_id]
        return max(0.0, self.config.horizon_hours - asset.maintenance_duration_hours)

    def _bounds_correction(self, individual: List[float]) -> None:
        for i in range(len(individual)):
            upper = self._max_start(i)
            individual[i] = max(0.0, min(upper, individual[i]))

    def _evaluate(self, individual: List[float]):
        self._bounds_correction(individual)
        schedule = {asset_id: float(individual[i]) for i, asset_id in enumerate(self.asset_ids)}
        result: SimulationResult = simulate_schedule(
            self.nodes,
            self.edges,
            self.assets,
            self.flows,
            schedule,
            self.config.horizon_hours,
            rng=self.rng,
        )
        w_condition, w_travel, w_cost = self.config.weights
        condition_penalty = (1.0 - result.avg_condition) * w_condition
        travel_penalty = result.avg_travel_time * w_travel
        cost_penalty = result.total_cost * w_cost
        return condition_penalty, travel_penalty, cost_penalty

    def run(self):
        pop = self.toolbox.population(n=self.config.population_size)
        hall = tools.ParetoFront()
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", lambda fits: np.mean(fits, axis=0))
        stats.register("min", lambda fits: np.min(fits, axis=0))
        pop = tools.selNSGA2(pop, len(pop))
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        for gen in range(1, self.config.generations + 1):
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
            fits = map(self.toolbox.evaluate, invalid)
            for ind, fit in zip(invalid, fits):
                ind.fitness.values = fit
            pop = self.toolbox.select(pop + offspring, self.config.population_size)
            hall.update(pop)
            stats.compile(pop)
        best = hall[0]
        best_schedule = {asset_id: float(best[i]) for i, asset_id in enumerate(self.asset_ids)}
        best_metrics = self.toolbox.evaluate(best)
        return best_schedule, best_metrics, hall
