"""Генетический алгоритм: эволюционный цикл, счёт оценок, лучший допустимый план."""

import math
import time
from dataclasses import dataclass

import numpy as np

from procurement_ga.constraints import (
    constraint_violations,
    is_feasible,
    penalized_fitness,
    repair,
)
from procurement_ga.data_generator import validate_data
from procurement_ga.models import (
    ConstraintStrategy,
    GAConfig,
    ProcurementData,
    RunResult,
)
from procurement_ga.objective import total_cost, total_utility
from procurement_ga.operators import (
    initialize_population,
    mutate,
    tournament_select,
    uniform_crossover,
)


@dataclass(frozen=True, eq=False)
class _Evaluated:
    """Оценки группы особей; одна особь = одна оценка целевой функции."""

    fitness: np.ndarray
    utility: np.ndarray
    cost: np.ndarray
    feasible: np.ndarray

    def concat(self, other: "_Evaluated") -> "_Evaluated":
        return _Evaluated(
            np.concatenate([self.fitness, other.fitness]),
            np.concatenate([self.utility, other.utility]),
            np.concatenate([self.cost, other.cost]),
            np.concatenate([self.feasible, other.feasible]),
        )

    def take(self, index: np.ndarray) -> "_Evaluated":
        return _Evaluated(
            self.fitness[index],
            self.utility[index],
            self.cost[index],
            self.feasible[index],
        )


def _evaluate(
    data: ProcurementData, config: GAConfig, population: np.ndarray
) -> _Evaluated:
    size = population.shape[0]
    fitness = np.empty(size)
    utility = np.empty(size)
    cost = np.empty(size)
    feasible = np.empty(size, dtype=bool)
    repair_mode = config.constraint_strategy is ConstraintStrategy.REPAIR
    for k, x in enumerate(population):
        utility[k] = total_utility(data, x)
        cost[k] = total_cost(data, x)
        feasible[k] = constraint_violations(data, x).feasible
        if repair_mode:
            fitness[k] = utility[k]
        else:
            fitness[k] = penalized_fitness(
                data, x, config.penalty_shortage, config.penalty_budget
            )
    if repair_mode and not feasible.all():
        raise RuntimeError("После repair в популяции осталась недопустимая особь")
    return _Evaluated(fitness, utility, cost, feasible)


class _BestTracker:
    """Лучший fitness (допустим или нет) и лучший допустимый план за весь запуск."""

    def __init__(self) -> None:
        self.fitness = -math.inf
        self.fitness_feasible = False
        self.x: np.ndarray | None = None
        self.utility = -math.inf
        self.cost = math.nan
        self.generation: int | None = None

    def update(
        self, generation: int, population: np.ndarray, ev: _Evaluated
    ) -> bool:
        """Учитывает оценённых особей; True, если рекорд улучшился."""
        improved = False
        top = int(np.argmax(ev.fitness))
        if ev.fitness[top] > self.fitness:
            self.fitness = float(ev.fitness[top])
            self.fitness_feasible = bool(ev.feasible[top])
            improved = True
        feasible_index = np.flatnonzero(ev.feasible)
        if feasible_index.size:
            best = int(feasible_index[np.argmax(ev.utility[feasible_index])])
            if ev.utility[best] > self.utility:
                self.utility = float(ev.utility[best])
                self.cost = float(ev.cost[best])
                self.x = population[best].copy()
                self.generation = generation
                improved = True
        return improved


def _make_offspring(
    data: ProcurementData,
    config: GAConfig,
    population: np.ndarray,
    fitness: np.ndarray,
    n_children: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_pairs = (n_children + 1) // 2
    parents = tournament_select(fitness, 2 * n_pairs, config.tournament_size, rng)
    children: list[np.ndarray] = []
    for p in range(n_pairs):
        a = population[parents[2 * p]]
        b = population[parents[2 * p + 1]]
        if rng.random() < config.crossover_prob:
            first, second = uniform_crossover(a, b, rng)
        else:
            first, second = a.copy(), b.copy()
        children.extend((first, second))

    repair_mode = config.constraint_strategy is ConstraintStrategy.REPAIR
    offspring: list[np.ndarray] = []
    for child in children[:n_children]:
        if rng.random() < config.mutation_prob:
            child = mutate(
                child,
                data.upper_bounds,
                config.mutation_type,
                config.mutation_genes,
                rng,
            )
        if repair_mode:
            child = repair(data, child)
        offspring.append(child)
    return np.array(offspring, dtype=np.int64)


def _next_generation(
    population: np.ndarray,
    ev: _Evaluated,
    offspring: np.ndarray,
    child_ev: _Evaluated,
    elite_size: int,
) -> tuple[np.ndarray, _Evaluated]:
    """Элита + потомки; если потомков меньше, места занимают лучшие из старых."""
    size = population.shape[0]
    order = np.lexsort((np.arange(size), -ev.fitness))  # лучшие первыми
    n_fill = size - elite_size - offspring.shape[0]
    keep = order[: elite_size + n_fill]
    new_population = np.vstack([population[keep], offspring])
    return new_population, ev.take(keep).concat(child_ev)


def run_ga(data: ProcurementData, config: GAConfig) -> RunResult:
    """Один запуск ГА. Число оценок не превышает config.max_evaluations."""
    validate_data(data)
    started = time.perf_counter()
    rng = np.random.default_rng(config.seed)
    repair_mode = config.constraint_strategy is ConstraintStrategy.REPAIR

    population = initialize_population(
        data, config.population_size, rng, feasible=repair_mode
    )
    ev = _evaluate(data, config, population)
    evaluations = config.population_size

    tracker = _BestTracker()
    tracker.update(0, population, ev)
    fitness_history: list[float] = []
    utility_history: list[float] = []
    feasible_history: list[float] = []

    def record() -> None:
        fitness_history.append(float(ev.fitness.max()))
        utility_history.append(tracker.utility if tracker.x is not None else math.nan)
        feasible_history.append(float(ev.feasible.mean()))

    record()
    stagnation = 0
    for generation in range(1, config.max_generations + 1):
        remaining = config.max_evaluations - evaluations
        if remaining <= 0:
            break
        n_children = min(config.population_size - config.elite_size, remaining)
        offspring = _make_offspring(
            data, config, population, ev.fitness, n_children, rng
        )
        child_ev = _evaluate(data, config, offspring)
        evaluations += n_children
        improved = tracker.update(generation, offspring, child_ev)
        population, ev = _next_generation(
            population, ev, offspring, child_ev, config.elite_size
        )
        record()
        stagnation = 0 if improved else stagnation + 1
        if config.stagnation_limit is not None and stagnation >= config.stagnation_limit:
            break

    found = tracker.x is not None
    if found and not is_feasible(data, tracker.x):
        raise RuntimeError("Лучший план оказался недопустимым")
    return RunResult(
        method="GA",
        seed=config.seed,
        config=config,
        best_x=tracker.x,
        best_utility=tracker.utility if found else None,
        best_cost=tracker.cost if found else None,
        found_feasible=found,
        best_feasible_generation=tracker.generation,
        best_fitness=tracker.fitness,
        best_fitness_feasible=tracker.fitness_feasible,
        fitness_history=fitness_history,
        utility_history=utility_history,
        feasible_fraction_history=feasible_history,
        evaluations=evaluations,
        elapsed_seconds=time.perf_counter() - started,
    )