"""Базовая жадная эвристика закупки."""

import time

import numpy as np

from procurement_ga.constraints import BUDGET_TOLERANCE, is_feasible
from procurement_ga.data_generator import validate_data
from procurement_ga.models import ProcurementData, RunResult
from procurement_ga.objective import item_utility, minimum_purchase, total_cost


def run_greedy(data: ProcurementData) -> RunResult:
    """Жадное добавление единиц с максимальным приростом полезности на рубль.

    evaluations - число вычислений полезности позиции (item_utility):
    это оценки кандидатов, а не оценки целых планов, как в генетическом алгоритме.
    """
    validate_data(data)
    started = time.perf_counter()

    plan = minimum_purchase(data).astype(np.int64)
    cost = total_cost(data, plan)
    evaluations = 0

    current = np.empty(data.n)
    for i in range(data.n):
        current[i] = item_utility(data, i, int(plan[i]))
        evaluations += 1
    utility = float(current.sum())
    history = [utility]

    while True:
        best_index = -1
        best_ratio = 0.0
        best_utility = 0.0
        for i in range(data.n):
            price = float(data.prices[i])
            if plan[i] >= data.upper_bounds[i] or cost + price > data.budget + BUDGET_TOLERANCE:
                continue
            candidate = item_utility(data, i, int(plan[i]) + 1)
            evaluations += 1
            ratio = (candidate - current[i]) / price
            if ratio > best_ratio:
                best_index, best_ratio, best_utility = i, ratio, candidate
        if best_index < 0:
            break
        utility += best_utility - current[best_index]
        current[best_index] = best_utility
        plan[best_index] += 1
        cost += float(data.prices[best_index])
        history.append(utility)

    if not is_feasible(data, plan):
        raise RuntimeError("Жадный план оказался недопустимым")
    final_cost = total_cost(data, plan)
    return RunResult(
        method="greedy",
        seed=None,
        config=None,
        best_x=plan,
        best_utility=utility,
        best_cost=final_cost,
        found_feasible=True,
        best_feasible_generation=None,
        best_fitness=utility,
        best_fitness_feasible=True,
        fitness_history=list(history),
        utility_history=list(history),
        feasible_fraction_history=[1.0] * len(history),
        evaluations=evaluations,
        elapsed_seconds=time.perf_counter() - started,
    )