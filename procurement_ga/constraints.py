"""Обработка ограничений: проверка допустимости, штрафы и восстановление."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from procurement_ga.models import ProcurementData
from procurement_ga.objective import (
    evaluate_plan,
    item_utilities,
    minimum_purchase,
    total_cost,
    total_utility,
    validate_plan,
)

BUDGET_TOLERANCE = 1e-9  # допуск на погрешность суммирования вещественных цен


@dataclass(frozen=True)
class Violations:
    """Объём нарушений ограничений для одного плана."""

    shortage: int  # sum(max(0, q_i - x_i))
    budget_excess: float  # max(0, C(x) - B)
    out_of_bounds: int  # число генов выше верхней границы u_i

    @property
    def feasible(self) -> bool:
        return (
            self.shortage == 0
            and self.out_of_bounds == 0
            and self.budget_excess <= BUDGET_TOLERANCE
        )


@dataclass(frozen=True)
class PlanDiagnostics:
    """Сводка по плану: показатели, допустимость и причины недопустимости."""

    cost: float
    utility: float
    shortage: int
    budget_excess: float
    out_of_bounds: int
    feasible: bool
    messages: tuple[str, ...]


def constraint_violations(data: ProcurementData, x: ArrayLike) -> Violations:
    plan = validate_plan(data, x)
    required = minimum_purchase(data)
    return Violations(
        shortage=int(np.maximum(0, required - plan).sum()),
        budget_excess=max(0.0, total_cost(data, plan) - data.budget),
        out_of_bounds=int((plan > data.upper_bounds).sum()),
    )


def is_feasible(data: ProcurementData, x: ArrayLike) -> bool:
    return constraint_violations(data, x).feasible


def penalized_fitness(
    data: ProcurementData,
    x: ArrayLike,
    penalty_shortage: float,
    penalty_budget: float,
) -> float:
    """F(x) = U(x) - lambda_q * shortage(x) - lambda_B * budget_excess(x) / B.

    Границы 0..u_i гарантируются операторами и штрафом не учитываются.
    """
    if penalty_shortage <= 0 or penalty_budget <= 0:
        raise ValueError("Штрафные коэффициенты должны быть > 0")
    plan = validate_plan(data, x)
    violations = constraint_violations(data, plan)
    return (
        total_utility(data, plan)
        - penalty_shortage * violations.shortage
        - penalty_budget * violations.budget_excess / data.budget
    )


def diagnose(data: ProcurementData, x: ArrayLike) -> PlanDiagnostics:
    evaluation = evaluate_plan(data, x)
    violations = constraint_violations(data, evaluation.purchase)
    messages: list[str] = []
    if violations.shortage:
        messages.append(
            f"недостаток обязательных закупок: {violations.shortage} ед."
        )
    if violations.budget_excess > BUDGET_TOLERANCE:
        messages.append(
            f"превышение бюджета на {violations.budget_excess:.2f} "
            f"(стоимость {evaluation.total_cost:.2f}, бюджет {data.budget:.2f})"
        )
    if violations.out_of_bounds:
        messages.append(
            f"генов выше верхней границы u: {violations.out_of_bounds}"
        )
    return PlanDiagnostics(
        cost=evaluation.total_cost,
        utility=evaluation.total_utility,
        shortage=violations.shortage,
        budget_excess=violations.budget_excess,
        out_of_bounds=violations.out_of_bounds,
        feasible=violations.feasible,
        messages=tuple(messages),
    )


def _cheapest_reduction(
    data: ProcurementData, plan: np.ndarray, required: np.ndarray
) -> tuple[int, int]:
    """Выбирает позицию для сокращения и число единиц с неизменной потерей.

    Потеря при снятии одной единицы считается по текущему значению гена:
    v_i(x_i) - v_i(x_i - 1). Критерий выбора: минимум потери на единицу
    сэкономленных средств (потеря / p_i); при равенстве - более дорогая
    позиция, затем меньший номер.
    """
    extra = plan - required
    candidates = extra > 0
    if not candidates.any():
        raise ValueError(
            "Бюджет не покрывает обязательные закупки: восстановить допустимость невозможно"
        )
    loss = item_utilities(data, plan) - item_utilities(
        data, np.maximum(plan - 1, 0)
    )
    score = np.where(candidates, loss / data.prices, np.inf)
    order = np.lexsort((np.arange(data.n), -data.prices, score))
    index = int(order[0])

    cap = int(data.reserve_caps[index])
    current_extra = int(extra[index])
    # Потеря постоянна, пока не кончится избыток над h_i или сам резерв.
    removable = current_extra - cap if current_extra > cap else current_extra
    return index, removable


def repair(data: ProcurementData, x: ArrayLike) -> np.ndarray:
    """Возвращает допустимый план, близкий к x; исходный массив не меняется."""
    plan = validate_plan(data, np.maximum(np.asarray(x), 0))
    required = minimum_purchase(data)
    plan = np.maximum(np.minimum(plan, data.upper_bounds), required)

    while True:
        excess = total_cost(data, plan) - data.budget
        if excess <= BUDGET_TOLERANCE:
            return plan
        index, removable = _cheapest_reduction(data, plan, required)
        needed = max(1, math.ceil(excess / data.prices[index] - 1e-9))
        plan[index] -= min(removable, needed)