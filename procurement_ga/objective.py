"""Математическая модель: стоимость, остатки и полезность плана закупки.

Все формулы задачи вычисляются только здесь; остальные модули вызывают
эти функции и не повторяют вычисления.
"""

import operator
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from procurement_ga.models import ProcurementData


def validate_plan(data: ProcurementData, x: ArrayLike) -> np.ndarray:
    """Проверяет размерность и тип вектора, возвращает его как int64."""
    plan = np.asarray(x)
    if plan.ndim != 1 or plan.shape[0] != data.n:
        raise ValueError(
            f"Ожидался вектор длины {data.n}, получена форма {plan.shape}"
        )
    if plan.dtype.kind not in "iu":
        raise ValueError(
            f"План должен состоять из целых чисел, получен тип {plan.dtype}"
        )
    plan = plan.astype(np.int64, copy=False)
    if (plan < 0).any():
        raise ValueError("Количество закупаемых единиц не может быть отрицательным")
    return plan


def minimum_purchase(data: ProcurementData) -> np.ndarray:
    """q_i = max(0, d_i + m_i - s_i)."""
    return np.maximum(0, data.demands + data.min_remainings - data.stocks)


def _extra_reserve(units: ArrayLike, required: ArrayLike) -> np.ndarray:
    """Дополнительный резерв сверх обязательной закупки: max(0, x_i - q_i)."""
    return np.maximum(0, np.asarray(units) - np.asarray(required))


def _utility_from_extra(
    weight: ArrayLike, cap: ArrayLike, extra: ArrayLike
) -> np.ndarray:
    """v_i = w_i * min(extra_i / h_i, 1)."""
    return weight * np.minimum(extra / cap, 1.0)


def _item_costs(data: ProcurementData, plan: np.ndarray) -> np.ndarray:
    return data.prices * plan


def _remaining(data: ProcurementData, plan: np.ndarray) -> np.ndarray:
    """r_i = s_i + x_i - d_i."""
    return data.stocks + plan - data.demands


def total_cost(data: ProcurementData, x: ArrayLike) -> float:
    """C(x) = sum(p_i * x_i)."""
    return float(_item_costs(data, validate_plan(data, x)).sum())


def remaining_stock(data: ProcurementData, x: ArrayLike) -> np.ndarray:
    """Остаток каждого материала после покрытия месячного расхода."""
    return _remaining(data, validate_plan(data, x))


def item_utility(data: ProcurementData, index: int, units: int) -> float:
    """Полезность одной позиции при закупке `units` единиц материала `index`."""
    if not 0 <= index < data.n:
        raise IndexError(f"Индекс материала {index} вне диапазона 0..{data.n - 1}")
    units = operator.index(units)
    if units < 0:
        raise ValueError("Количество закупаемых единиц не может быть отрицательным")
    required = minimum_purchase(data)[index]
    extra = _extra_reserve(units, required)
    return float(
        _utility_from_extra(data.weights[index], data.reserve_caps[index], extra)
    )


def item_utilities(data: ProcurementData, x: ArrayLike) -> np.ndarray:
    """Вектор полезностей v_i(x_i) по всем материалам."""
    plan = validate_plan(data, x)
    extra = _extra_reserve(plan, minimum_purchase(data))
    return _utility_from_extra(data.weights, data.reserve_caps, extra)


def total_utility(data: ProcurementData, x: ArrayLike) -> float:
    """U(x) = sum(v_i(x_i))."""
    return float(item_utilities(data, x).sum())


@dataclass(frozen=True, eq=False)
class PlanEvaluation:
    """Детальная оценка плана закупки (без проверки бюджета)."""

    purchase: np.ndarray
    required: np.ndarray  # q_i
    extra: np.ndarray  # max(0, x_i - q_i)
    item_costs: np.ndarray
    remaining: np.ndarray  # r_i
    item_utilities: np.ndarray
    total_cost: float
    total_utility: float
    meets_minimum: bool  # выполнены минимальные потребности


def evaluate_plan(data: ProcurementData, x: ArrayLike) -> PlanEvaluation:
    plan = validate_plan(data, x)
    required = minimum_purchase(data)
    extra = _extra_reserve(plan, required)
    item_costs = _item_costs(data, plan)
    remaining = _remaining(data, plan)
    utilities = _utility_from_extra(data.weights, data.reserve_caps, extra)

    by_purchase = bool((plan >= required).all())
    by_remaining = bool((remaining >= data.min_remainings).all())
    if by_purchase != by_remaining:
        raise RuntimeError(
            "Нарушена эквивалентность условий x_i >= q_i и r_i >= m_i: "
            "проверьте исходные данные"
        )

    return PlanEvaluation(
        purchase=plan,
        required=required,
        extra=extra,
        item_costs=item_costs,
        remaining=remaining,
        item_utilities=utilities,
        total_cost=float(item_costs.sum()),
        total_utility=float(utilities.sum()),
        meets_minimum=by_purchase,
    )