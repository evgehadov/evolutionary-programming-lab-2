"""Генетические операторы: инициализация, кроссовер, мутации, селекция.

Операторы работают только с целочисленными векторами в границах 0..u_i.
Бюджет и минимальные запасы здесь не обрабатываются: потомок затем проходит
repair либо оценивается штрафом (см. constraints.py).
"""

import numpy as np
from numpy.typing import ArrayLike

from procurement_ga.constraints import repair
from procurement_ga.models import MutationType, ProcurementData
from procurement_ga.objective import minimum_purchase


def _as_individual(x: ArrayLike, upper: np.ndarray) -> np.ndarray:
    """Проверяет особь (целые числа, длина, границы 0..u) и возвращает копию."""
    individual = np.asarray(x)
    if individual.ndim != 1 or individual.shape != upper.shape:
        raise ValueError(
            f"Ожидался вектор длины {upper.shape[0]}, получена форма {individual.shape}"
        )
    if individual.dtype.kind not in "iu":
        raise ValueError(
            f"Особь должна состоять из целых чисел, получен тип {individual.dtype}"
        )
    individual = individual.astype(np.int64)
    if (individual < 0).any() or (individual > upper).any():
        raise ValueError("Гены особи выходят за границы 0..u")
    return individual


def initialize_population(
    data: ProcurementData,
    size: int,
    rng: np.random.Generator,
    feasible: bool,
) -> np.ndarray:
    """Начальная популяция формы (size, n).

    feasible=True: каждая особь допустима (минимумы выполнены, бюджет не превышен).
    Разнообразие задаётся случайной долей позиций с резервом у каждой особи.
    feasible=False: гены равномерно случайны в 0..u_i, ограничения могут нарушаться.
    """
    if size < 1:
        raise ValueError("Размер популяции должен быть >= 1")
    upper = data.upper_bounds
    if not feasible:
        return rng.integers(0, upper + 1, size=(size, data.n))

    required = minimum_purchase(data)
    room = upper - required
    if (room < 0).any():
        raise ValueError("Верхняя граница u меньше обязательной закупки q")
    population = np.empty((size, data.n), dtype=np.int64)
    for k in range(size):
        density = rng.random()
        take = rng.random(data.n) < density
        extra = np.where(take, rng.integers(0, room + 1), 0)
        population[k] = repair(data, required + extra)
    return population


def uniform_crossover(
    parent_a: ArrayLike, parent_b: ArrayLike, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Равномерный кроссовер: каждый ген берётся от одного из родителей."""
    a = np.asarray(parent_a)
    b = np.asarray(parent_b)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError("Родители должны быть векторами одинаковой длины")
    mask = rng.random(a.shape[0]) < 0.5
    return np.where(mask, a, b), np.where(mask, b, a)


def _pick_genes(
    candidates: np.ndarray, n_genes: int, rng: np.random.Generator
) -> np.ndarray:
    count = min(n_genes, candidates.size)
    return rng.choice(candidates, size=count, replace=False)


def step_mutation(
    x: ArrayLike, upper: np.ndarray, n_genes: int, rng: np.random.Generator
) -> np.ndarray:
    """Шаговая мутация: до n_genes генов меняются на +1 или -1 в границах 0..u."""
    if n_genes < 1:
        raise ValueError("n_genes должен быть >= 1")
    child = _as_individual(x, upper)
    changeable = np.flatnonzero(upper > 0)
    if changeable.size == 0:
        return child
    index = _pick_genes(changeable, n_genes, rng)
    values = child[index]
    go_up = np.where(
        values == 0,
        True,
        np.where(values >= upper[index], False, rng.random(index.size) < 0.5),
    )
    child[index] += np.where(go_up, 1, -1)
    return child


def reassign_mutation(
    x: ArrayLike, upper: np.ndarray, n_genes: int, rng: np.random.Generator
) -> np.ndarray:
    """Мутация переназначением: до n_genes генов получают новое значение из 0..u."""
    if n_genes < 1:
        raise ValueError("n_genes должен быть >= 1")
    child = _as_individual(x, upper)
    changeable = np.flatnonzero(upper > 0)
    if changeable.size == 0:
        return child
    index = _pick_genes(changeable, n_genes, rng)
    # Берём число из 0..u-1 и сдвигаем его через старое значение: новое != старое.
    draw = rng.integers(0, upper[index])
    child[index] = draw + (draw >= child[index])
    return child


def mutate(
    x: ArrayLike,
    upper: np.ndarray,
    mutation_type: MutationType,
    n_genes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if mutation_type is MutationType.STEP:
        return step_mutation(x, upper, n_genes, rng)
    if mutation_type is MutationType.REASSIGN:
        return reassign_mutation(x, upper, n_genes, rng)
    raise ValueError(f"Неизвестный тип мутации: {mutation_type}")


def tournament_select(
    fitness: ArrayLike,
    n_select: int,
    tournament_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Турнирная селекция; возвращает индексы победителей.

    Побеждает особь с наибольшим fitness; при равенстве - с меньшим индексом.
    """
    scores = np.asarray(fitness, dtype=float)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("fitness должен быть непустым одномерным массивом")
    if np.isnan(scores).any():
        raise ValueError("fitness содержит NaN")
    if n_select < 1 or tournament_size < 1:
        raise ValueError("n_select и tournament_size должны быть >= 1")

    contenders = rng.integers(0, scores.size, size=(n_select, tournament_size))
    order = np.lexsort((np.arange(scores.size), -scores))  # лучшие первыми
    rank = np.empty(scores.size, dtype=np.int64)
    rank[order] = np.arange(scores.size)
    best_column = rank[contenders].argmin(axis=1)
    return contenders[np.arange(n_select), best_column]