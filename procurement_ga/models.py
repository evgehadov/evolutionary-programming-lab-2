"""Структуры данных проекта: материалы, параметры ГА, эксперименты, результаты."""

from dataclasses import dataclass, field, replace
from enum import Enum
from functools import cached_property

import numpy as np


class ConstraintStrategy(str, Enum):
    REPAIR = "repair"
    PENALTY = "penalty"


class MutationType(str, Enum):
    STEP = "step"
    REASSIGN = "reassign"


@dataclass(frozen=True)
class Material:
    """Один вид расходного материала (обозначения как в постановке задачи)."""

    id: int
    name: str
    price: float  # p_i, цена единицы
    stock: int  # s_i, текущий запас
    demand: int  # d_i, расход за месяц
    min_remaining: int  # m_i, минимальный остаток
    weight: float  # w_i, вес полезности
    reserve_cap: int  # h_i, резерв, при котором полезность насыщается
    max_purchase: int  # u_i, максимум к закупке

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError(f"Материал {self.id}: пустое название")
        if self.price <= 0:
            raise ValueError(f"Материал {self.id}: цена должна быть > 0")
        if self.stock < 0 or self.demand < 0 or self.min_remaining < 0:
            raise ValueError(
                f"Материал {self.id}: запас, расход и минимальный остаток должны быть >= 0"
            )
        if self.weight <= 0:
            raise ValueError(f"Материал {self.id}: вес полезности должен быть > 0")
        if self.reserve_cap < 1:
            raise ValueError(f"Материал {self.id}: размер резерва h должен быть >= 1")
        if self.max_purchase < 0:
            raise ValueError(f"Материал {self.id}: верхняя граница u должна быть >= 0")


def _readonly(values: list, dtype: type) -> np.ndarray:
    array = np.array(values, dtype=dtype)
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class ProcurementData:
    """Набор исходных данных: материалы и общий бюджет."""

    materials: tuple[Material, ...]
    budget: float
    seed: int | None = None  # seed генерации, если данные сгенерированы

    def __post_init__(self) -> None:
        object.__setattr__(self, "materials", tuple(self.materials))
        if not self.materials:
            raise ValueError("Набор данных не содержит материалов")
        ids = [m.id for m in self.materials]
        if len(set(ids)) != len(ids):
            raise ValueError("Идентификаторы материалов должны быть уникальны")
        if self.budget <= 0:
            raise ValueError("Бюджет должен быть > 0")

    @property
    def n(self) -> int:
        return len(self.materials)

    @cached_property
    def prices(self) -> np.ndarray:
        return _readonly([m.price for m in self.materials], float)

    @cached_property
    def stocks(self) -> np.ndarray:
        return _readonly([m.stock for m in self.materials], np.int64)

    @cached_property
    def demands(self) -> np.ndarray:
        return _readonly([m.demand for m in self.materials], np.int64)

    @cached_property
    def min_remainings(self) -> np.ndarray:
        return _readonly([m.min_remaining for m in self.materials], np.int64)

    @cached_property
    def weights(self) -> np.ndarray:
        return _readonly([m.weight for m in self.materials], float)

    @cached_property
    def reserve_caps(self) -> np.ndarray:
        return _readonly([m.reserve_cap for m in self.materials], np.int64)

    @cached_property
    def upper_bounds(self) -> np.ndarray:
        return _readonly([m.max_purchase for m in self.materials], np.int64)


@dataclass(frozen=True)
class GAConfig:
    """Параметры одного запуска генетического алгоритма."""

    population_size: int = 100
    max_generations: int = 1000  # страховочный предел числа поколений
    max_evaluations: int = 30_000  # основной критерий остановки
    stagnation_limit: int | None = None  # поколений без улучшения; None = выключено
    crossover_prob: float = 0.9
    mutation_prob: float = 0.3  # вероятность мутации одной особи
    mutation_genes: int = 3  # сколько генов меняется при мутации
    mutation_type: MutationType = MutationType.STEP
    constraint_strategy: ConstraintStrategy = ConstraintStrategy.REPAIR
    tournament_size: int = 3
    elite_size: int = 2
    penalty_shortage: float = 20.0  # lambda_q
    penalty_budget: float = 2000.0  # lambda_B
    seed: int = 0

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("Размер популяции должен быть >= 2")
        if self.max_generations < 1:
            raise ValueError("Число поколений должно быть >= 1")
        if self.max_evaluations < self.population_size:
            raise ValueError("Лимит оценок меньше размера популяции")
        if self.stagnation_limit is not None and self.stagnation_limit < 1:
            raise ValueError("stagnation_limit должен быть None или >= 1")
        if not 0.0 <= self.crossover_prob <= 1.0:
            raise ValueError("Вероятность кроссовера должна быть в [0, 1]")
        if not 0.0 <= self.mutation_prob <= 1.0:
            raise ValueError("Вероятность мутации должна быть в [0, 1]")
        if self.mutation_genes < 1:
            raise ValueError("mutation_genes должен быть >= 1")
        if self.tournament_size < 1:
            raise ValueError("Размер турнира должен быть >= 1")
        if not 0 <= self.elite_size < self.population_size:
            raise ValueError("Размер элиты должен быть в [0, population_size)")
        if self.penalty_shortage <= 0 or self.penalty_budget <= 0:
            raise ValueError("Штрафные коэффициенты должны быть > 0")


@dataclass(frozen=True)
class ExperimentConfig:
    """Серия независимых запусков с одними и теми же параметрами ГА."""

    name: str
    ga: GAConfig
    runs: int = 20
    base_seed: int = 1000

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("У эксперимента должно быть имя")
        if self.runs < 1:
            raise ValueError("Число запусков должно быть >= 1")

    def seed_for_run(self, run_index: int) -> int:
        return self.base_seed + run_index

    def config_for_run(self, run_index: int) -> GAConfig:
        return replace(self.ga, seed=self.seed_for_run(run_index))


@dataclass(frozen=True, eq=False)
class RunResult:
    """Результат одного запуска ГА или жадной эвристики."""

    method: str  # "GA" или "greedy"
    seed: int | None
    config: GAConfig | None
    best_x: np.ndarray | None  # лучший допустимый план; None, если не найден
    best_utility: float | None
    best_cost: float | None
    found_feasible: bool
    best_feasible_generation: int | None
    best_fitness: float  # лучший fitness за запуск (особь может быть недопустимой)
    best_fitness_feasible: bool  # допустима ли особь с лучшим fitness
    fitness_history: list[float] = field(default_factory=list)
    utility_history: list[float] = field(default_factory=list)  # nan, пока нет допустимых
    feasible_fraction_history: list[float] = field(default_factory=list)
    evaluations: int = 0
    elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class SeriesStats:
    """Статистика серии запусков. None означает отсутствие допустимых решений."""

    label: str
    n_runs: int
    n_feasible_found: int
    best_utility: float | None
    mean_utility: float | None
    median_utility: float | None
    std_utility: float | None
    worst_utility: float | None
    mean_cost: float | None
    final_feasible_share: float  # доля запусков, где лучшая по fitness особь допустима
    feasible_found_share: float  # доля запусков, где найден хоть один допустимый план
    mean_best_fitness: float | None
    mean_evaluations: float
    mean_time_seconds: float