"""Эксперименты A-D, статистика серий, сравнение с жадной эвристикой, демонстрации."""

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from procurement_ga.baseline import run_greedy
from procurement_ga.constraints import PlanDiagnostics, diagnose, is_feasible, repair
from procurement_ga.genetic_algorithm import run_ga
from procurement_ga.models import (
    ConstraintStrategy,
    ExperimentConfig,
    GAConfig,
    MutationType,
    ProcurementData,
    RunResult,
    SeriesStats,
)
from procurement_ga.objective import minimum_purchase
from procurement_ga.operators import initialize_population, mutate, uniform_crossover

DEFAULT_RUNS = 20
DEFAULT_BASE_SEED = 1000

# Обработка ограничений и тип мутации для каждого эксперимента.
EXPERIMENT_TABLE: dict[str, tuple[ConstraintStrategy, MutationType]] = {
    "A": (ConstraintStrategy.REPAIR, MutationType.STEP),
    "B": (ConstraintStrategy.REPAIR, MutationType.REASSIGN),
    "C": (ConstraintStrategy.PENALTY, MutationType.STEP),
    "D": (ConstraintStrategy.PENALTY, MutationType.REASSIGN),
}

ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True, eq=False)
class SuiteResult:
    """Результаты полной серии A-D и жадной эвристики."""

    experiments: dict[str, ExperimentConfig]
    runs: dict[str, list[RunResult]]
    stats: dict[str, SeriesStats]
    greedy: RunResult
    greedy_stats: SeriesStats


@dataclass(frozen=True, eq=False)
class PlanExample:
    """Пример плана с диагностикой; для недопустимых - ещё и результат repair."""

    label: str
    x: np.ndarray
    diagnostics: PlanDiagnostics
    repaired: np.ndarray | None


def build_experiments(
    base_ga: GAConfig | None = None,
    runs: int = DEFAULT_RUNS,
    base_seed: int = DEFAULT_BASE_SEED,
) -> dict[str, ExperimentConfig]:
    """Четыре конфигурации с одинаковыми базовыми параметрами ГА."""
    base = base_ga if base_ga is not None else GAConfig()
    return {
        name: ExperimentConfig(
            name=name,
            ga=replace(base, constraint_strategy=strategy, mutation_type=mutation),
            runs=runs,
            base_seed=base_seed,
        )
        for name, (strategy, mutation) in EXPERIMENT_TABLE.items()
    }


def run_series(
    data: ProcurementData,
    experiment: ExperimentConfig,
    progress: ProgressCallback | None = None,
) -> list[RunResult]:
    results: list[RunResult] = []
    for k in range(experiment.runs):
        results.append(run_ga(data, experiment.config_for_run(k)))
        if progress is not None:
            progress(experiment.name, k + 1, experiment.runs)
    return results


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def summarize_series(label: str, results: list[RunResult]) -> SeriesStats:
    """Статистика серии; без допустимых решений показатели полезности равны None."""
    if not results:
        raise ValueError("Серия не содержит запусков")
    feasible = [r for r in results if r.found_feasible]
    utilities = np.array([r.best_utility for r in feasible], dtype=float)
    n = len(results)
    return SeriesStats(
        label=label,
        n_runs=n,
        n_feasible_found=len(feasible),
        best_utility=float(utilities.max()) if feasible else None,
        mean_utility=float(utilities.mean()) if feasible else None,
        median_utility=float(np.median(utilities)) if feasible else None,
        std_utility=(
            (float(np.std(utilities, ddof=1)) if len(feasible) > 1 else 0.0)
            if feasible
            else None
        ),
        worst_utility=float(utilities.min()) if feasible else None,
        mean_cost=_mean([r.best_cost for r in feasible]),
        final_feasible_share=sum(r.best_fitness_feasible for r in results) / n,
        feasible_found_share=len(feasible) / n,
        mean_best_fitness=_mean([r.best_fitness for r in results]),
        mean_evaluations=float(np.mean([r.evaluations for r in results])),
        mean_time_seconds=float(np.mean([r.elapsed_seconds for r in results])),
    )


def _check_comparable(experiments: dict[str, ExperimentConfig]) -> None:
    if not experiments:
        raise ValueError("Не задано ни одного эксперимента")
    budgets = {e.ga.max_evaluations for e in experiments.values()}
    if len(budgets) != 1:
        raise ValueError(
            "Лимит вычислений целевой функции должен быть одинаков во всех "
            f"экспериментах, получено: {sorted(budgets)}"
        )


def run_suite(
    data: ProcurementData,
    experiments: dict[str, ExperimentConfig],
    progress: ProgressCallback | None = None,
) -> SuiteResult:
    """Все эксперименты на одних данных плюс жадная эвристика."""
    _check_comparable(experiments)
    runs = {
        name: run_series(data, experiment, progress)
        for name, experiment in experiments.items()
    }
    stats = {name: summarize_series(name, results) for name, results in runs.items()}
    greedy = run_greedy(data)
    return SuiteResult(
        experiments=experiments,
        runs=runs,
        stats=stats,
        greedy=greedy,
        greedy_stats=summarize_series("greedy", [greedy]),
    )


def best_ga_run(suite: SuiteResult) -> RunResult | None:
    """Запуск ГА с наибольшей полезностью допустимого плана; None, если таких нет."""
    candidates = [
        r for results in suite.runs.values() for r in results if r.found_feasible
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.best_utility)


def build_plan_examples(data: ProcurementData, seed: int = 0) -> list[PlanExample]:
    """Два допустимых и два недопустимых плана; ожидаемые свойства проверяются."""
    required = minimum_purchase(data)
    rng = np.random.default_rng(seed)
    examples: list[PlanExample] = []

    def add(label: str, x: np.ndarray, expected_feasible: bool) -> PlanDiagnostics:
        diagnostics = diagnose(data, x)
        if diagnostics.feasible != expected_feasible:
            raise RuntimeError(f"Пример «{label}» не соответствует ожидаемой допустимости")
        repaired = None
        if not expected_feasible:
            repaired = repair(data, x)
            if not is_feasible(data, repaired):
                raise RuntimeError(f"repair не восстановил допустимость: «{label}»")
        examples.append(PlanExample(label, x, diagnostics, repaired))
        return diagnostics

    add("допустимый: только обязательные закупки", required.copy(), True)
    add(
        "допустимый: случайный план после repair",
        repair(data, rng.integers(0, data.upper_bounds + 1)),
        True,
    )

    short = required.copy()
    index = int(np.argmax(required))
    if short[index] == 0:
        raise ValueError("Нет материала с обязательной закупкой: пример нехватки невозможен")
    short[index] -= 1
    diagnostics = add("недопустимый: недостаток обязательной закупки", short, False)
    if diagnostics.shortage != 1 or diagnostics.budget_excess > 0:
        raise RuntimeError("Нехватка распознана неверно")

    diagnostics = add(
        "недопустимый: превышение бюджета", data.upper_bounds.copy(), False
    )
    if diagnostics.budget_excess <= 0 or diagnostics.shortage != 0:
        raise RuntimeError("Превышение бюджета распознано неверно")
    return examples


def check_operators(data: ProcurementData, trials: int = 200, seed: int = 0) -> int:
    """Проверяет целочисленность, длину и границы после операторов.

    Возвращает число проверенных особей; при нарушении бросает RuntimeError.
    """
    rng = np.random.default_rng(seed)
    upper = data.upper_bounds
    checked = 0

    def verify(x: np.ndarray, source: str) -> None:
        nonlocal checked
        valid = (
            x.dtype.kind == "i"
            and x.shape == (data.n,)
            and bool((x >= 0).all())
            and bool((x <= upper).all())
        )
        if not valid:
            raise RuntimeError(f"{source}: нарушены целочисленность, длина или границы")
        checked += 1

    for feasible in (True, False):
        for x in initialize_population(data, 20, rng, feasible):
            verify(x, "инициализация")

    population = initialize_population(data, 20, rng, feasible=False)
    for _ in range(trials):
        i, j = rng.integers(0, population.shape[0], size=2)
        parent_a, parent_b = population[i], population[j]
        for child in uniform_crossover(parent_a, parent_b, rng):
            verify(child, "кроссовер")
        for mutation_type in MutationType:
            child = mutate(parent_a, upper, mutation_type, 3, rng)
            verify(child, f"мутация {mutation_type.value}")
            if (child == parent_a).all():
                raise RuntimeError(f"мутация {mutation_type.value} не изменила особь")
    return checked