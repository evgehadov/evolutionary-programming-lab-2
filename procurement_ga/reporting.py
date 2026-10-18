"""Сохранение результатов: CSV и графики в каталог results/."""

import csv
from dataclasses import fields
from enum import Enum
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from procurement_ga.constraints import is_feasible
from procurement_ga.experiments import SuiteResult, best_ga_run
from procurement_ga.models import GAConfig, ProcurementData, RunResult, SeriesStats
from procurement_ga.objective import evaluate_plan, minimum_purchase, total_cost

DEFAULT_RESULTS_DIR = Path("results")
GREEDY_LABEL = "greedy"
UTILITY_TOLERANCE = 1e-9
COLORS = {"A": "C0", "B": "C1", "C": "C2", "D": "C3", GREEDY_LABEL: "black"}

GA_PARAM_NAMES = [f.name for f in fields(GAConfig) if f.name != "seed"]
RESULT_COLUMNS = [
    "found_feasible",
    "best_utility",
    "best_cost",
    "best_feasible_generation",
    "best_fitness",
    "best_fitness_feasible",
    "evaluations",
    "elapsed_seconds",
]
SUMMARY_COLUMNS = [f.name for f in fields(SeriesStats)]


def ensure_results_dir(path: str | Path = DEFAULT_RESULTS_DIR) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _cell(value: object) -> object:
    """Приводит значение к виду для CSV: None и NaN дают пустую ячейку."""
    if value is None:
        return ""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return "" if np.isnan(value) else float(value)
    return value


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> Path:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in rows:
            writer.writerow([_cell(value) for value in row])
    return path


def _result_cells(result: RunResult) -> list[object]:
    return [getattr(result, column) for column in RESULT_COLUMNS]


def save_data_csv(data: ProcurementData, path: Path) -> Path:
    required = minimum_purchase(data)
    header = [
        "id", "name", "price", "stock", "demand", "min_remaining", "weight",
        "reserve_cap_h", "min_purchase_q", "max_purchase_u", "budget", "data_seed",
    ]
    rows = [
        [
            m.id, m.name, m.price, m.stock, m.demand, m.min_remaining, m.weight,
            m.reserve_cap, required[i], m.max_purchase, data.budget, data.seed,
        ]
        for i, m in enumerate(data.materials)
    ]
    return _write_csv(path, header, rows)


def save_experiments_config_csv(
    data: ProcurementData, suite: SuiteResult, path: Path
) -> Path:
    header = ["experiment", "runs", "base_seed", *GA_PARAM_NAMES,
              "n_materials", "budget", "data_seed"]
    rows = [
        [name, experiment.runs, experiment.base_seed,
         *[getattr(experiment.ga, p) for p in GA_PARAM_NAMES],
         data.n, data.budget, data.seed]
        for name, experiment in suite.experiments.items()
    ]
    return _write_csv(path, header, rows)


def save_runs_csv(suite: SuiteResult, path: Path) -> Path:
    header = ["experiment", "run", "seed", *GA_PARAM_NAMES, *RESULT_COLUMNS]
    rows: list[list[object]] = []
    for name, results in suite.runs.items():
        for k, result in enumerate(results):
            params = [getattr(result.config, p) for p in GA_PARAM_NAMES]
            rows.append([name, k, result.seed, *params, *_result_cells(result)])
    rows.append(
        [GREEDY_LABEL, 0, None, *[None] * len(GA_PARAM_NAMES),
         *_result_cells(suite.greedy)]
    )
    return _write_csv(path, header, rows)


def save_summary_csv(suite: SuiteResult, path: Path) -> Path:
    all_stats = [*suite.stats.values(), suite.greedy_stats]
    rows = [[getattr(s, column) for column in SUMMARY_COLUMNS] for s in all_stats]
    return _write_csv(path, SUMMARY_COLUMNS, rows)


def _describe_ga_run(suite: SuiteResult, run: RunResult) -> str:
    for name, results in suite.runs.items():
        for k, candidate in enumerate(results):
            if candidate is run:
                return f"GA {name}, запуск {k}, seed {run.seed}"
    raise ValueError("Запуск не принадлежит набору результатов")


def select_best_plan(suite: SuiteResult) -> tuple[RunResult, str]:
    """Лучший допустимый план: из ГА; жадный - только если строго лучше."""
    best = best_ga_run(suite)
    if best is not None and not (
        suite.greedy.best_utility > best.best_utility + UTILITY_TOLERANCE
    ):
        return best, _describe_ga_run(suite, best)
    return suite.greedy, GREEDY_LABEL


def save_best_plan_csv(
    data: ProcurementData, run: RunResult, source: str, path: Path
) -> Path:
    if run.best_x is None or not is_feasible(data, run.best_x):
        raise ValueError("Сохранять можно только допустимый план закупок")
    evaluation = evaluate_plan(data, run.best_x)
    header = [
        "id", "name", "price", "stock", "demand", "min_remaining", "required_q",
        "purchase", "final_stock", "item_cost", "utility",
    ]
    rows: list[list[object]] = [
        [m.id, m.name, m.price, m.stock, m.demand, m.min_remaining,
         evaluation.required[i], evaluation.purchase[i], evaluation.remaining[i],
         evaluation.item_costs[i], evaluation.item_utilities[i]]
        for i, m in enumerate(data.materials)
    ]
    blank = [None] * 5
    rows.append(["ИТОГО", "", *blank, int(evaluation.purchase.sum()),
                 None, evaluation.total_cost, evaluation.total_utility])
    rows.append(["БЮДЖЕТ", "", *blank, None, None, data.budget, None])
    rows.append(["ДОПУСТИМОСТЬ", "да", *blank, None, None, None, None])
    rows.append(["ИСТОЧНИК", source, *blank, None, None, None, None])
    return _write_csv(path, header, rows)


def _config_label(suite: SuiteResult, name: str) -> str:
    ga = suite.experiments[name].ga
    return f"{name}\n{ga.constraint_strategy.value}\n{ga.mutation_type.value}"


def _mean_curve(runs: list[RunResult], attribute: str) -> np.ndarray:
    """Среднее по запускам для каждого поколения; NaN игнорируются."""
    histories = [list(getattr(r, attribute)) for r in runs]
    length = max(len(h) for h in histories)
    padded = np.array(
        [h + [h[-1]] * (length - len(h)) for h in histories], dtype=float
    )
    counts = (~np.isnan(padded)).sum(axis=0)
    sums = np.nansum(padded, axis=0)
    return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def _finish(fig: plt.Figure, ax: plt.Axes, path: Path) -> Path:
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_curves(
    suite: SuiteResult, attribute: str, title: str, ylabel: str, path: Path,
    greedy_line: bool,
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, results in suite.runs.items():
        ga = suite.experiments[name].ga
        label = f"{name}: {ga.constraint_strategy.value}, {ga.mutation_type.value}"
        ax.plot(_mean_curve(results, attribute), label=label, color=COLORS[name])
    if greedy_line:
        ax.axhline(suite.greedy.best_utility, color=COLORS[GREEDY_LABEL],
                   linestyle="--", label="Жадная эвристика")
    ax.set_title(title)
    ax.set_xlabel("Поколение")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    return _finish(fig, ax, path)


def plot_convergence(suite: SuiteResult, path: Path) -> Path:
    return _plot_curves(
        suite, "utility_history",
        "Динамика лучшей допустимой полезности (среднее по запускам)",
        "Полезность лучшего допустимого плана", path, greedy_line=True,
    )


def plot_feasible_share(suite: SuiteResult, path: Path) -> Path:
    path = _plot_curves(
        suite, "feasible_fraction_history",
        "Доля допустимых особей в популяции (среднее по запускам)",
        "Доля допустимых особей", path, greedy_line=False,
    )
    return path


def _set_config_ticks(ax: plt.Axes, suite: SuiteResult, extra: list[str]) -> None:
    labels = [_config_label(suite, name) for name in suite.runs] + extra
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.5, len(labels) + 0.5)


def _no_feasible_note(ax: plt.Axes, position: float) -> None:
    ax.text(position, 0.5, "нет допустимых\nпланов", ha="center",
            transform=ax.get_xaxis_transform())


def plot_utility_comparison(suite: SuiteResult, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for position, (name, results) in enumerate(suite.runs.items(), start=1):
        values = [r.best_utility for r in results if r.found_feasible]
        if values:
            ax.boxplot([values], positions=[position], widths=0.5, showmeans=True)
        else:
            _no_feasible_note(ax, position)
    ax.axhline(suite.greedy.best_utility, color=COLORS[GREEDY_LABEL],
               linestyle="--", label="Жадная эвристика")
    _set_config_ticks(ax, suite, [])
    ax.set_title("Полезность лучшего допустимого плана по запускам")
    ax.set_xlabel("Конфигурация")
    ax.set_ylabel("Полезность")
    ax.grid(axis="y", alpha=0.3)
    return _finish(fig, ax, path)


def plot_cost_vs_budget(
    data: ProcurementData, suite: SuiteResult, path: Path
) -> Path:
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8, 5))
    for position, (name, results) in enumerate(suite.runs.items(), start=1):
        costs = [r.best_cost for r in results if r.found_feasible]
        if costs:
            jitter = rng.uniform(-0.15, 0.15, len(costs))
            ax.scatter(position + jitter, costs, color=COLORS[name], alpha=0.7)
        else:
            _no_feasible_note(ax, position)
    greedy_position = len(suite.runs) + 1
    ax.scatter([greedy_position], [suite.greedy.best_cost], marker="D",
               color=COLORS[GREEDY_LABEL], label="Жадная эвристика")
    mandatory = total_cost(data, minimum_purchase(data))
    ax.axhline(data.budget, color="red", linestyle="--", label="Бюджет")
    ax.axhline(mandatory, color="gray", linestyle=":", label="Обязательные закупки")
    _set_config_ticks(ax, suite, ["greedy"])
    ax.set_title("Стоимость лучшего допустимого плана и бюджет")
    ax.set_xlabel("Конфигурация (каждая точка - один запуск)")
    ax.set_ylabel("Стоимость закупки")
    ax.grid(axis="y", alpha=0.3)
    return _finish(fig, ax, path)


def plot_ga_vs_greedy(suite: SuiteResult, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(suite.runs) + [GREEDY_LABEL]
    all_stats = [*suite.stats.values(), suite.greedy_stats]
    best_labeled = False
    for position, (name, stats) in enumerate(zip(names, all_stats), start=1):
        if stats.mean_utility is None:
            _no_feasible_note(ax, position)
            continue
        ax.bar(position, stats.mean_utility, yerr=stats.std_utility, capsize=4,
               color=COLORS[name], alpha=0.8)
        ax.scatter([position], [stats.best_utility], marker="*", s=120,
                   color="gold", edgecolor="black", zorder=3,
                   label=None if best_labeled else "Лучший запуск серии")
        best_labeled = True
    labels = [_config_label(suite, name) for name in suite.runs] + [GREEDY_LABEL]
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_title("Генетический алгоритм и жадная эвристика: средняя полезность")
    ax.set_xlabel("Метод (столбец - среднее, усы - стандартное отклонение)")
    ax.set_ylabel("Полезность")
    ax.grid(axis="y", alpha=0.3)
    return _finish(fig, ax, path)


def save_all(
    data: ProcurementData,
    suite: SuiteResult,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> dict[str, Path]:
    """Сохраняет все CSV и графики; возвращает пути по именам."""
    directory = ensure_results_dir(results_dir)
    best, source = select_best_plan(suite)
    return {
        "data": save_data_csv(data, directory / "data.csv"),
        "experiments_config": save_experiments_config_csv(
            data, suite, directory / "experiments_config.csv"
        ),
        "runs": save_runs_csv(suite, directory / "runs.csv"),
        "summary": save_summary_csv(suite, directory / "summary.csv"),
        "best_plan": save_best_plan_csv(data, best, source, directory / "best_plan.csv"),
        "convergence": plot_convergence(suite, directory / "convergence.png"),
        "utility_comparison": plot_utility_comparison(
            suite, directory / "utility_comparison.png"
        ),
        "cost_vs_budget": plot_cost_vs_budget(
            data, suite, directory / "cost_vs_budget.png"
        ),
        "feasible_share": plot_feasible_share(
            suite, directory / "feasible_share.png"
        ),
        "ga_vs_greedy": plot_ga_vs_greedy(suite, directory / "ga_vs_greedy.png"),
    }