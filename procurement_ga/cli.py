"""Интерфейс командной строки."""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from procurement_ga.constraints import diagnose
from procurement_ga.data_generator import (
    DEFAULT_DATA_SEED,
    DEFAULT_N_MATERIALS,
    generate_data,
)
from procurement_ga.experiments import (
    DEFAULT_BASE_SEED,
    DEFAULT_RUNS,
    EXPERIMENT_TABLE,
    SuiteResult,
    build_experiments,
    build_plan_examples,
    check_operators,
    run_series,
    run_suite,
    summarize_series,
)
from procurement_ga.genetic_algorithm import run_ga
from procurement_ga.models import (
    ConstraintStrategy,
    GAConfig,
    MutationType,
    ProcurementData,
    SeriesStats,
)
from procurement_ga.objective import (
    evaluate_plan,
    minimum_purchase,
    total_cost,
    total_utility,
)


def _print_table(
    headers: list[str], rows: list[list[object]], left: tuple[int, ...] = (0,)
) -> None:
    columns = list(zip(headers, *rows))
    widths = [max(len(str(cell)) for cell in column) for column in columns]
    for line in [headers, *rows]:
        cells = []
        for index, (cell, width) in enumerate(zip(line, widths)):
            text = str(cell)
            cells.append(text.ljust(width) if index in left else text.rjust(width))
        print("  ".join(cells))


def _num(value: float | None, spec: str = ".3f") -> str:
    return "-" if value is None else format(value, spec)


def _stats_row(s: SeriesStats) -> list[object]:
    return [
        s.label,
        f"{s.n_feasible_found}/{s.n_runs}",
        _num(s.best_utility),
        _num(s.mean_utility),
        _num(s.median_utility),
        _num(s.std_utility),
        _num(s.worst_utility),
        _num(s.mean_cost, ".2f"),
        f"{s.mean_evaluations:.0f}",
        f"{s.mean_time_seconds:.2f}",
    ]


def _print_stats(stats: list[SeriesStats], greedy: SeriesStats | None = None) -> None:
    print("Полезность лучшего допустимого плана (по запускам, где он найден):")
    rows = [_stats_row(s) for s in stats]
    if greedy is not None:
        rows.append(_stats_row(greedy))
    _print_table(
        ["Метод", "Допуст. найдено", "Лучшая", "Средняя", "Медиана", "СКО",
         "Худшая", "Стоимость", "Оценок", "Время, с"],
        rows,
    )
    print()
    headers = ["Метод", "Итог допуст.", "Ср. fitness"]
    if greedy is not None:
        headers.append("Ср. - greedy")
    fitness_rows = []
    for s in stats:
        row: list[object] = [s.label, f"{s.final_feasible_share:.2f}",
                             _num(s.mean_best_fitness)]
        if greedy is not None:
            diff = None if s.mean_utility is None else s.mean_utility - greedy.best_utility
            row.append("-" if diff is None else f"{diff:+.3f}")
        fitness_rows.append(row)
    _print_table(headers, fitness_rows)
    print()
    print("Итог допуст. - доля запусков, где лучшая по fitness особь допустима;")
    print("fitness штрафного метода не равен полезности.")
    if greedy is not None:
        print("Оценок: для greedy - число вычислений полезности позиции, для ГА -")
        print("число оценок плана; напрямую эти числа сравнивать нельзя.")


def _print_plan(data: ProcurementData, x: object) -> None:
    evaluation = evaluate_plan(data, x)
    rows = [
        [m.id, m.name, int(evaluation.required[i]), int(evaluation.purchase[i]),
         f"{evaluation.item_costs[i]:.2f}", f"{evaluation.item_utilities[i]:.3f}"]
        for i, m in enumerate(data.materials)
    ]
    _print_table(["id", "Материал", "q", "x", "Стоимость", "Полезность"], rows, left=(1,))


def _progress(name: str, done: int, total: int) -> None:
    print(f"\r{name}: запуск {done}/{total}", end="", flush=True)
    if done == total:
        print()


def _load_data(args: argparse.Namespace) -> ProcurementData:
    return generate_data(n_materials=args.materials, seed=args.data_seed)


def _base_ga_config(args: argparse.Namespace) -> GAConfig:
    return GAConfig(
        population_size=args.population,
        max_generations=args.generations,
        max_evaluations=args.evaluations,
        crossover_prob=args.crossover,
        mutation_prob=args.mutation_prob,
        elite_size=args.elite,
        penalty_shortage=args.lambda_q,
        penalty_budget=args.lambda_b,
    )


def _run_suite(data: ProcurementData, args: argparse.Namespace) -> SuiteResult:
    experiments = build_experiments(_base_ga_config(args), args.runs, args.base_seed)
    print(f"Серии A-D: {args.runs} запусков на конфигурацию, "
          f"лимит {args.evaluations} оценок, seed данных {data.seed}")
    return run_suite(data, experiments, _progress)


def _cmd_check_data(args: argparse.Namespace) -> int:
    data = _load_data(args)
    required = minimum_purchase(data)
    mandatory = total_cost(data, required)
    print(f"Материалов: {data.n} (seed данных {data.seed})")
    print(f"Бюджет: {data.budget:.2f}")
    print(f"Стоимость обязательных закупок: {mandatory:.2f}")
    print(f"Остаток бюджета на дополнительные закупки: {data.budget - mandatory:.2f}")
    print(f"Обязательных единиц всего: {int(required.sum())}")
    print(f"Максимальная полезность (все x_i = u_i): "
          f"{total_utility(data, data.upper_bounds):.3f}")
    print("Данные корректны.")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    data = _load_data(args)
    print(f"Бюджет: {data.budget:.2f}")
    for example in build_plan_examples(data):
        d = example.diagnostics
        print(f"- {example.label}")
        print(f"    стоимость {d.cost:.2f}, полезность {d.utility:.3f}, "
              f"допустим: {'да' if d.feasible else 'нет'}")
        for message in d.messages:
            print(f"    нарушение: {message}")
        if example.repaired is not None:
            r = diagnose(data, example.repaired)
            print(f"    после repair: стоимость {r.cost:.2f}, полезность {r.utility:.3f}, "
                  f"допустим: {'да' if r.feasible else 'нет'}")
    return 0


def _cmd_check_operators(args: argparse.Namespace) -> int:
    data = _load_data(args)
    checked = check_operators(data)
    print(f"Проверено особей: {checked}")
    print("Целочисленность, длина и границы 0..u сохраняются после инициализации, "
          "кроссовера и обеих мутаций.")
    return 0


def _cmd_run_ga(args: argparse.Namespace) -> int:
    data = _load_data(args)
    config = replace(
        _base_ga_config(args),
        constraint_strategy=ConstraintStrategy(args.strategy),
        mutation_type=MutationType(args.mutation),
        seed=args.seed,
    )
    result = run_ga(data, config)
    print(f"Ограничения: {args.strategy}, мутация: {args.mutation}, seed: {args.seed}")
    print(f"Оценок целевой функции: {result.evaluations} (лимит {config.max_evaluations})")
    if result.found_feasible:
        print(f"Допустимый план найден в поколении {result.best_feasible_generation}")
        print(f"Полезность: {result.best_utility:.3f}, "
              f"стоимость: {result.best_cost:.2f} из бюджета {data.budget:.2f}")
    else:
        print("Допустимый план не найден")
    print(f"Лучший fitness: {result.best_fitness:.3f} "
          f"({'допустима' if result.best_fitness_feasible else 'недопустима'} особь)")
    print(f"Время: {result.elapsed_seconds:.2f} с")
    if args.show_plan and result.found_feasible:
        print()
        _print_plan(data, result.best_x)
    return 0


def _cmd_run_experiment(args: argparse.Namespace) -> int:
    data = _load_data(args)
    experiment = build_experiments(_base_ga_config(args), args.runs, args.base_seed)[args.name]
    ga = experiment.ga
    print(f"Эксперимент {args.name}: {ga.constraint_strategy.value}, "
          f"{ga.mutation_type.value}, {args.runs} запусков, seed {args.base_seed}+k")
    results = run_series(data, experiment, _progress)
    _print_stats([summarize_series(args.name, results)])
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    data = _load_data(args)
    suite = _run_suite(data, args)
    _print_stats(list(suite.stats.values()), suite.greedy_stats)
    return 0


def _cmd_run_all(args: argparse.Namespace) -> int:
    from procurement_ga.reporting import save_all, select_best_plan

    data = _load_data(args)
    suite = _run_suite(data, args)
    paths = save_all(data, suite, args.results_dir)
    _print_stats(list(suite.stats.values()), suite.greedy_stats)
    best, source = select_best_plan(suite)
    print()
    print(f"Лучший допустимый план: {source}, полезность {best.best_utility:.3f}, "
          f"стоимость {best.best_cost:.2f} из {data.budget:.2f}")
    print(f"Результаты сохранены в: {Path(args.results_dir)}")
    for path in paths.values():
        print(f"  {path.name}")
    return 0


def _data_args() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    group = parent.add_argument_group("исходные данные")
    group.add_argument("--materials", type=int, default=DEFAULT_N_MATERIALS,
                       help="число материалов, не менее 30 (по умолчанию %(default)s)")
    group.add_argument("--data-seed", type=int, default=DEFAULT_DATA_SEED,
                       help="seed генерации данных (по умолчанию %(default)s)")
    return parent


def _ga_args() -> argparse.ArgumentParser:
    defaults = GAConfig()
    parent = argparse.ArgumentParser(add_help=False)
    group = parent.add_argument_group("параметры генетического алгоритма")
    group.add_argument("--population", type=int, default=defaults.population_size,
                       help="размер популяции (по умолчанию %(default)s)")
    group.add_argument("--generations", type=int, default=defaults.max_generations,
                       help="предельное число поколений (по умолчанию %(default)s)")
    group.add_argument("--evaluations", type=int, default=defaults.max_evaluations,
                       help="лимит оценок целевой функции (по умолчанию %(default)s)")
    group.add_argument("--crossover", type=float, default=defaults.crossover_prob,
                       help="вероятность кроссовера (по умолчанию %(default)s)")
    group.add_argument("--mutation-prob", type=float, default=defaults.mutation_prob,
                       help="вероятность мутации особи (по умолчанию %(default)s)")
    group.add_argument("--elite", type=int, default=defaults.elite_size,
                       help="число элитных особей (по умолчанию %(default)s)")
    group.add_argument("--lambda-q", type=float, default=defaults.penalty_shortage,
                       help="штраф за недостаток закупки (по умолчанию %(default)s)")
    group.add_argument("--lambda-b", type=float, default=defaults.penalty_budget,
                       help="штраф за превышение бюджета (по умолчанию %(default)s)")
    return parent


def _series_args() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    group = parent.add_argument_group("серия запусков")
    group.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                       help="запусков на конфигурацию (по умолчанию %(default)s)")
    group.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED,
                       help="seed запуска k равен base-seed + k (по умолчанию %(default)s)")
    return parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="procurement_ga",
        description="Генетический алгоритм для задачи закупки расходных материалов.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="команда")
    data, ga, series = _data_args(), _ga_args(), _series_args()

    def add(name: str, help_text: str, handler, *parents) -> argparse.ArgumentParser:
        sub = subparsers.add_parser(
            name, help=help_text, description=help_text, parents=list(parents)
        )
        sub.set_defaults(handler=handler)
        return sub

    add("check-data", "проверить исходные данные", _cmd_check_data, data)
    add("demo", "показать допустимые и недопустимые планы", _cmd_demo, data)
    add("check-operators", "проверить генетические операторы", _cmd_check_operators, data)

    run = add("run-ga", "один запуск генетического алгоритма", _cmd_run_ga, data, ga)
    run.add_argument("--strategy", choices=[s.value for s in ConstraintStrategy],
                     default=ConstraintStrategy.REPAIR.value,
                     help="обработка ограничений (по умолчанию %(default)s)")
    run.add_argument("--mutation", choices=[m.value for m in MutationType],
                     default=MutationType.STEP.value,
                     help="тип мутации (по умолчанию %(default)s)")
    run.add_argument("--seed", type=int, default=DEFAULT_BASE_SEED,
                     help="seed запуска (по умолчанию %(default)s)")
    run.add_argument("--show-plan", action="store_true",
                     help="напечатать найденный план закупок")

    experiment = add("run-experiment", "одна экспериментальная конфигурация",
                     _cmd_run_experiment, data, ga, series)
    experiment.add_argument("name", choices=list(EXPERIMENT_TABLE),
                            help="конфигурация: A, B, C или D")

    add("compare", "серии A-D и жадная эвристика, без сохранения файлов",
        _cmd_compare, data, ga, series)
    run_all = add("run-all", "полная серия A-D, сравнение с эвристикой, сохранение результатов",
                  _cmd_run_all, data, ga, series)
    run_all.add_argument("--results-dir", default="results",
                         help="каталог результатов (по умолчанию %(default)s)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except ValueError as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 2
    except RuntimeError as error:
        print(f"Внутренняя ошибка проверки: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())