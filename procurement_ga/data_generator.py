"""Воспроизводимая генерация и проверка синтетических исходных данных."""

import math
from dataclasses import replace

import numpy as np

from procurement_ga.models import Material, ProcurementData
from procurement_ga.objective import minimum_purchase, total_cost

MIN_MATERIALS = 30
DEFAULT_N_MATERIALS = 30
DEFAULT_DATA_SEED = 2026
# Доля стоимости полного насыщения резервов, которую бюджет покрывает сверх обязательных закупок.
DEFAULT_BUDGET_EXTRA_SHARE = 0.4

_MATERIAL_NAMES = (
    "Бумага офисная А4", "Картридж лазерный", "Тонер-порошок", "Ручки шариковые",
    "Карандаши простые", "Маркеры для доски", "Стикеры для заметок",
    "Папки-скоросшиватели", "Файлы-вкладыши", "Степлерные скобы",
    "Скрепки канцелярские", "Клейкая лента", "Клей-карандаш", "Блокноты",
    "Тетради", "Перчатки нитриловые", "Маски медицинские", "Антисептик для рук",
    "Салфетки бумажные", "Бумажные полотенца", "Туалетная бумага",
    "Мешки для мусора", "Моющее средство", "Чистящий порошок",
    "Губки для посуды", "Батарейки AA", "Батарейки AAA", "Лампы светодиодные",
    "Удлинители", "Кабели USB", "Флеш-накопители", "Одноразовые стаканы",
    "Кофе в зёрнах", "Чай в пакетиках", "Сахар порционный", "Вода питьевая 19 л",
    "Фильтры для воды", "Этикетки самоклеящиеся", "Упаковочные коробки",
    "Скотч двусторонний",
)


def _material_name(index: int) -> str:
    if index < len(_MATERIAL_NAMES):
        return _MATERIAL_NAMES[index]
    return f"Материал {index + 1}"


def generate_data(
    n_materials: int = DEFAULT_N_MATERIALS,
    seed: int = DEFAULT_DATA_SEED,
    budget_extra_share: float = DEFAULT_BUDGET_EXTRA_SHARE,
) -> ProcurementData:
    """Генерирует набор данных; при одинаковых аргументах результат одинаков."""
    if n_materials < MIN_MATERIALS:
        raise ValueError(
            f"Нужно не менее {MIN_MATERIALS} материалов, запрошено {n_materials}"
        )
    if not 0.0 < budget_extra_share <= 1.0:
        raise ValueError("budget_extra_share должна быть в диапазоне (0, 1]")

    rng = np.random.default_rng(seed)
    n = n_materials
    prices = np.round(rng.uniform(5.0, 200.0, n), 2)
    demands = rng.integers(20, 201, n)
    stocks = rng.integers(0, np.floor(1.5 * demands).astype(np.int64) + 1)
    min_remainings = rng.integers(demands // 10, demands * 2 // 5 + 1)
    weights = np.round(rng.uniform(1.0, 10.0, n), 2)
    reserve_caps = rng.integers(3, 21, n)
    extra_room = rng.integers(1, 2 * reserve_caps + 1)  # u_i = q_i + extra_room_i

    # Сначала собираем материалы без верхних границ, чтобы q_i посчитал objective.py.
    preliminary = ProcurementData(
        materials=tuple(
            Material(
                id=i + 1,
                name=_material_name(i),
                price=float(prices[i]),
                stock=int(stocks[i]),
                demand=int(demands[i]),
                min_remaining=int(min_remainings[i]),
                weight=float(weights[i]),
                reserve_cap=int(reserve_caps[i]),
                max_purchase=0,
            )
            for i in range(n)
        ),
        budget=1.0,
    )
    required = minimum_purchase(preliminary)
    upper_bounds = required + extra_room
    materials = tuple(
        replace(material, max_purchase=int(upper))
        for material, upper in zip(preliminary.materials, upper_bounds)
    )

    mandatory = total_cost(preliminary, required)
    saturated = required + np.minimum(reserve_caps, extra_room)
    saturation_cost = total_cost(preliminary, saturated)
    raw_budget = mandatory + budget_extra_share * (saturation_cost - mandatory)
    budget = math.ceil(raw_budget * 100) / 100

    data = ProcurementData(materials=materials, budget=budget, seed=seed)
    validate_data(data)
    return data


def validate_data(data: ProcurementData) -> None:
    """Проверяет, что обязательные закупки возможны и укладываются в бюджет."""
    required = minimum_purchase(data)
    too_small = [
        material.id
        for material, q in zip(data.materials, required)
        if material.max_purchase < q
    ]
    if too_small:
        raise ValueError(
            "Верхняя граница u меньше обязательной закупки q "
            f"для материалов с id: {too_small}"
        )
    mandatory = total_cost(data, required)
    if mandatory > data.budget:
        raise ValueError(
            f"Бюджет {data.budget:.2f} не покрывает обязательные закупки "
            f"стоимостью {mandatory:.2f}: допустимых планов не существует"
        )