# -*- coding: utf-8 -*-
"""特征工程：形态学组合生成与概率混合"""
from __future__ import annotations

from functools import lru_cache
from itertools import product

import numpy as np

from .utils import DIGIT_N, normalize_proba


PRIME_DIGITS = frozenset({2, 3, 5, 7})

ROAD_DIGITS = (
    (0, 3, 6, 9),
    (1, 4, 7),
    (2, 5, 8),
)


@lru_cache(maxsize=1)
def all_digit_combinations() -> tuple[tuple[int, int, int], ...]:
    return tuple(product(range(10), repeat=3))


def ac_value(combo: tuple[int, int, int]) -> int:
    """计算 AC 值（复式差值个数）"""
    a, b, c = combo
    diffs = {abs(a - b), abs(a - c), abs(b - c)}
    diffs.discard(0)
    return len(diffs)


@lru_cache(maxsize=None)
def combos_by_parity(odd_count: int) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        combo
        for combo in all_digit_combinations()
        if sum(d % 2 for d in combo) == odd_count
    )


@lru_cache(maxsize=None)
def combos_by_prime(prime_count: int) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        combo
        for combo in all_digit_combinations()
        if sum(1 for d in combo if d in PRIME_DIGITS) == prime_count
    )


@lru_cache(maxsize=None)
def combos_by_span(span: int) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        combo
        for combo in all_digit_combinations()
        if max(combo) - min(combo) == span
    )


@lru_cache(maxsize=None)
def combos_by_ac(ac: int) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        combo
        for combo in all_digit_combinations()
        if ac_value(combo) == ac
    )


@lru_cache(maxsize=None)
def combos_by_road_pattern(
    r0: int,
    r1: int,
    r2: int
) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        product(
            ROAD_DIGITS[r0],
            ROAD_DIGITS[r1],
            ROAD_DIGITS[r2],
        )
    )


def position_proba_from_combos(
    combos: tuple[tuple[int, int, int], ...],
    n_states: int = DIGIT_N,
) -> list[np.ndarray]:
    """从组合列表计算各位置概率分布（拉普拉斯平滑）"""
    counts = [np.ones(n_states, dtype=float) for _ in range(3)]

    for combo in combos:
        for pos, digit in enumerate(combo):
            counts[pos][digit] += 1.0

    return [normalize_proba(x) for x in counts]


def mixture_by_category(
    category_proba: np.ndarray,
    combos_func,
    n_states: int = DIGIT_N,
) -> list[np.ndarray]:
    """按类别概率混合组合生成的位置概率"""
    category_proba = normalize_proba(category_proba)

    pos_counts = [np.ones(n_states, dtype=float) for _ in range(3)]

    for cat, prob in enumerate(category_proba):
        if prob <= 1e-12:
            continue

        combos = combos_func(int(cat))

        if not combos:
            continue

        freqs = position_proba_from_combos(combos, n_states=n_states)

        for pos in range(3):
            pos_counts[pos] += float(prob) * freqs[pos]

    return [normalize_proba(x) for x in pos_counts]