# -*- coding: utf-8 -*-
"""核心数学工具函数"""
from __future__ import annotations

from functools import lru_cache
from itertools import product
import math

import numpy as np
from loguru import logger


DIGIT_N = 10
SUM_N = 28


# 基准值（随机猜测）
BASELINE = {
    "logloss_per_position": 2.302585,  # log(10)
    "logloss_total": 6.907755,         # 3 * log(10)
    "brier_per_position": 0.9,
    "brier_total": 2.7,
}


def normalize_proba(
    prob: np.ndarray,
    uniform_size: int | None = None,
    eps: float = 1e-12
) -> np.ndarray:
    """概率归一化，处理 NaN/Inf/负值，支持均匀分布回退"""
    prob = np.asarray(prob, dtype=float)
    prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
    prob = np.clip(prob, 0.0, None)

    total = prob.sum()

    if total <= eps:
        n = uniform_size if uniform_size is not None else prob.size
        return np.full(n, 1.0 / n, dtype=float)

    return prob / total


def softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax with temperature"""
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    if temperature <= 0:
        temperature = 1.0

    z = (values - values.max()) / temperature
    prob = np.exp(z)

    return normalize_proba(prob)


def neighbor_proba(
    center: int,
    n_states: int,
    wrap: bool = False,
    center_weight: float = 0.5,
    side_weight: float = 0.25,
) -> np.ndarray:
    """邻域概率分布"""
    center = int(center)

    if center < 0:
        center = 0
    if center >= n_states:
        center = n_states - 1

    prob = np.zeros(n_states, dtype=float)
    prob[center] += center_weight

    if center > 0:
        prob[center - 1] += side_weight
    elif wrap:
        prob[-1] += side_weight
    else:
        prob[center] += side_weight

    if center < n_states - 1:
        prob[center + 1] += side_weight
    elif wrap:
        prob[0] += side_weight
    else:
        prob[center] += side_weight

    return normalize_proba(prob)


def top_k_from_proba(prob: np.ndarray, k: int = 3) -> list[int]:
    """从概率分布中取 Top-K"""
    prob = normalize_proba(prob)
    idx = np.argsort(prob)[::-1][:k]
    result = [int(x) for x in idx]

    for i in range(prob.size):
        if len(result) >= k:
            break
        if i not in result:
            result.append(i)

    return result[:k]


@lru_cache(maxsize=128)
def _ewma_cached(values_tuple: tuple, alpha: float) -> float:
    """EWMA 缓存实现"""
    values = np.array(values_tuple, dtype=float)
    if values.size == 0:
        return 0.0
    state = float(values[0])
    for x in values[1:]:
        state = alpha * float(x) + (1.0 - alpha) * state
    return float(state)


def ewma(values: np.ndarray, alpha: float = 0.3) -> float:
    """指数加权移动平均（带缓存）"""
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return 0.0
    
    # 对于短序列直接使用向量化计算
    if values.size <= 50:
        weights = alpha * np.power(1 - alpha, np.arange(values.size - 1, -1, -1))
        weights /= weights.sum()
        return float(np.dot(values, weights))
    
    # 长序列使用缓存版本
    try:
        return _ewma_cached(tuple(values[-50:]), alpha)
    except TypeError:
        # 如果无法哈希，回退到原始实现
        state = float(values[0])
        for x in values[1:]:
            state = alpha * float(x) + (1.0 - alpha) * state
        return float(state)


def categorical_recent_proba(
    values: np.ndarray,
    n_states: int,
    window: int | None = 100,
    decay: float = 0.03,
    smooth: float = 1.0,
) -> np.ndarray:
    """加权近期频率概率"""
    values = np.asarray(values, dtype=int)

    if window is not None and values.size > window:
        values = values[-window:]

    values = values[(values >= 0) & (values < n_states)]

    counts = np.full(n_states, float(smooth), dtype=float)
    n = values.size

    if n > 0:
        ages = np.arange(n - 1, -1, -1, dtype=float)
        weights = np.exp(-decay * ages)
        np.add.at(counts, values, weights)

    return normalize_proba(counts)


def markov_proba(
    states: np.ndarray,
    last_state: int,
    n_states: int,
    smooth: float = 1.0,
) -> np.ndarray:
    """一阶马尔可夫转移概率"""
    states = np.asarray(states, dtype=int)
    states = states[(states >= 0) & (states < n_states)]

    trans = np.full((n_states, n_states), float(smooth), dtype=float)

    if states.size >= 2:
        prev = states[:-1]
        curr = states[1:]
        np.add.at(trans, (prev, curr), 1.0)

    last_state = int(last_state)
    if last_state < 0 or last_state >= n_states:
        last_state = 0

    return normalize_proba(trans[last_state])


def state_omission_stats(
    states: np.ndarray,
    n_states: int
) -> tuple[np.ndarray, np.ndarray]:
    """遗漏统计：返回 (z_scores, current_omissions)"""
    states = np.asarray(states, dtype=int)
    n = states.size

    current = np.full(n_states, float(n), dtype=float)
    last_seen = np.full(n_states, -1, dtype=int)
    gaps = [[] for _ in range(n_states)]

    for i, s in enumerate(states):
        if 0 <= s < n_states:
            if last_seen[s] != -1:
                gaps[s].append(i - last_seen[s])
            last_seen[s] = i

    z = np.zeros(n_states, dtype=float)

    for s in range(n_states):
        if last_seen[s] != -1:
            current[s] = float(n - 1 - last_seen[s])

        gap_arr = np.array(gaps[s], dtype=float)

        if gap_arr.size > 0:
            mean = float(gap_arr.mean())
            std = float(gap_arr.std())
        else:
            mean = float(n)
            std = 1.0

        if std < 1e-9:
            std = 1.0

        z[s] = (current[s] - mean) / std

    return z, current


def combine_categorical_and_neighbor(
    values: np.ndarray,
    n_states: int,
    window: int | None,
    decay: float,
    center: int,
    wrap: bool = False,
    categorical_weight: float = 0.75,
    neighbor_weight: float = 0.25,
) -> np.ndarray:
    """分类频率 + 邻域平滑混合"""
    cat_prob = categorical_recent_proba(
        values,
        n_states=n_states,
        window=window,
        decay=decay,
        smooth=1.0,
    )

    try:
        center_int = int(round(float(center)))
    except (ValueError, TypeError) as e:
        logger.debug(f"Failed to convert center {center!r} to int: {e}")
        return cat_prob

    neigh_prob = neighbor_proba(
        center=center_int,
        n_states=n_states,
        wrap=wrap,
        center_weight=0.5,
        side_weight=0.25,
    )

    return normalize_proba(
        categorical_weight * cat_prob
        + neighbor_weight * neigh_prob
    )


@lru_cache(maxsize=1)
def sum_theory_prior() -> np.ndarray:
    """和值理论先验（三位独立均匀分布）"""
    counts = np.zeros(SUM_N, dtype=float)

    for a, b, c in product(range(10), repeat=3):
        counts[a + b + c] += 1.0

    return normalize_proba(counts)


def sum_proba_mixture(
    sums: np.ndarray,
    window: int | None = 100,
    decay: float = 0.03,
    theory_weight: float = 0.25,
    recent_weight: float = 0.55,
    neighbor_weight: float = 0.20,
    center: int | None = None,
) -> np.ndarray:
    """和值概率混合：理论先验 + 近期频率 + 邻域平滑"""
    theory = sum_theory_prior()

    sums = np.asarray(sums, dtype=int)

    if sums.size == 0:
        return theory.copy()

    recent = categorical_recent_proba(
        sums,
        n_states=SUM_N,
        window=window,
        decay=decay,
        smooth=0.5,
    )

    if center is None:
        center = int(round(ewma(sums[-window:] if window else sums, alpha=decay)))

    center = int(np.clip(center, 0, SUM_N - 1))

    neigh = neighbor_proba(
        center=center,
        n_states=SUM_N,
        wrap=False,
        center_weight=0.5,
        side_weight=0.25,
    )

    return normalize_proba(
        recent_weight * recent
        + neighbor_weight * neigh
        + theory_weight * theory
    )


def uniform_proba_result() -> dict:
    """均匀分布回退结果"""
    return {
        "positions_proba": [
            np.full(DIGIT_N, 1.0 / DIGIT_N, dtype=float)
            for _ in range(3)
        ],
        "sum_proba": np.full(SUM_N, 1.0 / SUM_N, dtype=float),
        "meta": {},
    }


def temperature_scale(
    prob: np.ndarray,
    temperature: float | None = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """温度缩放"""
    prob = normalize_proba(prob, eps=eps)

    if temperature is None or abs(float(temperature) - 1.0) < 1e-12:
        return prob

    temperature = float(temperature)
    if temperature <= 0:
        temperature = 1.0

    z = np.log(np.clip(prob, eps, None)) / temperature
    z = z - z.max()
    exp_z = np.exp(z)

    return normalize_proba(exp_z, eps=eps)


def temperature_grid() -> np.ndarray:
    """默认温度搜索网格"""
    return np.unique(
        np.concatenate([
            np.arange(0.10, 2.00, 0.05),
            np.arange(2.00, 5.00, 0.10),
            np.arange(5.00, 20.01, 0.50),
        ])
    )


def find_temperature(
    records: list[dict],
    grid: np.ndarray | None = None,
    eps: float = 1e-12,
) -> float:
    """在验证集上寻找最优温度"""
    if not records:
        return 1.0

    if grid is None:
        grid = temperature_grid()

    best_temperature = 1.0
    best_loss = mean_logloss(records, temperature=1.0, eps=eps)

    for t in grid:
        t = float(t)
        loss = mean_logloss(records, temperature=t, eps=eps)

        if loss < best_loss - 1e-12:
            best_loss = loss
            best_temperature = t

    return best_temperature


def find_position_temperatures(
    records: list[dict],
    grid: np.ndarray | None = None,
    eps: float = 1e-12,
) -> dict[int, float]:
    """为每个位置单独寻找最优温度"""
    result = {}

    for pos in range(3):
        pos_records = [r for r in records if r["pos"] == pos]
        result[pos] = find_temperature(pos_records, grid=grid, eps=eps)

    return result


def mean_logloss(
    records: list[dict],
    temperature: float | None = 1.0,
    position_temperatures: dict[int, float] | None = None,
    eps: float = 1e-12,
) -> float:
    """计算平均 LogLoss"""
    if not records:
        return float("nan")

    total = 0.0
    for record in records:
        if position_temperatures is not None:
            t = position_temperatures[record["pos"]]
        else:
            t = temperature

        q = temperature_scale(record["proba"], t, eps=eps)
        total += -math.log(float(q[record["actual"]]) + eps)

    return total / len(records)


def mean_brier(
    records: list[dict],
    temperature: float | None = 1.0,
    position_temperatures: dict[int, float] | None = None,
    eps: float = 1e-12,
) -> float:
    """计算平均 Brier Score"""
    if not records:
        return float("nan")

    total = 0.0
    for record in records:
        if position_temperatures is not None:
            t = position_temperatures[record["pos"]]
        else:
            t = temperature

        q = temperature_scale(record["proba"], t, eps=eps)

        onehot = np.zeros(10, dtype=float)
        onehot[record["actual"]] = 1.0

        total += float(np.sum((q - onehot) ** 2))

    return total / len(records)


def group_by_target(records: list[dict]) -> tuple[dict[int, list[dict]], list[int]]:
    """按目标期分组记录"""
    from collections import defaultdict
    by_target: dict[int, list[dict]] = defaultdict(list)

    for record in records:
        by_target[record["target_idx"]].append(record)

    targets = sorted(by_target.keys())

    return by_target, targets


def topk_accuracy(records: list[dict], k: int) -> float:
    """Top-K 准确率"""
    if not records:
        return float("nan")

    hits = 0
    for record in records:
        top = np.argsort(record["proba"])[::-1][:k]
        if record["actual"] in top:
            hits += 1

    return hits / len(records)


def evaluate_records(
    records: list[dict],
    temperature: float | None = 1.0,
    position_temperatures: dict[int, float] | None = None,
) -> dict:
    """综合评估"""
    ll = mean_logloss(
        records,
        temperature=temperature,
        position_temperatures=position_temperatures,
    )

    br = mean_brier(
        records,
        temperature=temperature,
        position_temperatures=position_temperatures,
    )

    return {
        "records": len(records),
        "periods": len(set(r["target_idx"] for r in records)),
        "top1": topk_accuracy(records, 1),
        "top3": topk_accuracy(records, 3),
        "logloss_per_position": ll,
        "logloss_total": 3.0 * ll,
        "brier_per_position": br,
        "brier_total": 3.0 * br,
    }