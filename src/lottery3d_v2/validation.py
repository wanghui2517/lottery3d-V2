# -*- coding: utf-8 -*-
"""回测验证模块"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .history import ensure_history
from .schema import Prediction
from .utils import normalize_proba


def normal_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def binomial_z_test(
    hits: int,
    trials: int,
    p0: float
) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 1.0

    phat = hits / trials
    se = math.sqrt(p0 * (1.0 - p0) / trials)

    if se <= 0:
        return 0.0, 1.0

    z = (phat - p0) / se
    p = 2.0 * normal_sf(abs(z))

    return z, p


def backtest_predictor(
    predictor,
    history,
    window: int = 100,
    periods: int = 200,
    train_window: int | None = None,
) -> dict | None:
    """单预测器回测

    Args:
        predictor: 预测器实例
        history: 历史数据列表
        window: 最小训练窗口
        periods: 回测期数
        train_window: 滚动训练窗口，None 表示使用全部历史

    Returns:
        包含各项指标的字典，样本不足返回 None
    """
    history = ensure_history(history)

    if len(history) <= window:
        return None

    start = max(window, len(history) - periods)

    records = []
    eps = 1e-12

    for target_idx in range(start, len(history)):
        end = target_idx

        if train_window is None:
            begin = 0
        else:
            begin = max(0, end - train_window)

        train = history[begin:end]

        if len(train) < window:
            continue

        pred = predictor.predict(train, top_k=3)
        actual = history[target_idx].digits

        top1_hits = 0
        top3_hits = 0

        log_loss = 0.0
        brier = 0.0

        for pos in range(3):
            candidates = pred.positions[pos]

            if candidates and candidates[0] == actual[pos]:
                top1_hits += 1

            if actual[pos] in candidates:
                top3_hits += 1

            p = normalize_proba(pred.positions_proba[pos], uniform_size=10)
            actual_digit = actual[pos]

            log_loss += -math.log(float(p[actual_digit]) + eps)
            brier += float(np.sum(p * p) - 2.0 * p[actual_digit] + 1.0)

        pair = pred.best_pair

        if pair is not None:
            pair_hit = (
                actual[pair["pos_a"]] == pair["digit_a"]
                and actual[pair["pos_b"]] == pair["digit_b"]
            )

            pair_any_hit = (
                actual[pair["pos_a"]] == pair["digit_a"]
                or actual[pair["pos_b"]] == pair["digit_b"]
            )
        else:
            pair_hit = False
            pair_any_hit = False

        records.append(
            {
                "issue": history[target_idx].issue,
                "top1_hits": top1_hits,
                "top3_hits": top3_hits,
                "pair_hit": int(pair_hit),
                "pair_any_hit": int(pair_any_hit),
                "log_loss": log_loss,
                "brier": brier,
            }
        )

    if not records:
        return None

    n = len(records)

    top1_hits = sum(r["top1_hits"] for r in records)
    top3_hits = sum(r["top3_hits"] for r in records)

    pair_hits = sum(r["pair_hit"] for r in records)
    pair_any_hits = sum(r["pair_any_hit"] for r in records)

    top1_trials = 3 * n
    top3_trials = 3 * n

    top1_rate = top1_hits / top1_trials
    top3_rate = top3_hits / top3_trials

    pair_strict_rate = pair_hits / n
    pair_any_rate = pair_any_hits / n

    mean_log_loss = float(np.mean([r["log_loss"] for r in records]))
    mean_brier = float(np.mean([r["brier"] for r in records]))

    top1_z, top1_p = binomial_z_test(top1_hits, top1_trials, 0.10)
    top3_z, top3_p = binomial_z_test(top3_hits, top3_trials, 0.30)

    pair_strict_z, pair_strict_p = binomial_z_test(pair_hits, n, 0.01)
    pair_any_z, pair_any_p = binomial_z_test(pair_any_hits, n, 0.19)

    return {
        "predictor": predictor.name,
        "periods": n,
        "top1_rate": top1_rate,
        "top3_rate": top3_rate,
        "pair_strict_rate": pair_strict_rate,
        "pair_any_rate": pair_any_rate,
        "mean_log_loss": mean_log_loss,
        "mean_brier": mean_brier,
        "top1_z": top1_z,
        "top1_p": top1_p,
        "top3_z": top3_z,
        "top3_p": top3_p,
        "pair_strict_z": pair_strict_z,
        "pair_strict_p": pair_strict_p,
        "pair_any_z": pair_any_z,
        "pair_any_p": pair_any_p,
        "records": records,
    }


def print_backtest_report(result: dict | None, title: str | None = None) -> None:
    print("=" * 90)

    if title:
        print(title)
        print("=" * 90)

    if result is None:
        print("回测样本不足。")
        return

    print(f"预测器: {result['predictor']}")
    print(f"回测期数: {result['periods']}")
    print()

    print("【命中率】")
    print(
        f"Top1: {result['top1_rate']:.4f} | 基准=0.1000 | "
        f"z={result['top1_z']:.3f} | p={result['top1_p']:.4f}"
    )
    print(
        f"Top3: {result['top3_rate']:.4f} | 基准=0.3000 | "
        f"z={result['top3_z']:.3f} | p={result['top3_p']:.4f}"
    )
    print(
        f"两位严格: {result['pair_strict_rate']:.4f} | 基准=0.0100 | "
        f"z={result['pair_strict_z']:.3f} | p={result['pair_strict_p']:.4f}"
    )
    print(
        f"两位任一: {result['pair_any_rate']:.4f} | 基准=0.1900 | "
        f"z={result['pair_any_z']:.3f} | p={result['pair_any_p']:.4f}"
    )

    print()
    print("【概率质量】")
    print(
        f"LogLoss: {result['mean_log_loss']:.4f} | "
        f"随机基准=6.9078"
    )
    print(
        f"Brier: {result['mean_brier']:.4f} | "
        f"随机基准=2.7000"
    )

    print()


def compare_predictors(
    history,
    predictors: list,
    window: int = 100,
    periods: int = 200,
    train_window: int | None = None,
) -> list[dict]:
    """批量回测多个预测器并返回排序结果"""
    results = []

    for pred in predictors:
        result = backtest_predictor(
            predictor=pred,
            history=history,
            window=window,
            periods=periods,
            train_window=train_window,
        )
        if result:
            results.append(result)

    # 按 LogLoss 排序（越小越好）
    results.sort(key=lambda r: r["mean_log_loss"])

    return results


def print_comparison_table(results: list[dict]) -> None:
    """打印对比表格"""
    print("=" * 120)
    print(f"{'预测器':<15} {'期数':>4} {'Top1':>8} {'Top3':>8} {'两位严格':>10} {'两位任一':>10} {'LogLoss':>10} {'Brier':>8}")
    print("-" * 120)

    for r in results:
        print(
            f"{r['predictor']:<15} {r['periods']:>4} "
            f"{r['top1_rate']:>8.4f} {r['top3_rate']:>8.4f} "
            f"{r['pair_strict_rate']:>10.4f} {r['pair_any_rate']:>10.4f} "
            f"{r['mean_log_loss']:>10.4f} {r['mean_brier']:>8.4f}"
        )

    print("=" * 120)