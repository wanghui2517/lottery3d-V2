# -*- coding: utf-8 -*-
"""温度校准模块：Holdout、Rolling 校准，支持位置特定温度"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .utils import (
    BASELINE,
    evaluate_records,
    find_position_temperatures,
    find_temperature,
    group_by_target,
    mean_brier,
    mean_logloss,
    normalize_proba,
    temperature_grid,
    temperature_scale,
)


def collect_calibration_records(
    history,
    model,
    periods: int,
    min_train: int,
    train_window: int | None,
) -> list[dict]:
    """收集校准用的概率记录"""
    start = max(min_train, len(history) - periods)

    records = []

    for target_idx in range(start, len(history)):
        end = target_idx

        if train_window is None:
            begin = 0
        else:
            begin = max(0, end - train_window)

        train = history[begin:end]

        if len(train) < min_train:
            continue

        proba = model.predict_proba(train)
        actual = history[target_idx].digits

        for pos in range(3):
            p = normalize_proba(proba.positions_proba[pos])

            records.append(
                {
                    "target_idx": int(target_idx),
                    "pos": int(pos),
                    "proba": p,
                    "actual": int(actual[pos]),
                }
            )

    return records


def holdout_calibrate(
    records: list[dict],
    calib_ratio: float = 0.5,
    position_specific: bool = True,
) -> dict[str, Any]:
    """Holdout 校准：前半段找温度，后半段验证"""
    by_target, targets = group_by_target(records)

    split = int(len(targets) * calib_ratio)

    calib_targets = targets[:split]
    test_targets = targets[split:]

    calib_records = [
        record
        for target in calib_targets
        for record in by_target[target]
    ]

    test_records = [
        record
        for target in test_targets
        for record in by_target[target]
    ]

    if position_specific:
        position_temperatures = find_position_temperatures(calib_records)
        global_temperature = None
    else:
        global_temperature = find_temperature(calib_records)
        position_temperatures = None

    result = {
        "mode": "holdout",
        "position_specific": position_specific,
        "calib_periods": len(calib_targets),
        "test_periods": len(test_targets),
        "global_temperature": global_temperature,
        "position_temperatures": position_temperatures,
        "baseline": BASELINE,
        "records": len(calib_records) + len(test_records),
        "calib": {
            "uncalibrated": evaluate_records(calib_records, temperature=1.0),
            "calibrated": evaluate_records(
                calib_records,
                temperature=global_temperature,
                position_temperatures=position_temperatures,
            ),
        },
        "test": {
            "uncalibrated": evaluate_records(test_records, temperature=1.0),
            "calibrated": evaluate_records(
                test_records,
                temperature=global_temperature,
                position_temperatures=position_temperatures,
            ),
        },
    }

    return result


def rolling_calibrate(
    records: list[dict],
    warmup_periods: int = 50,
    window_periods: int = 100,
    position_specific: bool = True,
) -> dict[str, Any]:
    """滚动校准：每期用前 window 期数据找温度，预测当期"""
    by_target, targets = group_by_target(records)

    if len(targets) <= warmup_periods:
        return {
            "mode": "rolling",
            "error": "期数不足，无法进行滚动校准。",
        }

    uncal_logloss = []
    cal_logloss = []

    uncal_brier = []
    cal_brier = []

    top1_hits = 0
    top3_hits = 0
    n = 0

    temperature_log = []

    for k in range(warmup_periods, len(targets)):
        current_target = targets[k]

        start = max(0, k - window_periods)
        calib_targets = targets[start:k]

        calib_records = [
            record
            for target in calib_targets
            for record in by_target[target]
        ]

        if position_specific:
            position_temperatures = find_position_temperatures(calib_records)
            global_temperature = None
            temperature_log.append(position_temperatures)
        else:
            global_temperature = find_temperature(calib_records)
            position_temperatures = None
            temperature_log.append(global_temperature)

        for record in by_target[current_target]:
            if position_temperatures is not None:
                t = position_temperatures[record["pos"]]
            else:
                t = global_temperature

            p = normalize_proba(record["proba"])
            q = temperature_scale(p, t)

            actual = record["actual"]

            uncal_logloss.append(
                -np.log(float(p[actual]) + 1e-12)
            )

            cal_logloss.append(
                -np.log(float(q[actual]) + 1e-12)
            )

            onehot = np.zeros(10, dtype=float)
            onehot[actual] = 1.0

            uncal_brier.append(float(np.sum((p - onehot) ** 2)))
            cal_brier.append(float(np.sum((q - onehot) ** 2)))

            top = np.argsort(p)[::-1]

            if actual == top[0]:
                top1_hits += 1

            if actual in top[:3]:
                top3_hits += 1

            n += 1

    if position_specific:
        temperature_summary = {}

        for pos in range(3):
            values = np.array(
                [item[pos] for item in temperature_log],
                dtype=float,
            )

            temperature_summary[str(pos)] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
    else:
        values = np.array(temperature_log, dtype=float)

        temperature_summary = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    uncal_ll = float(np.mean(uncal_logloss))
    cal_ll = float(np.mean(cal_logloss))

    uncal_br = float(np.mean(uncal_brier))
    cal_br = float(np.mean(cal_brier))

    return {
        "mode": "rolling",
        "position_specific": position_specific,
        "warmup_periods": warmup_periods,
        "window_periods": window_periods,
        "eval_periods": len(targets) - warmup_periods,
        "records": n,
        "top1": top1_hits / n if n else float("nan"),
        "top3": top3_hits / n if n else float("nan"),
        "baseline": BASELINE,
        "global_temperature": None if position_specific else global_temperature,
        "position_temperatures": position_temperatures if position_specific else None,
        "uncalibrated": {
            "logloss_per_position": uncal_ll,
            "logloss_total": 3.0 * uncal_ll,
            "brier_per_position": uncal_br,
            "brier_total": 3.0 * uncal_br,
        },
        "calibrated": {
            "logloss_per_position": cal_ll,
            "logloss_total": 3.0 * cal_ll,
            "brier_per_position": cal_br,
            "brier_total": 3.0 * cal_br,
        },
        "temperature_summary": temperature_summary,
    }


def print_calibration_report(result: dict[str, Any], title: str | None = None) -> None:
    print("=" * 90)

    if title:
        print(title)
        print("=" * 90)

    if "error" in result:
        print(result["error"])
        return

    print(f"模式: {result['mode']}")
    print(f"位置特定: {result['position_specific']}")

    if result["mode"] == "holdout":
        print(f"校准期数: {result['calib_periods']}")
        print(f"测试期数: {result['test_periods']}")
    else:
        print(f"预热期数: {result['warmup_periods']}")
        print(f"滚动窗口: {result['window_periods']}")
        print(f"评估期数: {result['eval_periods']}")

    print(f"记录数: {result['records']}")

    if result.get("global_temperature") is not None:
        print(f"全局温度: {result['global_temperature']:.4f}")
    elif result.get("position_temperatures"):
        print("位置温度:")
        for pos, temp in result["position_temperatures"].items():
            print(f"  位置{pos}: {temp:.4f}")

    print()
    print("【校准集/评估集】")
    print("  未校准:", end=" ")
    u = result.get("calib", {}).get("uncalibrated", result.get("uncalibrated", {}))
    if u:
        print(f"LL={u.get('logloss_per_position', float('nan')):.4f} Brier={u.get('brier_per_position', float('nan')):.4f}")
    else:
        print("N/A")
    print("  已校准:", end=" ")
    c = result.get("calib", {}).get("calibrated", result.get("calibrated", {}))
    if c:
        print(f"LL={c.get('logloss_per_position', float('nan')):.4f} Brier={c.get('brier_per_position', float('nan')):.4f}")
    else:
        print("N/A")

    if "temperature_summary" in result:
        print()
        print("【温度统计】")
        ts = result["temperature_summary"]
        if isinstance(list(ts.values())[0], dict):
            for pos, stats in ts.items():
                print(f"  位置{pos}: 均值={stats['mean']:.3f} 中位数={stats['median']:.3f} "
                      f"范围=[{stats['min']:.3f}, {stats['max']:.3f}]")
        else:
            print(f"  全局: 均值={ts['mean']:.3f} 中位数={ts['median']:.3f} "
                  f"范围=[{ts['min']:.3f}, {ts['max']:.3f}]")

    print()


def save_calibration_result(
    result: dict[str, Any],
    path: str,
    model_name: str = "ensemble",
) -> None:
    """保存校准结果到 JSON"""
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 转换 numpy 类型
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(x) for x in obj]
        return obj

    result = convert(result)
    result["model_name"] = model_name

    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def load_calibration_result(path: str) -> dict[str, Any]:
    """加载校准结果"""
    import json

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def apply_calibration(
    proba_result,
    calibration: dict[str, Any],
) -> Any:
    """将校准温度应用到预测结果"""
    from .schema import ProbaResult

    position_temperatures = calibration.get("position_temperatures")
    global_temperature = calibration.get("global_temperature")

    if position_temperatures is None and global_temperature is None:
        return proba_result

    calibrated_positions = []
    for pos in range(3):
        p = normalize_proba(proba_result.positions_proba[pos])
        if position_temperatures is not None:
            t = position_temperatures.get(str(pos), position_temperatures.get(pos, 1.0))
        else:
            t = global_temperature
        q = temperature_scale(p, t)
        calibrated_positions.append(q)

    sum_proba = normalize_proba(proba_result.sum_proba)
    # 和值暂不校准（可扩展）

    return ProbaResult(
        positions_proba=calibrated_positions,
        sum_proba=sum_proba,
        meta=proba_result.meta,
    )