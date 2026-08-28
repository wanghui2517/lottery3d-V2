# -*- coding: utf-8 -*-
"""随机对照检验命令实现"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from ...base import build_predictor
from ...history import load_history_from_csv
from ...validation import backtest_predictor

if TYPE_CHECKING:
    import argparse


def randomized_test(
    predictor,
    history,
    trials: int = 100,
    window: int = 100,
    periods: int = 200,
    train_window: int | None = None,
    seed: int | None = None,
) -> dict:
    """随机对照检验：打乱历史数据顺序，验证预测器是否真的学到了规律
    
    Args:
        predictor: 预测器实例
        history: 历史数据列表
        trials: 随机试验次数
        window: 最小训练窗口
        periods: 回测期数
        train_window: 滚动训练窗口
        seed: 随机种子
        
    Returns:
        包含统计结果的字典
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # 真实数据回测
    real_result = backtest_predictor(
        predictor=predictor,
        history=history,
        window=window,
        periods=periods,
        train_window=train_window,
    )
    
    if real_result is None:
        return {
            "error": "样本不足，无法进行随机对照检验",
            "trials": 0,
        }
    
    real_metrics = {
        "top1_rate": real_result["top1_rate"],
        "top3_rate": real_result["top3_rate"],
        "mean_log_loss": real_result["mean_log_loss"],
        "mean_brier": real_result["mean_brier"],
    }
    
    # 随机试验
    random_top1_rates = []
    random_top3_rates = []
    random_loglosses = []
    random_briers = []
    
    for trial in range(trials):
        # 打乱历史数据（保持 issue 不变，只打乱 digits）
        shuffled_history = history.copy()
        digits_list = [row.digits for row in shuffled_history]
        random.shuffle(digits_list)
        
        # 创建新的 HistoryRow 对象
        from ...schema import HistoryRow
        shuffled_rows = []
        for i, row in enumerate(shuffled_history):
            shuffled_rows.append(
                HistoryRow(
                    issue=row.issue,
                    digits=digits_list[i],
                )
            )
        
        # 在打乱的数据上回测
        random_result = backtest_predictor(
            predictor=predictor,
            history=shuffled_rows,
            window=window,
            periods=periods,
            train_window=train_window,
        )
        
        if random_result:
            random_top1_rates.append(random_result["top1_rate"])
            random_top3_rates.append(random_result["top3_rate"])
            random_loglosses.append(random_result["mean_log_loss"])
            random_briers.append(random_result["mean_brier"])
    
    # 计算统计量
    def compute_stats(values: list) -> dict:
        if not values:
            return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
        arr = np.array(values)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "percentile_95": float(np.percentile(arr, 95)) if len(arr) > 1 else float("nan"),
        }
    
    # 计算 p 值（单侧检验：真实值是否显著优于随机）
    def compute_p_value(real_value: float, random_values: list, higher_is_better: bool = True) -> float:
        if not random_values:
            return float("nan")
        count = sum(
            1 if higher_is_better else -1
            for rv in random_values
            if (rv >= real_value if higher_is_better else rv <= real_value)
        )
        return (count + 1) / (len(random_values) + 1)
    
    return {
        "predictor": predictor.name,
        "trials": trials,
        "real": real_metrics,
        "random_top1": compute_stats(random_top1_rates),
        "random_top3": compute_stats(random_top3_rates),
        "random_logloss": compute_stats(random_loglosses),
        "random_brier": compute_stats(random_briers),
        "p_values": {
            "top1": compute_p_value(real_metrics["top1_rate"], random_top1_rates, higher_is_better=True),
            "top3": compute_p_value(real_metrics["top3_rate"], random_top3_rates, higher_is_better=True),
            "logloss": compute_p_value(real_metrics["mean_log_loss"], random_loglosses, higher_is_better=False),
            "brier": compute_p_value(real_metrics["mean_brier"], random_briers, higher_is_better=False),
        },
        "interpretation": interpret_randomized_test(real_metrics, compute_stats(random_top1_rates), compute_stats(random_top3_rates)),
    }


def interpret_randomized_test(real: dict, top1_stats: dict, top3_stats: dict) -> str:
    """解释随机对照检验结果"""
    if any(not np.isfinite(v) for v in [real["top1_rate"], real["top3_rate"]]):
        return "无法解释：数据不足"
    
    top1_mean = top1_stats.get("mean", 0.1)
    top3_mean = top3_stats.get("mean", 0.3)
    
    if real["top1_rate"] > top1_mean * 1.2 and real["top3_rate"] > top3_mean * 1.1:
        return "✓ 预测器表现显著优于随机，可能捕捉到了真实规律"
    elif real["top1_rate"] > top1_mean * 1.1 or real["top3_rate"] > top3_mean * 1.05:
        return "△ 预测器略有优势，但需要更多数据验证"
    else:
        return "✗ 预测器与随机无显著差异，建议重新审视模型"


def print_randomized_report(result: dict, title: str | None = None) -> None:
    """打印随机对照检验报告"""
    print("=" * 90)
    
    if title:
        print(title)
        print("=" * 90)
    
    if "error" in result:
        print(result["error"])
        return
    
    print(f"预测器：{result['predictor']}")
    print(f"随机试验次数：{result['trials']}")
    print()
    
    print("【真实数据表现】")
    real = result["real"]
    print(f"  Top1 命中率：{real['top1_rate']:.4f}")
    print(f"  Top3 命中率：{real['top3_rate']:.4f}")
    print(f"  LogLoss:      {real['mean_log_loss']:.4f}")
    print(f"  Brier Score:  {real['mean_brier']:.4f}")
    print()
    
    print("【随机数据表现（均值±标准差）】")
    top1 = result["random_top1"]
    top3 = result["random_top3"]
    logloss = result["random_logloss"]
    brier = result["random_brier"]
    
    print(f"  Top1 命中率：{top1['mean']:.4f} ± {top1['std']:.4f}  [{top1['min']:.4f}, {top1['max']:.4f}]")
    print(f"  Top3 命中率：{top3['mean']:.4f} ± {top3['std']:.4f}  [{top3['min']:.4f}, {top3['max']:.4f}]")
    print(f"  LogLoss:      {logloss['mean']:.4f} ± {logloss['std']:.4f}")
    print(f"  Brier Score:  {brier['mean']:.4f} ± {brier['std']:.4f}")
    print()
    
    print("【显著性检验（p 值）】")
    pvals = result["p_values"]
    print(f"  Top1 p-value:  {pvals['top1']:.4f}  {'***' if pvals['top1'] < 0.01 else '**' if pvals['top1'] < 0.05 else '*' if pvals['top1'] < 0.1 else ''}")
    print(f"  Top3 p-value:  {pvals['top3']:.4f}  {'***' if pvals['top3'] < 0.01 else '**' if pvals['top3'] < 0.05 else '*' if pvals['top3'] < 0.1 else ''}")
    print(f"  LogLoss p-val: {pvals['logloss']:.4f}  {'***' if pvals['logloss'] < 0.01 else '**' if pvals['logloss'] < 0.05 else '*' if pvals['logloss'] < 0.1 else ''}")
    print(f"  Brier p-val:   {pvals['brier']:.4f}  {'***' if pvals['brier'] < 0.01 else '**' if pvals['brier'] < 0.05 else '*' if pvals['brier'] < 0.1 else ''}")
    print()
    
    print("【结论】")
    print(f"  {result['interpretation']}")
    print()


def randomize_cmd(args: argparse.Namespace) -> int:
    """随机对照检验命令
    
    Args:
        args: CLI 参数
        
    Returns:
        int: 退出码 (0=成功，1=失败)
    """
    try:
        logger.info(f"Loading history from {args.csv}")
        history = load_history_from_csv(args.csv, ascending=True)

        if not history:
            logger.error("历史数据为空")
            print("历史数据为空，无法进行随机对照检验。", file=sys.stderr)
            return 1

        logger.info(f"Building predictor: {args.predictor}")
        predictor = build_predictor(args.predictor, window=args.model_window)

        trials = getattr(args, 'trials', 100)
        periods = getattr(args, 'periods', 200)
        seed = getattr(args, 'seed', None)
        
        logger.info(f"Running randomized test with {trials} trials")
        result = randomized_test(
            predictor=predictor,
            history=history,
            trials=trials,
            window=args.model_window or 100,
            periods=periods,
            train_window=None,
            seed=seed,
        )

        # 打印报告
        print_randomized_report(result, title=f"随机对照检验：{args.predictor}")
        
        # 保存结果
        if hasattr(args, 'output') and args.output:
            import json
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 转换 numpy 类型
            def convert(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.floating, float)) and not np.isfinite(obj):
                    return None
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert(x) for x in obj]
                return obj
            
            result_clean = convert(result)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result_clean, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {output_path}")
            print(f"\n结果已保存到：{output_path}")

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"错误：文件未找到 - {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        logger.error(f"Value error: {e}")
        print(f"错误：参数无效 - {e}", file=sys.stderr)
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error during randomized test: {e}")
        print(f"错误：随机对照检验失败 - {e}", file=sys.stderr)
        return 1
