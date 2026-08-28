# -*- coding: utf-8 -*-
"""回测命令实现"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from ...base import build_predictor
from ...history import load_history_from_csv
from ...validation import (
    backtest_predictor,
    compare_predictors,
    print_backtest_report,
    print_comparison_table,
)

if TYPE_CHECKING:
    import argparse


def backtest_cmd(args: argparse.Namespace) -> int:
    """回测验证命令
    
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
            print("历史数据为空，无法回测。", file=sys.stderr)
            return 1

        train_window = None if args.train_window <= 0 else args.train_window

        if args.compare:
            # 批量对比
            logger.info("Running predictor comparison")
            from ...predictors import (
                MarkovPredictor,
                ParityPredictor,
                PrimePredictor,
                SpanPredictor,
                ZScorePredictor,
            )
            
            predictors = [
                MarkovPredictor(),
                SpanPredictor(window=args.model_window),
                ParityPredictor(window=args.model_window),
                PrimePredictor(window=args.model_window),
                ZScorePredictor(window=args.model_window),
            ]

            results = compare_predictors(
                history,
                predictors,
                window=args.model_window,
                periods=args.test_window,
                train_window=train_window,
            )

            print_comparison_table(results)

            if args.out:
                import json
                out_path = Path(args.out)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                export_data = {r["name"]: r for r in results}
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2, default=float)
                logger.info(f"Comparison results saved to {out_path}")
                print(f"\n结果已保存到：{out_path}")

        else:
            # 单个预测器回测
            logger.info(f"Backtesting predictor: {args.predictor}")
            predictor = build_predictor(args.predictor, window=args.model_window)

            result = backtest_predictor(
                predictor,
                history,
                window=args.model_window,
                periods=args.test_window,
                train_window=train_window,
            )

            print_backtest_report(result)

            if args.out:
                import json
                out_path = Path(args.out)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2, default=float)
                logger.info(f"Backtest results saved to {out_path}")
                print(f"\n结果已保存到：{out_path}")

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
        logger.exception(f"Unexpected error during backtest: {e}")
        print(f"错误：回测失败 - {e}", file=sys.stderr)
        return 1
