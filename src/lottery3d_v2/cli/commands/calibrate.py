# -*- coding: utf-8 -*-
"""校准命令实现"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from ...base import build_predictor
from ...calibration import (
    holdout_calibrate,
    collect_calibration_records,
    save_calibration_result,
    print_calibration_report,
)
from ...history import load_history_from_csv

if TYPE_CHECKING:
    import argparse


def calibrate_cmd(args: argparse.Namespace) -> int:
    """温度校准命令
    
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
            print("历史数据为空，无法校准。", file=sys.stderr)
            return 1

        logger.info(f"Building predictor: {args.predictor}")
        predictor = build_predictor(args.predictor, window=args.model_window)

        logger.info("Collecting calibration records")
        records = collect_calibration_records(
            history,
            predictor,
            periods=max(100, len(history) // 2),
            min_train=args.model_window or 50,
            train_window=None,
        )

        if not records:
            logger.error("校准记录为空")
            print("校准记录为空，无法校准。", file=sys.stderr)
            return 1

        logger.info("Running temperature calibration")
        result = holdout_calibrate(
            records,
            calib_ratio=args.n_samples / 1000 if hasattr(args, 'n_samples') and args.n_samples else 0.5,
            position_specific=True,
        )

        # 保存结果
        calib_dir = Path(args.calibration_dir)
        calib_dir.mkdir(parents=True, exist_ok=True)
        output_path = calib_dir / f"{args.predictor}_calibration.json"
        
        save_calibration_result(result, output_path)
        logger.info(f"Calibration saved to {output_path}")

        # 打印报告
        print_calibration_report(result)
        print(f"\n校准结果已保存到：{output_path}")

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
        logger.exception(f"Unexpected error during calibration: {e}")
        print(f"错误：校准失败 - {e}", file=sys.stderr)
        return 1
