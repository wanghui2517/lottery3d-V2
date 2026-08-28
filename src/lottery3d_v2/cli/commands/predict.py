# -*- coding: utf-8 -*-
"""预测命令实现"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

# 确保预测器已注册
from ... import predictors  # noqa: F401
from ...base import BasePredictor, build_predictor
from ...calibration import (
    apply_calibration,
    load_calibration_result,
)
from ...config import get_config
from ...history import load_history_incremental, print_snapshot
from ...schema import Prediction
from ...utils import normalize_proba, top_k_from_proba

if TYPE_CHECKING:
    import argparse


def build_ensemble(model_window: int = 100, ensemble_name: str = "full"):
    """构建集成模型"""
    from ...ensemble import EnsemblePredictor
    return EnsemblePredictor(ensemble_name=ensemble_name, top_k=3)


def predict_cmd(args: argparse.Namespace) -> int:
    """下一期预测命令
    
    Args:
        args: CLI 参数
        
    Returns:
        int: 退出码 (0=成功，1=失败)
    """
    try:
        # 加载历史数据
        logger.info(f"Loading history from {args.csv}")
        history, _ = load_history_incremental(args.csv)

        if not history:
            logger.error("历史数据为空，无法预测")
            print("历史数据为空，无法预测。", file=sys.stderr)
            return 1

        latest = history[-1]

        print("=" * 80)
        print("最新数据")
        print("=" * 80)
        print(f"最新期号：{latest.issue}")
        print(f"最新日期：{latest.date}")
        print(f"最新号码：{latest.digits}")
        print(f"总期数：{len(history)}")
        print()

        # 构建模型
        logger.info(f"Building model: {args.ensemble} (window={args.model_window})")
        if args.ensemble == "full":
            model = build_ensemble(args.model_window, "full")
        elif args.ensemble == "simple":
            model = build_ensemble(args.model_window, "simple")
        else:
            # 单个预测器
            model = build_predictor(args.ensemble, window=args.model_window)

        # 应用温度校准
        calibration = None
        if args.calibrated:
            calib_path = Path(args.calibration_dir) / f"{args.ensemble}_calibration.json"
            if calib_path.exists():
                calibration = load_calibration_result(calib_path)
                logger.info(f"Loaded calibration from {calib_path}")
                print(f"已加载温度校准：{calib_path}")
            else:
                logger.warning(f"校准文件不存在：{calib_path}")
                print(f"警告：校准文件不存在 {calib_path}，跳过校准", file=sys.stderr)

        # 预测
        logger.info("Running prediction")
        prediction = model.predict(history, top_k=args.top_k)

        # 如果有校准，应用到概率
        if calibration:
            logger.info("Applying temperature calibration")
            proba_result = model.predict_proba(history)
            calibrated_proba = apply_calibration(proba_result, calibration)
            
            positions_proba = [normalize_proba(p) for p in calibrated_proba.positions_proba]
            sum_proba = normalize_proba(calibrated_proba.sum_proba)

            positions = [top_k_from_proba(p, k=args.top_k) for p in positions_proba]
            sum_value = top_k_from_proba(sum_proba, k=args.top_k)

            best_pair = BasePredictor.choose_best_pair(positions_proba)

            prediction = Prediction(
                name=f"{model.name}_calibrated",
                positions=positions,
                sum_value=sum_value,
                positions_proba=positions_proba,
                sum_proba=sum_proba,
                best_pair=best_pair,
                meta=calibrated_proba.meta,
            )

        result = prediction.to_dict()

        # 输出结果
        print("=" * 80)
        print("下一期参考")
        print("=" * 80)
        print(f"模型：{prediction.name}")
        print(f"model_window: {args.model_window}")
        print(f"top_k: {args.top_k}")
        if calibration:
            print("温度校准：已应用")
        print()

        print("【定位候选】")
        print(f"百位 Top{args.top_k}: {result['positions'][0]}")
        print(f"十位 Top{args.top_k}: {result['positions'][1]}")
        print(f"个位 Top{args.top_k}: {result['positions'][2]}")
        print()

        print("【和值候选】")
        print(f"和值 Top{args.top_k}: {result['sum_value']}")
        print()

        if result.get("best_pair"):
            pair = result["best_pair"]
            pos_names = {0: "百位", 1: "十位", 2: "个位"}
            print("【最佳两位】")
            print(
                f"{pos_names[pair['pos_a']]} + {pos_names[pair['pos_b']]}: "
                f"{pair['digit_a']} + {pair['digit_b']}"
            )
            print(f"score: {pair['score']:.6f}")
            print()

        # 自动记录到前向日志
        if not args.no_log:
            _log_prediction(args, prediction)

        # 保存 JSON
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=float)
            logger.info(f"Results saved to {out_path}")
            print(f"结果已保存到：{out_path}")

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
        logger.exception(f"Unexpected error during prediction: {e}")
        print(f"错误：预测失败 - {e}", file=sys.stderr)
        return 1


def _log_prediction(args: argparse.Namespace, prediction: Prediction) -> None:
    """自动追加预测到前向日志"""
    from ...history import load_history_from_csv

    try:
        history = load_history_from_csv(args.csv, ascending=True)
        if not history:
            return

        latest = history[-1]
        target_issue = str(int(latest.issue) + 1) if latest.issue and latest.issue.isdigit() else "unknown"

        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # 准备行数据
        row = {
            "预测生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "数据截至期号": latest.issue,
            "目标期号": target_issue,
            "模型": prediction.name,
            "百位 Top1": prediction.positions[0][0] if prediction.positions[0] else "",
            "十位 Top1": prediction.positions[1][0] if prediction.positions[1] else "",
            "个位 Top1": prediction.positions[2][0] if prediction.positions[2] else "",
            "百位 Top3": ",".join(map(str, prediction.positions[0])),
            "十位 Top3": ",".join(map(str, prediction.positions[1])),
            "个位 Top3": ",".join(map(str, prediction.positions[2])),
            "和值 Top3": ",".join(map(str, prediction.sum_value)),
            "最佳两位": "",
            "实际号码": "",
            "Top1 命中数": "",
            "Top3 命中数": "",
            "和值命中": "",
            "两位严格命中": "",
        }

        if prediction.best_pair:
            pos_names = {0: "百位", 1: "十位", 2: "个位"}
            pair = prediction.best_pair
            row["最佳两位"] = (
                f"{pos_names[pair['pos_a']]}+{pos_names[pair['pos_b']]}"
                f":{pair['digit_a']}+{pair['digit_b']}"
            )

        # 写入 CSV
        fieldnames = list(row.keys())
        file_exists = log_path.exists()

        import csv
        with open(log_path, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        logger.info(f"Prediction logged to {log_path}")
        print(f"预测已记录到前向日志：{log_path}")
        
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")
        print(f"警告：记录日志失败 - {e}", file=sys.stderr)
