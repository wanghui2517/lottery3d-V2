# -*- coding: utf-8 -*-
"""统一 CLI 入口"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import get_config
from .history import (
    load_history_from_csv,
    load_history_incremental,
    print_snapshot,
)
from .predictors import (
    PREDICTOR_REGISTRY,
    EnsemblePredictor,
    MarkovPredictor,
    ParityPredictor,
    PrimePredictor,
    SpanPredictor,
    ZScorePredictor,
)
from .calibration import (
    apply_calibration,
    collect_calibration_records,
    holdout_calibrate,
    load_calibration_result,
    print_calibration_report,
    rolling_calibrate,
    save_calibration_result,
)
from .validation import (
    backtest_predictor,
    compare_predictors,
    print_backtest_report,
    print_comparison_table,
)
from .schema import Prediction


def build_ensemble(model_window: int = 100, ensemble_name: str = "full") -> EnsemblePredictor:
    """构建集成模型"""
    return EnsemblePredictor(ensemble_name=ensemble_name, top_k=3)


def build_simple_ensemble(model_window: int = 100) -> EnsemblePredictor:
    """构建简化集成模型"""
    return EnsemblePredictor(ensemble_name="simple", top_k=3)


# ==================== 命令实现 ====================

def cmd_snapshot(args: argparse.Namespace) -> int:
    """数据快照"""
    print_snapshot(args.csv)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    """下一期预测"""
    # 加载历史数据
    history, _ = load_history_incremental(args.csv)

    if not history:
        print("历史数据为空，无法预测。", file=sys.stderr)
        return 1

    latest = history[-1]

    print("=" * 80)
    print("最新数据")
    print("=" * 80)
    print(f"最新期号: {latest.issue}")
    print(f"最新日期: {latest.date}")
    print(f"最新号码: {latest.digits}")
    print(f"总期数: {len(history)}")
    print()

    # 构建模型
    if args.ensemble == "full":
        model = build_ensemble(args.model_window, "full")
    elif args.ensemble == "simple":
        model = build_simple_ensemble(args.model_window)
    else:
        # 单个预测器
        from .predictors import build_predictor
        model = build_predictor(args.ensemble, window=args.model_window)

    # 应用温度校准
    calibration = None
    if args.calibrated:
        calib_path = Path(args.calibration_dir) / f"{args.ensemble}_calibration.json"
        if calib_path.exists():
            calibration = load_calibration_result(calib_path)
            print(f"已加载温度校准: {calib_path}")
        else:
            print(f"警告: 校准文件不存在 {calib_path}，跳过校准", file=sys.stderr)

    # 预测
    prediction = model.predict(history, top_k=args.top_k)

    # 如果有校准，应用到概率
    if calibration:
        proba_result = model.predict_proba(history)
        calibrated_proba = apply_calibration(proba_result, calibration)
        # 重新计算 Top-K
        from .utils import normalize_proba, top_k_from_proba
        positions_proba = [normalize_proba(p) for p in calibrated_proba.positions_proba]
        sum_proba = normalize_proba(calibrated_proba.sum_proba)

        positions = [top_k_from_proba(p, k=args.top_k) for p in positions_proba]
        sum_value = top_k_from_proba(sum_proba, k=args.top_k)

        # 重新创建 Prediction
        from .base import BasePredictor
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

    print("=" * 80)
    print("下一期参考")
    print("=" * 80)
    print(f"模型: {prediction.name}")
    print(f"model_window: {args.model_window}")
    print(f"top_k: {args.top_k}")
    if calibration:
        print("温度校准: 已应用")
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
        log_prediction(args, prediction)

    # 保存 JSON
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=float)
        print(f"结果已保存到: {out_path}")

    return 0


def log_prediction(args: argparse.Namespace, prediction: Prediction) -> None:
    """自动追加预测到前向日志"""
    from .history import load_history_from_csv

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
        "百位Top1": prediction.positions[0][0] if prediction.positions[0] else "",
        "十位Top1": prediction.positions[1][0] if prediction.positions[1] else "",
        "个位Top1": prediction.positions[2][0] if prediction.positions[2] else "",
        "百位Top3": ",".join(map(str, prediction.positions[0])),
        "十位Top3": ",".join(map(str, prediction.positions[1])),
        "个位Top3": ",".join(map(str, prediction.positions[2])),
        "和值Top3": ",".join(map(str, prediction.sum_value)),
        "最佳两位": "",
        "实际号码": "",
        "Top1命中数": "",
        "Top3命中数": "",
        "和值命中": "",
        "两位严格命中": "",
    }

    if prediction.best_pair:
        pos_names = {0: "百位", 1: "十位", 2: "个位"}
        pair = prediction.best_pair
        row["最佳两位"] = (
            f"{pos_names[pair['pos_a']]}+{pos_names[pair['pos_b']]}:"
            f"{pair['digit_a']}+{pair['digit_b']}"
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

    print(f"预测已记录到前向日志: {log_path}")


def cmd_backtest(args: argparse.Namespace) -> int:
    """回测验证"""
    history = load_history_from_csv(args.csv, ascending=True)

    train_window = None if args.train_window <= 0 else args.train_window

    if args.compare:
        # 批量对比
        predictors = [
            MarkovPredictor(),
            SpanPredictor(window=args.model_window),
            ParityPredictor(window=args.model_window),
            PrimePredictor(window=args.model_window),
            ZScorePredictor(window=args.model_window),
            build_ensemble(args.model_window, "full"),
            build_simple_ensemble(args.model_window),
        ]

        results = compare_predictors(
            history,
            predictors,
            window=args.min_train,
            periods=args.periods,
            train_window=train_window,
        )

        print_comparison_table(results)
    else:
        # 单模型回测
        if args.ensemble == "full":
            model = build_ensemble(args.model_window, "full")
        elif args.ensemble == "simple":
            model = build_simple_ensemble(args.model_window)
        else:
            from .predictors import build_predictor
            model = build_predictor(args.ensemble, window=args.model_window)

        result = backtest_predictor(
            predictor=model,
            history=history,
            window=args.min_train,
            periods=args.periods,
            train_window=train_window,
        )

        title = (
            f"模型回测: {model.name} | "
            f"model_window={args.model_window} | "
            f"train_window={args.train_window or 'all'} | "
            f"periods={args.periods}"
        )

        print_backtest_report(result, title)

    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """温度校准"""
    history = load_history_from_csv(args.csv, ascending=True)

    train_window = None if args.train_window <= 0 else args.train_window

    if args.ensemble == "full":
        model = build_ensemble(args.model_window, "full")
    elif args.ensemble == "simple":
        model = build_simple_ensemble(args.model_window)
    else:
        from .predictors import build_predictor
        model = build_predictor(args.ensemble, window=args.model_window)

    print("收集校准数据...")
    records = collect_calibration_records(
        history,
        model,
        periods=args.periods,
        min_train=args.min_train,
        train_window=train_window,
    )

    print(f"收集到 {len(records)} 条记录 ({len(set(r['target_idx'] for r in records))} 期)")

    if args.mode == "holdout":
        result = holdout_calibrate(
            records,
            calib_ratio=args.calib_ratio,
            position_specific=args.position_specific,
        )
    else:
        result = rolling_calibrate(
            records,
            warmup_periods=args.warmup,
            window_periods=args.window,
            position_specific=args.position_specific,
        )

    title = f"温度校准: {model.name} | mode={args.mode}"
    print_calibration_report(result, title)

    if args.save:
        save_path = Path(args.save)
        save_calibration_result(result, save_path, model.name)
        print(f"校准结果已保存: {save_path}")

    return 0


def cmd_randomize(args: argparse.Namespace) -> int:
    """随机对照检验"""
    import numpy as np
    from .history import HistoryRow

    history = load_history_from_csv(args.csv, ascending=True)

    train_window = None if args.train_window <= 0 else args.train_window

    if args.ensemble == "full":
        model = build_ensemble(args.model_window, "full")
    else:
        model = build_simple_ensemble(args.model_window)

    print("正在回测真实数据...")

    observed = backtest_predictor(
        predictor=model,
        history=history,
        window=args.min_train,
        periods=args.periods,
        train_window=train_window,
    )

    if observed is None:
        print("真实数据回测失败。", file=sys.stderr)
        return 1

    print("=" * 80)
    print("实验配置")
    print("=" * 80)
    print(f"CSV: {args.csv}")
    print(f"ensemble: {args.ensemble}")
    print(f"model_window: {args.model_window}")
    print(f"train_window: {args.train_window or 'all'}")
    print(f"periods: {args.periods}")
    print(f"sims: {args.sims}")
    print(f"seed: {args.seed}")
    print()

    print("=" * 80)
    print("真实数据结果")
    print("=" * 80)
    print(f"Top1: {observed['top1_rate']:.4f}")
    print(f"Top3: {observed['top3_rate']:.4f}")
    print(f"两位任一: {observed['pair_any_rate']:.4f}")
    print(f"LogLoss: {observed['mean_log_loss']:.4f}")
    print()

    rng = np.random.default_rng(args.seed)

    top1_count = 0
    top3_count = 0
    pair_any_count = 0

    print(f"开始随机对照模拟，共 {args.sims} 次...")

    def randomize_history(hist, rng):
        randomized = []
        for row in hist:
            digits = tuple(int(x) for x in rng.integers(0, 10, size=3))
            randomized.append(
                HistoryRow(
                    issue=row.issue,
                    digits=digits,
                    sales=row.sales,
                    prize=row.prize,
                    single_wins=row.single_wins,
                    date=row.date,
                )
            )
        return randomized

    for i in range(args.sims):
        random_history = randomize_history(history, rng)

        result = backtest_predictor(
            predictor=model,
            history=random_history,
            window=args.min_train,
            periods=args.periods,
            train_window=train_window,
        )

        if result is None:
            continue

        if result["top1_rate"] >= observed["top1_rate"]:
            top1_count += 1

        if result["top3_rate"] >= observed["top3_rate"]:
            top3_count += 1

        if result["pair_any_rate"] >= observed["pair_any_rate"]:
            pair_any_count += 1

        if (i + 1) % 10 == 0:
            print(f"已完成 {i + 1}/{args.sims}")

    print("=" * 80)
    print("随机对照结果")
    print("=" * 80)

    print(
        f"随机数据中 Top1 >= 真实 Top1 的比例: "
        f"{top1_count / args.sims:.4f}"
    )

    print(
        f"随机数据中 Top3 >= 真实 Top3 的比例: "
        f"{top3_count / args.sims:.4f}"
    )

    print(
        f"随机数据中两位任一 >= 真实两位任一的比例: "
        f"{pair_any_count / args.sims:.4f}"
    )

    print()
    print("解释：")
    print("如果这些比例很高，例如 > 0.10，说明真实结果可能只是随机波动。")
    print("如果这些比例很低，例如 < 0.05，说明真实数据中可能存在局部可利用结构。")

    return 0


def cmd_update_log(args: argparse.Namespace) -> int:
    """更新前向日志实际开奖结果"""
    import csv

    POS_MAP = {"百位": 0, "十位": 1, "个位": 2}

    def parse_int_list(value: str) -> list[int]:
        return [int(x.strip()) for x in str(value).split(",") if x.strip() != ""]

    def parse_best_pair(value: str):
        if not value:
            return None
        try:
            pos_part, digit_part = value.split(":")
            pos_a_name, pos_b_name = pos_part.split("+")
            digit_a, digit_b = digit_part.split("+")
            return (
                POS_MAP[pos_a_name.strip()],
                POS_MAP[pos_b_name.strip()],
                int(digit_a.strip()),
                int(digit_b.strip()),
            )
        except Exception:
            return None

    def is_empty_actual(value: str) -> bool:
        value = str(value or "").strip()
        return value == "" or set(value) <= {",", " "}

    def yes_no(flag: bool) -> str:
        return "是" if flag else "否"

    actual = tuple(int(x.strip()) for x in args.actual.split(","))
    if len(actual) != 3:
        raise ValueError("实际号码必须是三个数字，例如: 4,7,0")

    actual_sum = sum(actual)
    actual_str = ",".join(str(x) for x in actual)

    path = Path(args.log)

    if not path.exists():
        raise FileNotFoundError(f"日志文件不存在: {path}")

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    updated = 0

    for row in rows:
        if row.get("目标期号") != args.target_issue:
            continue

        if args.model and row.get("模型") != args.model:
            continue

        if not is_empty_actual(row.get("实际号码", "")) and not args.force:
            continue

        row["实际号码"] = actual_str

        try:
            top1 = [
                int(row["百位Top1"]),
                int(row["十位Top1"]),
                int(row["个位Top1"]),
            ]
            row["Top1命中数"] = sum(
                1 for i, digit in enumerate(top1) if actual[i] == digit
            )
        except Exception:
            row["Top1命中数"] = ""

        try:
            top3 = [
                parse_int_list(row["百位Top3"]),
                parse_int_list(row["十位Top3"]),
                parse_int_list(row["个位Top3"]),
            ]
            row["Top3命中数"] = sum(
                1 for i, digits in enumerate(top3) if actual[i] in digits
            )
        except Exception:
            row["Top3命中数"] = ""

        try:
            sum_top3 = parse_int_list(row["和值Top3"])
            row["和值命中"] = yes_no(actual_sum in sum_top3)
        except Exception:
            row["和值命中"] = ""

        try:
            pair = parse_best_pair(row.get("最佳两位", ""))
            if pair is not None:
                pos_a, pos_b, digit_a, digit_b = pair
                row["两位严格命中"] = yes_no(
                    actual[pos_a] == digit_a and actual[pos_b] == digit_b
                )
            else:
                row["两位严格命中"] = ""
        except Exception:
            row["两位严格命中"] = ""

        updated += 1

    if updated == 0:
        print("没有更新任何记录。")
        return 0

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"已更新 {updated} 条记录。")
    print(f"目标期号: {args.target_issue}")
    print(f"实际号码: {actual_str}")
    print(f"实际和值: {actual_sum}")

    return 0


# ==================== 主入口 ====================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lottery3d",
        description="福彩3D智能推荐系统 v2.0",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 全局通用参数
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--csv",
        default="data/3d_full_history.csv",
        help="历史数据 CSV 路径",
    )
    parent_parser.add_argument(
        "--model-window",
        type=int,
        default=100,
        help="模型内部统计窗口",
    )

    # snapshot
    p_snap = subparsers.add_parser("snapshot", parents=[parent_parser], help="数据快照")
    p_snap.set_defaults(func=cmd_snapshot)

    # predict
    p_pred = subparsers.add_parser("predict", parents=[parent_parser], help="下一期预测")
    p_pred.add_argument(
        "--ensemble",
        choices=["full", "simple"] + sorted(PREDICTOR_REGISTRY.keys()),
        default="full",
        help="模型选择",
    )
    p_pred.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="每位置输出候选数",
    )
    p_pred.add_argument(
        "--calibrated",
        action="store_true",
        help="应用温度校准",
    )
    p_pred.add_argument(
        "--calibration-dir",
        default="results/calibration",
        help="校准文件目录",
    )
    p_pred.add_argument(
        "--out",
        default="",
        help="JSON 输出路径",
    )
    p_pred.add_argument(
        "--no-log",
        action="store_true",
        help="不记录前向日志",
    )
    p_pred.add_argument(
        "--log-file",
        default="results/forward_log.csv",
        help="前向日志路径",
    )
    p_pred.set_defaults(func=cmd_predict)

    # backtest
    p_bt = subparsers.add_parser("backtest", parents=[parent_parser], help="回测验证")
    p_bt.add_argument(
        "--ensemble",
        choices=["full", "simple"] + sorted(PREDICTOR_REGISTRY.keys()),
        default="full",
        help="模型选择",
    )
    p_bt.add_argument(
        "--train-window",
        type=int,
        default=0,
        help="滚动训练窗口，0=全部历史",
    )
    p_bt.add_argument(
        "--min-train",
        type=int,
        default=100,
        help="最小训练期数",
    )
    p_bt.add_argument(
        "--periods",
        type=int,
        default=200,
        help="回测期数",
    )
    p_bt.add_argument(
        "--compare",
        action="store_true",
        help="对比所有预测器",
    )
    p_bt.set_defaults(func=cmd_backtest)

    # calibrate
    p_cal = subparsers.add_parser("calibrate", parents=[parent_parser], help="温度校准")
    p_cal.add_argument(
        "--ensemble",
        choices=["full", "simple"] + sorted(PREDICTOR_REGISTRY.keys()),
        default="full",
        help="模型选择",
    )
    p_cal.add_argument(
        "--mode",
        choices=["holdout", "rolling"],
        default="rolling",
        help="校准模式",
    )
    p_cal.add_argument(
        "--train-window",
        type=int,
        default=0,
        help="滚动训练窗口，0=全部历史",
    )
    p_cal.add_argument(
        "--min-train",
        type=int,
        default=100,
        help="最小训练期数",
    )
    p_cal.add_argument(
        "--periods",
        type=int,
        default=200,
        help="校准数据期数",
    )
    p_cal.add_argument(
        "--calib-ratio",
        type=float,
        default=0.5,
        help="Holdout 校准集比例",
    )
    p_cal.add_argument(
        "--position-specific",
        action="store_true",
        default=True,
        help="位置特定温度",
    )
    p_cal.add_argument(
        "--no-position-specific",
        action="store_false",
        dest="position_specific",
        help="全局单一温度",
    )
    p_cal.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Rolling 预热期数",
    )
    p_cal.add_argument(
        "--window",
        type=int,
        default=100,
        help="Rolling 滑动窗口",
    )
    p_cal.add_argument(
        "--save",
        default="",
        help="校准结果保存路径",
    )
    p_cal.set_defaults(func=cmd_calibrate)

    # randomize
    p_rand = subparsers.add_parser("randomize", parents=[parent_parser], help="随机对照检验")
    p_rand.add_argument(
        "--ensemble",
        choices=["full", "simple"],
        default="full",
        help="模型选择",
    )
    p_rand.add_argument(
        "--train-window",
        type=int,
        default=0,
        help="滚动训练窗口，0=全部历史",
    )
    p_rand.add_argument(
        "--min-train",
        type=int,
        default=100,
        help="最小训练期数",
    )
    p_rand.add_argument(
        "--periods",
        type=int,
        default=300,
        help="回测期数",
    )
    p_rand.add_argument(
        "--sims",
        type=int,
        default=100,
        help="模拟次数",
    )
    p_rand.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    p_rand.set_defaults(func=cmd_randomize)

    # update-log
    p_log = subparsers.add_parser("update-log", help="更新前向日志实际开奖")
    p_log.add_argument("--log", default="results/forward_log.csv", help="日志文件")
    p_log.add_argument("--target-issue", required=True, help="目标期号")
    p_log.add_argument("--model", default="", help="模型名，空=全部")
    p_log.add_argument("--actual", required=True, help="实际号码，如 4,7,0")
    p_log.add_argument("--force", action="store_true", help="强制覆盖")
    p_log.set_defaults(func=cmd_update_log)

    return parser


def predict() -> int:
    """下一期预测 - 入口点"""
    import sys
    sys.argv = ["lottery3d", "predict"] + sys.argv[1:]
    return main()

def backtest() -> int:
    """回测验证 - 入口点"""
    import sys
    sys.argv = ["lottery3d", "backtest"] + sys.argv[1:]
    return main()

def calibrate() -> int:
    """温度校准 - 入口点"""
    import sys
    sys.argv = ["lottery3d", "calibrate"] + sys.argv[1:]
    return main()

def randomize() -> int:
    """随机对照检验 - 入口点"""
    import sys
    sys.argv = ["lottery3d", "randomize"] + sys.argv[1:]
    return main()

def snapshot() -> int:
    """数据快照 - 入口点"""
    import sys
    sys.argv = ["lottery3d", "snapshot"] + sys.argv[1:]
    return main()

def update_log() -> int:
    """更新前向日志 - 入口点"""
    import sys
    sys.argv = ["lottery3d", "update-log"] + sys.argv[1:]
    return main()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())