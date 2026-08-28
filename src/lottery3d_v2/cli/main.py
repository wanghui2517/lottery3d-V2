# -*- coding: utf-8 -*-
"""统一 CLI 入口 - 使用 Typer 框架"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from loguru import logger

from ..config import get_config

# 配置日志
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

app = typer.Typer(
    name="lottery3d",
    help="福彩 3D 智能推荐系统 v2.0 - 模块化、可配置、可验证的预测框架",
    add_completion=False,
)


def version_callback(value: bool):
    if value:
        print("lottery3d v2.0.0")
        raise typer.Exit()


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="启用详细日志"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="静默模式"),
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="显示版本号",
    ),
):
    """福彩 3D 智能推荐系统 CLI 入口"""
    if verbose:
        logger.level("DEBUG")
    if quiet:
        logger.disable("lottery3d_v2")


@app.command("snapshot")
def cmd_snapshot(
    csv: Path = typer.Option(
        Path("data/3d_full_history.csv"),
        "--csv",
        help="历史数据 CSV 文件路径",
    ),
):
    """数据快照 - 查看最新数据状态"""
    from lottery3d_v2.history import print_snapshot

    try:
        print_snapshot(csv)
        return 0
    except FileNotFoundError as e:
        logger.error(f"文件未找到：{e}")
        raise typer.Exit(code=1) from e
    except Exception as e:
        logger.error(f"显示快照失败：{e}")
        raise typer.Exit(code=1) from e


@app.command("predict")
def cmd_predict(
    csv: Path = typer.Option(
        Path("data/3d_full_history.csv"),
        "--csv",
        help="历史数据 CSV 文件路径",
    ),
    ensemble: str = typer.Option(
        "full",
        "--ensemble",
        "-e",
        help="集成模型名称或单个预测器名称",
    ),
    model_window: int = typer.Option(
        100,
        "--window",
        "-w",
        help="历史窗口大小",
    ),
    top_k: int = typer.Option(
        3,
        "--top-k",
        "-k",
        help="Top-K 候选数量",
    ),
    calibrated: bool = typer.Option(
        False,
        "--calibrated",
        "-c",
        help="应用温度校准",
    ),
    calibration_dir: Path = typer.Option(
        Path("results/calibration"),
        "--calib-dir",
        help="校准结果目录",
    ),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        "-o",
        help="输出 JSON 文件路径",
    ),
    no_log: bool = typer.Option(
        False,
        "--no-log",
        help="不记录到前向日志",
    ),
    log_file: Path = typer.Option(
        Path("results/forward_log.csv"),
        "--log-file",
        help="前向日志文件路径",
    ),
):
    """下一期预测"""
    from lottery3d_v2.cli.commands.predict import predict_cmd
    
    class Args:
        pass
    
    args = Args()
    args.csv = csv
    args.ensemble = ensemble
    args.model_window = model_window
    args.top_k = top_k
    args.calibrated = calibrated
    args.calibration_dir = calibration_dir
    args.out = out
    args.no_log = no_log
    args.log_file = log_file
    
    return predict_cmd(args)


@app.command("backtest")
def cmd_backtest(
    csv: Path = typer.Option(
        Path("data/3d_full_history.csv"),
        "--csv",
        help="历史数据 CSV 文件路径",
    ),
    predictor: str = typer.Option(
        "markov",
        "--predictor",
        "-p",
        help="预测器名称",
    ),
    model_window: int = typer.Option(
        100,
        "--window",
        "-w",
        help="历史窗口大小",
    ),
    train_window: int = typer.Option(
        0,
        "--train-window",
        help="训练窗口 (0=全部)",
    ),
    test_window: int = typer.Option(
        50,
        "--test-window",
        "-t",
        help="测试窗口",
    ),
    compare: bool = typer.Option(
        False,
        "--compare",
        help="批量对比多个预测器",
    ),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        "-o",
        help="输出 JSON 文件路径",
    ),
):
    """回测验证"""
    from .commands.backtest import backtest_cmd
    
    class Args:
        pass
    
    args = Args()
    args.csv = csv
    args.predictor = predictor
    args.model_window = model_window
    args.train_window = train_window
    args.test_window = test_window
    args.compare = compare
    args.out = out
    
    return backtest_cmd(args)


@app.command("calibrate")
def cmd_calibrate(
    csv: Path = typer.Option(
        Path("data/3d_full_history.csv"),
        "--csv",
        help="历史数据 CSV 文件路径",
    ),
    predictor: str = typer.Option(
        "markov",
        "--predictor",
        "-p",
        help="预测器名称",
    ),
    model_window: int = typer.Option(
        100,
        "--window",
        "-w",
        help="历史窗口大小",
    ),
    n_samples: int = typer.Option(
        500,
        "--samples",
        "-n",
        help="校准样本数",
    ),
    temp_min: float = typer.Option(
        0.5,
        "--temp-min",
        help="最小温度",
    ),
    temp_max: float = typer.Option(
        2.0,
        "--temp-max",
        help="最大温度",
    ),
    cv_folds: int = typer.Option(
        5,
        "--cv-folds",
        help="交叉验证折数",
    ),
    calibration_dir: Path = typer.Option(
        Path("results/calibration"),
        "--calib-dir",
        help="校准结果目录",
    ),
):
    """温度校准"""
    from .commands.calibrate import calibrate_cmd
    
    class Args:
        pass
    
    args = Args()
    args.csv = csv
    args.predictor = predictor
    args.model_window = model_window
    args.n_samples = n_samples
    args.temp_min = temp_min
    args.temp_max = temp_max
    args.cv_folds = cv_folds
    args.calibration_dir = calibration_dir
    
    return calibrate_cmd(args)


@app.command("randomize")
def cmd_randomize(
    csv: Path = typer.Option(
        Path("data/3d_full_history.csv"),
        "--csv",
        help="历史数据 CSV 文件路径",
    ),
    predictor: str = typer.Option(
        "markov",
        "--predictor",
        "-p",
        help="预测器名称",
    ),
    model_window: int = typer.Option(
        100,
        "--window",
        "-w",
        help="历史窗口大小",
    ),
    n_simulations: int = typer.Option(
        1000,
        "--simulations",
        "-n",
        help="模拟次数",
    ),
    test_window: int = typer.Option(
        50,
        "--test-window",
        "-t",
        help="测试窗口",
    ),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        "-o",
        help="输出 JSON 文件路径",
    ),
):
    """随机对照检验"""
    from .commands.randomize import randomize_cmd
    
    class Args:
        pass
    
    args = Args()
    args.csv = csv
    args.predictor = predictor
    args.model_window = model_window
    args.n_simulations = n_simulations
    args.test_window = test_window
    args.out = out
    
    return randomize_cmd(args)


if __name__ == "__main__":
    app()
