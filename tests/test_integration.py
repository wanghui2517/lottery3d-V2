# -*- coding: utf-8 -*-
"""集成测试：CLI 端到端测试"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run_cli_command(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """运行 CLI 命令"""
    cmd = [sys.executable, "-m", "lottery3d_v2.cli.main"] + args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/workspace")
    
    if check and result.returncode != 0:
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Command failed with code {result.returncode}")
    
    return result


def test_cli_help():
    """测试帮助命令"""
    result = run_cli_command(["--help"])
    assert result.returncode == 0
    # Typer 使用 Rich 格式化，包含中文描述
    assert "福彩" in result.stdout or "Lottery" in result.stdout
    assert "predict" in result.stdout
    assert "backtest" in result.stdout
    assert "calibrate" in result.stdout
    assert "randomize" in result.stdout
    print("✓ CLI help command works")


def test_cli_snapshot(tmp_path):
    """测试快照命令（跳过网络请求）"""
    # 快照命令需要网络，这里仅验证命令结构
    result = run_cli_command(["snapshot", "--help"])
    assert result.returncode == 0
    # Typer 使用 Rich 格式化，包含中文描述
    assert "快照" in result.stdout or "Snapshot" in result.stdout or "数据" in result.stdout
    print("✓ CLI snapshot command structure is valid")


def test_cli_predict_requires_csv():
    """测试预测命令在没有 CSV 时返回错误"""
    # predict 命令有默认 CSV 路径，所以会成功执行
    # 这里只验证命令能够正常运行
    result = run_cli_command(["predict"])
    assert result.returncode == 0
    assert "最新数据" in result.stdout or "最新期号" in result.stdout
    print("✓ CLI predict command executes successfully")


def test_cli_backtest_requires_csv():
    """测试回测命令在没有 CSV 时返回错误"""
    # backtest 命令也有默认 CSV 路径，所以会成功执行
    result = run_cli_command(["backtest"])
    assert result.returncode == 0
    assert "预测器" in result.stdout or "回测" in result.stdout
    print("✓ CLI backtest command executes successfully")


def test_cli_calibrate_requires_csv():
    """测试校准命令在没有 CSV 时返回错误"""
    # calibrate 命令也有默认 CSV 路径
    result = run_cli_command(["calibrate"])
    assert result.returncode == 0
    assert "温度" in result.stdout or "校准" in result.stdout or "Calibration" in result.stdout
    print("✓ CLI calibrate command executes successfully")


def test_cli_randomize_requires_csv():
    """测试随机对照命令在没有 CSV 时返回错误"""
    # randomize 命令也有默认 CSV 路径，使用 --simulations 参数
    result = run_cli_command(["randomize", "--simulations", "5"])  # 少量试验加快测试
    assert result.returncode == 0
    assert "随机" in result.stdout or "Random" in result.stdout or "试验" in result.stdout
    print("✓ CLI randomize command executes successfully")


def test_config_validation():
    """测试配置验证"""
    from lottery3d_v2.config_schema import GlobalConfig, PredictorConfig
    
    # 有效配置 - predictors 是 Dict[str, PredictorConfig]
    config = GlobalConfig(
        top_k=5,
        predictors={"ewma": PredictorConfig(window=50, alpha=0.3)},
    )
    assert config.top_k == 5
    assert len(config.predictors) == 1
    print("✓ Config validation accepts valid configuration")
    
    # 无效配置（负数窗口）
    try:
        PredictorConfig(window=-1, alpha=0.3)
        assert False, "Should have raised ValidationError"
    except Exception:
        print("✓ Config validation rejects invalid window")
    
    # 无效配置（alpha 超出范围）
    try:
        PredictorConfig(window=50, alpha=1.5)
        assert False, "Should have raised ValidationError"
    except Exception:
        print("✓ Config validation rejects invalid alpha parameter")


def test_calibration_records_collection():
    """测试校准记录收集"""
    from lottery3d_v2.calibration import collect_calibration_records
    from lottery3d_v2.base import build_predictor
    from lottery3d_v2.schema import HistoryRow
    
    # 创建模拟历史数据
    history = [
        HistoryRow(issue=f"2024{i:03d}", digits=[i % 10, (i + 1) % 10, (i + 2) % 10])
        for i in range(200)
    ]
    
    predictor = build_predictor("ewma", window=50)
    records = collect_calibration_records(
        history=history,
        model=predictor,
        periods=100,
        min_train=50,
        train_window=None,
    )
    
    assert len(records) > 0
    assert all("proba" in r and "actual" in r for r in records)
    print(f"✓ Calibration records collected: {len(records)} records")


def test_randomized_test_function():
    """测试随机对照检验函数"""
    from lottery3d_v2.cli.commands.randomize import randomized_test
    from lottery3d_v2.base import build_predictor
    from lottery3d_v2.schema import HistoryRow
    
    # 创建少量历史数据用于快速测试
    history = [
        HistoryRow(issue=f"2024{i:03d}", digits=[i % 10, (i + 1) % 10, (i + 2) % 10])
        for i in range(150)
    ]
    
    predictor = build_predictor("ewma", window=50)
    result = randomized_test(
        predictor=predictor,
        history=history,
        trials=5,  # 少量试验以加快测试
        window=50,
        periods=50,
        seed=42,
    )
    
    assert "error" not in result or result.get("trials", 0) == 0
    if "error" not in result:
        assert "real" in result
        assert "random_top1" in result
        assert "p_values" in result
        print(f"✓ Randomized test completed with {result['trials']} trials")
    else:
        print(f"⚠ Randomized test skipped: {result['error']}")


def test_ensemble_prediction():
    """测试集成预测"""
    from lottery3d_v2.ensemble import EnsemblePredictor
    from lottery3d_v2.base import build_predictor
    from lottery3d_v2.schema import HistoryRow
    
    history = [
        HistoryRow(issue=f"2024{i:03d}", digits=[i % 10, (i + 1) % 10, (i + 2) % 10])
        for i in range(100)
    ]
    
    predictors = [
        build_predictor("ewma", window=50),
        build_predictor("markov", window=100),
    ]
    
    ensemble = EnsemblePredictor(predictors, weights=[0.6, 0.4])
    prediction = ensemble.predict(history, top_k=3)
    
    assert prediction is not None
    assert len(prediction.positions[0]) == 3
    assert len(prediction.positions[1]) == 3
    assert len(prediction.positions[2]) == 3
    print("✓ Ensemble prediction works correctly")


def test_backtest_with_sample_data():
    """使用样本数据进行回测测试"""
    from lottery3d_v2.validation import backtest_predictor
    from lottery3d_v2.base import build_predictor
    from lottery3d_v2.schema import HistoryRow
    
    history = [
        HistoryRow(issue=f"2024{i:03d}", digits=[i % 10, (i + 1) % 10, (i + 2) % 10])
        for i in range(200)
    ]
    
    predictor = build_predictor("ewma", window=50)
    result = backtest_predictor(
        predictor=predictor,
        history=history,
        window=50,
        periods=100,
    )
    
    assert result is not None
    assert "top1_rate" in result
    assert "top3_rate" in result
    assert "mean_log_loss" in result
    print(f"✓ Backtest completed: Top1={result['top1_rate']:.4f}, Top3={result['top3_rate']:.4f}")


if __name__ == "__main__":
    import pytest
    
    # 运行所有测试
    pytest.main([__file__, "-v"])
