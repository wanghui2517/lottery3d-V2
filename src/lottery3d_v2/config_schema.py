# -*- coding: utf-8 -*-
"""Pydantic 配置模型 - 提供类型验证和自动文档"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, PositiveInt, confloat, validator
from pathlib import Path


class PredictorConfig(BaseModel):
    """预测器基础配置"""
    window: PositiveInt = Field(default=100, description="历史窗口大小")
    alpha: confloat(ge=0.0, le=1.0) = Field(default=0.3, description="EWMA 平滑系数")
    top_k: PositiveInt = Field(default=3, description="Top-K 候选数量")
    
    # 形态类权重
    categorical_weight: confloat(ge=0.0, le=1.0) = Field(default=0.75)
    neighbor_weight: confloat(ge=0.0, le=1.0) = Field(default=0.25)
    
    # 和值权重
    sum_theory_weight: confloat(ge=0.0, le=1.0) = Field(default=0.25)
    sum_recent_weight: confloat(ge=0.0, le=1.0) = Field(default=0.55)
    sum_neighbor_weight: confloat(ge=0.0, le=1.0) = Field(default=0.20)
    
    class Config:
        extra = "allow"  # 允许额外字段
        validate_assignment = True


class MarkovConfig(PredictorConfig):
    """马尔可夫预测器配置"""
    order: PositiveInt = Field(default=1, description="马尔可夫阶数")
    smoothing: confloat(ge=0.0, le=1.0) = Field(default=0.1, description="拉普拉斯平滑系数")


class OmissionConfig(PredictorConfig):
    """遗漏分析预测器配置"""
    max_omission: PositiveInt = Field(default=50, description="最大遗漏期数")
    weight_recent: confloat(ge=0.0, le=1.0) = Field(default=0.6, description="近期权重")


class TrendConfig(PredictorConfig):
    """趋势分析预测器配置"""
    trend_window: PositiveInt = Field(default=20, description="趋势窗口")
    momentum_factor: confloat(ge=0.0, le=1.0) = Field(default=0.5, description="动量因子")


class MorphologyConfig(PredictorConfig):
    """形态学预测器配置"""
    ac_threshold: PositiveInt = Field(default=5, description="AC 值阈值")
    span_range: List[int] = Field(default=[3, 9], description="跨度范围")


class EnsembleConfig(BaseModel):
    """集成模型配置"""
    name: str = Field(default="full", description="集成模型名称")
    predictors: List[str] = Field(default=[], description="预测器列表")
    weights: Dict[str, float] = Field(default={}, description="预测器权重")
    top_k: PositiveInt = Field(default=3)
    
    @validator('weights')
    def validate_weights(cls, v):
        if v:
            total = sum(v.values())
            if abs(total - 1.0) > 0.01:
                raise ValueError(f"权重总和必须为 1.0，当前为 {total}")
        return v


class CalibrationConfig(BaseModel):
    """温度校准配置"""
    method: str = Field(default="temperature", description="校准方法")
    n_samples: PositiveInt = Field(default=500, description="校准样本数")
    temperature_range: List[float] = Field(default=[0.5, 2.0], description="温度范围")
    cv_folds: PositiveInt = Field(default=5, description="交叉验证折数")


class BacktestConfig(BaseModel):
    """回测配置"""
    train_window: Optional[PositiveInt] = Field(default=None, description="训练窗口")
    test_window: PositiveInt = Field(default=50, description="测试窗口")
    step: PositiveInt = Field(default=1, description="滑动步长")
    metrics: List[str] = Field(default=["logloss", "brier", "top1_hit", "top3_hit"], 
                               description="评估指标")


class PathConfig(BaseModel):
    """路径配置"""
    history_csv: Path = Field(default=Path("data/3d_full_history.csv"))
    results_dir: Path = Field(default=Path("results"))
    calibration_dir: Path = Field(default=Path("results/calibration"))
    log_file: Path = Field(default=Path("results/forward_log.csv"))
    
    @validator('history_csv', 'results_dir', 'calibration_dir', 'log_file', pre=True)
    def resolve_paths(cls, v):
        if isinstance(v, str):
            return Path(v)
        return v


class GlobalConfig(BaseModel):
    """全局配置根模型"""
    version: str = Field(default="2.0.0")
    random_seed: Optional[int] = Field(default=42, description="随机种子")
    
    predictors: Dict[str, PredictorConfig] = Field(default_factory=dict)
    ensemble: Dict[str, EnsembleConfig] = Field(default_factory=dict)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    
    class Config:
        arbitrary_types_allowed = True


def validate_config(config_dict: Dict[str, Any]) -> GlobalConfig:
    """验证并转换配置字典为 Pydantic 模型
    
    Args:
        config_dict: 从 YAML 加载的配置字典
        
    Returns:
        GlobalConfig: 验证后的配置模型
        
    Raises:
        ValidationError: 当配置不合法时
    """
    return GlobalConfig(**config_dict)


__all__ = [
    "PredictorConfig",
    "MarkovConfig",
    "OmissionConfig",
    "TrendConfig",
    "MorphologyConfig",
    "EnsembleConfig",
    "CalibrationConfig",
    "BacktestConfig",
    "PathConfig",
    "GlobalConfig",
    "validate_config",
]
