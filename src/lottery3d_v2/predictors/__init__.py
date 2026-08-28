# -*- coding: utf-8 -*-
"""预测器包：自动注册所有预测器"""
from __future__ import annotations

# 导入触发 @register_predictor 装饰器
from . import trend, morphology, omission, market  # noqa: F401

from .trend import EWMAPredictor, AmplitudePredictor, MarkovPredictor
from .morphology import (
    ACValuePredictor,
    ParityPredictor,
    PrimePredictor,
    Road012Predictor,
    SpanPredictor,
)
from .omission import ZScorePredictor
from .market import PayoutMarketAnalyzer, PayoutAdjustedPredictor

from ..ensemble import EnsemblePredictor
from ..base import (
    PREDICTOR_REGISTRY,
    build_predictor,
    build_all_predictors,
)

__all__ = [
    "EWMAPredictor",
    "AmplitudePredictor",
    "MarkovPredictor",
    "ACValuePredictor",
    "ParityPredictor",
    "PrimePredictor",
    "Road012Predictor",
    "SpanPredictor",
    "ZScorePredictor",
    "PayoutMarketAnalyzer",
    "PayoutAdjustedPredictor",
    "EnsemblePredictor",
    "PREDICTOR_REGISTRY",
    "build_predictor",
    "build_all_predictors",
]