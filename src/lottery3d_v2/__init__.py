# -*- coding: utf-8 -*-
"""福彩 3D 智能推荐系统 v2.0"""
from __future__ import annotations

__version__ = "2.0.0"

# 导出核心组件
from .base import BasePredictor, PREDICTOR_REGISTRY, register_predictor, build_predictor
from .config import get_config, ConfigManager
from .config_schema import GlobalConfig, validate_config
from .schema import Prediction, ProbaResult

__all__ = [
    "BasePredictor",
    "PREDICTOR_REGISTRY",
    "register_predictor",
    "build_predictor",
    "get_config",
    "ConfigManager",
    "GlobalConfig",
    "validate_config",
    "Prediction",
    "ProbaResult",
    "__version__",
]
