# -*- coding: utf-8 -*-
"""预测器基类与注册表"""
from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import combinations
from typing import Any

import numpy as np

from .config import get_config, merge_configs
from .history import ensure_history
from .schema import Prediction, ProbaResult
from .utils import (
    DIGIT_N,
    SUM_N,
    normalize_proba,
    top_k_from_proba,
    uniform_proba_result,
)


class BasePredictor(ABC):
    """所有预测器基类。

    子类只需要实现 _predict_proba。
    配置自动从 configs/predictors/{name}.yaml 加载，可被 kwargs 覆盖。
    """

    name: str = "base"

    def __init__(self, top_k: int = 3, **kwargs):
        self.top_k = int(top_k)
        # 加载默认配置
        self.config = self._load_default_config()
        # 用 kwargs 覆盖（支持运行时调整）
        self.config = merge_configs(self.config, kwargs)
        self.params = kwargs

    def _load_default_config(self) -> dict:
        """从 YAML 加载默认配置"""
        cfg = get_config()
        try:
            return cfg.load_predictor(self.name)
        except FileNotFoundError:
            return {}

    @abstractmethod
    def _predict_proba(self, history: list) -> ProbaResult:
        """返回概率分布结果"""
        raise NotImplementedError

    def predict_proba(self, history) -> ProbaResult:
        history = ensure_history(history)
        return self._predict_proba(history)

    def predict(self, history, **kwargs) -> Prediction:
        top_k = int(kwargs.get("top_k", self.top_k))

        proba_result = self.predict_proba(history)

        positions_proba = [
            normalize_proba(p, uniform_size=DIGIT_N)
            for p in proba_result.positions_proba
        ]

        sum_proba = normalize_proba(
            proba_result.sum_proba,
            uniform_size=SUM_N
        )

        positions = [
            top_k_from_proba(p, k=top_k)
            for p in positions_proba
        ]

        sum_value = top_k_from_proba(sum_proba, k=top_k)

        best_pair = self.choose_best_pair(positions_proba)

        return Prediction(
            name=self.name,
            positions=positions,
            sum_value=sum_value,
            positions_proba=positions_proba,
            sum_proba=sum_proba,
            best_pair=best_pair,
            meta=proba_result.meta,
        )

    @staticmethod
    def choose_best_pair(positions_proba: list[np.ndarray]) -> dict | None:
        if len(positions_proba) != 3:
            return None

        best = None

        for pos_a, pos_b in combinations(range(3), 2):
            score_matrix = np.outer(
                positions_proba[pos_a],
                positions_proba[pos_b]
            )

            da, db = np.unravel_index(
                np.argmax(score_matrix),
                score_matrix.shape
            )

            da = int(da)
            db = int(db)

            score = float(score_matrix[da, db])

            if best is None or score > best["score"]:
                best = {
                    "pos_a": pos_a,
                    "pos_b": pos_b,
                    "digit_a": da,
                    "digit_b": db,
                    "score": score,
                }

        return best


# 预测器注册表
PREDICTOR_REGISTRY: dict[str, type[BasePredictor]] = {}


def register_predictor(cls: type[BasePredictor]) -> type[BasePredictor]:
    """装饰器：注册预测器类"""
    PREDICTOR_REGISTRY[cls.name] = cls
    return cls


def build_predictor(name: str, **kwargs) -> BasePredictor:
    """按名称构建预测器实例"""
    if name not in PREDICTOR_REGISTRY:
        raise ValueError(
            f"未知预测器: {name}. 可用: {sorted(PREDICTOR_REGISTRY.keys())}"
        )
    return PREDICTOR_REGISTRY[name](**kwargs)


def build_all_predictors() -> list[BasePredictor]:
    """构建所有已注册预测器的默认实例"""
    return [cls() for cls in PREDICTOR_REGISTRY.values()]


__all__ = [
    "BasePredictor",
    "PREDICTOR_REGISTRY",
    "register_predictor",
    "build_predictor",
    "build_all_predictors",
]