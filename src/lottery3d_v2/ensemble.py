# -*- coding: utf-8 -*-
"""集成预测器：加权融合 + 动态权重支持"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from loguru import logger

from .base import BasePredictor, build_predictor
from .config import get_config
from .schema import Prediction, ProbaResult
from .utils import DIGIT_N, SUM_N, normalize_proba, uniform_proba_result


class EnsemblePredictor(BasePredictor):
    name = "ensemble"

    def __init__(
        self,
        predictors: list[BasePredictor] | None = None,
        weights: list[float] | np.ndarray | None = None,
        top_k: int = 3,
        ensemble_name: str = "full",
        auto_weight: bool = False,
        weight_update_window: int = 50,
    ):
        super().__init__(top_k=top_k)

        # 如果未提供预测器，从配置构建
        if predictors is None:
            cfg = get_config().load_ensemble(ensemble_name)
            predictor_configs = cfg.get("predictors", [])
            predictors = []
            weights = []

            for pc in predictor_configs:
                pred_name = pc["name"]
                weight = pc["weight"]
                config_override = pc.get("config_override", {})
                pred = build_predictor(pred_name, **config_override)
                predictors.append(pred)
                weights.append(weight)

        self.predictors = list(predictors)

        if weights is None:
            weights = np.ones(len(self.predictors), dtype=float)
        else:
            weights = np.asarray(weights, dtype=float)

        if weights.size != len(self.predictors):
            weights = np.ones(len(self.predictors), dtype=float)

        self.weights = normalize_proba(weights)
        self.ensemble_name = ensemble_name
        self.auto_weight = auto_weight
        self.weight_update_window = weight_update_window
        self._weight_history: list[np.ndarray] = []

    def _predict_proba(self, history: list) -> ProbaResult:
        if not self.predictors:
            return ProbaResult(**uniform_proba_result())

        pos_accum = [np.zeros(DIGIT_N, dtype=float) for _ in range(3)]
        sum_accum = np.zeros(SUM_N, dtype=float)

        used = []

        for weight, predictor in zip(self.weights, self.predictors):
            try:
                result = predictor._predict_proba(history)
            except (ValueError, RuntimeError, KeyError, IndexError) as e:
                logger.warning(
                    f"Predictor {predictor.name} failed: {type(e).__name__}: {e}"
                )
                continue

            positions_proba = result.positions_proba
            sum_proba = result.sum_proba

            if positions_proba is None or sum_proba is None:
                continue

            if len(positions_proba) != 3:
                continue

            for pos in range(3):
                pos_accum[pos] += float(weight) * normalize_proba(
                    positions_proba[pos],
                    uniform_size=DIGIT_N,
                )

            sum_accum += float(weight) * normalize_proba(
                sum_proba,
                uniform_size=SUM_N,
            )

            used.append(predictor.name)

        if not used:
            return ProbaResult(**uniform_proba_result())

        return ProbaResult(
            positions_proba=[normalize_proba(x) for x in pos_accum],
            sum_proba=normalize_proba(sum_accum),
            meta={"members": used},
        )

    def update_weights_from_performance(
        self,
        history: list,
        window: int | None = None,
    ) -> np.ndarray:
        """基于近期回测表现动态调整权重（简单版：LogLoss 反比加权）"""
        if not self.auto_weight or len(self.predictors) < 2:
            return self.weights

        from .validation import backtest_predictor

        window = window or self.weight_update_window
        performances = []

        for pred in self.predictors:
            result = backtest_predictor(
                predictor=pred,
                history=history,
                window=min(50, len(history) // 2),
                periods=window,
                train_window=None,
            )
            if result:
                # 使用 LogLoss 倒数作为权重（越小越好）
                ll = result.get("mean_log_loss", float("inf"))
                performances.append(1.0 / (ll + 1e-6))
            else:
                performances.append(1e-6)

        perf_array = np.array(performances, dtype=float)
        new_weights = normalize_proba(perf_array)

        # 平滑更新：新旧权重混合
        self.weights = normalize_proba(0.7 * self.weights + 0.3 * new_weights)
        self._weight_history.append(self.weights.copy())

        return self.weights

    def save_weights(self, path: str | Path) -> None:
        """保存当前权重到文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "ensemble_name": self.ensemble_name,
            "weights": self.weights.tolist(),
            "members": [p.name for p in self.predictors],
            "weight_history": [w.tolist() for w in self._weight_history],
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True)

    @classmethod
    def load_weights(cls, path: str | Path, **kwargs) -> "EnsemblePredictor":
        """从文件加载权重"""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        ensemble = cls(ensemble_name=data.get("ensemble_name", "full"), **kwargs)
        ensemble.weights = np.array(data["weights"], dtype=float)
        ensemble._weight_history = [np.array(w) for w in data.get("weight_history", [])]
        return ensemble