# -*- coding: utf-8 -*-
"""市场/赔付分析预测器"""
from __future__ import annotations

import numpy as np

from ..base import BasePredictor, register_predictor
from ..history import ensure_history
from ..schema import ProbaResult


class PayoutMarketAnalyzer:
    """市场赔付状态分析器。

    注意：
        它不预测开奖号码。
        只判断近期赔付是否偏高、偏低或中性。
    """

    def __init__(
        self,
        window: int = 20,
        hot_threshold: float = 0.5,
        cold_threshold: float = -0.5,
    ):
        self.window = int(window)
        self.hot_threshold = float(hot_threshold)
        self.cold_threshold = float(cold_threshold)

    def analyze(self, history) -> dict:
        history = ensure_history(history)

        if len(history) < max(5, self.window // 2):
            return {
                "bias": 0.0,
                "zscore": 0.0,
                "direction": "neutral",
                "stake_multiplier": 1.0,
            }

        recent = history[-self.window:]

        ratios = []

        for row in recent:
            if row.sales is not None and row.sales > 0 and row.prize is not None:
                ratios.append(row.prize / row.sales)

        if len(ratios) < 5:
            return {
                "bias": 0.0,
                "zscore": 0.0,
                "direction": "neutral",
                "stake_multiplier": 1.0,
            }

        arr = np.asarray(ratios, dtype=float)

        mean = float(arr.mean())
        std = float(arr.std())

        latest = float(arr[-1])

        if std < 1e-9:
            z = 0.0
        else:
            z = (latest - mean) / std

        if z > self.hot_threshold:
            direction = "hot"
        elif z < self.cold_threshold:
            direction = "cold"
        else:
            direction = "neutral"

        stake_multiplier = float(np.clip(1.0 - 0.2 * z, 0.5, 1.5))

        return {
            "bias": float(latest - mean),
            "zscore": float(z),
            "direction": direction,
            "stake_multiplier": stake_multiplier,
        }


@register_predictor
class PayoutAdjustedPredictor(BasePredictor):
    """包装一个号码预测器，并附加市场状态信息。

    默认不修改号码概率。
    """

    name = "payout_adjusted"

    def __init__(
        self,
        base_predictor: BasePredictor,
        analyzer: PayoutMarketAnalyzer | None = None,
        top_k: int = 3,
    ):
        super().__init__(top_k=top_k)
        self.base_predictor = base_predictor
        self.analyzer = analyzer or PayoutMarketAnalyzer()

    def _predict_proba(self, history: list) -> ProbaResult:
        result = self.base_predictor._predict_proba(history)

        meta = dict(result.meta)
        meta["market"] = self.analyzer.analyze(history)
        meta["base_predictor"] = self.base_predictor.name

        return ProbaResult(
            positions_proba=result.positions_proba,
            sum_proba=result.sum_proba,
            meta=meta,
        )