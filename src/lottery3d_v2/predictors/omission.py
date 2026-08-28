# -*- coding: utf-8 -*-
"""遗漏/统计类预测器：Z-Score"""
from __future__ import annotations

import numpy as np

from ..base import BasePredictor, register_predictor
from ..history import digits_matrix
from ..utils import (
    DIGIT_N,
    SUM_N,
    categorical_recent_proba,
    normalize_proba,
    softmax,
    state_omission_stats,
    sum_proba_mixture,
    sum_theory_prior,
    uniform_proba_result,
)
from ..schema import ProbaResult


@register_predictor
class ZScorePredictor(BasePredictor):
    name = "zscore"

    def __init__(
        self,
        threshold: float = 2.0,
        temperature: float = 2.0,
        window: int = 200,
        top_k: int = 3,
        **kwargs
    ):
        super().__init__(top_k=top_k, **kwargs)
        # 从配置加载默认值（super().__init__ 后可用 self.config）
        cfg = self.config
        threshold = float(cfg.get("threshold", threshold))
        temperature = float(cfg.get("temperature", temperature))
        window = int(cfg.get("window", window))
        cold_weight = float(cfg.get("cold_weight", 0.65))
        recent_weight = float(cfg.get("recent_weight", 0.25))
        uniform_weight = float(cfg.get("uniform_weight", 0.10))
        sum_cold_weight = float(cfg.get("sum_cold_weight", 0.45))
        sum_recent_weight = float(cfg.get("sum_recent_weight", 0.30))
        sum_theory_weight = float(cfg.get("sum_theory_weight", 0.25))
        recent_decay = float(cfg.get("recent_decay", 0.03))
        sum_recent_decay = float(cfg.get("sum_recent_decay", 0.03))
        sum_recent_smooth = float(cfg.get("sum_recent_smooth", 0.5))

        self.threshold = threshold
        self.temperature = temperature
        self.window = window
        self.cold_weight = cold_weight
        self.recent_weight = recent_weight
        self.uniform_weight = uniform_weight
        self.sum_cold_weight = sum_cold_weight
        self.sum_recent_weight = sum_recent_weight
        self.sum_theory_weight = sum_theory_weight
        self.recent_decay = recent_decay
        self.sum_recent_decay = sum_recent_decay
        self.sum_recent_smooth = sum_recent_smooth

    def _predict_proba(self, history: list) -> ProbaResult:
        X = digits_matrix(history)

        if X.shape[0] == 0:
            return ProbaResult(**uniform_proba_result())

        positions_proba = []

        for pos in range(3):
            states = X[:, pos]

            z, _ = state_omission_stats(states, DIGIT_N)

            cold_proba = softmax(z, temperature=self.temperature)

            recent_proba = categorical_recent_proba(
                states,
                n_states=DIGIT_N,
                window=self.window,
                decay=self.recent_decay,
                smooth=1.0,
            )

            uniform = np.full(DIGIT_N, 1.0 / DIGIT_N)

            prob = normalize_proba(
                self.cold_weight * cold_proba
                + self.recent_weight * recent_proba
                + self.uniform_weight * uniform
            )

            positions_proba.append(prob)

        sums = X.sum(axis=1)

        sum_z, _ = state_omission_stats(sums, SUM_N)

        sum_cold = softmax(sum_z, temperature=self.temperature)

        sum_recent = categorical_recent_proba(
            sums,
            n_states=SUM_N,
            window=self.window,
            decay=self.sum_recent_decay,
            smooth=self.sum_recent_smooth,
        )

        theory = sum_theory_prior()

        sum_proba = normalize_proba(
            self.sum_cold_weight * sum_cold
            + self.sum_recent_weight * sum_recent
            + self.sum_theory_weight * theory
        )

        return ProbaResult(
            positions_proba=positions_proba,
            sum_proba=sum_proba,
        )