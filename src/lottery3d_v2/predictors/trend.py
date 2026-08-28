# -*- coding: utf-8 -*-
"""趋势类预测器：EWMA、振幅、马尔可夫"""
from __future__ import annotations

import numpy as np

from ..base import BasePredictor, register_predictor
from ..history import digits_matrix
from ..utils import (
    DIGIT_N,
    SUM_N,
    combine_categorical_and_neighbor,
    ewma,
    markov_proba,
    sum_proba_mixture,
    uniform_proba_result,
)
from ..schema import ProbaResult


@register_predictor
class EWMAPredictor(BasePredictor):
    name = "ewma"

    def __init__(
        self,
        alpha: float = 0.3,
        window: int = 100,
        top_k: int = 3,
        **kwargs
    ):
        super().__init__(top_k=top_k, **kwargs)
        # 从配置加载默认值（super().__init__ 后可用 self.config）
        cfg = self.config
        alpha = float(cfg.get("alpha", alpha))
        window = int(cfg.get("window", window))
        categorical_weight = float(cfg.get("categorical_weight", 0.75))
        neighbor_weight = float(cfg.get("neighbor_weight", 0.25))
        sum_theory_weight = float(cfg.get("sum_theory_weight", 0.25))
        sum_recent_weight = float(cfg.get("sum_recent_weight", 0.55))
        sum_neighbor_weight = float(cfg.get("sum_neighbor_weight", 0.20))

        self.alpha = alpha
        self.window = window
        self.categorical_weight = categorical_weight
        self.neighbor_weight = neighbor_weight
        self.sum_theory_weight = sum_theory_weight
        self.sum_recent_weight = sum_recent_weight
        self.sum_neighbor_weight = sum_neighbor_weight

    def _predict_proba(self, history: list) -> ProbaResult:
        X = digits_matrix(history)

        if X.shape[0] == 0:
            return ProbaResult(**uniform_proba_result())

        positions_proba = []

        for pos in range(3):
            values = X[:, pos]

            center = int(round(ewma(values[-self.window:], self.alpha)))
            center = int(np.clip(center, 0, DIGIT_N - 1))

            prob = combine_categorical_and_neighbor(
                values=values,
                n_states=DIGIT_N,
                window=self.window,
                decay=self.alpha,
                center=center,
                wrap=False,
                categorical_weight=self.categorical_weight,
                neighbor_weight=self.neighbor_weight,
            )

            positions_proba.append(prob)

        sums = X.sum(axis=1)

        sum_center = int(round(ewma(sums[-self.window:], self.alpha)))
        sum_center = int(np.clip(sum_center, 0, SUM_N - 1))

        sum_proba = sum_proba_mixture(
            sums=sums,
            window=self.window,
            decay=self.alpha,
            theory_weight=self.sum_theory_weight,
            recent_weight=self.sum_recent_weight,
            neighbor_weight=self.sum_neighbor_weight,
            center=sum_center,
        )

        return ProbaResult(
            positions_proba=positions_proba,
            sum_proba=sum_proba,
        )


@register_predictor
class AmplitudePredictor(BasePredictor):
    name = "amplitude"

    def __init__(
        self,
        alpha: float = 0.3,
        window: int = 100,
        top_k: int = 3,
        **kwargs
    ):
        super().__init__(top_k=top_k, **kwargs)
        # 从配置加载默认值（super().__init__ 后可用 self.config）
        cfg = self.config
        alpha = float(cfg.get("alpha", alpha))
        window = int(cfg.get("window", window))
        categorical_weight = float(cfg.get("categorical_weight", 0.70))
        neighbor_weight = float(cfg.get("neighbor_weight", 0.30))
        sum_theory_weight = float(cfg.get("sum_theory_weight", 0.25))
        sum_recent_weight = float(cfg.get("sum_recent_weight", 0.55))
        sum_neighbor_weight = float(cfg.get("sum_neighbor_weight", 0.20))

        self.alpha = alpha
        self.window = window
        self.categorical_weight = categorical_weight
        self.neighbor_weight = neighbor_weight
        self.sum_theory_weight = sum_theory_weight
        self.sum_recent_weight = sum_recent_weight
        self.sum_neighbor_weight = sum_neighbor_weight

    def _predict_proba(self, history: list) -> ProbaResult:
        X = digits_matrix(history)

        if X.shape[0] == 0:
            return ProbaResult(**uniform_proba_result())

        positions_proba = []

        for pos in range(3):
            if X.shape[0] < 2:
                positions_proba.append(np.full(DIGIT_N, 1.0 / DIGIT_N))
                continue

            values = X[:, pos]
            amps = np.abs(np.diff(values))

            amp_center = int(round(ewma(amps[-self.window:], self.alpha)))
            amp_center = int(np.clip(amp_center, 0, DIGIT_N - 1))

            amp_proba = combine_categorical_and_neighbor(
                values=amps,
                n_states=DIGIT_N,
                window=self.window,
                decay=self.alpha,
                center=amp_center,
                wrap=False,
                categorical_weight=self.categorical_weight,
                neighbor_weight=self.neighbor_weight,
            )

            last_digit = int(values[-1])

            digit_proba = np.ones(DIGIT_N, dtype=float) * 1e-3

            for amp, prob in enumerate(amp_proba):
                if prob <= 1e-12:
                    continue

                d1 = (last_digit + amp) % DIGIT_N
                d2 = (last_digit - amp) % DIGIT_N

                digit_proba[d1] += float(prob)

                if d2 != d1:
                    digit_proba[d2] += float(prob)

            positions_proba.append(normalize_proba(digit_proba))

        sums = X.sum(axis=1)

        sum_center = int(round(ewma(sums[-self.window:], self.alpha)))
        sum_center = int(np.clip(sum_center, 0, SUM_N - 1))

        sum_proba = sum_proba_mixture(
            sums=sums,
            window=self.window,
            decay=self.alpha,
            theory_weight=self.sum_theory_weight,
            recent_weight=self.sum_recent_weight,
            neighbor_weight=self.sum_neighbor_weight,
            center=sum_center,
        )

        return ProbaResult(
            positions_proba=positions_proba,
            sum_proba=sum_proba,
        )


@register_predictor
class MarkovPredictor(BasePredictor):
    name = "markov"

    def __init__(
        self,
        smooth: float = 1.0,
        window: int | None = None,
        top_k: int = 3,
        **kwargs
    ):
        super().__init__(top_k=top_k, **kwargs)
        # 从配置加载默认值（super().__init__ 后可用 self.config）
        cfg = self.config
        smooth = float(cfg.get("smooth", smooth))
        window = cfg.get("window", window)
        if window is not None:
            window = int(window)
        sum_smooth_factor = float(cfg.get("sum_smooth_factor", 0.5))
        sum_theory_weight = float(cfg.get("sum_theory_weight", 0.30))
        sum_markov_weight = float(cfg.get("sum_markov_weight", 0.70))

        self.smooth = smooth
        self.window = window
        self.sum_smooth_factor = sum_smooth_factor
        self.sum_theory_weight = sum_theory_weight
        self.sum_markov_weight = sum_markov_weight

    def _predict_proba(self, history: list) -> ProbaResult:
        X = digits_matrix(history)

        if self.window is not None and X.shape[0] > self.window:
            X = X[-self.window:]

        if X.shape[0] == 0:
            return ProbaResult(**uniform_proba_result())

        positions_proba = []

        for pos in range(3):
            states = X[:, pos]
            last_state = int(states[-1])

            prob = markov_proba(
                states=states,
                last_state=last_state,
                n_states=DIGIT_N,
                smooth=self.smooth,
            )

            positions_proba.append(prob)

        sums = X.sum(axis=1)
        last_sum = int(sums[-1])

        sum_markov = markov_proba(
            states=sums,
            last_state=last_sum,
            n_states=SUM_N,
            smooth=max(0.25, self.smooth * self.sum_smooth_factor),
        )

        from ..utils import sum_theory_prior
        theory = sum_theory_prior()

        sum_proba = normalize_proba(
            self.sum_markov_weight * sum_markov
            + self.sum_theory_weight * theory
        )

        return ProbaResult(
            positions_proba=positions_proba,
            sum_proba=sum_proba,
        )


# 导入 normalize_proba 用于 MarkovPredictor
from ..utils import normalize_proba