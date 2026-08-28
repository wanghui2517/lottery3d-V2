# -*- coding: utf-8 -*-
"""形态学预测器：奇偶、质数、AC值、跨度、012路"""
from __future__ import annotations

import numpy as np

from ..base import BasePredictor, register_predictor
from ..features import (
    PRIME_DIGITS,
    ac_value,
    combos_by_ac,
    combos_by_parity,
    combos_by_prime,
    combos_by_road_pattern,
    combos_by_span,
    mixture_by_category,
    position_proba_from_combos,
)
from ..history import digits_matrix
from ..utils import (
    DIGIT_N,
    SUM_N,
    categorical_recent_proba,
    combine_categorical_and_neighbor,
    ewma,
    normalize_proba,
    softmax,
    state_omission_stats,
    sum_proba_mixture,
    uniform_proba_result,
)
from ..schema import ProbaResult


class CategoryMixturePredictor(BasePredictor):
    """形态类预测器通用基类。

    子类需要提供：
        n_categories
        combos_func
        category_values()
    """

    n_categories: int = 4
    combos_func = None

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

    def category_values(
        self,
        X: np.ndarray,
        history: list
    ) -> np.ndarray:
        raise NotImplementedError

    def _category_proba(self, category_values: np.ndarray) -> np.ndarray:
        center = int(round(ewma(category_values[-self.window:], self.alpha)))
        center = int(np.clip(center, 0, self.n_categories - 1))

        return combine_categorical_and_neighbor(
            values=category_values,
            n_states=self.n_categories,
            window=self.window,
            decay=self.alpha,
            center=center,
            wrap=False,
            categorical_weight=self.categorical_weight,
            neighbor_weight=self.neighbor_weight,
        )

    def _predict_proba(self, history: list) -> ProbaResult:
        X = digits_matrix(history)

        if X.shape[0] == 0:
            return ProbaResult(**uniform_proba_result())

        category_values = self.category_values(X, history)
        category_proba = self._category_proba(category_values)

        combos_func = type(self).combos_func

        if combos_func is None:
            raise NotImplementedError("combos_func 未定义")

        positions_proba = mixture_by_category(
            category_proba=category_proba,
            combos_func=combos_func,
            n_states=DIGIT_N,
        )

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
            meta={"category_proba": category_proba.tolist()},
        )


@register_predictor
class ParityPredictor(CategoryMixturePredictor):
    name = "parity"
    n_categories = 4
    combos_func = staticmethod(combos_by_parity)

    def category_values(
        self,
        X: np.ndarray,
        history: list
    ) -> np.ndarray:
        return np.array(
            [int((row % 2).sum()) for row in X],
            dtype=int,
        )


@register_predictor
class PrimePredictor(CategoryMixturePredictor):
    name = "prime"
    n_categories = 4
    combos_func = staticmethod(combos_by_prime)

    def category_values(
        self,
        X: np.ndarray,
        history: list
    ) -> np.ndarray:
        return np.array(
            [
                sum(1 for d in row if d in PRIME_DIGITS)
                for row in X
            ],
            dtype=int,
        )


@register_predictor
class ACValuePredictor(CategoryMixturePredictor):
    name = "ac_value"
    n_categories = 4
    combos_func = staticmethod(combos_by_ac)

    def category_values(
        self,
        X: np.ndarray,
        history: list
    ) -> np.ndarray:
        return np.array(
            [ac_value(tuple(row)) for row in X],
            dtype=int,
        )


@register_predictor
class SpanPredictor(CategoryMixturePredictor):
    name = "span"
    n_categories = 10
    combos_func = staticmethod(combos_by_span)

    def __init__(
        self,
        alpha: float = 0.3,
        window: int = 100,
        method: str = "recent",
        temperature: float = 2.0,
        top_k: int = 3,
        **kwargs
    ):
        super().__init__(
            alpha=alpha,
            window=window,
            top_k=top_k,
            **kwargs
        )
        # 从配置加载默认值（super().__init__ 后可用 self.config）
        cfg = self.config
        method = str(cfg.get("method", method)).lower()
        temperature = float(cfg.get("temperature", temperature))
        categorical_weight = float(cfg.get("categorical_weight", 0.75))
        neighbor_weight = float(cfg.get("neighbor_weight", 0.25))
        zscore_cold_weight = float(cfg.get("zscore_cold_weight", 0.60))
        zscore_recent_weight = float(cfg.get("zscore_recent_weight", 0.40))
        sum_theory_weight = float(cfg.get("sum_theory_weight", 0.25))
        sum_recent_weight = float(cfg.get("sum_recent_weight", 0.55))
        sum_neighbor_weight = float(cfg.get("sum_neighbor_weight", 0.20))

        self.method = method
        self.temperature = temperature
        self.categorical_weight = categorical_weight
        self.neighbor_weight = neighbor_weight
        self.zscore_cold_weight = zscore_cold_weight
        self.zscore_recent_weight = zscore_recent_weight
        self.sum_theory_weight = sum_theory_weight
        self.sum_recent_weight = sum_recent_weight
        self.sum_neighbor_weight = sum_neighbor_weight

    def category_values(
        self,
        X: np.ndarray,
        history: list
    ) -> np.ndarray:
        return np.array(
            [int(row.max() - row.min()) for row in X],
            dtype=int,
        )

    def _category_proba(self, category_values: np.ndarray) -> np.ndarray:
        if self.method == "zscore":
            z, _ = state_omission_stats(
                category_values,
                self.n_categories
            )

            cold = softmax(z, temperature=self.temperature)

            recent = categorical_recent_proba(
                category_values,
                n_states=self.n_categories,
                window=self.window,
                decay=0.03,
                smooth=1.0,
            )

            return normalize_proba(
                self.zscore_cold_weight * cold
                + self.zscore_recent_weight * recent
            )

        return super()._category_proba(category_values)


@register_predictor
class Road012Predictor(BasePredictor):
    name = "road012"

    def __init__(
        self,
        alpha: float = 0.3,
        window: int = 100,
        roads_per_position: int = 2,
        top_k: int = 3,
        **kwargs
    ):
        super().__init__(top_k=top_k, **kwargs)
        # 从配置加载默认值（super().__init__ 后可用 self.config）
        cfg = self.config
        alpha = float(cfg.get("alpha", alpha))
        window = int(cfg.get("window", window))
        roads_per_position = int(cfg.get("roads_per_position", roads_per_position))
        roads_per_position = int(np.clip(roads_per_position, 1, 3))
        categorical_weight = float(cfg.get("categorical_weight", 0.80))
        neighbor_weight = float(cfg.get("neighbor_weight", 0.20))
        sum_theory_weight = float(cfg.get("sum_theory_weight", 0.25))
        sum_recent_weight = float(cfg.get("sum_recent_weight", 0.55))
        sum_neighbor_weight = float(cfg.get("sum_neighbor_weight", 0.20))

        self.alpha = alpha
        self.window = window
        self.roads_per_position = roads_per_position
        self.categorical_weight = categorical_weight
        self.neighbor_weight = neighbor_weight
        self.sum_theory_weight = sum_theory_weight
        self.sum_recent_weight = sum_recent_weight
        self.sum_neighbor_weight = sum_neighbor_weight

    def _predict_proba(self, history: list) -> ProbaResult:
        X = digits_matrix(history)

        if X.shape[0] == 0:
            return ProbaResult(**uniform_proba_result())

        roads = X % 3

        road_probas = []
        selected_roads = []

        for pos in range(3):
            values = roads[:, pos]

            center = int(round(ewma(values[-self.window:], self.alpha)))
            center = int(np.clip(center, 0, 2))

            road_proba = combine_categorical_and_neighbor(
                values=values,
                n_states=3,
                window=self.window,
                decay=self.alpha,
                center=center,
                wrap=False,
                categorical_weight=self.categorical_weight,
                neighbor_weight=self.neighbor_weight,
            )

            road_probas.append(road_proba)

            top_roads = np.argsort(road_proba)[::-1][:self.roads_per_position]
            selected_roads.append([int(r) for r in top_roads])

        pos_counts = [np.ones(DIGIT_N, dtype=float) for _ in range(3)]

        for r0 in selected_roads[0]:
            for r1 in selected_roads[1]:
                for r2 in selected_roads[2]:
                    pattern_prob = (
                        road_probas[0][r0]
                        * road_probas[1][r1]
                        * road_probas[2][r2]
                    )

                    if pattern_prob <= 1e-12:
                        continue

                    combos = combos_by_road_pattern(r0, r1, r2)

                    if not combos:
                        continue

                    freqs = position_proba_from_combos(combos)

                    for pos in range(3):
                        pos_counts[pos] += float(pattern_prob) * freqs[pos]

        positions_proba = [normalize_proba(x) for x in pos_counts]

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
            meta={
                "road_probas": [p.tolist() for p in road_probas],
                "selected_roads": selected_roads,
            },
        )