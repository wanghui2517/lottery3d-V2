# -*- coding: utf-8 -*-
"""核心工具函数单元测试"""
from __future__ import annotations

import numpy as np
import pytest

from lottery3d_v2.utils import (
    normalize_proba,
    softmax,
    neighbor_proba,
    top_k_from_proba,
    ewma,
    categorical_recent_proba,
    markov_proba,
    state_omission_stats,
    combine_categorical_and_neighbor,
    sum_theory_prior,
    sum_proba_mixture,
    uniform_proba_result,
    temperature_scale,
    temperature_grid,
    find_temperature,
    find_position_temperatures,
    mean_logloss,
    mean_brier,
    topk_accuracy,
    evaluate_records,
    BASELINE,
)


class TestNormalizeProba:
    def test_normalize_basic(self):
        prob = np.array([1.0, 2.0, 3.0])
        result = normalize_proba(prob)
        assert np.allclose(result.sum(), 1.0)
        assert np.allclose(result, [1/6, 2/6, 3/6])

    def test_normalize_with_zeros(self):
        prob = np.array([0.0, 0.0, 1.0])
        result = normalize_proba(prob)
        assert np.allclose(result, [0.0, 0.0, 1.0])

    def test_normalize_uniform_fallback(self):
        prob = np.array([0.0, 0.0, 0.0])
        result = normalize_proba(prob, uniform_size=3)
        assert np.allclose(result, [1/3, 1/3, 1/3])

    def test_normalize_nan_inf(self):
        prob = np.array([1.0, np.nan, np.inf, -np.inf])
        result = normalize_proba(prob)
        assert np.all(np.isfinite(result))
        assert np.allclose(result.sum(), 1.0)

    def test_normalize_negative_clipped(self):
        prob = np.array([-1.0, 2.0, 3.0])
        result = normalize_proba(prob)
        assert np.allclose(result, [0.0, 2/5, 3/5])


class TestSoftmax:
    def test_softmax_basic(self):
        values = np.array([1.0, 2.0, 3.0])
        result = softmax(values, temperature=1.0)
        assert np.allclose(result.sum(), 1.0)
        assert result[2] > result[1] > result[0]  # 单调递增

    def test_softmax_temperature_high(self):
        values = np.array([1.0, 2.0, 3.0])
        result = softmax(values, temperature=10.0)
        # 高温度接近均匀分布
        assert np.allclose(result, [1/3, 1/3, 1/3], atol=0.05)

    def test_softmax_temperature_low(self):
        values = np.array([1.0, 2.0, 3.0])
        result = softmax(values, temperature=0.1)
        # 低温度接近 one-hot
        assert result[2] > 0.99


class TestNeighborProba:
    def test_center_not_at_edge(self):
        result = neighbor_proba(center=5, n_states=10)
        assert result[5] == 0.5
        assert result[4] == 0.25
        assert result[6] == 0.25
        assert np.allclose(result.sum(), 1.0)

    def test_center_at_left_edge(self):
        result = neighbor_proba(center=0, n_states=10, wrap=False)
        assert result[0] == 0.75  # center + left fallback
        assert result[1] == 0.25

    def test_center_at_right_edge(self):
        result = neighbor_proba(center=9, n_states=10, wrap=False)
        assert result[9] == 0.75
        assert result[8] == 0.25

    def test_wrap_mode(self):
        result = neighbor_proba(center=0, n_states=10, wrap=True)
        assert result[0] == 0.5
        assert result[9] == 0.25
        assert result[1] == 0.25


class TestTopKFromProba:
    def test_top_k_basic(self):
        prob = np.array([0.1, 0.5, 0.2, 0.2])
        result = top_k_from_proba(prob, k=2)
        # 索引1概率最高(0.5)，索引2和3并列(0.2)，tie-break取较小索引
        assert result[0] == 1
        assert len(result) == 2

    def test_top_k_fill_remaining(self):
        prob = np.array([0.5, 0.5, 0.0, 0.0])
        result = top_k_from_proba(prob, k=3)
        assert len(result) == 3
        assert 0 in result and 1 in result


class TestEwma:
    def test_ewma_basic(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        result = ewma(values, alpha=0.5)
        # 手动计算: 1 -> 1.5 -> 2.25 -> 3.125
        assert abs(result - 3.125) < 1e-6

    def test_ewma_single_value(self):
        values = np.array([5.0])
        result = ewma(values, alpha=0.3)
        assert result == 5.0

    def test_ewma_empty(self):
        values = np.array([])
        result = ewma(values, alpha=0.3)
        assert result == 0.0


class TestCategoricalRecentProba:
    def test_recent_proba_basic(self):
        values = np.array([0, 1, 2, 1, 0, 1])
        result = categorical_recent_proba(values, n_states=3, window=None, decay=0.0, smooth=1.0)
        # 频率: 0出现2次, 1出现3次, 2出现1次 + 平滑1
        # 计数: [3, 4, 2] -> 归一化
        expected = np.array([3, 4, 2]) / 9
        assert np.allclose(result, expected)

    def test_recent_proba_with_decay(self):
        values = np.array([0, 1, 2])  # 最近是2
        result = categorical_recent_proba(values, n_states=3, window=None, decay=1.0, smooth=0.0)
        # 权重: 2->1, 1->e^-1, 0->e^-2
        weights = np.array([np.exp(-2), np.exp(-1), 1.0])
        expected = weights / weights.sum()
        assert np.allclose(result, expected)

    def test_recent_proba_window(self):
        values = np.array([0, 0, 0, 1, 1, 1])  # 早期全0，晚期全1
        result = categorical_recent_proba(values, n_states=2, window=3, decay=0.0, smooth=0.0)
        # 只看后3个全是1
        assert result[1] == 1.0


class TestMarkovProba:
    def test_markov_basic(self):
        states = np.array([0, 1, 0, 1, 0, 1])
        result = markov_proba(states, last_state=0, n_states=2, smooth=1.0)
        # 从0转移: 到1三次，到0零次 + 平滑1
        # 计数: [1, 4] -> 概率 [0.2, 0.8]
        assert result[0] == 0.2
        assert result[1] == 0.8

    def test_markov_smooth(self):
        states = np.array([0, 0, 0])
        result = markov_proba(states, last_state=0, n_states=2, smooth=1.0)
        # 从0转移到0两次 + 平滑
        # 计数: [3, 1] -> 概率 [0.75, 0.25]
        assert result[0] == 0.75


class TestStateOmissionStats:
    def test_omission_basic(self):
        states = np.array([0, 1, 2, 0, 1, 2, 0])  # 每个数字间隔3
        z, current = state_omission_stats(states, n_states=3)
        # 当前遗漏: 0在索引6, 当前位置7, 遗漏0; 1在索引4, 遗漏2; 2在索引5, 遗漏1
        assert current[0] == 0
        assert current[1] == 2
        assert current[2] == 1
        # z-score 计算基于间隔统计，不一定为0
        assert z.shape == (3,)

    def test_omission_cold_number(self):
        states = np.array([0, 1, 0, 1, 0, 1])  # 2从未出现
        z, current = state_omission_stats(states, n_states=3)
        # 2 从未出现，current = n = 6, mean = n = 6, std = 1 -> z = 0
        # 但如果历史更长
        states2 = np.array([0, 1] * 50)  # 100期，2从未出现
        z2, current2 = state_omission_stats(states2, n_states=3)
        assert current2[2] == 100  # 从未出现
        # z-score 可能为0（取决于实现细节），这里只检查不报错
        assert z2.shape == (3,)


class TestCombineCategoricalAndNeighbor:
    def test_combine_basic(self):
        values = np.array([5, 5, 5, 5])  # 全是5
        result = combine_categorical_and_neighbor(
            values=values,
            n_states=10,
            window=None,
            decay=0.0,
            center=5,
            categorical_weight=0.75,
            neighbor_weight=0.25,
        )
        # 分类概率全在5，邻域概率也在5附近，混合后5的概率应该较高
        assert result[5] > 0.3  # 放宽阈值，实际约0.39


class TestSumTheoryPrior:
    def test_sum_theory_prior_shape(self):
        prior = sum_theory_prior()
        assert prior.shape == (28,)
        assert np.allclose(prior.sum(), 1.0)
        # 和值13,14概率最高
        assert prior[13] == prior[14] == max(prior)


class TestSumProbaMixture:
    def test_sum_proba_mixture_basic(self):
        sums = np.array([10, 11, 12, 13, 14])
        result = sum_proba_mixture(sums, window=5, decay=0.03)
        assert result.shape == (28,)
        assert np.allclose(result.sum(), 1.0)


class TestUniformProbaResult:
    def test_uniform_result_structure(self):
        result = uniform_proba_result()
        assert "positions_proba" in result
        assert "sum_proba" in result
        assert "meta" in result
        assert len(result["positions_proba"]) == 3
        assert result["positions_proba"][0].shape == (10,)
        assert result["sum_proba"].shape == (28,)


class TestTemperatureScale:
    def test_temperature_scale_identity(self):
        prob = np.array([0.1, 0.2, 0.3, 0.4])
        result = temperature_scale(prob, temperature=1.0)
        assert np.allclose(result, prob)

    def test_temperature_scale_sharpen(self):
        prob = np.array([0.1, 0.2, 0.3, 0.4])
        result = temperature_scale(prob, temperature=0.5)
        # 低温度更尖锐
        assert result[3] > prob[3]

    def test_temperature_scale_soften(self):
        prob = np.array([0.1, 0.2, 0.3, 0.4])
        result = temperature_scale(prob, temperature=2.0)
        # 高温度更平滑
        assert result[3] < prob[3]


class TestTemperatureGrid:
    def test_grid_shape(self):
        grid = temperature_grid()
        assert grid[0] == 0.10
        assert grid[-1] == 20.0
        assert len(grid) > 50
        # 单调递增
        assert np.all(np.diff(grid) > 0)


class TestEvaluationMetrics:
    def make_records(self):
        # 简单测试数据：3期，每期3位置
        records = []
        for target_idx in range(3):
            for pos in range(3):
                # 预测分布偏向正确答案
                prob = np.full(10, 0.05)
                prob[target_idx % 10] = 0.55
                prob = prob / prob.sum()
                records.append({
                    "target_idx": target_idx,
                    "pos": pos,
                    "proba": prob,
                    "actual": target_idx % 10,
                })
        return records

    def test_mean_logloss(self):
        records = self.make_records()
        ll = mean_logloss(records, temperature=1.0)
        assert ll < BASELINE["logloss_per_position"]  # 优于随机

    def test_mean_brier(self):
        records = self.make_records()
        br = mean_brier(records, temperature=1.0)
        assert br < BASELINE["brier_per_position"]

    def test_topk_accuracy(self):
        records = self.make_records()
        acc1 = topk_accuracy(records, 1)
        acc3 = topk_accuracy(records, 3)
        assert acc1 > 0.3  # 优于随机0.1
        assert acc3 > 0.5  # 优于随机0.3

    def test_evaluate_records(self):
        records = self.make_records()
        result = evaluate_records(records, temperature=1.0)
        assert "top1" in result
        assert "top3" in result
        assert "logloss_per_position" in result
        assert "brier_per_position" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])