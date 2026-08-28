# -*- coding: utf-8 -*-
"""历史数据与预测器单元测试"""
from __future__ import annotations

import numpy as np
import pytest

from lottery3d_v2.history import (
    parse_digits,
    to_float,
    ensure_history,
    load_history_from_csv,
    digits_matrix,
)
from lottery3d_v2.schema import HistoryRow, Prediction
from lottery3d_v2.predictors import (
    PREDICTOR_REGISTRY,
    build_predictor,
    EWMAPredictor,
    AmplitudePredictor,
    MarkovPredictor,
    ZScorePredictor,
    ParityPredictor,
    PrimePredictor,
    ACValuePredictor,
    SpanPredictor,
    Road012Predictor,
)
from lottery3d_v2.ensemble import EnsemblePredictor
from lottery3d_v2.utils import DIGIT_N, SUM_N


class TestParseDigits:
    def test_parse_string_with_comma(self):
        result = parse_digits("4,1,5")
        assert result == (4, 1, 5)

    def test_parse_string_with_spaces(self):
        result = parse_digits("4 1 5")
        assert result == (4, 1, 5)

    def test_parse_tuple(self):
        result = parse_digits((4, 1, 5))
        assert result == (4, 1, 5)

    def test_parse_list(self):
        result = parse_digits([4, 1, 5])
        assert result == (4, 1, 5)

    def test_parse_3digit_string(self):
        result = parse_digits("415")
        assert result == (4, 1, 5)

    def test_parse_with_quotes(self):
        result = parse_digits('"4,1,5"')
        assert result == (4, 1, 5)

    def test_parse_invalid(self):
        with pytest.raises(ValueError):
            parse_digits("invalid")

    def test_parse_none(self):
        with pytest.raises(ValueError):
            parse_digits(None)


class TestToFloat:
    def test_valid_float(self):
        assert to_float("123.45") == 123.45

    def test_valid_int(self):
        assert to_float("100") == 100.0

    def test_with_comma(self):
        assert to_float("1,000,000") == 1000000.0

    def test_none(self):
        assert to_float(None) is None

    def test_empty(self):
        assert to_float("") is None

    def test_invalid(self):
        assert to_float("abc") is None


class TestEnsureHistory:
    def test_empty(self):
        result = ensure_history([])
        assert result == []

    def test_none(self):
        result = ensure_history(None)
        assert result == []

    def test_history_rows(self):
        rows = [
            HistoryRow(issue="1", digits=(1, 2, 3)),
            HistoryRow(issue="2", digits=(4, 5, 6)),
        ]
        result = ensure_history(rows)
        assert len(result) == 2
        assert all(isinstance(r, HistoryRow) for r in result)

    def test_dicts(self):
        rows = [
            {"issue": "1", "开奖号码": "1,2,3"},
            {"issue": "2", "开奖号码": "4,5,6"},
        ]
        result = ensure_history(rows)
        assert len(result) == 2
        assert all(isinstance(r, HistoryRow) for r in result)

    def test_tuples(self):
        rows = [(1, 2, 3), (4, 5, 6)]
        result = ensure_history(rows)
        assert len(result) == 2
        assert all(isinstance(r, HistoryRow) for r in result)


class TestDigitsMatrix:
    def test_basic(self):
        history = [
            HistoryRow(issue="1", digits=(1, 2, 3)),
            HistoryRow(issue="2", digits=(4, 5, 6)),
        ]
        matrix = digits_matrix(history)
        assert matrix.shape == (2, 3)
        assert np.array_equal(matrix[0], [1, 2, 3])
        assert np.array_equal(matrix[1], [4, 5, 6])

    def test_empty(self):
        matrix = digits_matrix([])
        assert matrix.shape == (0, 3)


class TestPredictorRegistry:
    def test_registry_not_empty(self):
        assert len(PREDICTOR_REGISTRY) >= 9

    def test_expected_predictors(self):
        expected = {
            "ewma", "amplitude", "markov", "zscore",
            "parity", "prime", "ac_value", "span", "road012",
            "payout_adjusted",
        }
        assert expected.issubset(set(PREDICTOR_REGISTRY.keys()))


class TestBuildPredictor:
    def test_build_by_name(self):
        pred = build_predictor("markov")
        assert isinstance(pred, MarkovPredictor)
        assert pred.name == "markov"

    def test_build_with_kwargs(self):
        pred = build_predictor("markov", smooth=0.5)
        assert pred.smooth == 0.5

    def test_build_unknown_raises(self):
        with pytest.raises(ValueError):
            build_predictor("nonexistent")


class TestPredictorInterface:
    """测试所有预测器实现统一接口"""

    @pytest.fixture
    def sample_history(self):
        return [
            HistoryRow(issue=str(i), digits=(i % 10, (i+1) % 10, (i+2) % 10))
            for i in range(50)
        ]

    def test_predict_proba_returns_proba_result(self, sample_history):
        for name, cls in PREDICTOR_REGISTRY.items():
            if name == "payout_adjusted":
                continue  # 需要基础预测器
            pred = cls()
            result = pred.predict_proba(sample_history)

            # 检查结构
            assert hasattr(result, "positions_proba")
            assert hasattr(result, "sum_proba")
            assert hasattr(result, "meta")

            # 检查形状
            assert len(result.positions_proba) == 3
            for p in result.positions_proba:
                assert p.shape == (DIGIT_N,)
                assert np.allclose(p.sum(), 1.0, atol=1e-6)
                assert np.all(p >= 0)

            assert result.sum_proba.shape == (SUM_N,)
            assert np.allclose(result.sum_proba.sum(), 1.0, atol=1e-6)
            assert np.all(result.sum_proba >= 0)

    def test_predict_returns_prediction(self, sample_history):
        for name, cls in PREDICTOR_REGISTRY.items():
            if name == "payout_adjusted":
                continue
            pred = cls(top_k=3)
            result = pred.predict(sample_history)

            assert isinstance(result, Prediction)
            assert result.name == name
            assert len(result.positions) == 3
            assert all(len(p) == 3 for p in result.positions)
            assert len(result.sum_value) == 3
            assert len(result.positions_proba) == 3
            assert result.sum_proba.shape == (SUM_N,)

    def test_predict_top_k_respected(self, sample_history):
        pred = build_predictor("markov", top_k=5)
        result = pred.predict(sample_history, top_k=5)
        assert all(len(p) == 5 for p in result.positions)
        assert len(result.sum_value) == 5


class TestSpecificPredictors:
    """测试特定预测器的特有行为"""

    @pytest.fixture
    def sample_history(self):
        return [
            HistoryRow(issue=str(i), digits=(i % 10, (i+1) % 10, (i+2) % 10))
            for i in range(100)
        ]

    def test_markov_no_window(self, sample_history):
        pred = MarkovPredictor(window=None)
        result = pred.predict_proba(sample_history)
        assert len(result.positions_proba) == 3

    def test_markov_with_window(self, sample_history):
        pred = MarkovPredictor(window=50)
        result = pred.predict_proba(sample_history)
        assert len(result.positions_proba) == 3

    def test_span_method_recent(self, sample_history):
        pred = SpanPredictor(method="recent")
        result = pred.predict_proba(sample_history)
        assert len(result.positions_proba) == 3

    def test_span_method_zscore(self, sample_history):
        pred = SpanPredictor(method="zscore", temperature=2.0)
        result = pred.predict_proba(sample_history)
        assert len(result.positions_proba) == 3

    def test_zscore_threshold(self, sample_history):
        pred = ZScorePredictor(threshold=1.5)
        result = pred.predict_proba(sample_history)
        assert len(result.positions_proba) == 3

    def test_road012_roads_per_position(self, sample_history):
        pred = Road012Predictor(roads_per_position=1)
        result = pred.predict_proba(sample_history)
        assert "selected_roads" in result.meta
        assert all(len(r) == 1 for r in result.meta["selected_roads"])


class TestEnsemblePredictor:
    @pytest.fixture
    def sample_history(self):
        return [
            HistoryRow(issue=str(i), digits=(i % 10, (i+1) % 10, (i+2) % 10))
            for i in range(100)
        ]

    def test_ensemble_from_config(self, sample_history):
        ensemble = EnsemblePredictor(ensemble_name="full")
        result = ensemble.predict(sample_history)
        assert isinstance(result, Prediction)
        assert "members" in result.meta
        assert len(result.meta["members"]) == 5

    def test_ensemble_simple(self, sample_history):
        ensemble = EnsemblePredictor(ensemble_name="simple")
        result = ensemble.predict(sample_history)
        assert len(result.meta["members"]) == 2

    def test_ensemble_custom_predictors(self, sample_history):
        preds = [
            build_predictor("markov"),
            build_predictor("zscore"),
        ]
        weights = [0.6, 0.4]
        ensemble = EnsemblePredictor(predictors=preds, weights=weights)
        result = ensemble.predict(sample_history)
        assert len(result.meta["members"]) == 2

    def test_ensemble_weight_normalization(self, sample_history):
        preds = [build_predictor("markov"), build_predictor("zscore")]
        weights = [10, 20]  # 不归一化
        ensemble = EnsemblePredictor(predictors=preds, weights=weights)
        assert np.allclose(ensemble.weights.sum(), 1.0)
        assert np.allclose(ensemble.weights, [1/3, 2/3])


class TestPredictionSchema:
    def test_prediction_to_dict(self):
        pred = Prediction(
            name="test",
            positions=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            sum_value=[10, 11, 12],
            positions_proba=[
                np.full(10, 0.1),
                np.full(10, 0.1),
                np.full(10, 0.1),
            ],
            sum_proba=np.full(28, 1/28),
        )
        d = pred.to_dict()
        assert d["name"] == "test"
        assert d["positions"] == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assert isinstance(d["positions_proba"][0], list)

    def test_prediction_with_numpy_arrays(self):
        pred = Prediction(
            name="test",
            positions=[[1], [2], [3]],
            sum_value=[6],
            positions_proba=[
                np.array([0.5, 0.5] + [0]*8),
                np.array([0.5, 0.5] + [0]*8),
                np.array([0.5, 0.5] + [0]*8),
            ],
            sum_proba=np.array([0.5, 0.5] + [0]*26),
        )
        d = pred.to_dict()
        assert all(isinstance(x, float) for x in d["positions_proba"][0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])