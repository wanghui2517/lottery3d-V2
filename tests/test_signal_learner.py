# -*- coding: utf-8 -*-
"""信号引擎 + 奖惩 + 学习机制测试（含 80/20 切分冒烟）"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from lottery3d_v2.history import load_history_from_csv
from lottery3d_v2.predictors.signal_engine import (
    BOUNDS, SignalEngine, annotate_signals, classify_signal,
    derive_rate_series, signal_weights, _rolling_avg5,
)
from lottery3d_v2.predictors.reward import (
    grade_outcome, payout_for_grade, reward_multiplier, settle_record,
)
from lottery3d_v2.predictors import learner


@pytest.fixture(scope="module")
def hist():
    return load_history_from_csv("data/3d_full_history.csv", ascending=True)


# ---------- signal_engine ----------

class TestSignalEngine:
    def test_rate_series(self, hist):
        rates = derive_rate_series(hist)
        assert len(rates) == len(hist)
        v = [r for r in rates if not np.isnan(r)]
        assert 0.0 <= min(v) and max(v) <= 1.0

    def test_six_states_all_reachable(self, hist):
        sigs = {a["signal"] for a in annotate_signals(hist)}
        assert sigs == {"cold_reversion", "normal", "hot_penalty",
                        "extreme_hot", "jackpot_defense", "greedy_momentum"}

    def test_jackpot_defense_event(self, hist):
        ann = {a["issue"]: a["signal"] for a in annotate_signals(hist)}
        # 2026242 期 rate≈0.986 且开奖前奖池满额被消耗 -> jackpot_defense
        assert ann["2026242"] == "jackpot_defense"

    def test_classify_layers(self):
        b = BOUNDS
        # 第0层优先：单期 rate>=0.85 且奖池未满 -> jackpot_defense
        assert classify_signal(0.1, 0.0, last_rate=0.90) == "jackpot_defense"
        # 第0层：rate>=0.70 且奖池满 -> extreme_hot（不触发 defense）
        assert classify_signal(0.1, 2e6, last_rate=0.75) == "extreme_hot"
        # 第1层：avgR 状态机
        assert classify_signal(0.55, 2e6, last_rate=None) == "greedy_momentum"
        assert classify_signal(0.55, 0.0, last_rate=None) == "hot_penalty"
        assert classify_signal(0.47, None, last_rate=None) == "hot_penalty"
        assert classify_signal(0.42, None, last_rate=None) == "normal"
        assert classify_signal(0.30, None, last_rate=None) == "cold_reversion"
        assert classify_signal(float("nan"), None, last_rate=None) == "normal"

    def test_compute_dict(self, hist):
        info = SignalEngine().compute(hist)
        assert info["signal"] in {"cold_reversion", "normal", "hot_penalty",
                                  "extreme_hot", "jackpot_defense", "greedy_momentum"}
        assert set(info["model_weights"]) == {"frequency", "markov",
                                              "cold_reversion", "sum_span", "ml"}
        assert abs(sum(info["model_weights"].values()) - 1.0) < 1e-9

    def test_weights_normalized(self):
        from lottery3d_v2.predictors.signal_engine import SIGNAL_NAMES
        for s in SIGNAL_NAMES:
            w = signal_weights(s)
            assert abs(sum(w.values()) - 1.0) < 1e-9


# ---------- reward ----------

class TestReward:
    def test_grades(self):
        assert grade_outcome(3) == "jackpot"
        assert grade_outcome(2, "normal") == "excellent"
        assert grade_outcome(2, "greedy_momentum") == "reckless"
        assert grade_outcome(1, "cold_reversion") == "bonus"
        assert grade_outcome(1, "normal") == "good"
        assert grade_outcome(0, "normal") == "poor"

    def test_payouts(self):
        assert payout_for_grade("jackpot", 3) == pytest.approx(1040 - 30)
        assert payout_for_grade("excellent", 2) == pytest.approx(20 - 30)
        assert payout_for_grade("good", 1) == pytest.approx(-30)

    def test_multiplier_bounds_and_streak(self):
        m = reward_multiplier(["poor"] * 5, current=1.0)
        assert m <= 0.5
        m2 = reward_multiplier(["jackpot"], current=1.0)
        assert m2 == pytest.approx(1.5)
        for _ in range(30):
            m2 = reward_multiplier(["jackpot"] * 30, current=m2)
        assert m2 <= 1.5
        m3 = reward_multiplier([], current=1.0)
        assert m3 == 1.0

    def test_settle_record(self):
        r = settle_record(3, "normal", [], multiplier=1.0)
        assert r["grade"] == "jackpot" and r["pnl"] > 0


# ---------- learner ----------

class TestLearner:
    def test_roundtrip(self, tmp_path):
        st = learner._default_state()
        p = tmp_path / "ls.json"
        learner.save_state(st, p)
        st2 = learner.load_state(p)
        assert st2["samples"] == 0

    def test_record_updates_stats(self):
        st = learner._default_state()
        for i in range(5):
            learner.record_settlement(st, f"T{i}", "normal", "good", -30.0, 1.0,
                                      model_hits={"frequency": 1})
        assert st["signal_stats"]["normal"]["n"] == 5
        assert st["model_stats"]["frequency"]["n"] == 5

    def test_baseline_clamps(self):
        st = learner._default_state()
        for i in range(10):
            learner.record_settlement(st, f"A{i}", "normal", "jackpot", 1010.0, 1.5)
            learner.record_settlement(st, f"B{i}", "cold_reversion", "poor", -30.0, 0.5)
        assert learner.BASE_MIN <= st["signal_baseline"]["normal"] <= learner.BASE_MAX
        assert learner.BASE_MIN <= st["signal_baseline"]["cold_reversion"] <= learner.BASE_MAX

    def test_effective_weights(self):
        st = learner._default_state()
        ew = signal_weights("normal")
        out = learner.effective_weights(st, "normal", ew)
        assert abs(sum(out["weights"].values()) - 1.0) < 1e-9


# ---------- 80/20 冒烟 ----------

class TestSplitSmoke:
    def test_split_execution(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from train_test_split import run, top3_positions, hit_count
        state = run(state_path=tmp_path / "ls.json")
        assert state["samples"] > 400          # 训练集结算量
        # 基线选号器 Top3 命中率应显著高于随机 30%
        hist = load_history_from_csv("data/3d_full_history.csv", ascending=True)
        hits = []
        for i in range(120, min(len(hist), 320)):
            tops = top3_positions(hist[:i])
            hits.append(hit_count(tops, hist[i].digits))
        rate = sum(1 for h in hits if h >= 1) / len(hits)
        assert rate > 0.40
