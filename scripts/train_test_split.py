# -*- coding: utf-8 -*-
"""80/20 时间序列切分 + 学习执行

规则：
- 按历史表（升序）取前 80% 为训练集、后 20% 为测试集。
- "学习"定义：对每一期 t（有实际开奖），用截至 t-1 的历史计算信号，
  以 Top3 命中口径评估基线选号器预测，
  训练集内逐条 record_settlement 更新 state/learning_state.json；
  测试集只做样本外评估，不更新状态。
- 输出训练/测试各自的命中率与派生收益汇总。
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from lottery3d_v2.history import load_history_from_csv
from lottery3d_v2.predictors.anti_prize import _morph_score, all_1000_numbers
from lottery3d_v2.predictors.reward import (
    grade_outcome, payout_for_grade, reward_multiplier,
)
from lottery3d_v2.predictors.signal_engine import SignalEngine, BASE_KEYS
from lottery3d_v2.predictors import learner


def top3_positions(history, w_morph: float = 1.0, window: int = 100):
    """基线选号器：位置边缘分 = 频率分布 × exp(形态加成) × 近期追逐。"""
    recent = history[-window:]
    pos_freq = [np.zeros(10) for _ in range(3)]
    for row in recent:
        for p, dg in enumerate(row.digits):
            pos_freq[p][dg] += 1.0
    pos_freq = [f / f.sum() for f in pos_freq]

    bonus = [np.zeros(10) for _ in range(3)]
    for d in all_1000_numbers():
        m = _morph_score(d)
        for p in range(3):
            bonus[p][d[p]] += m
    bonus = [b / b.max() for b in bonus]

    tops = []
    for p in range(3):
        score = pos_freq[p] * np.exp(w_morph * bonus[p])
        for lag, row in enumerate(recent[-5:]):
            score[row.digits[p]] *= (1 + 0.3 * (1 - lag / 5))
        order = np.argsort(-score)
        tops.append([int(x) for x in order[:3]])
    return tops


def hit_count(tops3, actual) -> int:
    """Top3 口径命中数：actual 各位数字是否落在对应位 Top3。"""
    return sum(1 for p in range(3) if int(actual[p]) in tops3[p])


def run(csv_path="data/3d_full_history.csv", ratio=0.8, state_path=None):
    hist = load_history_from_csv(csv_path, ascending=True)
    n = len(hist)
    split = int(n * ratio)
    train, test = hist[:split], hist[split:]
    print(f"总期数 {n} | 训练集 {len(train)} 期 ({hist[0].issue}~{train[-1].issue})"
          f" | 测试集 {len(test)} 期 ({test[0].issue}~{test[-1].issue})")

    state = learner._default_state()
    engine = SignalEngine()

    def evaluate_segment(seg, start_i, update: bool, label: str):
        signals, hits, pnls = [], [], []
        mult = state["stake_multiplier"]
        for j, row in enumerate(seg):
            i = start_i + j
            past = hist[:i]           # 严格只用 t-1 及更早（无未来泄漏）
            if len(past) < 60:
                continue
            signal = engine.compute(past)["signal"]
            tops = top3_positions(past)
            h = hit_count(tops, row.digits)
            g = grade_outcome(h, signal)
            pnl = payout_for_grade(g, h)
            signals.append(signal)
            hits.append(h)
            pnls.append(pnl)
            if update:
                new_mult = reward_multiplier(state["grades"] + [g], current=mult)
                model_hits = {k: h for k in BASE_KEYS}  # 基线选号器代表整体
                learner.record_settlement(
                    state, issue=row.issue, signal=signal, grade=g,
                    pnl=pnl, stake_multiplier=new_mult, model_hits=model_hits)
                mult = new_mult
        rate = sum(1 for x in hits if x >= 1) / max(len(hits), 1)
        full3 = sum(1 for x in hits if x == 3) / max(len(hits), 1)
        avg_hit = float(np.mean(hits)) if hits else 0.0
        total_pnl = float(np.sum(pnls))
        print(f"[{label}] 结算 {len(hits)} 期 | 至少一位命中率 {rate:.2%}"
              f" | 三位全中率 {full3:.2%} | 平均命中数 {avg_hit:.2f}/3"
              f" | 累计派生收益 {total_pnl:+,.0f} 元")
        print(f"         信号分布: {dict(Counter(signals))}")
        return hits, pnls

    print("\n== 训练阶段（更新学习状态） ==")
    evaluate_segment(train, 0, update=True, label="训练集(80%)")
    saved = learner.save_state(state, state_path)

    print("\n== 测试阶段（样本外，冻结状态） ==")
    evaluate_segment(test, split, update=False, label="测试集(20%)")

    print(f"\n学习状态已写入: {saved}")
    print("当前下注乘子:", round(state["stake_multiplier"], 3))
    print("\n" + learner.report(state))
    return state


if __name__ == "__main__":
    run(state_path=Path("state/learning_state.json"))
