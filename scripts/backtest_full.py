# -*- coding: utf-8 -*-
"""全量历史回测：策略信号驱动的选号 + 奖惩/学习闭环

流程（严格无未来泄漏）：
  对每期 t（升序，前 MIN_WARMUP 期作为引擎预热不结算）：
    1. 用截至 t-1 的历史计算六态信号（SignalEngine）；
    2. 按信号选择基线选号器参数（cold_reversion -> 加大形态/冷号权重；
       hot/extreme/defense/greedy -> 追逐近期热门），生成各位 Top3；
    3. 与 t 期实际开奖比对，Top3 口径命中数 -> 评级 -> 派生收益
       （单选 1040 元/注、两位 10 注共 20 元、本票成本 30 元）；
    4. 训练段（前 80%）逐条更新学习状态（信号基线 + 下注乘子），
       测试段冻结状态做样本外评估；
    5. 输出净值曲线、命中率、收益、回撤等指标。

用法：python scripts/backtest_full.py [--ratio 0.8] [--out results/backtest_full.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from lottery3d_v2.history import load_history_from_csv
from lottery3d_v2.predictors.anti_prize import _morph_score, all_1000_numbers
from lottery3d_v2.predictors.reward import (
    STAKE_COST, grade_outcome, payout_for_grade, reward_multiplier,
)
from lottery3d_v2.predictors.signal_engine import SignalEngine, BASE_KEYS
from lottery3d_v2.predictors import learner

MIN_WARMUP = 60


def top3_positions(history, w_morph: float = 1.0, window: int = 100,
                   chase_lag: float = 0.3):
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
            score[row.digits[p]] *= (1 + chase_lag * (1 - lag / 5))
        order = np.argsort(-score)
        tops.append([int(x) for x in order[:3]])
    return tops


# 信号 -> 选号风格参数（可解释的轻量策略开关）
SIGNAL_TILTS = {
    #                 w_morph, chase_lag
    "cold_reversion":   (1.6, 0.0),   # 冷回归：重形态、不追热
    "normal":           (1.0, 0.3),   # 基线
    "hot_penalty":      (0.6, 0.5),   # 追热避形态扎堆
    "extreme_hot":      (0.6, 0.5),
    "jackpot_defense":  (0.8, 0.4),
    "greedy_momentum":  (0.5, 0.8),   # 贪婪：强追近期动量
}


def hit_count(tops3, actual) -> int:
    return sum(1 for p in range(3) if int(actual[p]) in tops3[p])


def metrics_from_pnls(pnls, mults, hits):
    equity = np.cumsum(np.array(pnls) * np.array(mults))
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    max_dd = float(dd.min()) if len(dd) else 0.0
    total = float(equity[-1]) if len(equity) else 0.0
    return {
        "periods": len(pnls),
        "total_pnl": round(total, 1),
        "max_drawdown": round(max_dd, 1),
        "hit_rate_ge1": round(sum(1 for h in hits if h >= 1) / max(len(hits), 1), 4),
        "hit_rate_3": round(sum(1 for h in hits if h == 3) / max(len(hits), 1), 4),
        "avg_hit": round(float(np.mean(hits)), 3) if hits else 0.0,
        "win_rate_stake": round(sum(1 for i in range(len(pnls))
                                    if pnls[i] > 0) / max(len(pnls), 1), 4),
    }


def run(csv_path="data/3d_full_history.csv", ratio=0.8, out=None):
    hist = load_history_from_csv(csv_path, ascending=True)
    n = len(hist)
    split = int(n * ratio)
    print(f"总期数 {n} | 训练 {split} ({hist[0].issue}~{hist[split-1].issue})"
          f" | 测试 {n-split} ({hist[split].issue}~{hist[-1].issue})")

    state = learner._default_state()
    engine = SignalEngine()

    records = []
    mult = state["stake_multiplier"]

    for i in range(MIN_WARMUP, n):
        past = hist[:i]
        row = hist[i]
        sig_info = engine.compute(past)
        signal = sig_info["signal"]
        w_morph, chase = SIGNAL_TILTS.get(signal, SIGNAL_TILTS["normal"])
        tops = top3_positions(past, w_morph=w_morph, chase_lag=chase)
        h = hit_count(tops, row.digits)
        g = grade_outcome(h, signal)
        pnl = payout_for_grade(g, h)
        rec = {
            "issue": row.issue, "date": str(row.date),
            "actual": ",".join(map(str, row.digits)),
            "signal": signal, "tops": "/".join(",".join(map(str, t)) for t in tops),
            "hit": h, "grade": g, "pnl": pnl, "mult": mult,
            "net": round(pnl * mult, 1),
            "avgR": sig_info["avg_rate_5"], "last_rate": sig_info["last_rate"],
            "jackpot": sig_info["jackpot"],
        }
        records.append(rec)
        # 仅训练段更新学习状态
        if i < split:
            new_mult = reward_multiplier(state["grades"] + [g], current=mult)
            learner.record_settlement(
                state, issue=row.issue, signal=signal, grade=g,
                pnl=pnl, stake_multiplier=new_mult,
                model_hits={k: h for k in BASE_KEYS})
            mult = new_mult

    train_recs = [r for r in records if int(r["issue"]) <= int(hist[split-1].issue)]
    test_recs = [r for r in records if r not in train_recs]

    def agg(recs):
        return metrics_from_pnls([r["pnl"] for r in recs],
                                 [r["mult"] for r in recs],
                                 [r["hit"] for r in recs])

    m_all, m_tr, m_te = agg(records), agg(train_recs), agg(test_recs)

    print("\n== 全量回测 ==")
    for label, m in [("全部", m_all), ("训练段", m_tr), ("测试段(样本外)", m_te)]:
        print(f"[{label}] {m['periods']}期 | 净收益 {m['total_pnl']:+,.0f} 元"
              f" | 最大回撤 {m['max_drawdown']:+,.0f}"
              f" | ≥1位命中 {m['hit_rate_ge1']:.2%} | 全中率 {m['hit_rate_3']:.2%}"
              f" | 均命中 {m['avg_hit']:.2f}/3 | 单票胜率 {m['win_rate_stake']:.2%}")

    print("\n== 分信号统计（全量） ==")
    by_sig = defaultdict(list)
    for r in records:
        by_sig[r["signal"]].append(r)
    print(f"{'信号':<18}{'期数':>6}{'均命中':>8}{'全中':>6}{'>=1命中':>9}"
          f"{'累计净收益':>12}")
    for s, rs in sorted(by_sig.items(), key=lambda kv: -len(kv[1])):
        m = metrics_from_pnls([x["pnl"] for x in rs], [x["mult"] for x in rs],
                              [x["hit"] for x in rs])
        jacks = sum(1 for x in rs if x["hit"] == 3)
        print(f"{s:<18}{m['periods']:>6}{m['avg_hit']:>8.2f}{jacks:>6}"
              f"{m['hit_rate_ge1']:>9.2%}{m['total_pnl']:>+12,.0f}")

    print("\n评级分布:", dict(Counter(r["grade"] for r in records)))
    print("最终下注乘子:", round(state["stake_multiplier"], 3))

    result = {"summary": {"all": m_all, "train": m_tr, "test": m_te},
              "signal_stats": {s: metrics_from_pnls(
                  [x["pnl"] for x in rs], [x["mult"] for x in rs],
                  [x["hit"] for x in rs]) for s, rs in by_sig.items()},
              "records": records}
    if out:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float),
                     encoding="utf-8")
        print(f"\n明细已保存: {p}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/3d_full_history.csv")
    ap.add_argument("--ratio", type=float, default=0.8)
    ap.add_argument("--out", default="results/backtest_full.json")
    a = ap.parse_args()
    run(a.csv, a.ratio, a.out)
