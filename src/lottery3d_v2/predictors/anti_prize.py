# -*- coding: utf-8 -*-
"""单选反奖率：号码拥挤度（竞争强度）分析

原始逻辑恢复（依据 data/forward_log.csv 历史字段标定，42 条记录）：

    单选占比 (single_ratio)
        = 预估全网单选中奖注数 / 预估总投注注数
        由历史行为模型估计：n(i) ∝ exp(频率分 + 形态分 + 重复追逐分)，
        豹子/顺子/对子等大众偏爱形态与近期开奖号获得加成。

    反奖率 (anti_prize_rate) —— 从日志经验标定恢复：
        rate ≈ 506 × single_ratio + 0.024      （R² = 0.989）
        截距 0.024 ≈ SINGLE_PRIZE(1040) / 平均销售额(≈43,000,000)，
        即"该号若命中，仅摊到本票的固定奖金占销售额比例"；
        斜率 506 ≈ 全网派奖放大系数（其他玩家同押该号的摊薄效应）。
        rate 越高 → 该号越拥挤、开出后人均派奖越低 → 选号惩罚越大。

    划分标准（按反奖率区间，档位上限取整自日志分布）：
        cold     反奖率 <  0.25   冷门，优先选择
        normal   0.25 <= r < 0.40 中性，不惩罚
        warm     0.40 <= r < 0.60 偏热，轻度惩罚
        hot      0.60 <= r < 0.85 大热，重度惩罚
        crowded  r >= 0.85        拥挤，强烈回避（如 Jackpot_Defense 期 0.9889）

    贪婪模式：奖池水位 jackpot >= 2,000,000 时触发，
        放弃反奖率惩罚直接追热（日志中 greed_momentum/prize_pool 记录均为 True）。

结算口径：单选中奖奖金 1040 元/注；两位命中共 10 注、每注 2 元共 20 元。
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from ..history import ensure_history
from ..utils import normalize_proba

SINGLE_PRIZE = 1040.0          # 单选每注奖金（元）
TWO_DIGIT_BETS = 10            # 两位命中共 10 注
TWO_DIGIT_STAKE = 20.0         # 每注 2 元共 20 元

# 经验标定参数（由 forward_log.csv 42 条记录线性拟合得到）
RATE_INTERCEPT = 0.024         # ≈ SINGLE_PRIZE / 平均销售额
RATE_SLOPE = 506.0             # 全网派奖放大系数
UNIFORM_SHARE = 0.001          # 均匀分布基准占比

# (档位名, 反奖率下限, 反奖率上限)
ANTI_PRIZE_BUCKETS: list[tuple[str, float, float]] = [
    ("cold", float("-inf"), 0.25),
    ("normal", 0.25, 0.40),
    ("warm", 0.40, 0.60),
    ("hot", 0.60, 0.85),
    ("crowded", 0.85, 1.0),
]


def all_1000_numbers() -> list[tuple[int, int, int]]:
    return [(a, b, c) for a in range(10) for b in range(10) for c in range(10)]


def number_index(digits: Iterable[int]) -> int:
    d = tuple(int(x) for x in digits)
    return d[0] * 100 + d[1] * 10 + d[2]


def _morph_score(d: tuple[int, int, int]) -> float:
    """大众偏好形态加分：豹子/顺子/对子更受偏爱，更拥挤。"""
    a, b, c = d
    if a == b == c:
        return 3.0                      # 豹子（如 555、777）
    s = sorted(d)
    if s[2] - s[1] == 1 and s[1] - s[0] == 1:
        return 2.0                      # 顺子
    if a == b or b == c or a == c:
        return 1.5                      # 对子
    if len({a, b, c}) == 3 and (max(d) - min(d)) <= 2:
        return 0.5                      # 连号邻域
    return 0.0


def estimate_crowding(
    history,
    w_freq: float = 1.0,
    w_morph: float = 1.0,
    window: int = 100,
) -> np.ndarray:
    """估计 1000 个号码的相对拥挤度（单选占比向量，和为 1）。

    频率分：近 window 期各位置数字出现频率之和（追热是大众主流行为）；
    形态分：豹子/顺子/对子等大众偏爱形态加分；
    最近 5 期开奖号有额外重复追逐加成。
    """
    history = ensure_history(history)
    recent = history[-window:]

    pos_freq = [np.zeros(10) for _ in range(3)]
    for row in recent:
        for p, dg in enumerate(row.digits):
            pos_freq[p][dg] += 1.0
    pos_freq = [normalize_proba(f) for f in pos_freq]

    freq_score = np.zeros(1000)
    morph_score = np.zeros(1000)
    for idx, (a, b, c) in enumerate(all_1000_numbers()):
        freq_score[idx] = pos_freq[0][a] + pos_freq[1][b] + pos_freq[2][c]
        morph_score[idx] = _morph_score((a, b, c))

    logits = w_freq * freq_score + w_morph * morph_score

    for lag, row in enumerate(recent[-5:]):
        logits[number_index(row.digits)] += 0.8 * (1.0 - lag / 5.0)

    return normalize_proba(np.exp(logits - logits.max()))


def share_to_rate(crowd_share: float) -> float:
    """单选占比 → 反奖率（经验标定线性映射，截断到 [0,1]）。"""
    if crowd_share <= 0:
        return 0.0
    rate = RATE_INTERCEPT + RATE_SLOPE * crowd_share
    return float(min(max(rate, 0.0), 1.0))


def classify_anti_prize_rate(rate: float) -> str:
    """反奖率 → 档位。"""
    bucket = ANTI_PRIZE_BUCKETS[0][0]
    for name, lo, _hi in ANTI_PRIZE_BUCKETS:
        if lo == float("-inf") or rate >= lo:
            bucket = name
    return bucket


def classify_crowd_share(crowd_share: float) -> str:
    """单选占比 → 档位（先转反奖率再分档）。"""
    return classify_anti_prize_rate(share_to_rate(crowd_share))


def anti_prize_rate(
    crowd_share: float,
    sales_estimate: float | None = None,
    total_bets_estimate: float | None = None,
) -> float:
    """反奖率。

    默认使用经验标定映射 rate = 0.024 + 506 * share。
    若显式给定销量口径（total_bets ≈ 销售额/2），可用第一性原理折算：
        rate = SINGLE_PRIZE/sales + crowd_share*total_bets*SINGLE_PRIZE/sales
    两者在典型销量 (~1200万) 下数值接近。
    """
    if sales_estimate and total_bets_estimate and sales_estimate > 0:
        own = SINGLE_PRIZE / sales_estimate
        pool = crowd_share * total_bets_estimate * SINGLE_PRIZE / sales_estimate
        return float(min(max(own + pool, 0.0), 1.0))
    return share_to_rate(crowd_share)


def anti_prize_penalty(rate: float, strength: float = 1.0) -> float:
    """把反奖率转成候选号打分惩罚系数（乘子 ∈ (0,1]）。

    反奖率 <= 0.35（normal 及以下）不惩罚；之上线性加重。
    """
    excess = max(0.0, rate - 0.35)
    return float(max(1e-6, 1.0 - strength * excess))


class AntiPrizeAnalyzer:
    """单选反奖率分析器（供预测/日志结算调用）。"""

    def __init__(self, w_freq: float = 1.0, w_morph: float = 1.0, window: int = 100):
        self.w_freq = w_freq
        self.w_morph = w_morph
        self.window = window

    def analyze(
        self,
        history,
        candidate_digits: tuple[int, int, int],
        positions_proba: list[np.ndarray] | None = None,
    ) -> dict:
        """返回 {single_ratio, anti_prize_rate, bucket, best_value_number,
        jackpot_level, greedy_mode}。

        positions_proba：模型三位置概率分布。若提供，则用
        P_model(i)/P_crowd(i) 的价值比在模型联合概率前 50 的号中挑选
        "性价比最高"（反奖率最低）的代表号——即原逻辑中
        "通过反奖率惩罚规避高竞争号码"的实现方式。
        """
        history = ensure_history(history)
        crowd = estimate_crowding(
            history, w_freq=self.w_freq, w_morph=self.w_morph, window=self.window
        )

        chosen_idx = number_index(candidate_digits)

        if positions_proba is not None and len(positions_proba) == 3:
            a, b, c = np.meshgrid(*(np.arange(10),) * 3, indexing="ij")
            pa = normalize_proba(positions_proba[0])[a.ravel()]
            pb = normalize_proba(positions_proba[1])[b.ravel()]
            pc = normalize_proba(positions_proba[2])[c.ravel()]
            model_p = pa * pb * pc

            value = model_p / (crowd + 1e-12)
            top_ids = np.argsort(-model_p)[:50]
            chosen_idx = int(top_ids[np.argmax(value[top_ids])])

        share = float(crowd[chosen_idx])

        latest = history[-1] if history else None
        jackpot = getattr(latest, "jackpot", None) if latest else None
        greedy = bool(jackpot is not None and float(jackpot) >= 2_000_000.0)

        rate = share_to_rate(share)
        # 贪婪模式（奖池水位达标）：跳过惩罚，直接追热——rate 仅作记录

        return {
            "single_ratio": round(share, 4),
            "anti_prize_rate": round(rate, 4),
            "bucket": classify_crowd_share(share),
            "best_value_number": (chosen_idx // 100, (chosen_idx // 10) % 10, chosen_idx % 10),
            "jackpot_level": float(jackpot) if jackpot is not None else None,
            "greedy_mode": greedy,
        }
