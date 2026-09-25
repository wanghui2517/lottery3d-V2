# -*- coding: utf-8 -*-
"""奖惩机制：结算结果 → 评级 + 下注乘子

口径（与 anti_prize.py 一致）：
    单选命中奖金 1040 元/注；两位命中共 10 注、每注 2 元共 20 元。
    本票成本按 30 元计（1 注单选 2 元 + 10 注两位中除覆盖部分，简化固定）。

评级 grade_outcome(hit_count, signal)：
    hit3            -> jackpot
    hit2            -> excellent
    hit1            -> good
    hit0            -> poor
    叠加信号语境：
      hot/extreme/crowded/greedy 类信号下 hit>=2 -> reckless（追热高险中奖，不奖励策略本身）
      cold 类信号下 hit>=1        -> bonus     （冷门命中，策略价值最高）

奖惩乘子 reward_multiplier(performance, history)：
    基线 1.0；连续 poor >= 3 次 -> 0.5（降杠杆）；
    最近一次为 bonus/excellent   -> 1.2（升杠杆）；
    jackpot                      -> 1.5（上限）；reckless -> 0.75（回撤惩罚）。
    乘子限幅 [0.25, 1.5]。
"""
from __future__ import annotations

SINGLE_PRIZE = 1040.0
TWO_DIGIT_BETS = 10
TWO_DIGIT_STAKE = 20.0
STAKE_COST = 30.0

HOT_SIGNALS = {"hot_penalty", "extreme_hot", "jackpot_defense",
               "greedy_momentum", "hot", "crowded"}
COLD_SIGNALS = {"cold_reversion", "cold"}


def grade_outcome(hit_count: int, signal: str | None = None) -> str:
    """根据 Top3 命中数与当期信号给出评级。"""
    sig = (signal or "").lower()
    if hit_count >= 3:
        return "jackpot"
    if hit_count == 2:
        if sig in HOT_SIGNALS:
            return "reckless"
        return "excellent"
    if hit_count == 1:
        if sig in COLD_SIGNALS:
            return "bonus"
        return "good"
    # hit 0
    if sig in COLD_SIGNALS:
        return "poor_cold"
    return "poor"


def payout_for_grade(grade: str, hit_count: int) -> float:
    """本票派生收益（不含奖池加成，按日志口径）。"""
    win = SINGLE_PRIZE * max(hit_count - 2, 0) if hit_count >= 3 else (
        20.0 if hit_count == 2 else (2.0 * 34.0 if hit_count == 1 else 0.0))
    # 两位命中 10 注返 20 元；单选按位命中每号约返 34 元（两位组合摊分口径），
    # 全中 3 位视为单选命中：1040 元。
    if hit_count >= 3:
        win = SINGLE_PRIZE
    elif hit_count == 2:
        win = TWO_DIGIT_STAKE
    elif hit_count == 1:
        win = 0.0
    else:
        win = 0.0
    return float(win - STAKE_COST)


GRADE_MULTIPLIER = {
    "jackpot": 1.5,
    "excellent": 1.2,
    "bonus": 1.2,
    "good": 1.0,
    "reckless": 0.75,
    "poor": 0.9,
    "poor_cold": 0.9,
}

MULT_MIN, MULT_MAX = 0.25, 1.5


def reward_multiplier(
    grades: list[str],
    current: float = 1.0,
    poor_streak_threshold: int = 3,
) -> float:
    """由近期评级序列计算新的下注乘子。

    grades 按时间顺序（最新在末尾）。
    """
    m = float(current)
    if not grades:
        return m

    last = grades[-1]
    poor_streak = 0
    for g in reversed(grades):
        if g.startswith("poor"):
            poor_streak += 1
        else:
            break

    if poor_streak >= poor_streak_threshold:
        m = min(m, 0.5)          # 连续表现差 -> 降杠杆
    base = GRADE_MULTIPLIER.get(last, 1.0)
    m = m * (base / 1.0) if base != 1.0 else m
    # 归一：以 1.0 为中心做比例调整，避免复利式漂移
    m = float(current) * base if poor_streak < poor_streak_threshold else min(float(current) * base, 0.5)
    return float(min(max(m, MULT_MIN), MULT_MAX))


def settle_record(
    hit_count: int,
    signal: str | None,
    grades_history: list[str],
    multiplier: float = 1.0,
) -> dict:
    """结算单条预测：返回评级、收益与建议新乘子。"""
    grade = grade_outcome(hit_count, signal)
    pnl = payout_for_grade(grade, hit_count) * multiplier
    new_mult = reward_multiplier(grades_history + [grade], current=multiplier)
    return {
        "grade": grade,
        "hit_count": int(hit_count),
        "pnl": round(pnl, 2),
        "stake_multiplier": round(new_mult, 3),
    }
