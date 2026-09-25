# -*- coding: utf-8 -*-
"""策略信号引擎：基于 3d_full_history.csv 的反奖率六态状态机

信号完全由历史表派生（不依赖 forward_log），输入为截至"数据截至期号"的历史序列，
输出本期预测应使用的策略信号。

核心指标（每期 t 结算后可得）：
    派生反奖率 rate(t) = SINGLE_PRIZE * single_wins(t) / sales(t)
        —— 该期全部单选派奖占销售额比例，衡量当期全网拥挤程度。
    近5期均值 avgR(t)   = mean(rate(t-4..t))
    销量z分 zS(t)       = (sales(t) - mean_60) / std_60
    单选占比 share(t)   ≈ rate(t) / 506 （经验标定逆映射，仅作展示）

划分标准（综合 反奖率 + 单选号码占比 + 奖池金额）：
    阈值按 3d_full_history.csv 全量回放的分位数标定（rate 均值≈0.42，
    p90≈0.53、p99≈0.62；单期 rate 峰值 0.99 出现在豹子号期），
    并保证六态在 704 期样本内均有非零触发：

    第 0 层 事件覆盖（优先级最高，用当期实测值）：
      jackpot_defense   last_rate >= EVENT_RATE_HI(0.85) 且 奖池未补满
                        （派奖打穿奖池水位，如 2026242 期 rate=0.986）
      extreme_hot       last_rate >= EVENT_RATE_LO(0.70) 或 (豹子号 且 avgR >= 分位线)
                        （极端拥挤开奖，如 2026251 豹子 777 期 rate=0.87）
    第 1 层 滚动状态机（avgR = 近5期 rate 均值）：
      greedy_momentum   奖池 >= 2,000,000 且 avgR >= P_HI(0.50)
                        （满池高消耗区→大众追热，赔率被摊薄前顺势）
      hot_penalty       avgR >= P_MID(0.45)                （偏热，惩罚追号）
      normal            P_LOW(0.40) <= avgR < P_MID        （中性带）
      cold_reversion    avgR < P_LOW(0.40)                 （冷回归）
另有两个叠加标记（不改变主信号）：
    sales_anomaly  |zS| > 1.5
    leopard_flag   最新一期为豹子号（三位相同）

模型权重建议（五基模型：frequency/markov/cold_reversion/sum_span/ml）：
    以 DEFAULT_WEIGHTS 为基线，按信号做方向性调整；
    greedy/hot/extreme 压制冷回归、抬升频率与马尔可夫；
    cold_reversion 反之；ML 仅在 normal 下参与。
"""
from __future__ import annotations

import numpy as np

from ..history import ensure_history
from .anti_prize import SINGLE_PRIZE

# ---- 阈值常量（可在调用处覆盖） ----
RATE_ROLL_WINDOW = 5          # 反奖率滚动窗口
SALES_STAT_WINDOW = 60        # 销量均值/方差窗口
JACKPAT_FULL = 2_000_000.0    # 满额奖池水位
RATE_SLOPE = 506.0            # share→rate 经验标定斜率（逆映射用）

BOUNDS = {
    # 第 1 层：avgR 滚动状态机分位线（704 期回放标定）
    "p_low": 0.40,        # < p_low -> cold_reversion   (~35%)
    "p_mid": 0.45,        # [p_low, p_mid) -> normal    (~28%)
    "p_hi": 0.50,         # >= p_hi 且满池 -> greedy_momentum
    # 第 0 层：事件覆盖（单期实测 rate）
    "event_lo": 0.70,     # >= event_lo -> extreme_hot
    "event_hi": 0.85,     # >= event_hi 且奖池未补满 -> jackpot_defense
}

SIGNAL_NAMES = [
    "cold_reversion",
    "normal",
    "hot_penalty",
    "extreme_hot",
    "jackpot_defense",
    "greedy_momentum",
]

BASE_KEYS = ["frequency", "markov", "cold_reversion", "sum_span", "ml"]
DEFAULT_WEIGHTS = {k: 0.2 for k in BASE_KEYS}


def derive_rate_series(history) -> list[float]:
    """逐期派生反奖率 rate(t) = 1040 * 单选中奖注数 / 销售额。"""
    history = ensure_history(history)
    out: list[float] = []
    for row in history:
        if (
            row.single_wins is not None
            and row.sales is not None
            and float(row.sales) > 0
        ):
            r = SINGLE_PRIZE * float(row.single_wins) / float(row.sales)
            out.append(float(min(max(r, 0.0), 1.0)))
        else:
            out.append(float("nan"))
    return out


def _rolling_avg5(rates: list[float]) -> list[float]:
    """avgR(t)：最近5个有效 rate 的均值（不足则取已有均值）。"""
    out: list[float] = []
    valid: list[float] = []
    for r in rates:
        if not np.isnan(r):
            valid.append(r)
        window = valid[-RATE_ROLL_WINDOW:]
        out.append(float(np.mean(window)) if window else float("nan"))
    return out


def classify_signal(
    avg_r: float,
    jackpot: float | None,
    last_rate: float | None = None,
    leopard: bool = False,
    bounds: dict[str, float] | None = None,
) -> str:
    """六态主信号判定（第 0 层事件覆盖优先，第 1 层滚动状态机兜底）。"""
    b = bounds or BOUNDS

    # ---- 第 0 层：单期极端事件 ----
    if last_rate is not None and not (isinstance(last_rate, float) and np.isnan(last_rate)):
        pool_full = jackpot is not None and float(jackpot) >= JACKPAT_FULL
        if last_rate >= b["event_hi"] and not pool_full:
            return "jackpot_defense"      # 派奖打穿奖池水位
        if last_rate >= b["event_lo"]:
            return "extreme_hot"          # 极端拥挤开奖

    if np.isnan(avg_r):
        return "normal"

    # ---- 第 1 层：滚动状态机 ----
    if avg_r >= b["p_hi"]:
        if jackpot is not None and float(jackpot) >= JACKPAT_FULL:
            return "greedy_momentum"      # 满池 + 高消耗区 -> 顺势追热
        return "hot_penalty"
    if avg_r >= b["p_mid"]:
        return "hot_penalty"
    if avg_r >= b["p_low"]:
        return "normal"
    return "cold_reversion"


def signal_weights(signal: str) -> dict[str, float]:
    """信号 → 五基模型建议权重（归一化后返回）。"""
    w = dict(DEFAULT_WEIGHTS)
    if signal == "greedy_momentum":
        w.update({"frequency": 0.35, "markov": 0.30, "cold_reversion": 0.05,
                  "sum_span": 0.20, "ml": 0.10})
    elif signal == "jackpot_defense":
        w.update({"frequency": 0.30, "markov": 0.25, "cold_reversion": 0.05,
                  "sum_span": 0.25, "ml": 0.15})
    elif signal == "extreme_hot":
        w.update({"frequency": 0.28, "markov": 0.22, "cold_reversion": 0.08,
                  "sum_span": 0.22, "ml": 0.20})
    elif signal == "hot_penalty":
        w.update({"frequency": 0.24, "markov": 0.20, "cold_reversion": 0.14,
                  "sum_span": 0.22, "ml": 0.20})
    elif signal == "cold_reversion":
        w.update({"frequency": 0.12, "markov": 0.15, "cold_reversion": 0.35,
                  "sum_span": 0.20, "ml": 0.18})
    else:  # normal
        w = dict(DEFAULT_WEIGHTS)
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


class SignalEngine:
    """从历史序列计算当前策略信号。"""

    def __init__(self, bounds: dict[str, float] | None = None):
        self.bounds = dict(BOUNDS)
        if bounds:
            self.bounds.update(bounds)

    def compute(self, history) -> dict:
        """返回指标、信号、权重与叠加标记。

        history 需按期号升序排列（最后一期为最新已开奖期）。
        """
        history = ensure_history(history)
        if not history:
            raise ValueError("SignalEngine.compute 需要非空历史序列")

        rates = derive_rate_series(history)
        avgs = _rolling_avg5(rates)
        avg_r = avgs[-1]
        last_rate = rates[-1] if not np.isnan(rates[-1]) else None

        latest = history[-1]
        jackpot = getattr(latest, "jackpot", None)

        # 销量 z 分（近 SALES_STAT_WINDOW 期）
        sales = [float(r.sales) for r in history
                 if r.sales is not None and float(r.sales) > 0]
        z_s = 0.0
        if len(sales) >= 10:
            win = np.asarray(sales[-SALES_STAT_WINDOW:], dtype=float)
            mu, sd = float(win.mean()), float(win.std())
            if sd > 0:
                z_s = (sales[-1] - mu) / sd

        digits = tuple(int(x) for x in latest.digits)
        leopard = digits[0] == digits[1] == digits[2]

        signal = classify_signal(avg_r, jackpot, last_rate=last_rate,
                                 leopard=leopard, bounds=self.bounds)
        share_est = None
        if last_rate is not None and latest.sales:
            residual = max(float(last_rate) - SINGLE_PRIZE / float(latest.sales), 0.0)
            share_est = round(residual / RATE_SLOPE, 6)

        return {
            "signal": signal,
            "avg_rate_5": round(float(avg_r), 4) if not np.isnan(avg_r) else None,
            "last_rate": round(float(last_rate), 4) if last_rate is not None else None,
            "share_est": share_est,
            "jackpot": float(jackpot) if jackpot is not None else None,
            "sales_zscore": round(float(z_s), 3),
            "sales_anomaly": bool(abs(z_s) > 1.5),
            "leopard_flag": bool(leopard),
            "model_weights": signal_weights(signal),
        }


def annotate_signals(history) -> list[dict]:
    """对整段历史逐期回放信号（用于验证/回测分析）。

    返回 [{issue, signal, avg_rate_5}, ...]，第 i 项表示"用截至第 i 期的
    历史判定的信号"（即预测第 i+1 期时所用信号）。
    """
    history = ensure_history(history)
    rates = derive_rate_series(history)
    avgs = _rolling_avg5(rates)
    out: list[dict] = []
    for i, row in enumerate(history):
        jackpot = getattr(row, "jackpot", None)
        lr = rates[i] if not np.isnan(rates[i]) else None
        d = tuple(int(x) for x in row.digits)
        sig = classify_signal(avgs[i], jackpot, last_rate=lr,
                              leopard=d[0] == d[1] == d[2])
        out.append({
            "issue": row.issue,
            "signal": sig,
            "avg_rate_5": round(float(avgs[i]), 4) if not np.isnan(avgs[i]) else None,
        })
    return out
