# -*- coding: utf-8 -*-
"""学习机制：把结算评级转化为可持久化的权重学习

两部分：
1. 信号级基线权重（signal_baseline）——六态信号各自的权重乘子，
   按该信号历史累计表现（平均收益/命中率）以 lr 步长向目标方向调整。
2. 模型级权重（model_weights）——五基模型的全局乘性更新，
   依据各模型 Top3 命中率的 EWMA 与整体均值比较。

状态文件 state/learning_state.json：
{
  "updated_at": ...,
  "samples": int,
  "grades": [...],                  # 全部历史评级序列
  "stake_multiplier": float,        # 奖惩乘子（reward.reward_multiplier 维护）
  "signal_stats": {sig: {"n","hits","pnl_sum"}},
  "signal_baseline": {sig: mult},   # ∈ [0.5, 1.5]
  "model_stats": {key: {"n","hits"}},
  "model_weights": {key: w},        # 归一化
  "history": [{issue, grade, pnl, multiplier}, ...]
}
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .signal_engine import BASE_KEYS, DEFAULT_WEIGHTS, SIGNAL_NAMES

STATE_PATH = Path("state/learning_state.json")

LR_SIGNAL = 0.05          # 信号基线学习率
LR_MODEL = 0.15           # 模型权重学习率
BASE_MIN, BASE_MAX = 0.5, 1.5


def _default_state() -> dict:
    return {
        "updated_at": None,
        "samples": 0,
        "grades": [],
        "stake_multiplier": 1.0,
        "signal_stats": {s: {"n": 0, "hits": 0, "pnl_sum": 0.0} for s in SIGNAL_NAMES},
        "signal_baseline": {s: 1.0 for s in SIGNAL_NAMES},
        "model_stats": {k: {"n": 0, "hits": 0} for k in BASE_KEYS},
        "model_weights": dict(DEFAULT_WEIGHTS),
        "history": [],
    }


def load_state(path: Path | str | None = None) -> dict:
    p = Path(path) if path else STATE_PATH
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            base = _default_state()
            base.update(data)
            # 补齐新增键
            for s in SIGNAL_NAMES:
                base["signal_stats"].setdefault(s, {"n": 0, "hits": 0, "pnl_sum": 0.0})
                base["signal_baseline"].setdefault(s, 1.0)
            for k in BASE_KEYS:
                base["model_stats"].setdefault(k, {"n": 0, "hits": 0})
                base["model_weights"].setdefault(k, 0.2)
            return base
        except (json.JSONDecodeError, OSError):
            return _default_state()
    return _default_state()


def save_state(state: dict, path: Path | str | None = None) -> Path:
    p = Path(path) if path else STATE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    return p


def update_signal_baseline(state: dict) -> dict:
    """按各信号平均收益相对全局均值的偏差调整基线乘子。"""
    stats = state["signal_stats"]
    baselines = state["signal_baseline"]
    pnls_per_signal = {}
    for s, st in stats.items():
        if st["n"] >= 3:
            pnls_per_signal[s] = st["pnl_sum"] / st["n"]
    if not pnls_per_signal:
        return state
    gmean = sum(pnls_per_signal.values()) / len(pnls_per_signal)
    scale = max(abs(gmean), 1.0)
    for s, avg in pnls_per_signal.items():
        delta = LR_SIGNAL * ((avg - gmean) / scale)
        baselines[s] = float(min(max(baselines[s] + delta, BASE_MIN), BASE_MAX))
    return state


def update_model_weights(state: dict) -> dict:
    """按各模型 Top3 命中率做乘性更新并归一化。"""
    ms = state["model_stats"]
    w = dict(state["model_weights"])
    hit_rates = {k: (v["hits"] / v["n"]) for k, v in ms.items() if v["n"] >= 5}
    if not hit_rates:
        return state
    mean_hr = sum(hit_rates.values()) / len(hit_rates)
    for k, hr in hit_rates.items():
        factor = 1.0 + LR_MODEL * (hr - mean_hr) / max(mean_hr, 1e-6) * 0.5
        w[k] = float(min(max(w.get(k, 0.2) * factor, 0.02), 0.6))
    total = sum(w.values())
    state["model_weights"] = {k: v / total for k, v in w.items()}
    return state


def record_settlement(
    state: dict,
    issue: str,
    signal: str,
    grade: str,
    pnl: float,
    stake_multiplier: float,
    model_hits: dict[str, int] | None = None,
) -> dict:
    """写入一次结算并触发两级学习更新。

    model_hits: {model_key: top3_hit_count(0..3)}，可选。
    """
    st = state["signal_stats"].setdefault(signal, {"n": 0, "hits": 0, "pnl_sum": 0.0})
    st["n"] += 1
    st["pnl_sum"] += float(pnl)
    if grade in ("jackpot", "excellent", "bonus", "good"):
        st["hits"] += 1

    state["grades"].append(grade)
    state["samples"] = int(state.get("samples", 0)) + 1
    state["stake_multiplier"] = float(stake_multiplier)
    state["history"].append({
        "issue": issue, "signal": signal, "grade": grade,
        "pnl": round(float(pnl), 2), "multiplier": round(float(stake_multiplier), 3),
    })

    if model_hits:
        for k, h in model_hits.items():
            m = state["model_stats"].setdefault(k, {"n": 0, "hits": 0})
            m["n"] += 1
            m["hits"] += int(h)

    update_signal_baseline(state)
    update_model_weights(state)
    return state


def effective_weights(state: dict, signal: str, engine_weights: dict[str, float]) -> dict[str, float]:
    """最终权重 = 信号引擎建议权重 × 信号基线 × 模型学习权重（再归一化）。"""
    base = state["signal_baseline"].get(signal, 1.0)
    learned = state["model_weights"]
    w = {k: engine_weights.get(k, 0.2) * learned.get(k, 0.2) for k in BASE_KEYS}
    total = sum(w.values()) or 1.0
    w = {k: v / total for k, v in w.items()}
    # 信号基线只影响下注规模（通过 stake），不改变形状；此处保留给调用方使用
    return {"weights": w, "signal_baseline": float(base)}


def report(state: dict) -> str:
    lines = [
        f"学习状态报告  (样本 {state['samples']} 条, 更新时间 {state['updated_at']})",
        f"当前下注乘子: {state['stake_multiplier']:.3f}",
        "",
        "信号统计:",
    ]
    for s, st in state["signal_stats"].items():
        if st["n"]:
            lines.append(
                f"  {s:<18} n={st['n']:<4} 命中率={st['hits']/st['n']:.2f} "
                f"均益={st['pnl_sum']/st['n']:+.1f} 基线={state['signal_baseline'][s]:.3f}")
    lines.append("")
    lines.append("模型权重:")
    for k, v in state["model_weights"].items():
        ms = state["model_stats"].get(k, {"n": 0, "hits": 0})
        hr = f"{ms['hits']/ms['n']:.2f}" if ms["n"] else "-"
        lines.append(f"  {k:<15} w={v:.3f}  n={ms['n']}  Top3命中率={hr}")
    return "\n".join(lines)
