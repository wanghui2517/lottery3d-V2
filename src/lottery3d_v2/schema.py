# -*- coding: utf-8 -*-
"""数据模型定义"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _to_jsonable(obj: Any) -> Any:
    """转换为 JSON 可序列化对象"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return obj


@dataclass(frozen=True)
class HistoryRow:
    """统一历史开奖行

    digits 固定为三位：
        (百位, 十位, 个位)
    """

    issue: str | None
    digits: tuple[int, int, int]
    sales: float | None = None
    prize: float | None = None
    single_wins: float | None = None
    date: str | None = None
    jackpot: float | None = None        # 奖池金额（可为空以兼容旧数据）


@dataclass
class Prediction:
    """统一预测结果"""

    name: str
    positions: list[list[int]]  # [[百位topk], [十位topk], [个位topk]]
    sum_value: list[int]        # 和值topk
    positions_proba: list[np.ndarray]  # [位置0概率分布, 位置1, 位置2]
    sum_proba: np.ndarray       # 和值概率分布
    best_pair: dict[str, Any] | None = None  # 最佳两位组合
    meta: dict[str, Any] = field(default_factory=dict)  # 额外元信息

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable({
            "name": self.name,
            "positions": self.positions,
            "sum_value": self.sum_value,
            "positions_proba": self.positions_proba,
            "sum_proba": self.sum_proba,
            "best_pair": self.best_pair,
            "meta": self.meta,
        })


@dataclass
class ProbaResult:
    """预测器内部概率结果"""

    positions_proba: list[np.ndarray]  # 长度3，每个 shape (10,)
    sum_proba: np.ndarray              # shape (28,)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable({
            "positions_proba": self.positions_proba,
            "sum_proba": self.sum_proba,
            "meta": self.meta,
        })