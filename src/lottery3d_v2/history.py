# -*- coding: utf-8 -*-
"""历史数据加载与处理"""
from __future__ import annotations

import csv
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from .schema import HistoryRow
from .utils import DIGIT_N


def parse_digits(value: Any) -> tuple[int, int, int]:
    """解析开奖号码为三位元组"""
    if value is None:
        raise ValueError("开奖号码为空")

    if isinstance(value, (tuple, list, np.ndarray)):
        parts = [str(x) for x in value]
    else:
        s = str(value).strip()
        s = s.replace('"', "").replace("'", "")
        s = s.replace(",", " ")
        parts = s.split()

    if len(parts) == 1:
        token = parts[0].strip()
        if len(token) == 3 and token.isdigit():
            parts = list(token)

    if len(parts) != 3:
        raise ValueError(f"无法解析开奖号码: {value}")

    return tuple(int(float(x)) for x in parts)


def to_float(value: Any) -> float | None:
    """安全转换为浮点数"""
    if value is None:
        return None
    try:
        s = str(value).strip().replace(",", "")
        if s == "":
            return None
        return float(s)
    except (ValueError, TypeError) as e:
        logger.debug(f"Failed to convert {value!r} to float: {e}")
        return None


def _row_from_mapping(row: dict) -> HistoryRow | None:
    """从字典映射构建 HistoryRow"""
    raw = (
        row.get("front_winning_num")
        or row.get("开奖号码")
        or row.get("digits")
    )

    if raw is None:
        return None

    digits = parse_digits(raw)

    issue = row.get("期号") or row.get("issue")
    date = row.get("开奖日期") or row.get("date")

    sales = to_float(row.get("sales_amount", row.get("销售总额")))
    prize = to_float(row.get("prize_amount", row.get("中奖总额")))
    single_wins = to_float(
        row.get("single_winning_count", row.get("单选中奖注数"))
    )

    return HistoryRow(
        issue=str(issue) if issue is not None else None,
        digits=digits,
        sales=sales,
        prize=prize,
        single_wins=single_wins,
        date=str(date) if date is not None else None,
    )


def ensure_history(history: Any) -> list[HistoryRow]:
    """统一转换为 HistoryRow 列表"""
    if history is None:
        return []

    if hasattr(history, "to_dict"):
        history = history.to_dict("records")

    result: list[HistoryRow] = []

    for item in history:
        if isinstance(item, HistoryRow):
            result.append(item)
        elif isinstance(item, dict):
            row = _row_from_mapping(item)
            if row is not None:
                result.append(row)
        else:
            result.append(
                HistoryRow(
                    issue=None,
                    digits=parse_digits(item),
                    sales=None,
                    prize=None,
                    single_wins=None,
                    date=None,
                )
            )

    return result


def load_history_from_csv(
    path: str | Path,
    ascending: bool = True,
) -> list[HistoryRow]:
    """读取 CSV 历史数据

    ascending=True: 从旧到新，history[-1] 是最新一期
    """
    path = Path(path)
    rows: list[HistoryRow] = []

    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for raw in reader:
            row = _row_from_mapping(raw)
            if row is not None:
                rows.append(row)

    def sort_key(row: HistoryRow):
        issue_num = (
            int(row.issue)
            if row.issue is not None and str(row.issue).isdigit()
            else -1
        )
        return row.date or "", issue_num

    rows.sort(key=sort_key, reverse=not ascending)
    return rows


def load_history_incremental(
    path: str | Path,
    state_file: str | Path | None = None,
) -> tuple[list[HistoryRow], dict]:
    """增量加载历史数据

    返回: (history_rows, state_dict)
    state_dict 包含文件 mtime、行数等，用于下次判断是否需重新加载
    """
    path = Path(path)
    state_file = Path(state_file) if state_file else path.with_suffix(".state.json")

    current_mtime = path.stat().st_mtime
    current_size = path.stat().st_size

    # 尝试读取状态
    last_state = {}
    if state_file.exists():
        import json
        with open(state_file, encoding="utf-8") as f:
            last_state = json.load(f)

    # 检查是否需要重新加载
    if (last_state.get("mtime") == current_mtime
            and last_state.get("size") == current_size
            and "history" in last_state):
        # 直接返回缓存的历史数据
        history = [HistoryRow(**row) for row in last_state["history"]]
        return history, last_state

    # 全量加载
    history = load_history_from_csv(path, ascending=True)

    # 保存状态
    new_state = {
        "mtime": current_mtime,
        "size": current_size,
        "rows": len(history),
        "latest_issue": history[-1].issue if history else None,
        "history": [row.__dict__ for row in history],
    }

    import json
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(new_state, f, ensure_ascii=False, indent=2)

    return history, new_state


def digits_matrix(history: list[HistoryRow]) -> np.ndarray:
    """转换为数字矩阵 (n_samples, 3)"""
    if not history:
        return np.empty((0, 3), dtype=int)
    return np.array([row.digits for row in history], dtype=int)


def file_sha256(path: str | Path, chunk_size: int = 65536) -> str:
    """计算文件 SHA256"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def short_hash(hash_value: str, length: int = 16) -> str:
    return hash_value[:length]


def digits_sha256(history: list[HistoryRow]) -> str:
    """只对开奖号码核心字段计算逻辑哈希"""
    h = hashlib.sha256()
    rows = sorted(history, key=lambda r: (r.date or "", r.issue or ""))

    for row in rows:
        line = (
            f"{row.issue or ''}"
            f"|{row.date or ''}"
            f"|{row.digits[0]},{row.digits[1]},{row.digits[2]}\n"
        )
        h.update(line.encode("utf-8"))
    return h.hexdigest()


def market_sha256(history: list[HistoryRow]) -> str:
    """对开奖号码 + 市场字段计算逻辑哈希"""
    h = hashlib.sha256()
    rows = sorted(history, key=lambda r: (r.date or "", r.issue or ""))

    for row in rows:
        line = (
            f"{row.issue or ''}"
            f"|{row.date or ''}"
            f"|{row.digits[0]},{row.digits[1]},{row.digits[2]}"
            f"|{row.single_wins if row.single_wins is not None else ''}"
            f"|{row.sales if row.sales is not None else ''}"
            f"|{row.prize if row.prize is not None else ''}\n"
        )
        h.update(line.encode("utf-8"))
    return h.hexdigest()


def snapshot_meta(csv_path: str | Path) -> dict:
    """生成数据快照元信息"""
    csv_path = Path(csv_path)
    history = load_history_from_csv(csv_path, ascending=True)

    if not history:
        return {
            "csv_path": str(csv_path),
            "file_sha256": short_hash(file_sha256(csv_path)),
            "rows": 0,
            "latest_issue": None,
            "latest_date": None,
            "earliest_issue": None,
            "earliest_date": None,
            "digits_sha256": None,
            "market_sha256": None,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }

    sorted_history = sorted(history, key=lambda r: (r.date or "", r.issue or ""))
    earliest = sorted_history[0]
    latest = sorted_history[-1]

    return {
        "csv_path": str(csv_path),
        "file_sha256": short_hash(file_sha256(csv_path)),
        "rows": len(history),
        "latest_issue": latest.issue,
        "latest_date": latest.date,
        "earliest_issue": earliest.issue,
        "earliest_date": earliest.date,
        "digits_sha256": short_hash(digits_sha256(history)),
        "market_sha256": short_hash(market_sha256(history)),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def print_snapshot(csv_path: str | Path) -> dict:
    meta = snapshot_meta(csv_path)
    print("=" * 80)
    print("数据快照")
    print("=" * 80)
    for key, value in meta.items():
        print(f"{key}: {value}")
    print()
    return meta