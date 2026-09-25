# -*- coding: utf-8 -*-
"""单选反奖率逻辑单元测试"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lottery3d_v2.predictors.anti_prize import (  # noqa: E402
    ANTI_PRIZE_BUCKETS,
    AntiPrizeAnalyzer,
    anti_prize_penalty,
    anti_prize_rate,
    classify_crowd_share,
    estimate_crowding,
    number_index,
)
from lottery3d_v2.schema import HistoryRow  # noqa: E402


def make_history(n=120):
    rng = np.random.default_rng(7)
    rows = []
    for i in range(n):
        d = tuple(int(x) for x in rng.integers(0, 10, 3))
        rows.append(HistoryRow(issue=str(2026000 + i), digits=d,
                               sales=11_000_000.0, prize=5_000_000.0,
                               single_wins=4000.0, jackpot=2_000_000.0))
    return rows


def test_number_index():
    assert number_index((4, 8, 6)) == 486
    assert number_index((0, 0, 0)) == 0


def test_estimate_crowding_is_distribution():
    crowd = estimate_crowding(make_history())
    assert crowd.shape == (1000,)
    assert abs(crowd.sum() - 1.0) < 1e-9
    assert (crowd > 0).all()


def test_leopard_more_crowded_than_scatter():
    crowd = estimate_crowding(make_history())
    # 豹子形态加分应显著高于无形态号码（同数字频率量级下）
    assert crowd[number_index((5, 5, 5))] > crowd[number_index((1, 4, 7))]


def test_bucket_classification():
    # 按反奖率区间划分（经验标定后口径）
    assert classify_crowd_share(0.0002) == "cold"       # rate≈0.125
    assert classify_crowd_share(0.0006) == "normal"     # rate≈0.328
    assert classify_crowd_share(0.0009) == "warm"       # rate≈0.479
    assert classify_crowd_share(0.0013) == "hot"        # rate≈0.682
    assert classify_crowd_share(0.0019) == "crowded"    # rate≈0.985
    assert len(ANTI_PRIZE_BUCKETS) == 5


def test_anti_prize_rate_bucket_mapping():
    # 无销量口径时退化为线性标定映射
    assert abs(anti_prize_rate(0.0002) - (0.024 + 506 * 0.0002)) < 1e-9
    assert anti_prize_rate(0.0) == 0.0


def test_anti_prize_rate_with_sales():
    # 第一性原理折算：own + pool
    r = anti_prize_rate(0.0005, sales_estimate=10_000_000.0,
                        total_bets_estimate=5_000_000.0)
    expect = 1040 / 10_000_000 + 0.0005 * 5_000_000 * 1040 / 10_000_000
    assert abs(r - expect) < 1e-12


def test_penalty_monotonic():
    assert anti_prize_penalty(0.30) == 1.0
    p1, p2 = anti_prize_penalty(0.5), anti_prize_penalty(0.8)
    assert 0 < p2 < p1 < 1


def test_analyzer_output_and_greedy():
    hist = make_history()
    an = AntiPrizeAnalyzer()
    res = an.analyze(hist, (4, 5, 6))
    for key in ("single_ratio", "anti_prize_rate", "bucket",
                "best_value_number", "jackpot_level", "greedy_mode"):
        assert key in res
    assert res["greedy_mode"] is True  # jackpot=2,000,000 → 贪婪模式

    # 奖池为 0 时不触发贪婪
    for row in hist:
        row_jp = row
    hist2 = [HistoryRow(issue=r.issue, digits=r.digits, sales=r.sales,
                        prize=r.prize, jackpot=0.0) for r in hist]
    res2 = an.analyze(hist2, (4, 5, 6))
    assert res2["greedy_mode"] is False


def test_analyzer_with_positions_proba_picks_value_number():
    hist = make_history()
    an = AntiPrizeAnalyzer()
    proba = [np.full(10, 0.1) for _ in range(3)]
    res = an.analyze(hist, (5, 5, 5), positions_proba=proba)
    # 均匀模型概率下应避开拥挤的豹子号
    assert res["best_value_number"] != (5, 5, 5)
    assert res["bucket"] in {b[0] for b in ANTI_PRIZE_BUCKETS}


def test_calibrated_share_to_rate():
    # 与 forward_log.csv 经验标定一致：rate = 0.024 + 506*share
    from lottery3d_v2.predictors.anti_prize import share_to_rate
    assert abs(share_to_rate(0.0002) - 0.1252) < 1e-9   # 日志最小档
    assert abs(share_to_rate(0.0019) - 0.9854) < 1e-9   # 日志最大档附近
    assert share_to_rate(0.0) == 0.0


def test_calibrated_buckets():
    from lottery3d_v2.predictors.anti_prize import classify_anti_prize_rate
    assert classify_anti_prize_rate(0.1265) == "cold"
    assert classify_anti_prize_rate(0.30) == "normal"
    assert classify_anti_prize_rate(0.50) == "warm"
    assert classify_anti_prize_rate(0.70) == "hot"
    assert classify_anti_prize_rate(0.9889) == "crowded"


def test_forward_log_reproduction():
    """用重建逻辑复算 data/forward_log.csv 的 rate-share 线性关系。"""
    import csv
    from pathlib import Path
    from lottery3d_v2.predictors.anti_prize import share_to_rate
    p = Path(__file__).parent.parent / "data" / "forward_log.csv"
    if not p.exists():
        import pytest
        pytest.skip("forward_log.csv 不存在")
    errs = []
    with open(p, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                share = float(row["单选占比"]); rate = float(row["反奖率"])
            except (ValueError, KeyError):
                continue
            errs.append(abs(share_to_rate(share) - rate))
    assert errs, "日志中应有可解析的反奖率记录"
    assert max(errs) < 0.06, f"标定偏差过大: {max(errs)}"
