from dataclasses import replace
from math import isclose

import pytest

from bakerylot.leftover_curve import service as demand
from bakerylot.leftover_curve.replay import Trial, masked_station, window_selection
from bakerylot.pipeline import run


def receipts():
    data = []
    for slot in range(1, 20):
        for station, name in (("北柜", "青麦"), ("南柜", "碱水")):
            row = demand.DailyFinish.parse({
                "historical_lot": f"{station}-{slot}",
                "day": slot,
                "station": station,
                "sku": name,
                "remaining": slot+2,
                "minutes": 20+slot,
                "late_sales": slot//2,
                "daypart": slot % 7,
                "rain_mm": 0,
                "censored": 0,
                "observed_day": slot,
            })
            data.append(row)
    return data


def test_temporal_trial_recovers_late_volume_and_keeps_cabinet_filter():
    samples = receipts()
    kept = {("北柜", "青麦")}
    outcome, denied = demand.recent_replay(samples, 20, kept, 6)
    assert not denied
    assert [line.day for line in outcome] == list(range(14,20))
    assert all(line.sku == "青麦" and line.total == 1 for line in outcome)
    stat, = demand.replay_summary(outcome)
    assert stat["模型平均绝对误差"] < stat["历史中位绝对误差"]
    assert stat["回放日"] == 6
    assert stat["回放炉次"] == 6
    assert stat["误差改善率"] > 0.5


def test_retroactive_edit_can_only_affect_trial_after_publication():
    raw = receipts()
    admitted = {("北柜", "青麦")}
    historic = next(row for row in raw if row.historical_lot == "北柜-4")
    late = replace(historic, historical_lot="补录", observed_day=17, late_sales=6)
    baseline,_ = demand.recent_replay(raw,20,admitted,6)
    mutated,_ = demand.recent_replay(raw+[late],20,admitted,6)
    assert baseline[:4] == mutated[:4]
    assert baseline[4:] != mutated[4:]


def test_unknown_observations_and_sold_out_aggregates_stay_distinct():
    cases = [Trial("芝麻",10,2,0,1,1,0),Trial("芝麻",11,3,0,2,2,1)]
    view, = demand.replay_summary(cases)
    assert view["模型平均绝对误差"] == 0.6
    assert view["历史中位绝对误差"] == 0.0
    assert view["误差改善率"] is None
    assert view["原价售罄"] == 1
    evidence = cases[1].evidence()
    assert evidence["原价售罄"] == 1
    assert evidence["预估不足炉次"] == 2


def test_holdout_leaves_no_station_rows_in_training():
    before, after = masked_station(receipts(), "北柜", 18)
    assert len(before) == len(after) == 17
    assert {row.station for row in before} == {"南柜"}
    assert {row.station for row in after} == {"北柜"}
    accepted = demand.replay_held_stations(receipts(),20)
    assert {row["柜台"] for row in accepted} == {"南柜","北柜"}
    assert all(row["状态"]=="已验证" for row in accepted)
    assert isclose(accepted[0]["模型误差"], accepted[1]["模型误差"])


def test_short_history_has_explicit_cold_start_status():
    result = demand.replay_held_stations(receipts(),4,min_training=10)
    assert len(result)==2
    assert result[0]["验证量"]==3
    assert result[0]["模型误差"] is None
    trials, denied=demand.recent_replay(receipts(),4,{("南柜","碱水")},2)
    assert len(denied) == len(trials) == 1
    assert trials[0].day == 3
    assert {item["结果"] for item in denied}=={"训练样本不足"}


@pytest.mark.parametrize("budget", [1,26,69,100])
def test_live_workbench_budget_and_receipt_proofs_reconcile(budget):
    report=run(budget)
    summary=report["ranking_scenario"]
    official=sum(row["reserved"] for row in report["rows"])
    assert summary["budget_cents"] == budget*100
    assert summary["reservations"][-1]["累计分"] == official <= budget*100
    assert report["metrics"]["预留折扣（元）"] == official/100
    assert sum(record["登记实收（分）"] for record in report["stages"][-1]["records"] if "柜台" in record) >= report["metrics"]["通过收据实收（元）"]*100
    assert report["model_replay"]
    assert all(0<=item["原价售罄"]<=item["留出炉次数"] for item in report["model_replay"])



def test_dates_after_cutoff_do_not_add_trials():
    stream = receipts()
    late = replace(stream[-1], historical_lot="次日登记", day=25, observed_day=25)
    selection = {("北柜", "青麦"),("南柜", "碱水")}
    original = demand.recent_replay(stream, 20, selection, 5)
    injected = demand.recent_replay(stream + [late], 20, selection, 5)
    assert original == injected
    assert {trial.day for trial in injected[0]} == set(range(15, 20))


def test_censored_sales_never_create_a_perfect_baseline():
    sample = receipts()
    original = sample[-1]
    corrected = replace(original, historical_lot="停电停售", censored=1, late_sales=0)
    first = demand.recent_replay(sample, 20, {("南柜", "碱水")}, 5)
    second = demand.recent_replay(sample+[corrected], 20, {("南柜", "碱水")}, 5)
    assert first == second
    assert len(first[0]) == 5


def trial(sku, day, *, long_error, long_missed, short_error, short_missed, short_samples, total=1):
    return Trial(sku=sku, day=day, total=total, baseline_error=0.0, model_error=long_error,
                 missed_stock=long_missed, soldout=0, short_error=short_error,
                 short_missed=short_missed, short_samples=short_samples)


def test_short_window_only_wins_when_error_drops_without_more_under_prediction():
    cases = [
        trial("甲", 10, long_error=4.0, long_missed=1, short_error=3.0, short_missed=1, short_samples=7),
        trial("丙", 10, long_error=4.0, long_missed=1, short_error=2.0, short_missed=2, short_samples=7),
        trial("丁", 10, long_error=2.0, long_missed=0, short_error=3.0, short_missed=0, short_samples=7),
    ]
    selection = {row["商品"]: row for row in window_selection(cases)}
    assert selection["甲"]["训练窗口"] == "短窗"
    assert selection["丙"]["训练窗口"] == "长窗"
    assert selection["丁"]["训练窗口"] == "长窗"


def test_short_window_with_fewer_than_four_samples_falls_back_to_long():
    case = trial("戊", 10, long_error=4.0, long_missed=1, short_error=None,
                 short_missed=None, short_samples=3)
    row, = window_selection([case])
    assert row["训练窗口"] == "长窗"
    assert row["短窗可用日"] == 0
    assert row["短窗平均绝对误差"] == row["长窗平均绝对误差"]
    assert row["短窗低估炉次"] == row["长窗低估炉次"]


def test_both_windows_are_scored_on_the_same_held_out_lots():
    # 同一批留出炉次既进长窗误差，也进短窗误差；短窗训练样本严格来自最近一周。
    samples = receipts()
    outcome, _ = demand.recent_replay(samples, 20, {("北柜", "青麦")}, 6)
    assert all(line.short_samples <= 7 for line in outcome)
    assert all(line.short_samples >= 4 for line in outcome)
    row, = window_selection(outcome)
    assert row["回放炉次"] == sum(line.total for line in outcome)
    assert row["短窗平均绝对误差"] != row["长窗平均绝对误差"]


def test_short_window_does_not_learn_calibration_or_future_sales():
    samples = receipts()
    kept = {("北柜", "青麦")}
    baseline = demand.recent_replay(samples, 20, kept, 6)
    # 回放日 19 的短窗取 12-18 日；把 19 日之后的实销改大不应影响它。
    future = replace(samples[-1], historical_lot="未来", day=25, observed_day=25, late_sales=40)
    edited = demand.recent_replay(samples + [future], 20, kept, 6)
    assert edited == baseline


def history(*, days=27, sku="青麦"):
    return [demand.DailyFinish.parse({
        "historical_lot": f"L{day:02}", "day": day, "station": "北柜", "sku": sku,
        "remaining": 20 + day % 5, "minutes": 20 + day * 3, "late_sales": 4 + day // 3,
        "daypart": day % 7, "rain_mm": float(day % 4), "censored": 0, "observed_day": day,
    }) for day in range(1, days + 1)]


def test_calibration_sales_stay_out_of_training_but_still_set_the_band():
    # 校准实销不进训练：改动校准段（第 23–27 日）只改半径，不改曲线。
    from bakerylot.pipeline import assess_model
    rows = history(days=27)
    for window in ("长窗", "短窗"):
        curve, band, source = assess_model(rows, 28, "青麦", 0.8, window)
        assert source == window
        edited = [replace(row, late_sales=30) if row.day == 25 else row for row in rows]
        curve2, band2, _ = assess_model(edited, 28, "青麦", 0.8, window)
        assert curve2 == curve
        assert band2 != band
        trained = [replace(row, late_sales=30) if row.day == 22 else row for row in rows]
        assert assess_model(trained, 28, "青麦", 0.8, window)[0] != curve
