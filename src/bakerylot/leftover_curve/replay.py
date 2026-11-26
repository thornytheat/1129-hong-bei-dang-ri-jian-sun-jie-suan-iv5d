from collections import defaultdict
from dataclasses import dataclass
from math import fsum
from statistics import median

from .analysis import (
    MIN_SHORT_SAMPLES,
    MIN_TRAINING_SAMPLES,
    chronological_sample,
    curve_for,
    fit,
    one,
    points,
    short_window_sample,
)


@dataclass(frozen=True, slots=True)
class Trial:
    sku: str
    day: int
    total: int
    baseline_error: float
    model_error: float
    missed_stock: int
    soldout: int
    short_error: float | None = None
    short_missed: int | None = None
    short_samples: int = 0

    def evidence(self):
        return {
            "商品": self.sku,
            "业务日": self.day,
            "留出炉次数": self.total,
            "朴素基线总误差": round(self.baseline_error, 3),
            "模型总误差": round(self.model_error, 3),
            "短窗总误差": None if self.short_error is None else round(self.short_error, 3),
            "原价售罄": self.soldout,
            "预估不足炉次": self.missed_stock,
            "短窗预估不足炉次": self.short_missed,
            "短窗训练样本": self.short_samples,
        }


def recent_replay(history, day, station_skus, days):
    by_key = defaultdict(list)
    for item in chronological_sample(history, day):
        if (item.station, item.sku) in station_skus:
            by_key[item.sku, item.day].append(item)
    trials = []
    denied = []
    for (sku, date), fixtures in sorted(by_key.items()):
        if date < day - days:
            continue
        candidates = chronological_sample(history, date, product=sku)
        if len(candidates) < MIN_TRAINING_SAMPLES:
            denied.append({"商品": sku, "业务日": date, "结果": "训练样本不足"})
            continue
        regression = fit(candidates, date, product=sku)
        recent = short_window_sample(history, date, product=sku)
        short_regression = curve_for(recent) if len(recent) >= MIN_SHORT_SAMPLES else None
        naive = median(row.late_sales for row in candidates)
        raw_errors = []
        estimated_errors = []
        missed = 0
        soldout = 0
        short_errors = []
        short_missed = 0
        for row in fixtures:
            expectation = one(regression, points(row))
            raw_errors.append(abs(min(row.remaining, naive) - row.late_sales))
            estimated_errors.append(abs(expectation - row.late_sales))
            missed += expectation + 0.5 < row.late_sales
            soldout += row.remaining == row.late_sales
            if short_regression is not None:
                estimate = one(short_regression, points(row))
                short_errors.append(abs(estimate - row.late_sales))
                short_missed += estimate + 0.5 < row.late_sales
        trials.append(Trial(
            sku, date, len(fixtures), fsum(raw_errors), fsum(estimated_errors), missed, soldout,
            short_error=fsum(short_errors) if short_regression is not None else None,
            short_missed=short_missed if short_regression is not None else None,
            short_samples=len(recent),
        ))
    return trials, denied


def replay_summary(trials):
    groups = defaultdict(list)
    for trial in trials:
        groups[trial.sku].append(trial)
    outcomes = []
    for sku, group in sorted(groups.items()):
        count = sum(line.total for line in group)
        baseline = fsum(line.baseline_error for line in group)
        estimate = fsum(line.model_error for line in group)
        signed = (baseline - estimate) / baseline if baseline else None
        outcomes.append({
            "商品": sku,
            "回放日": len({line.day for line in group}),
            "回放炉次": count,
            "历史中位绝对误差": round(baseline/count, 3) if count else None,
            "模型平均绝对误差": round(estimate/count, 3) if count else None,
            "误差改善率": round(signed, 4) if signed is not None else None,
            "原价售罄": sum(line.soldout for line in group),
        })
    return outcomes


def projected(trial: Trial) -> tuple[float, int]:
    """该回放日实际会用的误差与低估炉次：短窗可用时用短窗，否则沿用长窗。"""
    if trial.short_error is None or trial.short_missed is None:
        return trial.model_error, trial.missed_stock
    return trial.short_error, trial.short_missed


def window_selection(trials):
    """长短窗用同一批留出炉次比实销；短窗误差更小且低估炉次没增加才启用。

    短窗少于四条同款训练样本的日子沿用长窗，一并计入短窗方案的总误差。
    """
    groups = defaultdict(list)
    for trial in trials:
        groups[trial.sku].append(trial)
    rows = []
    for sku, group in sorted(groups.items()):
        count = sum(line.total for line in group)
        short_days = sum(1 for line in group if line.short_error is not None)
        long_error = fsum(line.model_error for line in group)
        short_error = fsum(projected(line)[0] for line in group)
        long_missed = sum(line.missed_stock for line in group)
        short_missed = sum(projected(line)[1] for line in group)
        enable = bool(short_days and count and short_error < long_error
                      and short_missed <= long_missed)
        rows.append({
            "商品": sku,
            "回放日": len({line.day for line in group}),
            "回放炉次": count,
            "短窗可用日": short_days,
            "长窗平均绝对误差": round(long_error / count, 3) if count else None,
            "短窗平均绝对误差": round(short_error / count, 3) if count else None,
            "长窗低估炉次": long_missed,
            "短窗低估炉次": short_missed,
            "训练窗口": "短窗" if enable else "长窗",
        })
    return rows


def masked_station(history, station, day):
    eligible = chronological_sample(history, day)
    visible = [line for line in eligible if line.station != station]
    absent = [line for line in eligible if line.station == station]
    return visible, absent


def replay_held_stations(history, day, *, min_training=15):
    results = []
    for station in sorted({row.station for row in history}):
        anchors, trials = masked_station(history, station, day)
        if len(anchors) < min_training or not trials:
            results.append({"柜台": station, "验证量": len(trials), "模型误差": None, "状态": "样本量不足"})
            continue
        trained = fit(anchors, day)
        errors = [abs(one(trained, points(row)) - row.late_sales) for row in trials]
        count = len(errors)
        results.append({"柜台": station, "验证量": count, "模型误差": round(fsum(errors)/count, 3), "状态": "已验证"})
    return results
