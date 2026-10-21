import numpy as np
import pandas as pd
import pytest

import notes


def synthetic(years=30, seed=1, storms=True):
    """Daily flow: a seasonal cycle plus planted storms that decay with a known half-life."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1990-01-01", periods=365 * years + years // 4)
    doy = dates.dayofyear.to_numpy()
    base = 100 + 60 * np.sin((doy - 100) / 365 * 2 * np.pi) + rng.normal(0, 2, len(dates))
    q = np.maximum(base, 20)
    if storms:
        alpha = np.log(2) / 5.0                     # planted half-life: 5 days
        for start in rng.choice(np.arange(200, len(dates) - 40), size=len(dates) // 40, replace=False):
            t = np.arange(40)
            q[start:start + 40] += 300 * np.exp(-alpha * t)
    return pd.DataFrame({"date": dates, "value": q})


def test_normal_bands_are_ordered_and_follow_the_seasons():
    bands = notes.normal_bands(synthetic(storms=False))
    assert len(bands) == 366 and bands.index[0] == "01-01"
    assert (bands["p10"] <= bands["p50"]).all() and (bands["p50"] <= bands["p90"]).all()
    assert bands.loc["07-01", "p50"] > bands.loc["01-01", "p50"] + 20      # the planted seasonal swing shows
    smooth_step = bands["p50"].diff().abs().max()
    assert smooth_step < 15                                              # smoothed, no jumps


def test_band_for_dates_lines_up_with_the_calendar():
    bands = notes.normal_bands(synthetic(storms=False))
    dates = pd.date_range("2024-02-28", "2024-03-02")                     # includes Feb 29
    aligned = notes.band_for_dates(bands, dates)
    assert list(aligned.index) == list(dates) and aligned.notna().all().all()


def test_percentile_for_date_ranks_against_other_years():
    df = synthetic(storms=False)
    df.loc[df.index[-1], "value"] = 10_000                                # an absurdly high last day
    assert notes.percentile_for_date(df) == pytest.approx(100.0)
    df.loc[df.index[-1], "value"] = 0.1
    assert notes.percentile_for_date(df) == pytest.approx(0.0)
    assert notes.percentile_for_date(df.iloc[:200]) is None              # too little history


def test_events_find_planted_storms_with_rise_and_fall_times():
    dates = pd.date_range("2020-01-01", periods=200)
    q = np.full(200, 50.0)
    for start, height in ((40, 400), (120, 250)):
        t = np.arange(40)
        q[start:start + 40] += height * np.exp(-np.log(2) / 4 * t)
    s = pd.Series(q, index=dates)
    found = notes.find_events(s)
    assert [e.peak_date.strftime("%m-%d") for e in found] == [dates[40].strftime("%m-%d"), dates[120].strftime("%m-%d")]
    first = found[0]
    assert first.ratio > 5 and first.rise_days >= 1
    assert first.half_fall_days is not None and 3 <= first.half_fall_days <= 5     # planted half-life: 4 days
    assert notes.find_events(pd.Series(np.full(100, 5.0), index=pd.date_range("2020-01-01", periods=100))) == []


def test_flashiness_separates_steady_from_flashy():
    steady = pd.Series(100 + np.sin(np.arange(200) / 30) * 5)
    flashy = pd.Series(np.where(np.arange(200) % 7 == 0, 500.0, 20.0))
    assert notes.flashiness(steady) < 0.1 < 0.5 < notes.flashiness(flashy)
    assert notes.flashiness(pd.Series([1.0, 2.0])) is None


def test_recession_half_life_recovers_the_planted_value():
    t = np.arange(20)
    cycle = 300 * np.exp(-np.log(2) / 5.0 * t)                           # pure exponential decay, half-life 5 days
    s = pd.Series(np.tile(cycle, 12), index=pd.date_range("2020-01-01", periods=240))
    half_life, n = notes.recession_half_life(s, months=range(1, 13))
    assert n >= 8 and half_life == pytest.approx(5.0, rel=0.05)
    none, few = notes.recession_half_life(notes._flow(synthetic(years=1, storms=False)))
    assert none is None and few == 0


def test_a_steady_baseflow_makes_the_half_life_look_longer():
    # The fit is on total flow, so storms riding on a big baseflow recede "slowly" in log space.
    half_life, _ = notes.recession_half_life(notes._flow(synthetic()), months=range(1, 13), min_events=3)
    assert half_life is not None and half_life > 5.0


def test_answer_key_and_description():
    df = synthetic()
    key = notes.answer_key(df, drainage_area_mi2=251.0)
    assert key["years_of_record"] > 29 and key["events"] and 0 <= key["percentile_today"] <= 100
    lines = notes.describe(key)
    assert any(l.startswith("Latest daily flow") for l in lines) and any("half-life" in l for l in lines)
    assert any("251 mi²" in l for l in lines)
