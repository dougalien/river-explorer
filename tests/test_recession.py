import math

import numpy as np
import pandas as pd
import pytest

import recession


def storms(k=20.0, days=400, every=40, peak=500.0, base=50.0, flags=None):
    """Fake daily flow: a storm every `every` days (day 0 rising, day 1 the peak), then a pure exponential fall with constant k."""
    dates = pd.date_range("2015-01-01", periods=days)
    q = np.empty(days)
    for i in range(days):
        t = i % every
        q[i] = (q[i - 1] * 1.5 if i else base) if t == 0 else peak * math.exp(-(t - 1) / k)
    df = pd.DataFrame({"date": dates, "value": q, "approved": True, "flags": ""})
    if flags:
        for day, f in flags.items():
            df.loc[day, "flags"] = f
    return df


def test_segments_recover_k_and_skip_the_days_after_the_peak():
    table, series = recession.segments(storms(k=20.0), skip=2, min_len=5)
    assert len(table) == 10 and len(series) == 10
    assert table["k_days"].to_numpy() == pytest.approx(20.0, rel=1e-6)
    assert table["daily_K"].to_numpy() == pytest.approx(math.exp(-1 / 20), rel=1e-6)
    assert table["half_life_days"].to_numpy() == pytest.approx(20 * math.log(2), rel=1e-6)
    assert table["q_start"].to_numpy() == pytest.approx(500 * math.exp(-2 / 20))   # two days after the peak
    assert (table["days"] == 37).all()                                  # days 3 to 39 of each storm and (table["r2"] > 0.999).all()


def test_flagged_days_and_months_break_segments():
    table, _ = recession.segments(storms(flags={10: "ICE"}), min_len=5)
    first = table.iloc[0]
    assert first["end"] < pd.Timestamp("2015-01-11")             # stopped before the ice day
    kept, _ = recession.segments(storms(flags={10: "ICE"}), skip_flagged=False)
    assert kept.iloc[0]["end"] > pd.Timestamp("2015-01-11")
    summer, _ = recession.segments(storms(), months=(6, 7, 8))
    assert set(summer["month"]) <= {6, 7, 8} and len(summer) < 10


def test_short_and_rising_runs_are_not_segments():
    df = storms(every=6)                                          # only a few falling days per storm
    table, series = recession.segments(df, min_len=5)
    assert table.empty and series == []
    assert recession.master_curve(series) is None


def test_master_curve_lines_segments_up_on_one_exponential():
    df = storms(k=15.0, peak=500.0)
    df.loc[df.index % 80 == 1, "value"] = 300.0                   # every other storm is smaller
    df.loc[df.index % 80 > 1, "value"] = [300 * math.exp(-((i % 80) - 1) / 15) if (i // 40) % 2 == 0 else v
                                          for i, v in df.loc[df.index % 80 > 1, "value"].items()]
    table, series = recession.segments(df)
    mrc = recession.master_curve(series)
    assert mrc["k_days"] == pytest.approx(15.0, rel=1e-6)
    for part in mrc["shifted"]:
        on_curve = mrc["q_top"] * np.exp(-part["t"].to_numpy() / mrc["k_days"])
        assert part["q"].to_numpy() == pytest.approx(on_curve, rel=1e-6)
    assert min(p["t"].iloc[0] for p in mrc["shifted"]) == 0


def test_brutsaert_points_and_a_linear_reservoir_fit():
    _, series = recession.segments(storms(k=25.0))
    pts = recession.dqdt_points(series)
    assert (pts["dqdt"] > 0).all() and len(pts) == sum(len(s) - 1 for s in series)
    fit = recession.fit_power_law(pts)
    assert fit["all"]["b"] == pytest.approx(1.0, abs=1e-6)
    assert fit["all"]["a"] == pytest.approx(2 * math.tanh(1 / 50), rel=1e-6)  # one day's drop over the two days' mean flow
    assert fit["envelope"]["b"] == pytest.approx(1.0, abs=0.01)
    assert "linear reservoir" in recession.describe_power_law(fit)[0]


def test_power_law_recovers_b_from_a_nonlinear_reservoir():
    # -dQ/dt = a Q^b integrates to Q(t) = (Q0^(1-b) - (1-b) a t)^(1/(1-b)); sample it daily
    a, b = 0.002, 1.5
    t = np.arange(60)
    q = (200.0 ** (1 - b) - (1 - b) * a * t) ** (1 / (1 - b))
    s = pd.Series(q, index=pd.date_range("2020-06-01", periods=60))
    fit = recession.fit_power_law(recession.dqdt_points([s]))
    assert fit["all"]["b"] == pytest.approx(b, abs=0.05)
    assert "slows as flow drops" in recession.describe_power_law(fit)[1]


def test_too_little_data_is_said_plainly():
    assert recession.fit_power_law(pd.DataFrame(columns=["date", "q", "dqdt"])) is None
    assert "Too few" in recession.describe_power_law(None)[0]
    assert recession.by_season(pd.DataFrame(columns=recession.SEGMENT_COLS)).empty


def test_by_season_summary():
    table, _ = recession.segments(storms(days=365 * 2))
    out = recession.by_season(table)
    assert list(out["Season"]) == ["Winter", "Spring", "Summer", "Fall"]
    assert out["Segments"].sum() == len(table)
    assert out["Median k (days)"].to_numpy() == pytest.approx(20.0, rel=1e-6)


def test_one_storm_in_15_minute_data_gives_k_in_days():
    times = pd.date_range("2026-10-01", periods=7 * 96, freq="15min")
    hours = (times - times[0]) / pd.Timedelta(hours=1)
    rng = np.random.default_rng(0)
    q = np.where(hours < 24, 80.0 + hours * 30, 800 * np.exp(-(hours - 24) / 24 / 1.5)) * (1 + rng.normal(0, 0.001, len(times)))
    h = recession.hourly(pd.DataFrame({"time": times, "value": q}))
    assert len(h) == 7 * 24 and (h["flags"] == "").all()
    table, series = recession.segments(h, skip=0.5, min_len=1, tol=0.005)
    assert len(table) == 1
    assert table["k_days"].iloc[0] == pytest.approx(1.5, rel=0.02)
    assert 5 < table["days"].iloc[0] < 6                      # 6 days of fall, minus the half day skipped
    mrc = recession.master_curve(series)
    assert mrc["k_days"] == pytest.approx(1.5, rel=0.05)
    pts = recession.dqdt_points(series)
    fit = recession.fit_power_law(pts)
    assert fit["all"]["b"] == pytest.approx(1.0, abs=0.1)
    assert fit["all"]["k_days"] == pytest.approx(1.5, rel=0.15)   # -dQ/dt is per day, so 1/a is in days
