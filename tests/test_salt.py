import numpy as np
import pandas as pd
import pytest

import salt


def daily(start="2010-01-01", end="2019-12-31", slope=10.0, seed=0):
    """Fake daily conductance: rises `slope` per year, higher in summer and fall, with day-to-day noise."""
    dates = pd.date_range(start, end)
    rng = np.random.default_rng(seed)
    years = dates.year - dates.year[0]
    season = np.where(dates.month.isin(range(6, 12)), 100.0, 0.0)
    return pd.DataFrame({"date": dates, "value": 400 + slope * years + season + rng.normal(0, 5, len(dates)),
                         "approved": True, "flags": ""})


def test_season_days_keeps_only_the_chosen_months():
    d = salt.season_days(daily(), salt.SEASONS["summer_fall"][1])
    assert set(d["date"].dt.month) == {6, 7, 8, 9, 10, 11}


def test_annual_means_sd_se_and_counts():
    t = salt.aggregate(daily(), salt.SEASONS["summer_fall"][1], "year")
    assert len(t) == 10 and (t["n"] == 183).all() and (t["possible"] == 183).all()
    assert t["mean"].iloc[0] == pytest.approx(500, abs=2) and t["mean"].iloc[-1] == pytest.approx(590, abs=2)
    assert (t["sd"] > 3).all() and (t["sd"] < 7).all()
    assert np.allclose(t["se"], t["sd"] / np.sqrt(t["n"]))
    assert (t["period"].dt.month == 9).all()                          # plotted at mid-season
    assert (t["month"] == 0).all()


def test_monthly_means():
    t = salt.aggregate(daily(), salt.SEASONS["fall"][1], "month")
    assert len(t) == 30 and set(t["month"]) == {9, 10, 11}
    assert (t["period"].dt.day == 15).all()
    assert t.loc[t["month"] == 10, "possible"].eq(31).all()


def test_a_thin_year_is_left_out_unless_coverage_is_relaxed():
    d = daily()
    d = d[~((d["date"].dt.year == 2012) & (d["date"].dt.month != 9))]   # 2012 has only September
    months = salt.SEASONS["summer_fall"][1]
    assert 2012 not in salt.aggregate(d, months, "year")["year"].tolist()
    assert 2012 in salt.aggregate(d, months, "year", min_coverage=0)["year"].tolist()


def test_empty_and_bad_grouping():
    d = daily("2010-01-01", "2010-03-31")
    assert salt.aggregate(d, salt.SEASONS["summer"][1]).empty
    with pytest.raises(ValueError):
        salt.aggregate(d, (1,), "week")


def test_trend_finds_a_known_slope_and_significance():
    years = np.arange(2000, 2020)
    t = salt.trend(years, 300 + 7.5 * (years - 2000))
    assert t["slope"] == pytest.approx(7.5) and t["p"] < 0.001 and t["tau"] == pytest.approx(1.0)
    assert t["intercept"] + t["slope"] * 2000 == pytest.approx(300)
    assert (t["first_year"], t["last_year"], t["n"]) == (2000, 2019, 20)


def test_trend_on_noise_is_not_significant_and_needs_five_years():
    rng = np.random.default_rng(3)
    years = np.arange(2000, 2020)
    assert salt.trend(years, 500 + rng.normal(0, 20, 20))["p"] > 0.05
    assert salt.trend([2000, 2001, 2002, 2003], [1, 2, 3, 4]) is None
    assert salt.trend(years, np.full(20, 400.0))["slope"] == 0 and salt.trend(years, np.full(20, 400.0))["p"] == 1.0


def test_mann_kendall_matches_a_hand_count():
    # S = 8 for this series (9 pairs up, 1 down, 0 ties among 5 points), var = 5*4*15/18 = 16.67
    t = salt.trend([1, 2, 3, 4, 5], [1, 3, 2, 4, 5])
    assert t["tau"] == pytest.approx(0.8)
    assert t["p"] == pytest.approx(0.0864, abs=0.001)


def test_describe_is_careful():
    years = np.arange(2000, 2020)
    up = salt.describe(salt.trend(years, 300 + 7.5 * (years - 2000)))
    assert "rising" in up and "statistically significant" in up and "7.5 µS/cm per year" in up
    assert "Fewer than 5 years" in salt.describe(None)
