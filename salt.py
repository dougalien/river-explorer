"""Specific conductance over the years at a gage, to see whether road salt is building up in a stream.

The method is the one done by hand: take the daily mean specific conductance, keep only the summer and fall months
(base flow, when the stream is fed by groundwater rather than snowmelt or winter runoff), average each year or each
month, and plot those means with error bars against date. Chloride from winter salting soaks into groundwater and
comes back out as base flow for years, so a salted watershed shows base-flow conductance climbing year after year.

The trend is Sen's slope with a Mann-Kendall test, the usual nonparametric pair for water-quality trends, always
computed on the annual means (monthly means within a season rise and fall with flow, which would muddy a trend
test). Nothing here imports Streamlit.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

SEASONS = {
    "summer_fall": ("Summer and fall (Jun to Nov)", (6, 7, 8, 9, 10, 11)),
    "summer": ("Summer (Jun to Aug)", (6, 7, 8)),
    "fall": ("Fall (Sep to Nov)", (9, 10, 11)),
    "all": ("All months", tuple(range(1, 13))),
}
ERROR_BARS = {
    "sd": "Standard deviation (SD) of the daily means: how much conductance varied within the year or month",
    "se": "Standard error (SE) of the mean, SD / √n: how precisely the mean is known",
}


def season_days(daily: pd.DataFrame, months: tuple[int, ...]) -> pd.DataFrame:
    """Daily rows (date, value, ...) that fall in the chosen calendar months."""
    return daily[daily["date"].dt.month.isin(months)].reset_index(drop=True)


def aggregate(daily: pd.DataFrame, months: tuple[int, ...], by: str = "year", min_coverage: float = 0.5) -> pd.DataFrame:
    """Mean, SD, SE and day count of daily values per year (within the season) or per month.

    A year or month is kept only if at least `min_coverage` of its days have data, so a year with only a wet
    September is not set beside full seasons. Columns: period (a date to plot at), year, month (0 for annual),
    mean, sd, se, n, possible.
    """
    if by not in ("year", "month"):
        raise ValueError(f"unknown grouping {by!r}")
    s = season_days(daily, months)
    cols = ["period", "year", "month", "mean", "sd", "se", "n", "possible"]
    if s.empty:
        return pd.DataFrame(columns=cols)
    s = s.assign(year=s["date"].dt.year, month=s["date"].dt.month if by == "month" else 0)
    g = s.groupby(["year", "month"])["value"]
    out = pd.DataFrame({"mean": g.mean(), "sd": g.std(ddof=1), "n": g.count()}).reset_index()
    if by == "month":
        out["possible"] = [pd.Period(year=y, month=m, freq="M").days_in_month for y, m in zip(out["year"], out["month"])]
        out["period"] = pd.to_datetime({"year": out["year"], "month": out["month"], "day": 15})
    else:
        out["possible"] = [sum(pd.Period(year=y, month=m, freq="M").days_in_month for m in months) for y in out["year"]]
        mid = months[len(months) // 2]  # plot each year at the middle of its season
        out["period"] = pd.to_datetime({"year": out["year"], "month": mid, "day": 1})
    out["se"] = out["sd"] / np.sqrt(out["n"])
    out = out[out["n"] >= min_coverage * out["possible"]]
    return out[cols].sort_values("period").reset_index(drop=True)


def _norm_sf(z: float) -> float:
    """Upper-tail probability of a standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def trend(years: pd.Series | np.ndarray, values: pd.Series | np.ndarray) -> dict | None:
    """Sen's slope (units per year) and a two-sided Mann-Kendall p-value; None with fewer than 5 points.

    Keys: slope, intercept (value at year 0 on the Sen line), p, tau, n, first_year, last_year.
    """
    x = np.asarray(years, dtype=float)
    y = np.asarray(values, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    n = len(x)
    if n < 5:
        return None
    i, j = np.triu_indices(n, k=1)
    dx, dy = x[j] - x[i], y[j] - y[i]
    slope = float(np.median(dy[dx != 0] / dx[dx != 0]))
    intercept = float(np.median(y - slope * x))
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    s = float(np.sign(ys[j] - ys[i]).sum())
    _, counts = np.unique(ys, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - sum(t * (t - 1) * (2 * t + 5) for t in counts)) / 18.0
    z = 0.0 if s == 0 or var <= 0 else (s - math.copysign(1, s)) / math.sqrt(var)
    return {"slope": slope, "intercept": intercept, "p": min(1.0, 2 * _norm_sf(abs(z))), "tau": s / (n * (n - 1) / 2),
            "n": n, "first_year": int(xs[0]), "last_year": int(xs[-1])}


def describe(t: dict | None, unit: str = "µS/cm") -> str:
    """One plain sentence about the trend, careful not to overstate it."""
    if t is None:
        return "Fewer than 5 years with enough data, so no trend is computed."
    span = t["last_year"] - t["first_year"]
    total = t["slope"] * span
    direction = "rising" if t["slope"] > 0 else "falling" if t["slope"] < 0 else "flat"
    strength = ("a statistically significant" if t["p"] < 0.05 else "a weak" if t["p"] < 0.10 else "no clear")
    return (f"Base-flow conductance is {direction} by about {abs(t['slope']):.3g} {unit} per year (Sen's slope), "
            f"about {abs(total):,.0f} {unit} over {t['first_year']} to {t['last_year']}: {strength} trend "
            f"(Mann-Kendall p {'< 0.001' if t['p'] < 0.001 else '= ' + format(t['p'], '.3f')}, {t['n']} years).")
