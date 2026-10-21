"""The answer key: facts about a hydrograph that code can compute and a tutor must not blurt out.

Everything here is deterministic and tested, so the tutor (and the instructor's view) can rely on it instead of
asking a language model to eyeball a curve. The recession fit is adapted from HydroTobin's find_recessions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
BAND_COLS = ["p10", "p25", "p50", "p75", "p90"]
_LEAP_YEAR_DAYS = pd.date_range("2000-01-01", "2000-12-31").strftime("%m-%d")


def _flow(daily: pd.DataFrame) -> pd.Series:
    """Daily mean flow as a Series on a complete daily index (gaps are NaN)."""
    return daily.set_index("date")["value"].sort_index().asfreq("D")


# ---------------------------------------------------------------------------------------------------
# "Normal for this date"
# ---------------------------------------------------------------------------------------------------

def normal_bands(daily: pd.DataFrame, smooth_days: int = 7, min_years: int = 10) -> pd.DataFrame:
    """Percentiles of daily flow for each calendar day across all years, lightly smoothed round the year.

    Index 'MM-DD' (all 366 days); columns p10 p25 p50 p75 p90 and n (years behind that day).
    """
    s = daily.set_index("date")["value"].dropna()
    grouped = s.groupby(s.index.strftime("%m-%d"))
    table = grouped.quantile(list(QUANTILES)).unstack()
    table.columns = BAND_COLS
    table["n"] = grouped.count()
    table = table[table["n"] >= min_years].reindex(_LEAP_YEAR_DAYS)
    k = smooth_days
    padded = pd.concat([table.iloc[-k:], table, table.iloc[:k]])
    smooth = padded[BAND_COLS].rolling(2 * k + 1, center=True, min_periods=1).mean().iloc[k:-k]
    smooth = smooth.interpolate(limit_direction="both")
    smooth["n"] = table["n"]
    return smooth


def band_for_dates(bands: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """The normal band lined up with real dates (one row per date)."""
    out = bands.loc[dates.strftime("%m-%d"), BAND_COLS].copy()
    out.index = dates
    return out


def percentile_for_date(daily: pd.DataFrame, when: pd.Timestamp | None = None, window: int = 7) -> float | None:
    """Share (0-100) of other years' flows, within +/- window days of this date, that were lower than this day's flow."""
    s = daily.set_index("date")["value"].dropna()
    if s.empty:
        return None
    when = pd.Timestamp(when) if when is not None else s.index[-1]
    if when not in s.index:
        return None
    gap = np.abs(((s.index.dayofyear - when.dayofyear + 183) % 366) - 183)
    same = s[(gap <= window) & (s.index.year != when.year)]
    if len(same) < 30:
        return None
    return float((same < s.loc[when]).mean() * 100)


# ---------------------------------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------------------------------

@dataclass
class Event:
    peak_date: pd.Timestamp
    peak_q: float
    base_q: float
    ratio: float                # peak over the lowest flow in the 30 days before it
    rise_days: int              # from that low to the peak
    half_fall_days: int | None  # days after the peak until flow is back halfway to the low (None if it never got there)


def find_events(s: pd.Series, window: int = 30, min_ratio: float = 1.5, top: int = 5, min_sep: int = 10) -> list[Event]:
    """The biggest rises: local peaks at least min_ratio times the lowest flow of the previous `window` days."""
    q = s.interpolate(limit=3).to_numpy(dtype=float)
    n = len(q)
    found: list[Event] = []
    for i in range(1, n - 1):
        if np.isnan(q[i]) or not (q[i] >= q[i - 1] and q[i] > q[i + 1]):
            continue
        seg = q[max(0, i - window):i + 1]
        if np.all(np.isnan(seg)):
            continue
        j = max(0, i - window) + int(np.nanargmin(seg))
        base = q[j]
        if not base > 0 or q[i] / base < min_ratio:
            continue
        target = base + 0.5 * (q[i] - base)
        k = i + 1
        while k < n and not q[k] <= target:
            k += 1
        found.append(Event(s.index[i], float(q[i]), float(base), float(q[i] / base), i - j, (k - i) if k < n else None))
    found.sort(key=lambda e: e.peak_q - e.base_q, reverse=True)
    chosen: list[Event] = []
    for e in found:
        if all(abs((e.peak_date - c.peak_date).days) >= min_sep for c in chosen):
            chosen.append(e)
        if len(chosen) == top:
            break
    return sorted(chosen, key=lambda e: e.peak_date)


def flashiness(s: pd.Series) -> float | None:
    """Richards-Baker index: total day-to-day change over total flow. Near 0 is steady, above ~0.5 is flashy."""
    q = s.dropna().to_numpy(dtype=float)
    if len(q) < 30 or q[1:].sum() <= 0:
        return None
    return float(np.abs(np.diff(q)).sum() / q[1:].sum())


# ---------------------------------------------------------------------------------------------------
# Baseflow recession (adapted from HydroTobin)
# ---------------------------------------------------------------------------------------------------

def recession_half_life(s: pd.Series, skip: int = 2, min_len: int = 6, rise_pct: float = 20, tol: float = 0.0,
                        min_r2: float = 0.9, months: range | tuple = range(6, 12), min_events: int = 5):
    """Pooled exponential recession Q = Q0*exp(-alpha t) after rain peaks. Returns (half_life_days, n_events) or (None, n)."""
    q = s.to_numpy(dtype=float)
    idx = s.index
    n = len(q)
    num = den = 0.0
    events = 0
    for i in range(5, n - 1):
        if np.isnan(q[i]) or not (q[i] >= q[i - 1] and q[i] > q[i + 1]):
            continue
        before = q[i - 5:i]
        if np.isnan(before).any() or q[i] < np.min(before) * (1 + rise_pct / 100):
            continue
        j = i + 1
        while j < n and not np.isnan(q[j]) and q[j] <= q[j - 1] * (1 + tol):
            j += 1
        fit = q[i + skip:j]
        if len(fit) < min_len or not set(idx[i + skip:j].month) <= set(months) or np.any(fit <= 0):
            continue
        t = np.arange(len(fit), dtype=float)
        y = np.log(fit)
        slope, icpt = np.polyfit(t, y, 1)
        r2 = 1 - np.sum((y - (icpt + slope * t)) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
        if slope >= 0 or r2 < min_r2:
            continue
        yy = np.log(fit / fit[0])
        num += float(np.sum(t * yy))
        den += float(np.sum(t * t))
        events += 1
    if events < min_events or den == 0 or num >= 0:
        return None, events
    return float(np.log(2) / (-num / den)), events


# ---------------------------------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------------------------------

def answer_key(daily: pd.DataFrame, drainage_area_mi2: float | None = None) -> dict:
    s = _flow(daily)
    last_year = s.iloc[-365:]
    half_life, n_rec = recession_half_life(s.iloc[-365 * 20:])
    return {
        "latest_date": s.dropna().index[-1] if s.notna().any() else None,
        "latest_flow": float(s.dropna().iloc[-1]) if s.notna().any() else None,
        "percentile_today": percentile_for_date(daily),
        "years_of_record": round((s.index[-1] - s.index[0]).days / 365.25, 1),
        "events": find_events(last_year),
        "flashiness_rb": flashiness(last_year),
        "recession_half_life_days": half_life,
        "recession_events": n_rec,
        "drainage_area_mi2": drainage_area_mi2,
    }


def describe(key: dict) -> list[str]:
    """The key in plain sentences (for the instructor's view; the tutor gets its own wording)."""
    lines = []
    if key["latest_flow"] is not None:
        pct = key["percentile_today"]
        where = f"higher than {pct:.0f}% of other years on this date" if pct is not None else "(not enough years to compare)"
        lines.append(f"Latest daily flow {key['latest_flow']:,.0f} cfs on {key['latest_date']:%b %d, %Y}: {where}.")
    for e in key["events"]:
        fall = f"halfway back down after {e.half_fall_days} days" if e.half_fall_days is not None else "had not yet receded halfway"
        lines.append(f"Event {e.peak_date:%b %d, %Y}: peak {e.peak_q:,.0f} cfs, {e.ratio:.1f}x the low before it; "
                     f"rose for {e.rise_days} days, {fall}.")
    if key["flashiness_rb"] is not None:
        rb = key["flashiness_rb"]
        lines.append(f"Flashiness (Richards-Baker) {rb:.2f}: {'flashy, responds fast to rain' if rb > 0.5 else 'moderate' if rb > 0.2 else 'steady, slow to respond'}.")
    if key["recession_half_life_days"]:
        lines.append(f"Baseflow recession half-life about {key['recession_half_life_days']:.0f} days "
                     f"(from {key['recession_events']} summer and fall recessions in the last 20 years).")
    if key["drainage_area_mi2"]:
        lines.append(f"Drainage area {key['drainage_area_mi2']:,.0f} mi²; record spans {key['years_of_record']} years.")
    return lines
