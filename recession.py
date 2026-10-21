"""Base flow recession: how fast a stream drains its groundwater once the rain stops.

Two views of the same recession days, so they can be checked against each other:

1. Segments and k. Each unbroken run of falling daily flow after a peak is a recession segment. The first few days
   after the peak are skipped (that is quickflow draining off, not base flow), and the rest is fitted with
   Q = Q0 * exp(-t / k). k (days) is the recession constant; K = exp(-1/k) is the same thing as a daily ratio, and
   k * ln 2 is the half-life. A master recession curve (MRC) is built by strip matching: each segment is slid along
   the time axis until its first day sits on one exponential through all of them, pooled from every recession day.
2. -dQ/dt against Q (Brutsaert and Nieber, 1977). For each pair of recession days, -dQ/dt = Q[t] - Q[t+1] and
   Q = their mean. On log-log axes -dQ/dt = a * Q^b is a straight line. b = 1 is a linear reservoir (k = 1/a);
   b > 1 means the stream slows its fall as it gets lower. Brutsaert and Nieber fitted the lower envelope, the
   slowest recessions, which are the least touched by rain and evapotranspiration; both the envelope and a fit
   through all points are given.

Nothing here imports Streamlit.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Daily values USGS marks with these are not measured flow (ice backwater, estimated), so a segment stops there.
BAD_FLAGS = ("ICE", "ESTIMATED", "EQUIP", "BKW")
SEGMENT_COLS = ["start", "end", "days", "q_start", "q_end", "k_days", "daily_K", "half_life_days", "r2", "month"]
DAY = pd.Timedelta(days=1)


def _bad(flags: pd.Series) -> np.ndarray:
    text = flags.fillna("").astype(str).str.upper()
    return text.apply(lambda f: any(b in f for b in BAD_FLAGS)).to_numpy()


def _step(index: pd.DatetimeIndex) -> pd.Timedelta:
    """The record's time step (one day for daily values; an hour for hourly means of 15-minute data)."""
    return pd.Series(index).diff().median() if len(index) > 1 else DAY


def hourly(continuous: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """15-minute readings -> hourly means (date, value, flags), which damps sensor jitter enough to find recessions."""
    s = continuous.set_index(time_col)["value"].resample("h").mean()
    out = s.reset_index().rename(columns={time_col: "date"})
    out["flags"] = ""
    return out


def segments(daily: pd.DataFrame, months: tuple[int, ...] = tuple(range(1, 13)), skip: float = 2, min_len: float = 5,
             min_r2: float = 0.8, skip_flagged: bool = True, tol: float = 0.0) -> tuple[pd.DataFrame, list[pd.Series]]:
    """Recession segments in a regular record (date, value, flags): daily values, or hourly means for a single storm.

    A segment starts `skip` days after a local peak and runs while flow keeps falling (or rises by no more than `tol`,
    a fraction, to ride over sensor jitter) with no gap and no flagged step; it is kept if at least `min_len` days
    remain, every step is in `months`, all flows are above zero, and the exponential fit explains at least `min_r2`
    of the variance of ln Q. `skip` and `min_len` are in days whatever the time step; so are k and the half-life.
    Returns (table with one row per segment, list of the segments' flows as time-indexed Series).
    """
    d = daily.set_index("date").sort_index()
    step = _step(d.index)
    full = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq=step)) if len(d) else d
    per_day = DAY / step
    q = full["value"].to_numpy(dtype=float)
    bad = _bad(full["flags"]) if skip_flagged and "flags" in full else np.zeros(len(q), dtype=bool)
    q = np.where(bad, np.nan, q)
    idx = full.index
    n = len(q)
    skip_n, min_n = int(round(skip * per_day)), max(int(math.ceil(min_len * per_day)), 3)
    rows, series = [], []
    i = 1
    while i < n - 1:
        if np.isnan(q[i]) or np.isnan(q[i - 1]) or not (q[i] >= q[i - 1] and q[i] > q[i + 1]):
            i += 1
            continue
        j = i + 1
        while j < n and not np.isnan(q[j]) and q[j] <= q[j - 1] * (1 + tol):
            j += 1
        fit = q[i + skip_n:j]
        span = idx[i + skip_n:j]
        if len(fit) >= min_n and np.all(fit > 0) and set(span.month) <= set(months):
            t = np.arange(len(fit), dtype=float) / per_day
            y = np.log(fit)
            slope, icpt = np.polyfit(t, y, 1)
            ss = np.sum((y - y.mean()) ** 2)
            r2 = 1 - np.sum((y - (icpt + slope * t)) ** 2) / ss if ss > 0 else 0.0
            if slope < 0 and r2 >= min_r2:
                k = -1 / slope
                rows.append((span[0], span[-1], len(fit) if per_day == 1 else round(len(fit) / per_day, 2), float(fit[0]),
                             float(fit[-1]), float(k), float(math.exp(slope)), float(k * math.log(2)), float(r2),
                             int(span[len(span) // 2].month)))
                series.append(pd.Series(fit, index=span))
        i = max(j, i + 1)
    return pd.DataFrame(rows, columns=SEGMENT_COLS), series


def master_curve(series: list[pd.Series]) -> dict | None:
    """Strip-matched master recession curve. None without segments.

    k_days is pooled from every recession day (the mean of ln(Q[t+1] / Q[t]) over all segments). Each segment is
    shifted to start at t = k * ln(Q_top / Q_start), where Q_top is the highest starting flow, so all of them fall
    along Q = Q_top * exp(-t / k). Keys: k_days, daily_K, half_life_days, q_top, shifted (list of DataFrames t, q).
    """
    steps = [np.diff(np.log(s.to_numpy(dtype=float))) for s in series if len(s) > 1]
    if not steps:
        return None
    dt = _step(series[0].index) / DAY  # days per step
    mean_step = float(np.concatenate(steps).mean()) / dt
    if mean_step >= 0:
        return None
    k = -1 / mean_step
    q_top = max(float(s.iloc[0]) for s in series)
    shifted = [pd.DataFrame({"t": k * math.log(q_top / float(s.iloc[0])) + np.arange(len(s)) * dt, "q": s.to_numpy(dtype=float),
                             "start": s.index[0]}) for s in series]
    return {"k_days": k, "daily_K": math.exp(-1 / k), "half_life_days": k * math.log(2), "q_top": q_top, "shifted": shifted}


def dqdt_points(series: list[pd.Series]) -> pd.DataFrame:
    """Brutsaert-Nieber points from recession steps: q = (Q[t] + Q[t+1]) / 2, dqdt = (Q[t] - Q[t+1]) per day (> 0 only)."""
    parts = []
    for s in series:
        v = s.to_numpy(dtype=float)
        if len(v) < 2:
            continue
        drop = (v[:-1] - v[1:]) / (_step(s.index) / DAY)
        parts.append(pd.DataFrame({"date": s.index[:-1], "q": (v[:-1] + v[1:]) / 2, "dqdt": drop}))
    if not parts:
        return pd.DataFrame(columns=["date", "q", "dqdt"])
    out = pd.concat(parts, ignore_index=True)
    return out[out["dqdt"] > 0].reset_index(drop=True)


def _line(logq: np.ndarray, logd: np.ndarray) -> tuple[float, float, float]:
    b, loga = np.polyfit(logq, logd, 1)
    pred = loga + b * logq
    ss = np.sum((logd - logd.mean()) ** 2)
    return float(math.exp(loga)), float(b), float(1 - np.sum((logd - pred) ** 2) / ss) if ss > 0 else 0.0


def fit_power_law(points: pd.DataFrame, bins: int = 15, envelope_pct: float = 5, min_per_bin: int = 5) -> dict | None:
    """Fit -dQ/dt = a * Q^b in log-log space, through all points and along the lower envelope.

    The envelope is the `envelope_pct` percentile of -dQ/dt in each of `bins` equal-width bins of log Q (bins with
    fewer than `min_per_bin` points are skipped). Keys: all and envelope, each {a, b, r2, k_days (1/a, only
    meaningful when b is near 1)} or None; envelope_points (DataFrame q, dqdt); n.
    """
    p = points[(points["q"] > 0) & (points["dqdt"] > 0)]
    if len(p) < 10:
        return None
    lq, ld = np.log(p["q"].to_numpy(dtype=float)), np.log(p["dqdt"].to_numpy(dtype=float))
    a, b, r2 = _line(lq, ld)
    out = {"all": {"a": a, "b": b, "r2": r2, "k_days": 1 / a}, "envelope": None,
           "envelope_points": pd.DataFrame(columns=["q", "dqdt"]), "n": len(p)}
    if lq.max() > lq.min():
        edges = np.linspace(lq.min(), lq.max(), bins + 1)
        which = np.clip(np.digitize(lq, edges) - 1, 0, bins - 1)
        env = [(math.exp(np.median(lq[which == i])), math.exp(np.percentile(ld[which == i], envelope_pct)))
               for i in range(bins) if (which == i).sum() >= min_per_bin]
        if len(env) >= 3:
            e = pd.DataFrame(env, columns=["q", "dqdt"])
            ea, eb, er2 = _line(np.log(e["q"].to_numpy()), np.log(e["dqdt"].to_numpy()))
            out["envelope"] = {"a": ea, "b": eb, "r2": er2, "k_days": 1 / ea}
            out["envelope_points"] = e
    return out


def by_season(table: pd.DataFrame) -> pd.DataFrame:
    """Median k and count of segments per meteorological season (winter DJF, spring MAM, summer JJA, fall SON)."""
    names = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
             6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"}
    if table.empty:
        return pd.DataFrame(columns=["Season", "Segments", "Median k (days)", "Median half-life (days)"])
    g = table.assign(Season=table["month"].map(names)).groupby("Season")
    out = pd.DataFrame({"Segments": g.size(), "Median k (days)": g["k_days"].median(),
                        "Median half-life (days)": g["half_life_days"].median()})
    order = [s for s in ("Winter", "Spring", "Summer", "Fall") if s in out.index]
    return out.loc[order].reset_index()


def describe_power_law(fit: dict | None, unit: str = "cfs") -> list[str]:
    if fit is None:
        return ["Too few recession days for a -dQ/dt against Q fit (at least 10 are needed)."]
    lines = []
    for key, name in (("envelope", "Lower envelope"), ("all", "All recession days")):
        f = fit[key]
        if f is None:
            continue
        kind = ("close to a linear reservoir (b ≈ 1)" if abs(f["b"] - 1) <= 0.2 else
                "recession slows as flow drops (b > 1)" if f["b"] > 1 else "recession speeds up as flow drops (b < 1)")
        k = f"; k = 1/a ≈ {f['k_days']:.2g} days" if abs(f["b"] - 1) <= 0.2 else ""
        lines.append(f"{name}: -dQ/dt = {f['a']:.3g} · Q^{f['b']:.2f} ({unit}/day, r² {f['r2']:.2f}): {kind}{k}.")
    return lines
