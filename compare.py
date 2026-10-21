"""Putting two gages on one chart so a student can ask why they differ (city versus forest, small versus large, above versus below a dam).

Raw flow is useless for that: a big river is thousands of times bigger than a creek. So the default view divides each gage's
flow by its drainage area (cfs per square mile), which compares how much each acre of watershed yields. "Shape only"
divides by each gage's own peak so timing can be compared. Nothing here imports Streamlit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import notes

MODES = {
    "per_sq_mi": ("Per square mile", "Discharge per square mile of watershed (cfs per mi²)"),
    "shape": ("Shape only", "Share of each gage's own peak in this window"),
    "actual": ("Actual flow", "Discharge (cubic feet per second)"),
}


def daily_window(values: pd.Series | np.ndarray, x) -> pd.Series:
    """Daily mean series from a chart's values and x positions (works for 15-minute and daily data)."""
    return pd.Series(np.asarray(values, dtype=float), index=pd.DatetimeIndex(x)).resample("D").mean()


def normalize(values: pd.Series, mode: str, drainage_mi2: float | None) -> pd.Series:
    if mode == "actual":
        return values
    if mode == "per_sq_mi":
        if not drainage_mi2 or drainage_mi2 <= 0:
            raise ValueError("no drainage area on record")
        return values / drainage_mi2
    if mode == "shape":
        top = float(np.nanmax(values.to_numpy(dtype=float)))
        if not top > 0:
            raise ValueError("no flow in this window")
        return values / top
    raise ValueError(f"unknown mode {mode!r}")


def _facts(series: pd.Series, area: float | None) -> dict:
    s = series.dropna()
    ev = notes.find_events(series, window=min(30, max(len(s) // 2, 3)))
    biggest = max(ev, key=lambda e: e.peak_q) if ev else None
    return {
        "peak_cfs": float(s.max()) if len(s) else None,
        "peak_per_mi2": float(s.max() / area) if len(s) and area else None,
        "mean_per_mi2": float(s.mean() / area) if len(s) and area else None,
        "flashiness": notes.flashiness(series),
        "half_fall_days": biggest.half_fall_days if biggest else None,
        "area": area,
    }


def comparison_facts(a: pd.Series, area_a: float | None, b: pd.Series, area_b: float | None) -> dict:
    return {"a": _facts(a, area_a), "b": _facts(b, area_b)}


def describe(label_a: str, label_b: str, facts: dict) -> list[str]:
    """Plain sentences for the instructor's view; students see the chart first and ask why."""
    a, b = facts["a"], facts["b"]
    out = []
    if a["area"] and b["area"]:
        out.append(f"{label_a} drains {a['area']:,.0f} mi²; {label_b} drains {b['area']:,.0f} mi² ({max(a['area'], b['area']) / min(a['area'], b['area']):.0f}x difference in size).")
    if a["peak_per_mi2"] and b["peak_per_mi2"]:
        hi, lo = (label_a, label_b) if a["peak_per_mi2"] >= b["peak_per_mi2"] else (label_b, label_a)
        ratio = max(a["peak_per_mi2"], b["peak_per_mi2"]) / min(a["peak_per_mi2"], b["peak_per_mi2"])
        out.append(f"Peak yield per square mile: {label_a} {a['peak_per_mi2']:.2f} cfs/mi², {label_b} {b['peak_per_mi2']:.2f} cfs/mi²; {hi} is {ratio:.1f}x higher than {lo}.")
    if a["flashiness"] is not None and b["flashiness"] is not None:
        flasher = label_a if a["flashiness"] >= b["flashiness"] else label_b
        out.append(f"Flashiness (Richards-Baker): {label_a} {a['flashiness']:.2f}, {label_b} {b['flashiness']:.2f}; {flasher} responds faster to rain.")
    if a["half_fall_days"] is not None and b["half_fall_days"] is not None:
        out.append(f"Days for the biggest rise to fall halfway back: {label_a} {a['half_fall_days']}, {label_b} {b['half_fall_days']}.")
    return out
