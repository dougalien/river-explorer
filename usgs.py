"""USGS water-data access for River Explorer (no Streamlit in here, so it can be tested or reused anywhere).

Uses the Water Data OGC API at api.waterdata.usgs.gov, the successor to waterservices.usgs.gov, which USGS
plans to switch off in early 2027. No API key is needed. If USGS_API_KEY is set it is sent as X-Api-Key,
which raises the hourly request limit; the app works without it.

Daily values come back as the same numbers the old service gave (checked on Cedar River at Cedar Rapids,
45,307 days, 0 differences), plus an approved/provisional flag and qualifiers such as ESTIMATED or ICE.
"""

from __future__ import annotations

import math
import os
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterator

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:  # trust the operating system's certificate store; Python's bundled CA list can be stale
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

BASE = "https://api.waterdata.usgs.gov/ogcapi/v1/collections"
TIMEOUT = (10, 45)  # seconds to connect, seconds to wait for data (a whole 120-year record takes ~10 s)
PAGE = 50000  # rows per request; the API follows "next" links beyond this
DISCHARGE = "00060"
GAGE_HEIGHT = "00065"
CONDUCTANCE = "00095"  # specific conductance, microsiemens per cm at 25 degrees C
UNIT_LABEL = {"ft^3/s": "cfs", "ft": "ft", "uS/cm @25C": "µS/cm"}
PARAM_UNIT = {DISCHARGE: "cfs", GAGE_HEIGHT: "ft", CONDUCTANCE: "µS/cm"}
# USGS statistic codes for daily values: the day's mean, maximum and minimum of the instantaneous readings.
DAILY_STATS = {"00003": "mean", "00001": "max", "00002": "min"}

STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
    "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts",
    "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "PR": "Puerto Rico", "VI": "Virgin Islands",
    "GU": "Guam",
}
_STATE_BY_NAME = {v.lower(): k for k, v in STATES.items()}

# Words USGS abbreviates or that every stream has, so they say nothing about which stream is meant.
_GENERIC = {"RIVER", "R", "RIV", "CREEK", "CR", "CRK", "STREAM", "BROOK", "BK", "THE", "OF", "AT", "NEAR", "NR",
            "IN", "A", "AN", "FORK", "BRANCH", "AND"}
_ABBR = {"NR": "near", "BLW": "below", "BL": "below", "ABV": "above", "AB": "above", "R": "River", "RIV": "River",
         "CR": "Creek", "CRK": "Creek", "BK": "Brook", "TRIB": "Tributary", "RES": "Reservoir", "FK": "Fork",
         "FRK": "Fork", "MT": "Mount", "LK": "Lake", "CNL": "Canal", "DIV": "Diversion"}
_SMALL = {"at", "near", "below", "above", "of", "the", "and", "to", "in", "on", "from", "for", "by", "between"}
_TZ = {"EST": "America/New_York", "EDT": "America/New_York", "CST": "America/Chicago", "CDT": "America/Chicago",
       "MST": "America/Denver", "MDT": "America/Denver", "PST": "America/Los_Angeles", "PDT": "America/Los_Angeles",
       "AKST": "America/Anchorage", "AKDT": "America/Anchorage", "HST": "Pacific/Honolulu", "AST": "America/Puerto_Rico"}


class USGSError(RuntimeError):
    """A problem talking to USGS, worded so it can be shown to a student."""


class USGSRateLimit(USGSError):
    """USGS says this server has used up its requests for the hour (1,000 without a free API key)."""

    def __init__(self, retry_after: int | None = None):
        self.retry_after = retry_after
        when = f"about {max(1, math.ceil(retry_after / 60))} minutes" if retry_after else "a few minutes"
        super().__init__(f"USGS is limiting requests from this server right now. Please try again in {when}.")


# ---------------------------------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------------------------------

_session_obj: requests.Session | None = None


def _session() -> requests.Session:
    global _session_obj
    if _session_obj is None:
        s = requests.Session()
        # Retry brief server hiccups only. A 429 (hourly limit) says "come back in ~8 minutes": never sleep on that.
        retry = Retry(total=2, backoff_factor=1.0, status_forcelist=(500, 502, 503, 504),
                      allowed_methods=("GET",), respect_retry_after_header=False, raise_on_status=False)
        s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=10))
        s.headers["User-Agent"] = "RiverExplorer/1.0 (student teaching tool)"
        key = os.environ.get("USGS_API_KEY", "").strip()
        if key:
            s.headers["X-Api-Key"] = key
        _session_obj = s
    return _session_obj


def _get_json(url: str, params: dict | None = None) -> dict:
    try:
        r = _session().get(url, params=params, timeout=TIMEOUT)
    except requests.RequestException as exc:
        raise USGSError(f"Could not reach USGS ({type(exc).__name__}). Try again in a moment.") from exc
    if r.status_code == 429:
        wait = str(r.headers.get("Retry-After", ""))
        raise USGSRateLimit(int(wait) if wait.isdigit() else None)
    if r.status_code == 404:
        raise USGSError("USGS has no record of that site.")
    if r.status_code >= 400:
        detail = ""
        try:
            detail = str(r.json().get("description", ""))[:200]
        except ValueError:
            pass
        raise USGSError(f"USGS rejected the request (HTTP {r.status_code}). {detail}".strip())
    try:
        return r.json()
    except ValueError as exc:
        raise USGSError("USGS sent back something that was not data. Try again in a moment.") from exc


def _features(collection: str, params: dict, max_rows: int | None = None, page: int | None = None) -> Iterator[dict]:
    """Yield GeoJSON features from a collection, following 'next' links until the data or max_rows runs out."""
    size = min(page or PAGE, max_rows) if max_rows else (page or PAGE)
    url: str | None = f"{BASE}/{collection}/items"
    query: dict | None = {"f": "json", "limit": size, **params}
    n = 0
    while url:
        data = _get_json(url, query)
        for feat in data.get("features", []):
            yield feat
            n += 1
            if max_rows and n >= max_rows:
                return
        url = next((link["href"] for link in data.get("links", []) if link.get("rel") == "next"), None)
        query = None  # the next link already carries every parameter


def site_id(site_no: str) -> str:
    """'05464500' or 'USGS-05464500' -> 'USGS-05464500'."""
    s = str(site_no).strip()
    if not re.fullmatch(r"(USGS-)?\d{8,15}", s):
        raise ValueError(f"'{site_no}' is not a USGS site number (expected 8 or more digits).")
    return s if s.startswith("USGS-") else "USGS-" + s


# ---------------------------------------------------------------------------------------------------
# Small helpers (pure, unit-tested)
# ---------------------------------------------------------------------------------------------------


def state_full_name(state: str | None) -> str | None:
    """'ma' or 'Massachusetts' -> 'Massachusetts'; None if it isn't a state USGS covers."""
    if not state:
        return None
    s = state.strip()
    if s.upper() in STATES:
        return STATES[s.upper()]
    return STATES.get(_STATE_BY_NAME.get(s.lower(), ""))


def state_abbr(state_name: str | None) -> str:
    return _STATE_BY_NAME.get((state_name or "").strip().lower(), "")


def name_terms(name: str | None) -> list[str]:
    """Distinctive upper-case words from a stream name, safe to put inside a CQL LIKE filter."""
    if not name:
        return []
    cleaned = re.sub(r"[^A-Za-z0-9 .-]", " ", name).upper()
    words = [w.strip(".-") for w in cleaned.split()]
    keep = [w for w in words if len(w) >= 2 and w not in _GENERIC]
    return keep[:3]


def bbox_around(lat: float, lon: float, radius_km: float = 30.0) -> tuple[float, float, float, float]:
    """(west, south, east, north) box that contains a circle of radius_km around the point."""
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("Latitude or longitude out of range.")
    r = min(max(float(radius_km), 5.0), 150.0)
    dlat = r / 111.0
    dlon = r / (111.0 * max(math.cos(math.radians(lat)), 0.2))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def pretty_name(name: str) -> str:
    """USGS shouts and abbreviates ('MYSTIC R NR MEDFORD, MA'); make it readable without changing the meaning."""
    if not name or name != name.upper():
        return name or ""
    body, comma, tail = name.rpartition(",")
    state = ""
    if comma and re.fullmatch(r"\s*[A-Z]{2}\s*", tail):
        state, name = tail.strip(), body
    out = []
    for tok in re.findall(r"[A-Z0-9.'&/-]+|,", name):
        if tok == ",":
            out.append(",")
        elif tok in _ABBR:
            out.append(_ABBR[tok])
        elif tok.lower() in _SMALL:
            out.append(tok.lower())
        elif any(c.isdigit() for c in tok):
            out.append(tok)
        else:
            out.append(tok.capitalize())
    text = " ".join(out).replace(" ,", ",")
    if text:
        text = text[0].upper() + text[1:]
    return f"{text}, {state}" if state else text


def site_timezone(info: dict) -> str:
    """IANA zone for a site (so UTC timestamps can be shown in local time)."""
    abbr = (info.get("time_zone") or "").upper()
    if abbr == "MST" and info.get("uses_dst") == "N":
        return "America/Phoenix"
    return _TZ.get(abbr, "UTC")


def _flags(qualifier: Any) -> str:
    if not qualifier:
        return ""
    if isinstance(qualifier, str):
        return qualifier
    return ",".join(str(q) for q in qualifier)


def _number(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _to_utc(x: Any) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def daily_frame(features: list[dict] | Iterator[dict]) -> pd.DataFrame:
    """Features from the 'daily' collection -> DataFrame(date, value, approved, flags), one row per day."""
    rows = []
    for f in features:
        p = f.get("properties", {})
        v = _number(p.get("value"))
        if v is None or not p.get("time"):
            continue
        rows.append((str(p["time"])[:10], v, p.get("approval_status") == "Approved", _flags(p.get("qualifier"))))
    df = pd.DataFrame(rows, columns=["date", "value", "approved", "flags"])
    df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def daily_stats_frame(features: list[dict] | Iterator[dict]) -> pd.DataFrame:
    """Features from the 'daily' collection holding several statistics -> one row per day.

    Columns: date, mean, max, min (NaN where USGS reports no such statistic that day), approved, flags (from the mean).
    """
    rows = []
    for f in features:
        p = f.get("properties", {})
        v = _number(p.get("value"))
        col = DAILY_STATS.get(str(p.get("statistic_id")))
        if v is None or col is None or not p.get("time"):
            continue
        rows.append((str(p["time"])[:10], col, v, p.get("approval_status") == "Approved", _flags(p.get("qualifier"))))
    cols = ["date", *DAILY_STATS.values(), "approved", "flags"]
    if not rows:
        return pd.DataFrame(columns=cols)
    long = pd.DataFrame(rows, columns=["date", "stat", "value", "approved", "flags"]).drop_duplicates(["date", "stat"], keep="last")
    wide = long.pivot(index="date", columns="stat", values="value").reindex(columns=list(DAILY_STATS.values()))
    first = long.sort_values("stat", key=lambda c: c.ne("mean")).drop_duplicates("date")  # the mean row's flags if there is one
    out = wide.join(first.set_index("date")[["approved", "flags"]]).reset_index()
    out["date"] = pd.to_datetime(out["date"])
    out["approved"] = out["approved"].astype(bool)
    return out[cols].sort_values("date").reset_index(drop=True)


def continuous_frame(features: list[dict] | Iterator[dict]) -> pd.DataFrame:
    """Features from the 'continuous' collection -> DataFrame(time[UTC], value, approved, flags)."""
    rows = []
    for f in features:
        p = f.get("properties", {})
        v = _number(p.get("value"))
        if v is None or not p.get("time"):
            continue
        rows.append((p["time"], v, p.get("approval_status") == "Approved", _flags(p.get("qualifier")),
                     p.get("unit_of_measure")))
    df = pd.DataFrame(rows, columns=["time", "value", "approved", "flags", "unit"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    unit = df["unit"].dropna().iloc[0] if df["unit"].notna().any() else ""
    df = df.drop(columns="unit").drop_duplicates("time", keep="last").sort_values("time").reset_index(drop=True)
    df.attrs["unit"] = UNIT_LABEL.get(unit, unit)
    return df


def site_from_feature(feat: dict) -> dict:
    p = feat.get("properties", {})
    coords = (feat.get("geometry") or {}).get("coordinates") or [None, None]
    da = _number(p.get("drainage_area"))
    return {
        "site_no": str(p.get("id", "")).replace("USGS-", ""),
        "name": p.get("monitoring_location_name", ""),
        "label": pretty_name(p.get("monitoring_location_name", "")),
        "state": p.get("state_name", ""),
        "county": p.get("county_name", ""),
        "lat": coords[1],
        "lon": coords[0],
        "drainage_area_mi2": da,
        "huc": p.get("hydrologic_unit_code"),
        "time_zone": p.get("time_zone_abbreviation"),
        "uses_dst": p.get("uses_daylight_savings"),
        "site_type": p.get("site_type_code"),
    }


def _parse_day(x: Any) -> date | None:
    try:
        return pd.Timestamp(x).date() if x else None
    except (ValueError, TypeError):
        return None


def gage_rows(features: list[dict] | Iterator[dict], today: date | None = None, active_days: int = 60) -> list[dict]:
    """combined-metadata features -> one row per site (widest record kept), flagged active or not."""
    today = today or date.today()
    best: dict[str, dict] = {}
    for f in features:
        p = f.get("properties", {})
        sid = str(p.get("monitoring_location_id", ""))
        coords = (f.get("geometry") or {}).get("coordinates") or [None, None]
        begin, end = _parse_day(p.get("begin")), _parse_day(p.get("end"))
        if not sid.startswith("USGS-") or begin is None or end is None:
            continue
        row = {
            "site_no": sid[5:], "name": p.get("monitoring_location_name", ""),
            "label": pretty_name(p.get("monitoring_location_name", "")), "state": p.get("state_name", ""),
            "lat": coords[1], "lon": coords[0], "drainage_area_mi2": _number(p.get("drainage_area")),
            "begin": begin, "end": end,
        }
        old = best.get(row["site_no"])
        if old:
            row["begin"], row["end"] = min(old["begin"], begin), max(old["end"], end)
        best[row["site_no"]] = row
    rows = list(best.values())
    for r in rows:
        r["record_years"] = round((r["end"] - r["begin"]).days / 365.25, 1)
        r["active"] = (today - r["end"]).days <= active_days
    return rows


# ---------------------------------------------------------------------------------------------------
# Public calls
# ---------------------------------------------------------------------------------------------------


def fetch_daily(site_no: str, start: Any = None, end: Any = None, parameter: str = DISCHARGE) -> pd.DataFrame:
    """Daily mean values for a site (the whole record if start and end are None).

    Columns: date, value, approved (bool), flags (e.g. 'ESTIMATED,ICE'). df.attrs['unit'] holds the unit.
    """
    params: dict[str, Any] = {"monitoring_location_id": site_id(site_no), "parameter_code": parameter,
                              "statistic_id": "00003", "properties": "time,value,approval_status,qualifier"}
    if start or end:
        params["time"] = f"{pd.Timestamp(start).date() if start else '..'}/{pd.Timestamp(end).date() if end else '..'}"
    df = daily_frame(_features("daily", params))
    df.attrs["unit"] = PARAM_UNIT.get(parameter, "")
    if df.empty:
        raise USGSError(f"USGS has no daily values for site {site_no} in that range.")
    return df


def fetch_daily_stats(site_no: str, start: Any = None, end: Any = None, parameter: str = DISCHARGE) -> pd.DataFrame:
    """The daily mean, maximum and minimum USGS publishes for a site, in one request (paged like fetch_daily).

    Columns: date, mean, max, min, approved, flags. Many discharge gages publish only the mean: then max and min
    are all NaN, and the caller should say so rather than invent them.
    """
    params: dict[str, Any] = {"monitoring_location_id": site_id(site_no), "parameter_code": parameter,
                              "properties": "time,value,statistic_id,approval_status,qualifier"}
    if start or end:
        params["time"] = f"{pd.Timestamp(start).date() if start else '..'}/{pd.Timestamp(end).date() if end else '..'}"
    df = daily_stats_frame(_features("daily", params))
    df.attrs["unit"] = PARAM_UNIT.get(parameter, "")
    if df.empty:
        raise USGSError(f"USGS has no daily values for site {site_no} in that range.")
    return df


def fetch_continuous(site_no: str, start: Any = None, end: Any = None, parameter: str = DISCHARGE,
                     days: int = 7) -> pd.DataFrame:
    """15-minute values (UTC) for a site; defaults to the last `days` days.

    Columns: time (UTC), value, approved (bool), flags. df.attrs['unit'] holds the unit.
    """
    end_ts = _to_utc(end) if end is not None else pd.Timestamp.now(tz="UTC")
    start_ts = _to_utc(start) if start is not None else end_ts - timedelta(days=days)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    params = {"monitoring_location_id": site_id(site_no), "parameter_code": parameter,
              "time": f"{start_ts.strftime(fmt)}/{end_ts.strftime(fmt)}",
              "properties": "time,value,approval_status,qualifier,unit_of_measure"}
    df = continuous_frame(_features("continuous", params))
    if df.empty:
        raise USGSError(f"USGS has no 15-minute values for site {site_no} in that range.")
    return df


def fetch_site_info(site_no: str) -> dict:
    """Name, location, drainage area (mi²) and time zone for one site."""
    return site_from_feature(_get_json(f"{BASE}/monitoring-locations/items/{site_id(site_no)}", {"f": "json"}))


def find_gages(name: str | None = None, state: str | None = None,
               bbox: tuple[float, float, float, float] | None = None, active_days: int = 60,
               max_rows: int = 3000, site_no: str | None = None, parameter: str = DISCHARGE) -> list[dict]:
    """USGS stream gages that report a daily mean of `parameter` (discharge unless told otherwise), matching a name, a state and/or a bounding box.

    Needs at least one of name, state, bbox or site_no. Rows: site_no, name, label, state, lat, lon,
    drainage_area_mi2, begin, end, record_years, active. Unranked: ranking is the caller's business.
    """
    terms = name_terms(name)
    st = state_full_name(state)
    if state and not st:
        raise ValueError(f"'{state}' is not a state USGS covers.")
    if not (terms or st or bbox or site_no):
        raise ValueError("Give a stream name, a state, or a place to search around.")
    params: dict[str, Any] = {
        "parameter_code": parameter, "computation_identifier": "Mean", "site_type_code": "ST",
        "agency_code": "USGS",
        "properties": "monitoring_location_id,monitoring_location_name,state_name,drainage_area,begin,end",
    }
    if site_no:
        params["monitoring_location_id"] = site_id(site_no)
    if st:
        params["state_name"] = st
    if bbox:
        params["bbox"] = ",".join(f"{x:.5f}" for x in bbox)
    if terms:  # USGS names are ALL CAPS on older sites and Mixed Case on newer ones, so match without case
        params["filter"] = " AND ".join(f"casei(monitoring_location_name) LIKE casei('%{t.lower()}%')" for t in terms)
        params["filter-lang"] = "cql-text"
    return gage_rows(_features("combined-metadata", params, max_rows=max_rows), active_days=active_days)
