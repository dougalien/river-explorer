"""The page itself, run headlessly with USGS and the AI replaced by fakes: nothing here touches the network."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import usgs

APP = str(Path(__file__).resolve().parent.parent / "app.py")
SITE = "05464500"


def fake_daily(site_no, start=None, end=None, parameter="00060"):
    dates = pd.date_range(start or "1995-01-01", end or "2026-10-16")
    doy = dates.dayofyear.to_numpy()
    q = 3000 + 1500 * np.sin(doy / 365 * 2 * np.pi) + 4000 * (np.arange(len(dates)) % 90 < 5)
    df = pd.DataFrame({"date": dates, "value": q, "approved": True, "flags": ""})
    df.attrs["unit"] = "cfs"
    return df


def fake_daily_stats(site_no, start=None, end=None, parameter="00060"):
    d = fake_daily(site_no, start, end, parameter)
    df = pd.DataFrame({"date": d["date"], "mean": d["value"], "max": d["value"] * 1.2, "min": d["value"] * 0.8,
                       "approved": True, "flags": ""})
    df.attrs["unit"] = "cfs"
    return df


def fake_continuous(site_no, start=None, end=None, parameter="00060", days=7):
    end_ts = pd.Timestamp("2026-10-16", tz="UTC") if end is None else pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end)
    start_ts = end_ts - pd.Timedelta(days=days) if start is None else (pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start))
    times = pd.date_range(start_ts, end_ts, freq="15min", tz="UTC")
    df = pd.DataFrame({"time": times, "value": 5000 + 2000 * np.sin(np.arange(len(times)) / 20), "approved": False, "flags": ""})
    df.attrs["unit"] = "cfs"
    return df


def fake_info(site_no):
    return {"site_no": site_no, "name": "CEDAR RIVER AT CEDAR RAPIDS, IA", "label": "Cedar River at Cedar Rapids, IA", "state": "Iowa",
            "county": "Linn County", "lat": 41.97, "lon": -91.67, "drainage_area_mi2": 6510.0, "huc": None,
            "time_zone": "CST", "uses_dst": "Y", "site_type": "ST"}


@pytest.fixture
def app(monkeypatch):
    import streamlit as st

    st.cache_data.clear()                         # cached USGS answers from an earlier test would hide this test's fakes
    monkeypatch.setattr(usgs, "fetch_daily", fake_daily)
    monkeypatch.setattr(usgs, "fetch_daily_stats", fake_daily_stats)
    monkeypatch.setattr(usgs, "fetch_continuous", fake_continuous)
    monkeypatch.setattr(usgs, "fetch_site_info", fake_info)
    monkeypatch.setenv("OLLAMA_API_KEY", "")      # no AI: the built-in guide answers
    monkeypatch.setenv("AI_API_KEY", "")
    monkeypatch.setenv("TEACHER_MODE", "0")
    monkeypatch.setenv("CLASS_PASSCODE", "")
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["selected"] = SITE
    return at


def click(at, label):
    next(b for b in at.button if b.label == label).click().run()


def test_a_selected_gage_shows_the_chart_and_the_guide_invitation(app):
    app.run()
    assert not app.exception
    assert any("Cedar River at Cedar Rapids" in s.value for s in app.subheader)
    assert any(b.label == "Start guided exploration" for b in app.button)


def test_the_guide_runs_without_an_ai_and_the_report_appears(app):
    app.run()
    click(app, "Start guided exploration")
    assert not app.exception
    assert len(app.chat_message) == 1 and "each axis" in app.chat_message[0].markdown[0].value.lower()
    assert any(e.label == "Session report for your instructor" for e in app.expander)
    assert sum(b.label.startswith("Download report") or b.label.startswith("Download data") for b in app.get("download_button")) == 2
    click(app, "I'm stuck")
    assert not app.exception and len(app.chat_message) == 3        # the guide, the "I'm stuck" note, the nudge


def test_a_gallery_scene_opens_its_own_window_with_the_scene_view(app):
    app.run()
    click(app, "Explore")                                           # the first card: a small city stream
    assert not app.exception
    assert app.session_state["window"].id == "flashy_urban" and app.session_state["selected"] == "01104460"
    assert app.session_state["view"] == "Gallery window"


def test_a_search_after_a_scene_returns_to_the_normal_time_spans(app, monkeypatch):
    from datetime import date

    monkeypatch.setattr(usgs, "find_gages", lambda **kw: [{
        "site_no": SITE, "name": "CEDAR RIVER AT CEDAR RAPIDS, IA", "label": "Cedar River at Cedar Rapids, IA", "state": "Iowa",
        "lat": 41.97, "lon": -91.67, "drainage_area_mi2": 6510.0, "begin": date(1902, 10, 1), "end": date(2026, 10, 15),
        "record_years": 124.0, "active": True}])
    app.session_state["selected"] = None
    app.run()
    click(app, "Explore")                                           # a gallery scene: window and its own view
    assert app.session_state["view"] == "Gallery window"
    app.text_input(key="query_text").set_value(SITE).run()
    click(app, "Find gages")
    assert not app.exception
    assert app.session_state["window"] is None and app.session_state["view"] == "Last 30 days"
    assert app.session_state["selected"] == SITE


def test_two_gages_are_overlaid_and_the_numbers_appear_only_on_request(app, monkeypatch):
    asked = []

    def recording_daily(site_no, start=None, end=None, parameter="00060"):
        asked.append(site_no)
        return fake_daily(site_no, start, end, parameter)

    def two_sizes(site_no):
        info = fake_info(site_no)
        return {**info, "label": "Stony Brook, MA" if site_no == "01104460" else info["label"],
                "drainage_area_mi2": 22.0 if site_no == "01104460" else 6510.0}

    monkeypatch.setattr(usgs, "fetch_daily", recording_daily)
    monkeypatch.setattr(usgs, "fetch_site_info", two_sizes)
    app.session_state["view"] = "Last year"
    app.session_state["cmp_on"] = True
    app.session_state["cmp_site"] = "01104460"
    app.run()
    assert not app.exception and "01104460" in asked and SITE in asked
    assert app.radio(key="cmp_mode").value == "per_sq_mi"             # the fair default for a creek next to a big river
    assert len(app.dataframe) == 0                                    # numbers stay hidden until asked for
    app.checkbox(key="cmp_nums").check().run()
    assert not app.exception and len(app.dataframe) == 1
    table = app.dataframe[0].value
    assert list(table.columns)[1:] == ["Cedar River at Cedar Rapids, IA", "Stony Brook, MA"]
    app.radio(key="cmp_mode").set_value("shape").run()
    assert not app.exception


def test_comparing_a_gage_with_itself_is_refused_politely(app):
    app.session_state["cmp_on"] = True
    app.session_state["cmp_site"] = SITE
    app.run()
    assert not app.exception and any("same gage" in w.value for w in app.warning)


def test_a_second_gage_without_a_drainage_area_falls_back_to_actual_flow(app, monkeypatch):
    monkeypatch.setattr(usgs, "fetch_site_info", lambda s: {**fake_info(s), "drainage_area_mi2": None} if s == "01104460" else fake_info(s))
    app.session_state["cmp_on"] = True
    app.session_state["cmp_site"] = "01104460"
    app.run()
    assert not app.exception and any("Showing actual flow" in w.value for w in app.warning)


def test_the_guide_is_dropped_when_the_gage_changes(app):
    app.run()
    click(app, "Start guided exploration")
    assert app.session_state["guide"] is not None
    click(app, "Explore")                                           # pick a different gage from the gallery
    assert app.session_state["guide"] is None and not app.exception


def chart_traces(at, key):
    import json

    chart = next(c for c in at.get("plotly_chart") if json.loads(c.proto.spec)["layout"] and key in c.proto.id)
    return [t.get("name") for t in json.loads(chart.proto.spec)["data"]]


def test_daily_views_show_the_usgs_max_min_band_and_15_minute_views_show_daily_steps(app):
    app.session_state["view"] = "Last year"
    app.run()
    assert not app.exception
    names = chart_traces(app, "hydro_chart")
    assert {"Daily min (USGS)", "Daily max (USGS)", "Daily mean (USGS)"} <= set(names)
    assert any("daily mean, max and min" in b.label for b in app.get("download_button"))
    app.radio(key="view").set_value("Last 7 days").run()
    assert not app.exception
    names = chart_traces(app, "hydro_chart")
    assert {"Daily min (USGS)", "Daily max (USGS)", "Daily mean (USGS)", "Discharge"} <= set(names)
    app.checkbox(key="daily_stats").uncheck().run()
    assert "Daily max (USGS)" not in chart_traces(app, "hydro_chart")


def test_a_gage_with_only_daily_means_says_so(app, monkeypatch):
    monkeypatch.setattr(usgs, "fetch_daily_stats",
                        lambda *a, **k: fake_daily_stats(*a, **k).assign(max=float("nan"), min=float("nan")))
    app.session_state["view"] = "Last year"
    app.run()
    assert not app.exception
    assert any("only the daily mean" in c.value for c in app.caption)
    assert "Daily max (USGS)" not in chart_traces(app, "hydro_chart")


def test_conductance_trend_loads_only_when_asked_and_shows_a_trend(app, monkeypatch):
    asked = []

    def conductance(site_no, start=None, end=None, parameter="00060"):
        asked.append(parameter)
        if parameter != usgs.CONDUCTANCE:
            return fake_daily(site_no, start, end, parameter)
        dates = pd.date_range("2005-01-01", "2026-10-16")
        df = pd.DataFrame({"date": dates, "value": 300 + 8.0 * (dates.year - 2005) + 10 * np.sin(np.arange(len(dates))),
                           "approved": True, "flags": ""})
        df.attrs["unit"] = "µS/cm"
        return df

    monkeypatch.setattr(usgs, "fetch_daily", conductance)
    app.run()
    assert not app.exception and usgs.CONDUCTANCE not in asked
    app.toggle(key="sc_on").set_value(True).run()
    app.radio(key="sc_mode").set_value("long").run()
    assert not app.exception and usgs.CONDUCTANCE in asked
    assert any("rising" in m.value and "statistically significant" in m.value for m in app.markdown)
    assert any("standard deviation (SD)" in c.value for c in app.caption)
    names = chart_traces(app, "sc_chart")
    assert "Year mean ± SD" in names and "Trend (Sen's slope, annual means)" in names
    app.radio(key="sc_by").set_value("month").run()
    app.radio(key="sc_bars").set_value("se").run()
    app.checkbox(key="sc_raw").check().run()
    assert not app.exception and "Month mean ± SE" in chart_traces(app, "sc_chart")


def test_a_gage_without_conductance_offers_nearby_gages(app, monkeypatch):
    from datetime import date

    def no_conductance(site_no, start=None, end=None, parameter="00060"):
        if parameter == usgs.CONDUCTANCE:
            raise usgs.USGSError("USGS has no daily values for site 05464500 in that range.")
        return fake_daily(site_no, start, end, parameter)

    seen = {}
    monkeypatch.setattr(usgs, "fetch_daily", no_conductance)
    monkeypatch.setattr(usgs, "find_gages", lambda **kw: seen.update(kw) or [{
        "site_no": "05464420", "name": "CEDAR R AT BLAIRS FERRY RD", "label": "Cedar River at Blairs Ferry Road", "state": "Iowa",
        "lat": 42.0, "lon": -91.6, "drainage_area_mi2": None, "begin": date(2012, 1, 1), "end": date(2026, 10, 15),
        "record_years": 14.8, "active": True}])
    app.session_state["sc_on"] = True
    app.session_state["sc_mode"] = "long"
    app.run()
    assert not app.exception and any("no daily specific conductance" in i.value for i in app.info)
    click(app, "Find gages within 50 km that record conductance")
    assert seen["parameter"] == usgs.CONDUCTANCE
    click(app, "Use this gage")
    assert app.session_state["selected"] == "05464420" and not app.exception


def test_recession_loads_only_when_asked_and_both_views_draw(app):
    app.run()
    assert not any(c.proto.id and "rec_mrc_chart" in c.proto.id for c in app.get("plotly_chart"))
    app.toggle(key="rec_on").set_value(True).run()
    app.radio(key="rec_mode").set_value("long").run()
    assert not app.exception
    assert any(m.label == "Segments" for m in app.metric)
    names = chart_traces(app, "rec_mrc_chart")
    assert "Segments, slid in time" in names and any(n.startswith("Master curve, k =") for n in names)
    assert any(n.startswith("All days: b =") for n in chart_traces(app, "rec_bn_chart"))
    assert any(b.label == "Download the segments (CSV)" for b in app.get("download_button"))
    app.selectbox(key="rec_season").set_value("summer").run()
    app.number_input(key="rec_min_len").set_value(30).run()
    assert not app.exception


def storm_continuous(site_no, start=None, end=None, parameter="00060", days=7):
    """One storm at 15-minute resolution: flow rises for a day then falls with k = 1.5 days; conductance dips as it rises."""
    df = fake_continuous(site_no, start, end, parameter, days)
    hours = ((df["time"] - df["time"].iloc[0]) / pd.Timedelta(hours=1)).to_numpy()
    q = np.where(hours < 24, 80.0 + hours * 30, 800 * np.exp(-(hours - 24) / 36))
    df["value"] = 600 - 0.4 * q if parameter == usgs.CONDUCTANCE else q
    df.attrs["unit"] = "µS/cm" if parameter == usgs.CONDUCTANCE else "cfs"
    return df


def test_conductance_and_recession_follow_a_short_hydrograph_window(app, monkeypatch):
    asked = []

    def recording(site_no, start=None, end=None, parameter="00060", days=7):
        asked.append(parameter)
        return storm_continuous(site_no, start, end, parameter, days)

    monkeypatch.setattr(usgs, "fetch_continuous", recording)
    app.session_state["view"] = "Last 7 days"
    app.session_state["sc_on"] = True
    app.session_state["rec_on"] = True
    app.run()
    assert not app.exception
    assert app.radio(key="sc_mode").value == "window" and app.radio(key="rec_mode").value == "window"
    assert usgs.CONDUCTANCE in asked                                   # 15-minute conductance for the same week
    names = chart_traces(app, "sc_window_chart")
    assert "Specific conductance" in names and "Discharge" in names
    lowest = next(m for m in app.metric if m.label == "Lowest")
    assert lowest.value.endswith("µS/cm")
    k = next(m for m in app.metric if m.label == "Median k of segments")
    assert k.value == "1.5 days"                                         # k = 36 hours, found in hourly means
    assert any("Hourly means" in c.value for c in app.caption)


def test_conductance_follows_a_daily_window_with_the_usgs_max_and_min(app):
    app.session_state["view"] = "Last year"
    app.session_state["sc_on"] = True
    app.run()
    assert not app.exception
    names = chart_traces(app, "sc_window_chart")
    assert {"Daily mean conductance (USGS)", "Daily max (USGS)", "Daily min (USGS)", "Discharge"} <= set(names)
