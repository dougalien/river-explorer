"""
River Explorer: type any river, place or question, find a real USGS stream gage, and look at its hydrograph.

Run:  streamlit run app.py        (or double-click run.bat)
"""

from __future__ import annotations

import hmac
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import compare
import finder
import gallery
import llm
import notes
import recession
import report
import salt
import settings
import tutor
import usgs

settings.export_usgs_key()

BLUE, ORANGE = "#2a78d6", "#eb6834"
EXAMPLES = ["Charles River", "a flashy urban creek near Boston", "Cedar River in Iowa", "biggest river in Louisiana"]
VIEWS = ["Last 7 days", "Last 30 days", "Last year", "Full record"]
EVENT_VIEW = "Gallery window"

st.set_page_config(page_title="River Explorer", page_icon="🌊", layout="wide")


# ---------------------------------------------------------------------------------------------------
# Cached USGS calls (shared by every student on the server)
# ---------------------------------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False, max_entries=256)
def cached_info(site_no: str) -> dict:
    return usgs.fetch_site_info(site_no)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=64)
def cached_daily(site_no: str, start: str | None, end: str | None = None) -> "object":
    return usgs.fetch_daily(site_no, start=start, end=end)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=64)
def cached_daily_stats(site_no: str, start: str | None, end: str | None = None, parameter: str = usgs.DISCHARGE) -> "object":
    return usgs.fetch_daily_stats(site_no, start=start, end=end, parameter=parameter)


@st.cache_data(ttl=600, show_spinner=False, max_entries=64)
def cached_iv_param(site_no: str, start: str, end: str, parameter: str) -> "object":
    return usgs.fetch_continuous(site_no, start=start, end=end, parameter=parameter)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=64)
def cached_conductance(site_no: str) -> "object":
    return usgs.fetch_daily(site_no, parameter=usgs.CONDUCTANCE)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=64)
def cached_conductance_gages(lat: float, lon: float, radius_km: float) -> list:
    return usgs.find_gages(bbox=usgs.bbox_around(lat, lon, radius_km), parameter=usgs.CONDUCTANCE, active_days=365)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=32)
def cached_iv_range(site_no: str, start: str, end: str) -> "object":
    return usgs.fetch_continuous(site_no, start=start, end=pd.Timestamp(end) + pd.Timedelta(days=1))


@st.cache_data(ttl=600, show_spinner=False, max_entries=128)
def cached_recent(site_no: str, days: int) -> "object":
    return usgs.fetch_continuous(site_no, days=days)


@st.cache_resource
def server_budget() -> llm.Budget:
    return llm.Budget(settings.get_int("AI_CALLS_PER_DAY", 1500), daily=True, what="day")


# ---------------------------------------------------------------------------------------------------
# State and callbacks
# ---------------------------------------------------------------------------------------------------

def init_state() -> None:
    ss = st.session_state
    ss.setdefault("query_text", "")
    ss.setdefault("queued", False)
    ss.setdefault("result", None)
    ss.setdefault("selected", None)
    ss.setdefault("search_id", 0)
    ss.setdefault("unlocked", False)
    ss.setdefault("guide", None)
    ss.setdefault("window", None)
    ss.setdefault("budget", llm.Budget(settings.get_int("AI_CALLS_PER_SESSION", 80), what="session"))


def queue_search() -> None:
    st.session_state.queued = True


def use_example(text: str) -> None:
    st.session_state.query_text = text
    st.session_state.queued = True


def leave_gallery() -> None:
    """A gage chosen from a search is shown with the usual time spans, not a gallery scene's window."""
    ss = st.session_state
    ss.window = None
    if ss.get("view") == EVENT_VIEW:
        ss.view = VIEWS[1]


def select_gage(site_no: str) -> None:
    leave_gallery()
    st.session_state.selected = site_no


def pick_scene(scene_id: str) -> None:
    ss = st.session_state
    scene = gallery.BY_ID[scene_id]
    ss.window, ss.selected, ss.guide, ss.result = scene, scene.site_no, None, None
    ss.view = EVENT_VIEW


def surprise_scene() -> None:
    ss = st.session_state
    pick_scene(gallery.surprise(exclude=ss.window.id if ss.get("window") else None).id)


def map_clicked() -> None:
    ss = st.session_state
    event = ss.get(f"gage_map_{ss.search_id}")
    points = (event.selection.points if event and event.selection else []) if event is not None else []
    if points and ss.result:
        idx = points[0].get("point_index", points[0].get("point_number"))
        if isinstance(idx, int) and 0 <= idx < len(ss.result.gages):
            select_gage(ss.result.gages[idx].site_no)


def ai_client() -> tuple[llm.ChatClient | None, str]:
    """The AI helper if it is allowed right now, else None and a sentence saying why."""
    cfg = settings.llm_config()
    if cfg is None:
        return None, "No AI key is set, so searches use plain keywords."
    if settings.passcode() and not st.session_state.unlocked:
        return None, "Enter the class passcode in the sidebar to turn on the AI helper."
    return settings.make_client(cfg, st.session_state.budget, server_budget()), ""


def run_search() -> None:
    ss = st.session_state
    text = ss.query_text.strip()
    client, _ = ai_client()
    try:
        with st.spinner("Asking the AI to search USGS…" if client else "Searching USGS…"):
            result = finder.find(text, client)
    except usgs.USGSError as exc:
        st.error(str(exc))
        return
    leave_gallery()
    ss.result = result
    ss.search_id += 1
    ss.selected = result.gages[0].site_no if len(result.gages) == 1 else None


# ---------------------------------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------------------------------

def sidebar() -> None:
    ss = st.session_state
    with st.sidebar:
        st.header("🌊 River Explorer")
        st.caption("Real data from the U.S. Geological Survey. Look first, then ask questions.")
        cfg = settings.llm_config()
        code = settings.passcode()
        if cfg is not None and code and not ss.unlocked:
            entered = st.text_input("Class passcode", type="password", help="Ask your instructor.")
            if entered and hmac.compare_digest(entered.encode(), code.encode()):
                ss.unlocked = True
                st.rerun()
            elif entered:
                st.error("That passcode didn't match.")
        client, why_not = ai_client()
        if client:
            st.success(f"AI helper on · {ss.budget.left} AI steps left in this session")
        else:
            st.info(f"AI helper off. {why_not}")
        with st.expander("How the search works"):
            st.markdown(
                "- Type a river, a town, a storm, or a USGS site number.\n"
                "- The AI turns your words into a search. **USGS decides which gages exist**: "
                "the AI can only choose among real results.\n"
                "- Without the AI, a keyword search looks for the river name and a state (try *Charles River MA*)."
            )
        st.caption("Data: USGS Water Data APIs. Recent values are provisional and may be revised.")


# ---------------------------------------------------------------------------------------------------
# Finder: search results, map, cards
# ---------------------------------------------------------------------------------------------------

def map_view(gages: list) -> tuple[float, float, float]:
    lats, lons = [g.lat for g in gages], [g.lon for g in gages]
    lat, lon = (min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2
    span = max((max(lons) - min(lons)) * math.cos(math.radians(lat)), (max(lats) - min(lats)) * 1.6, 0.08)
    return lat, lon, min(max(math.log2(300 / span), 2.0), 11.0)


def gage_map(gages: list, selected: str | None) -> go.Figure:
    mapped = [g for g in gages if g.lat is not None and g.lon is not None]
    lat, lon, zoom = map_view(mapped)
    fig = go.Figure(go.Scattermap(
        lat=[g.lat for g in mapped], lon=[g.lon for g in mapped], mode="markers+text",
        text=[str(gages.index(g) + 1) for g in mapped], textfont=dict(color="white", size=13),
        marker=dict(size=26, color=[ORANGE if g.site_no == selected else BLUE for g in mapped]),
        customdata=[[g.label, g.facts] for g in mapped],
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>",
    ))
    fig.update_layout(map=dict(style="open-street-map", center=dict(lat=lat, lon=lon), zoom=zoom),
                      margin=dict(l=0, r=0, t=0, b=0), height=430, showlegend=False)
    return fig


def show_results() -> None:
    ss = st.session_state
    result = ss.result
    if result is None:
        return
    st.markdown(f"**{result.reading}**")
    for note in result.notes:
        st.caption(note)
    if not result.gages:
        return
    left, right = st.columns([3, 2], gap="large")
    with left:
        if any(g.lat is not None for g in result.gages):
            st.plotly_chart(gage_map(result.gages, ss.selected), key=f"gage_map_{ss.search_id}", on_select=map_clicked,
                            selection_mode="points", config={"displayModeBar": False})
    with right:
        for i, g in enumerate(result.gages, start=1):
            with st.container(border=True):
                st.markdown(f"**{i}. {g.label}**")
                st.caption(g.facts)
                if g.why:
                    st.write(g.why)
                chosen = g.site_no == ss.selected
                st.button("Exploring this one" if chosen else "Explore this gage", key=f"pick_{ss.search_id}_{g.site_no}",
                          type="primary" if chosen else "secondary", on_click=select_gage, args=(g.site_no,),
                          width="stretch")


def load_view(site_no: str, view: str, info: dict):
    """The data behind one time-span choice: (dataframe, x values, x-axis label, hover template)."""
    scene = st.session_state.get("window") if view == EVENT_VIEW else None
    if scene is not None:
        if scene.resolution == "15min":
            df = cached_iv_range(site_no, scene.start, scene.end)
            x = df["time"].dt.tz_convert(usgs.site_timezone(info)).dt.tz_localize(None)
            return df, x, "Local time at the gage", "%{x|%b %d, %H:%M}<br>%{y:,.0f} cfs<extra></extra>"
        df = cached_daily(site_no, scene.start, scene.end)
        return df, df["date"], "Date", "%{x|%b %d, %Y}<br>%{y:,.0f} cfs (daily mean)<extra></extra>"
    if view in ("Last 7 days", "Last 30 days"):
        df = cached_recent(site_no, 7 if view == "Last 7 days" else 30)
        x = df["time"].dt.tz_convert(usgs.site_timezone(info)).dt.tz_localize(None)
        return df, x, "Local time at the gage", "%{x|%b %d, %H:%M}<br>%{y:,.0f} cfs<extra></extra>"
    start = None if view == "Full record" else (date.today() - timedelta(days=365)).isoformat()
    df = cached_daily(site_no, start)
    return df, df["date"], "Date", "%{x|%b %d, %Y}<br>%{y:,.0f} cfs (daily mean)<extra></extra>"


# ---------------------------------------------------------------------------------------------------
# The guide (chat beside the chart). Callbacks run before the page redraws, so the chart sees the new highlight.
# ---------------------------------------------------------------------------------------------------

def chart_selection() -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """The stretch dragged across on the hydrograph, as (start, end) in the chart's own time, or None."""
    try:
        box = st.session_state.get("hydro_chart").selection.box or []
        x0, x1 = sorted(pd.Timestamp(v) for v in box[0]["x"])
        return x0, x1
    except Exception:  # no selection yet, or an older shape of the event
        return None


def focus_text() -> str:
    """A stretch the student drag-selected on the chart, in words (empty if none)."""
    sel = chart_selection()
    return f"{sel[0]:%b %d} to {sel[1]:%b %d}" if sel else ""


def guide_context(site_no: str) -> tutor.TutorContext:
    ss = st.session_state
    info = cached_info(site_no)
    view = ss.get("view", VIEWS[1])
    df, x, _, _ = load_view(site_no, view, info)
    window = pd.Series(df["value"].to_numpy(), index=pd.DatetimeIndex(x)).resample("D").mean()
    scene = ss.get("window") if view == EVENT_VIEW else None
    return tutor.build_context(info["label"], scene.window_text if scene else view, window, cached_daily(site_no, None),
                               bool(ss.get("compare")) and view != "Full record", info["drainage_area_mi2"], focus_text(),
                               scenario=scene.cause if scene else None)


def _record(result: tutor.TurnResult, student: str | None = None) -> None:
    g = st.session_state.guide
    if student:
        g["msgs"].append({"role": "student", "text": student})
    g["msgs"].append({"role": "assistant", "text": result.say})
    g["highlight"] = result.highlight
    g["notes"] = result.notes


def guide_start() -> None:
    ss = st.session_state
    client, _ = ai_client()
    ss.guide = {"site": ss.selected, "tutor": tutor.Tutor(client=client), "msgs": [], "highlight": None, "notes": []}
    try:
        _record(ss.guide["tutor"].start(guide_context(ss.selected)))
    except usgs.USGSError as exc:
        ss.guide["notes"] = [str(exc)]


def guide_send() -> None:
    ss = st.session_state
    text = (ss.get("guide_input") or "").strip()
    if not (ss.guide and text):
        return
    try:
        _record(ss.guide["tutor"].turn(guide_context(ss.guide["site"]), text), student=text)
    except usgs.USGSError as exc:
        ss.guide["notes"] = [str(exc)]


def guide_stuck() -> None:
    ss = st.session_state
    if ss.guide:
        try:
            _record(ss.guide["tutor"].turn(guide_context(ss.guide["site"]), "", hint=True), student="I'm stuck.")
        except usgs.USGSError as exc:
            ss.guide["notes"] = [str(exc)]


def guide_reset() -> None:
    st.session_state.guide = None


def guide_panel() -> None:
    g = st.session_state.guide
    st.markdown("#### River Guide")
    if g is None:
        st.write("Take a minute with the chart first. When you're ready, the guide will ask you questions, one at a time, "
                 "and help you work out what the river is doing. It won't hand you the answers.")
        st.button("Start guided exploration", type="primary", on_click=guide_start, width="stretch")
        return
    with st.container(height=420, border=True):
        for m in g["msgs"]:
            with st.chat_message("user" if m["role"] == "student" else "assistant"):
                st.markdown(m["text"])
    for note in g["notes"]:
        st.caption(note)
    t: tutor.Tutor = g["tutor"]
    st.chat_input("Type what you see or think…", key="guide_input", on_submit=guide_send)
    b1, b2, b3 = st.columns([1, 1, 1.4], vertical_alignment="center")
    b1.button("I'm stuck", on_click=guide_stuck, disabled=t.finished, width="stretch")
    b2.button("Restart", on_click=guide_reset, width="stretch")
    b3.caption("Finished" if t.finished else f"Step {min(t.index + 1, len(tutor.CONCEPTS))} of {len(tutor.CONCEPTS)}")
    if not t.finished:
        st.caption("Tip: drag across the chart to select a stretch, then ask about it.")
    with st.expander("Session report for your instructor"):
        st.text_input("Your name (it goes in the report only)", key="student_name", max_chars=60)
        meta = report_meta(g["site"])
        stem = report.file_stem(meta)
        st.download_button("Download report (.md)", report.build_markdown(meta, t), f"{stem}.md", "text/markdown",
                           width="stretch")
        st.download_button("Download data (.json)", report.to_json_text(meta, t), f"{stem}.json", "application/json",
                           width="stretch")
        st.caption("Includes your answers word for word, which steps you did on your own, and a suggested score. Nothing is stored on the server.")


def report_meta(site_no: str) -> dict:
    ss = st.session_state
    info = cached_info(site_no)
    view = ss.get("view", VIEWS[1])
    scene = ss.get("window") if view == EVENT_VIEW else None
    return {"student": (ss.get("student_name") or "").strip(), "site_label": info["label"], "site_no": site_no,
            "view": scene.window_text if scene else view, "scene_title": scene.title if scene else None,
            "compare_on": bool(ss.get("compare")) and view != "Full record"}


# ---------------------------------------------------------------------------------------------------
# Hydrograph
# ---------------------------------------------------------------------------------------------------

def _facts_column(f: dict) -> list[str]:
    def n(v, spec):
        return "n/a" if v is None else format(v, spec)
    return [n(f["area"], ",.0f"), n(f["peak_cfs"], ",.0f"), n(f["peak_per_mi2"], ".2f"), n(f["flashiness"], ".2f"),
            "did not fall halfway in view" if f["half_fall_days"] is None else str(f["half_fall_days"])]


def cmp_search() -> None:
    """Find candidates for the second gage from what the student typed."""
    ss = st.session_state
    client, _ = ai_client()
    try:
        result = finder.find(ss.get("cmp_query", ""), client)
    except usgs.USGSError as exc:
        ss.cmp_note, ss.cmp_candidates = str(exc), []
        return
    ss.cmp_candidates, ss.cmp_note = result.gages, result.reading
    ss.cmp_site = result.gages[0].site_no if len(result.gages) == 1 else None
    ss.cmp_pick = 0


def cmp_picked() -> None:
    ss = st.session_state
    options = ss.get("cmp_candidates") or []
    if options:
        ss.cmp_site = options[ss.get("cmp_pick", 0)].site_no


def cmp_gallery_picked() -> None:
    ss = st.session_state
    choice = ss.get("cmp_gallery")
    if choice and choice != "(none)":
        ss.cmp_site = gallery.BY_ID[choice].site_no


def second_gage_picker(site_no: str) -> str | None:
    """The 'compare with a second gage' controls. Returns the second site number when one is chosen."""
    ss = st.session_state
    if not st.checkbox("Compare with a second gage", key="cmp_on",
                       help="Overlay another gage to ask why two rivers behave differently: city versus forest, small versus large."):
        return None
    c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
    c1.text_input("Second gage: a river name or a USGS site number", key="cmp_query", on_change=cmp_search)
    c2.selectbox("or pick from the gallery", ["(none)"] + [s.id for s in gallery.SCENES], key="cmp_gallery", on_change=cmp_gallery_picked,
                 format_func=lambda i: i if i == "(none)" else gallery.BY_ID[i].title)
    options = ss.get("cmp_candidates") or []
    if len(options) > 1:
        st.selectbox("Which one?", range(len(options)), key="cmp_pick", on_change=cmp_picked,
                     format_func=lambda i: f"{options[i].label} ({options[i].facts})")
    if ss.get("cmp_note") and not ss.get("cmp_site"):
        st.caption(ss.cmp_note)
    other = ss.get("cmp_site")
    if other == site_no:
        st.warning("That is the same gage. Pick a different one.")
        return None
    if other:
        st.radio("Compare using", list(compare.MODES), format_func=lambda k: compare.MODES[k][0], horizontal=True, key="cmp_mode",
                 help="Raw flow hides small streams next to big rivers. 'Per square mile' divides by watershed size.")
    return other


def add_daily_stats(fig: go.Figure, stats: pd.DataFrame, daily_view: bool) -> bool:
    """Draw what USGS publishes for each day. Daily views get a max-to-min band around the mean line (the mean is the
    main trace); 15-minute views get each day's mean, max and min as steps over the readings. True if anything was drawn."""
    has_range = stats["max"].notna().any() or stats["min"].notna().any()
    if daily_view:
        if not has_range:
            return False
        fig.add_trace(go.Scatter(x=stats["date"], y=stats["min"], mode="lines", line=dict(width=0.8, color="rgba(42,120,214,0.5)"),
                                 name="Daily min (USGS)", hovertemplate="%{y:,.0f} cfs (daily min)<extra></extra>"))
        fig.add_trace(go.Scatter(x=stats["date"], y=stats["max"], mode="lines", line=dict(width=0.8, color="rgba(42,120,214,0.5)"),
                                 fill="tonexty", fillcolor="rgba(42,120,214,0.15)", name="Daily max (USGS)",
                                 hovertemplate="%{y:,.0f} cfs (daily max)<extra></extra>"))
        return True
    # one extra row so the last day's step reaches midnight
    steps = pd.concat([stats, stats.tail(1).assign(date=stats["date"].iloc[-1] + pd.Timedelta(days=1))], ignore_index=True)
    drawn = False
    for col, label, style in (("max", "Daily max (USGS)", dict(dash="dot", color="#1b4f8f")),
                              ("mean", "Daily mean (USGS)", dict(dash="dash", color="#333333")),
                              ("min", "Daily min (USGS)", dict(dash="dot", color="#1b4f8f"))):
        if steps[col].notna().any():
            fig.add_trace(go.Scatter(x=steps["date"], y=steps[col], mode="lines", line=dict(width=1.3, shape="hv", **style), name=label,
                                     hovertemplate="%{y:,.0f} cfs (" + label[:-7].lower() + ")<extra></extra>"))
            drawn = True
    return drawn


def hydrograph(site_no: str) -> dict | None:
    """Draw the hydrograph. Returns the time window it shows, for the conductance and recession panels (None on failure).

    Keys: fine (True for 15-minute data), flow (DataFrame date, value, flags in the gage's local time), start, end
    (the dragged selection if there is one, else the whole chart), selected (bool), utc (start, end) for 15-minute data,
    label (words for the window).
    """
    ss = st.session_state
    try:
        info = cached_info(site_no)
    except usgs.USGSError as exc:
        st.error(str(exc))
        return
    if ss.guide and ss.guide["site"] != site_no:
        ss.guide = None
    st.divider()
    st.subheader(info["label"])
    bits = [b for b in (info["county"], info["state"]) if b]
    if info["drainage_area_mi2"]:
        bits.append(f"drains {info['drainage_area_mi2']:,.0f} mi²")
    bits.append(f"USGS {site_no}")
    st.caption(" · ".join(bits))
    scene = ss.window if ss.window and ss.window.site_no == site_no else None
    if scene:
        cap_col, next_col = st.columns([4, 1], vertical_alignment="center")
        cap_col.caption(f"Gallery: {scene.title}. {scene.teaser}")
        next_col.button("Another scene", on_click=surprise_scene, width="stretch")
    views = ([EVENT_VIEW] if scene else []) + VIEWS
    if ss.get("view") not in views:
        ss.view = views[0] if scene else VIEWS[1]

    c1, c2, c3, c4 = st.columns([3, 1, 2, 2])
    view = c1.radio("Time span", views, index=0 if scene else 1, horizontal=True, label_visibility="collapsed", key="view")
    log_axis = c2.checkbox("Log scale", value=False, key="log", help="Log scale makes low flows easier to see.")
    normal_on = c3.checkbox("Compare with normal", value=False, key="compare", disabled=view == "Full record",
                          help="Shades the typical range (10th to 90th percentile) for each date, from every year on record. "
                               "Look at the shape first, then turn this on.") and view != "Full record"
    stats_on = c4.checkbox("USGS daily mean, max, min", value=True, key="daily_stats",
                           help="The daily statistics USGS publishes for this gage: the day's mean, and the highest and lowest "
                                "readings. Many gages publish only the mean for discharge.")
    other = second_gage_picker(site_no)
    compare_mode = ss.get("cmp_mode", "per_sq_mi")

    try:
        with st.spinner("Fetching data from USGS…" + (" The whole record can take several seconds." if view == "Full record" else "")):
            df, x, xlabel, hover = load_view(site_no, view, info)
    except usgs.USGSError as exc:
        st.error(str(exc))
        return

    second = None          # (info, dataframe, x) of the second gage, when one is chosen and has data in this window
    if other:
        try:
            info2 = cached_info(other)
            df2, x2, _, _ = load_view(other, view, info2)
            second = (info2, df2, x2)
        except usgs.USGSError as exc:
            st.warning(f"Could not load the second gage for this window: {exc}")

    stats = None           # USGS daily mean, max and min for the dates on the chart
    if stats_on and second is None:
        try:
            stats = cached_daily_stats(site_no, pd.Timestamp(x.min()).date().isoformat(), pd.Timestamp(x.max()).date().isoformat())
        except usgs.USGSError as exc:
            st.caption(f"Could not load the USGS daily statistics: {exc}")

    daily_all = None
    if normal_on or settings.get("TEACHER_MODE") == "1":
        try:
            with st.spinner("Loading the whole record to find what is normal… (a few seconds the first time)"):
                daily_all = cached_daily(site_no, None)
        except usgs.USGSError as exc:
            st.error(str(exc))

    left, right = st.columns([5, 4], gap="large")
    with left:
        fig = go.Figure()
        if normal_on and second is not None:
            st.caption("The normal band is hidden while two gages are shown.")
        if normal_on and daily_all is not None and second is None:
            bands = notes.normal_bands(daily_all)
            dates = pd.date_range(pd.Timestamp(x.min()).normalize(), pd.Timestamp(x.max()).normalize())
            b = notes.band_for_dates(bands, dates)
            shape = "hv" if view in ("Last 7 days", "Last 30 days") else "linear"
            fig.add_trace(go.Scatter(x=dates, y=b["p10"], mode="lines", line=dict(width=0, shape=shape), hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=dates, y=b["p90"], mode="lines", line=dict(width=0, shape=shape), fill="tonexty",
                                     fillcolor="rgba(130,130,130,0.25)", name="Typical range (10th to 90th percentile)",
                                     hovertemplate="%{y:,.0f} cfs (90th percentile)<extra></extra>"))
            fig.add_trace(go.Scatter(x=dates, y=b["p50"], mode="lines", line=dict(width=1.2, dash="dot", color="gray", shape=shape),
                                     name="Median for the date", hovertemplate="%{y:,.0f} cfs (median)<extra></extra>"))
        y_title = "Discharge (cubic feet per second)"
        daily_view = xlabel == "Date"
        stats_drawn = stats is not None and add_daily_stats(fig, stats, daily_view)
        if second is None:
            fig.add_trace(go.Scatter(x=x, y=df["value"], mode="lines", line=dict(width=1.6 if view == "Full record" else 2.2, color=BLUE),
                                     hovertemplate=hover, name="Daily mean (USGS)" if daily_view and stats_drawn else "Discharge"))
        else:
            info2, df2, x2 = second
            mode = compare_mode
            try:
                y1 = compare.normalize(pd.Series(df["value"].to_numpy(), index=x), mode, info["drainage_area_mi2"])
                y2 = compare.normalize(pd.Series(df2["value"].to_numpy(), index=x2), mode, info2["drainage_area_mi2"])
            except ValueError as exc:
                st.warning(f"Showing actual flow because {exc}.")
                mode = "actual"
                y1, y2 = pd.Series(df["value"].to_numpy(), index=x), pd.Series(df2["value"].to_numpy(), index=x2)
            y_title = compare.MODES[mode][1]
            unit = "" if mode == "shape" else " cfs/mi²" if mode == "per_sq_mi" else " cfs"
            for label, xs, ys, color in ((info["label"], x, y1, BLUE), (info2["label"], x2, y2, "#eb6834")):
                fig.add_trace(go.Scatter(x=xs, y=ys.to_numpy(), mode="lines", name=label, line=dict(width=2, color=color),
                                         hovertemplate="%{x|%b %d, %Y %H:%M}<br>%{y:.3g}" + unit + "<extra>" + label + "</extra>"))
        hl = ss.guide["highlight"] if ss.guide else None
        if hl:
            start, end, label = hl
            fig.add_shape(type="rect", xref="x", yref="paper", x0=start.isoformat(), x1=end.isoformat(), y0=0, y1=1,
                          fillcolor="rgba(255,200,61,0.28)", line_width=0, layer="below")
            fig.add_annotation(x=start.isoformat(), y=1, yref="paper", text=label, showarrow=False, xanchor="left", yanchor="bottom",
                               font=dict(color="#b8860b"))
        fig.update_layout(height=470, margin=dict(l=10, r=10, t=10, b=10), showlegend=normal_on or second is not None or stats_drawn, hovermode="x unified",
                          legend=dict(orientation="h", x=0, y=1.14), xaxis_title=xlabel, yaxis_title=y_title,
                          yaxis_type="log" if log_axis else "linear", dragmode="select")
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False}, key="hydro_chart",
                        on_select="rerun", selection_mode="box")
        if not df["approved"].iloc[-1]:
            st.caption("The newest values are provisional: USGS may revise them after checking.")
        if stats_on and second is not None:
            st.caption("The USGS daily mean, max and min are hidden while two gages are shown.")
        if stats is not None:
            if stats["max"].isna().all() and stats["min"].isna().all():
                st.caption("USGS publishes only the daily mean discharge at this gage, not a daily maximum or minimum.")
            st.download_button("Download the USGS daily mean, max and min (CSV)", stats.to_csv(index=False, date_format="%Y-%m-%d"),
                               f"USGS_{site_no}_daily_discharge.csv", "text/csv", key="stats_csv")
        if second is not None:
            facts = compare.comparison_facts(compare.daily_window(df["value"], x), info["drainage_area_mi2"],
                                             compare.daily_window(second[1]["value"], second[2]), second[0]["drainage_area_mi2"])
            if st.checkbox("Show the numbers behind the comparison", key="cmp_nums", help="Look at the shapes first, then check the numbers."):
                table = pd.DataFrame({
                    "": ["Watershed (mi²)", "Highest daily flow (cfs)", "Highest flow per mi² (cfs)", "Flashiness (0 steady, 1+ flashy)",
                         "Days for the biggest rise to fall halfway"],
                    info["label"]: _facts_column(facts["a"]), second[0]["label"]: _facts_column(facts["b"])})
                st.dataframe(table, hide_index=True, width="stretch")
    with right:
        guide_panel()

    if settings.get("TEACHER_MODE") == "1" and daily_all is not None:
        with st.sidebar:
            with st.expander("Instructor answer key", expanded=False):
                if scene:
                    st.markdown("- Gallery scene cause: " + scene.cause)
                if second is not None:
                    for line in compare.describe(info["label"], second[0]["label"], facts):
                        st.markdown("- " + line)
                for line in notes.describe(notes.answer_key(daily_all, info["drainage_area_mi2"])):
                    st.markdown("- " + line)

    fine = "time" in df.columns
    flow = pd.DataFrame({"date": pd.DatetimeIndex(x), "value": df["value"].to_numpy(), "flags": df["flags"].to_numpy()})
    sel = chart_selection()
    start, end = (sel if sel else (flow["date"].min(), flow["date"].max()))
    label = f"{start:%b %d, %Y} to {end:%b %d, %Y}" + (" (the stretch selected on the hydrograph)" if sel else f" ({view.lower()})")
    return {"fine": fine, "flow": flow[flow["date"].between(start, end)].reset_index(drop=True), "start": start, "end": end,
            "selected": sel is not None, "utc": (df["time"].min(), df["time"].max()) if fine else None, "label": label,
            "tz": usgs.site_timezone(info)}


# ---------------------------------------------------------------------------------------------------
# Specific conductance trend (road salt)
# ---------------------------------------------------------------------------------------------------

def conductance_gages_nearby(site_no: str) -> None:
    """When a gage has no conductance record, offer the gages around it that do."""
    ss = st.session_state
    try:
        info = cached_info(site_no)
    except usgs.USGSError:
        return
    if info["lat"] is None or info["lon"] is None:
        return
    if st.button("Find gages within 50 km that record conductance", key="sc_find"):
        try:
            with st.spinner("Asking USGS…"):
                rows = cached_conductance_gages(info["lat"], info["lon"], 50.0)
        except usgs.USGSError as exc:
            st.error(str(exc))
            return
        ss.sc_nearby = (site_no, sorted(rows, key=lambda r: -r["record_years"])[:10])
    found = ss.get("sc_nearby")
    if not found or found[0] != site_no:
        return
    if not found[1]:
        st.caption("No gage within 50 km reports daily specific conductance.")
    for r in found[1]:
        c1, c2 = st.columns([4, 1], vertical_alignment="center")
        status = "active" if r["active"] else f"ended {r['end']:%Y}"
        c1.markdown(f"**{r['label']}** · USGS {r['site_no']} · conductance {r['begin']:%Y} to {r['end']:%Y} ({status})")
        c2.button("Use this gage", key=f"sc_use_{r['site_no']}", on_click=select_gage, args=(r["site_no"],), width="stretch")


SPAN_MODES = {"window": "Same time span as the hydrograph", "long": "Long term"}


def span_mode(key: str, win: dict | None, long_label: str) -> str:
    """'window' (follow the hydrograph, or the stretch dragged across it) or 'long'."""
    if win is None:
        return "long"
    return st.radio("Time span", list(SPAN_MODES), key=key, horizontal=True,
                    format_func=lambda k: SPAN_MODES[k] if k == "window" else long_label,
                    help="Follow the hydrograph's time span (drag across the hydrograph to pick one storm), or look across years.")


def conductance_window(site_no: str, win: dict) -> None:
    """Conductance as measured over the hydrograph's window, with discharge alongside: how one storm dilutes or flushes salt."""
    unit = usgs.PARAM_UNIT[usgs.CONDUCTANCE]
    try:
        with st.spinner("Fetching specific conductance from USGS…"):
            if win["fine"]:
                u0, u1 = win["utc"]
                raw = cached_iv_param(site_no, u0.isoformat(), u1.isoformat(), usgs.CONDUCTANCE)
                sc = pd.DataFrame({"date": raw["time"].dt.tz_convert(win["tz"]).dt.tz_localize(None), "mean": raw["value"]})
            else:
                sc = cached_daily_stats(site_no, win["flow"]["date"].min().date().isoformat(),
                                        win["flow"]["date"].max().date().isoformat(), usgs.CONDUCTANCE)
    except usgs.USGSError:
        st.info("USGS has no specific conductance for this gage in this time span. Many gages measure flow only; "
                "try the long-term view, or a nearby gage.")
        conductance_gages_nearby(site_no)
        return
    sc = sc[sc["date"].between(win["start"].normalize() if not win["fine"] else win["start"], win["end"])]
    if sc.empty or sc["mean"].isna().all():
        st.info("No conductance readings in the selected stretch.")
        return
    st.caption(f"{win['label']}. " + ("15-minute readings." if win["fine"] else "USGS daily mean conductance, with the daily max and min where USGS publishes them."))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Lowest", f"{sc['mean'].min():,.0f} {unit}")
    m2.metric("Highest", f"{sc['mean'].max():,.0f} {unit}")
    m3.metric("Mean", f"{sc['mean'].mean():,.0f} {unit}")
    first, last = sc["mean"].dropna().iloc[0], sc["mean"].dropna().iloc[-1]
    m4.metric("Start to end", f"{last - first:+,.0f} {unit}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=win["flow"]["date"], y=win["flow"]["value"], mode="lines", yaxis="y2", name="Discharge",
                             line=dict(width=1.4, color="rgba(120,120,120,0.7)"), hovertemplate="%{y:,.0f} cfs<extra></extra>"))
    if not win["fine"] and "max" in sc and sc["max"].notna().any():
        fig.add_trace(go.Scatter(x=sc["date"], y=sc["min"], mode="lines", line=dict(width=0.8, color="rgba(235,104,52,0.5)"),
                                 name="Daily min (USGS)", hovertemplate="%{y:,.0f} " + unit + " (daily min)<extra></extra>"))
        fig.add_trace(go.Scatter(x=sc["date"], y=sc["max"], mode="lines", line=dict(width=0.8, color="rgba(235,104,52,0.5)"),
                                 fill="tonexty", fillcolor="rgba(235,104,52,0.15)", name="Daily max (USGS)",
                                 hovertemplate="%{y:,.0f} " + unit + " (daily max)<extra></extra>"))
    fig.add_trace(go.Scatter(x=sc["date"], y=sc["mean"], mode="lines", line=dict(width=2.2, color=ORANGE),
                             name="Specific conductance" if win["fine"] else "Daily mean conductance (USGS)",
                             hovertemplate="%{y:,.0f} " + unit + "<extra></extra>"))
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified", legend=dict(orientation="h", x=0, y=1.12),
                      xaxis_title="Local time at the gage" if win["fine"] else "Date",
                      yaxis=dict(title=f"Specific conductance ({unit} at 25 °C)", color=ORANGE),
                      yaxis2=dict(title="Discharge (cfs)", overlaying="y", side="right", showgrid=False, color="#777777"))
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False}, key="sc_window_chart")
    st.caption("A storm usually dilutes conductance as rain water arrives; a winter melt or the first flush after salting can "
               "push it up instead. Drag across the hydrograph above to look at one event.")
    out = sc.rename(columns={"mean": "conductance" if win["fine"] else "mean"})
    st.download_button("Download this conductance (CSV)", out.to_csv(index=False, date_format="%Y-%m-%d %H:%M"),
                       f"USGS_{site_no}_conductance_window.csv", "text/csv", key="sc_window_csv")


def conductance_panel(site_no: str, win: dict | None = None) -> None:
    st.divider()
    st.subheader("Specific conductance (road salt)")
    st.caption("Road salt washes into streams in winter, but much of the chloride soaks into groundwater and returns as base flow "
               "for years. Keeping only the summer and fall months, when streams run on base flow, and averaging each year shows "
               "whether that salt is building up. The same time span as the hydrograph shows what conductance did during one storm.")
    if not st.toggle("Show specific conductance for this gage", key="sc_on"):
        return
    if span_mode("sc_mode", win, "Long-term base-flow trend") == "window":
        conductance_window(site_no, win)
        return
    try:
        with st.spinner("Fetching daily specific conductance from USGS…"):
            daily = cached_conductance(site_no)
    except usgs.USGSError:
        st.info("USGS has no daily specific conductance for this gage. Many gages measure flow only.")
        conductance_gages_nearby(site_no)
        return
    unit = daily.attrs.get("unit") or "µS/cm"

    c1, c2, c3, c4 = st.columns([2, 1, 2, 1], vertical_alignment="bottom")
    season = c1.selectbox("Months to keep", list(salt.SEASONS), format_func=lambda k: salt.SEASONS[k][0], key="sc_season",
                          help="Summer and fall are base-flow months: little snowmelt and no fresh winter runoff.")
    by = c2.radio("Average by", ["year", "month"], format_func=str.capitalize, horizontal=True, key="sc_by")
    bars = c3.radio("Error bars", list(salt.ERROR_BARS), format_func=str.upper, horizontal=True, key="sc_bars",
                    help=" ".join(f"{k.upper()}: {v}." for k, v in salt.ERROR_BARS.items()))
    raw = c4.checkbox("Daily values", key="sc_raw", help="Show the daily means behind the averages.")
    label, months = salt.SEASONS[season]

    table = salt.aggregate(daily, months, by)
    dropped = len(salt.aggregate(daily, months, by, min_coverage=0)) - len(table)
    if table.empty:
        st.warning(f"No {by} has data on at least half its days in {label.lower()}.")
        return
    annual = table if by == "year" else salt.aggregate(daily, months, "year")
    t = salt.trend(annual["year"], annual["mean"]) if len(annual) else None

    fig = go.Figure()
    if raw:
        days = salt.season_days(daily, months)
        fig.add_trace(go.Scatter(x=days["date"], y=days["value"], mode="markers", marker=dict(size=3, color="rgba(120,120,120,0.35)"),
                                 name="Daily mean", hovertemplate="%{x|%b %d, %Y}<br>%{y:,.0f} " + unit + "<extra></extra>"))
    when = "%{x|%Y}" if by == "year" else "%{x|%b %Y}"
    fig.add_trace(go.Scatter(x=table["period"], y=table["mean"], mode="markers+lines", line=dict(width=1, color=BLUE),
                             marker=dict(size=8, color=BLUE), error_y=dict(type="data", array=table[bars].fillna(0), thickness=1.2),
                             customdata=table[["sd", "se", "n"]],
                             name=f"{by.capitalize()} mean ± {bars.upper()}",
                             hovertemplate=when + "<br>mean %{y:,.0f} " + unit + "<br>SD %{customdata[0]:,.0f} · SE %{customdata[1]:,.1f}"
                                           " · %{customdata[2]} days<extra></extra>"))
    if t is not None:
        ends = annual.iloc[[0, -1]]
        fig.add_trace(go.Scatter(x=ends["period"], y=t["intercept"] + t["slope"] * ends["year"], mode="lines",
                                 line=dict(width=2, dash="dash", color=ORANGE), name="Trend (Sen's slope, annual means)",
                                 hoverinfo="skip"))
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10), hovermode="closest", showlegend=True,
                      legend=dict(orientation="h", x=0, y=1.12), xaxis_title="Date",
                      yaxis_title=f"Specific conductance ({unit} at 25 °C)")
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False}, key="sc_chart")
    st.markdown(salt.describe(t, unit))
    st.caption(f"Each point is the mean of the USGS daily mean conductance for one {by}, using {label} only. "
               f"Error bars are ± one {'standard deviation (SD)' if bars == 'sd' else 'standard error (SE = SD / √n)'} of those daily values. "
               f"A {by} needs data on at least half its days"
               + (f"; {dropped} {by}{'s were' if dropped != 1 else ' was'} left out." if dropped else ".")
               + " Daily values one day apart are not independent, so SE makes the means look more certain than they are; "
               "SD is the safer choice for judging whether years differ.")
    if not daily["approved"].iloc[-1]:
        st.caption("The newest values are provisional: USGS may revise them after checking.")
    st.download_button(f"Download the {by} means (CSV)", table.to_csv(index=False, date_format="%Y-%m-%d"),
                       f"USGS_{site_no}_conductance_{season}_{by}.csv", "text/csv", key="sc_csv")


# ---------------------------------------------------------------------------------------------------
# Base flow recession
# ---------------------------------------------------------------------------------------------------

def _days(v: float) -> str:
    return f"{v:.1f} days" if v < 10 else f"{v:.0f} days"


def _loglog_line(fit: dict, q: pd.Series) -> tuple[list, list]:
    lo, hi = float(q.min()), float(q.max())
    return [lo, hi], [fit["a"] * lo ** fit["b"], fit["a"] * hi ** fit["b"]]


def recession_panel(site_no: str, win: dict | None = None) -> None:
    st.divider()
    st.subheader("Base flow recession")
    st.caption("After rain, a stream falls back toward the flow its groundwater can supply. How fast it falls says how big "
               "and how leaky that groundwater store is. Each unbroken fall after a peak is a recession segment.")
    if not st.toggle("Show base flow recession for this gage", key="rec_on"):
        return
    fine = False
    if span_mode("rec_mode", win, "Choose the years") == "window":
        fine = win["fine"]
        window = recession.hourly(win["flow"], time_col="date") if fine else win["flow"]
        months = salt.SEASONS["all"][1]
        st.caption(f"{win['label']}. " + ("Hourly means of the 15-minute readings, so one storm's fall can be fitted; "
                                          "k is still in days." if fine else "Daily mean flow."))
    else:
        try:
            with st.spinner("Loading the whole daily record… (a few seconds the first time)"):
                daily = cached_daily(site_no, None)
        except usgs.USGSError as exc:
            st.error(str(exc))
            return
        first, last = int(daily["date"].dt.year.min()), int(daily["date"].dt.year.max())
        c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
        if first < last:
            years = c1.slider("Years", first, last, (max(first, last - 29), last), key="rec_years",
                              help="Thirty years by default; the whole record mixes old and new land use and climate.")
        else:
            years = (first, last)
        season = c2.selectbox("Months", list(salt.SEASONS), index=list(salt.SEASONS).index("all"), key="rec_season",
                              format_func=lambda k: salt.SEASONS[k][0],
                              help="Evapotranspiration speeds summer recessions, so k is often shorter in summer than in winter.")
        window = daily[daily["date"].dt.year.between(*years)]
        months = salt.SEASONS[season][1]
    with st.expander("Segment rules"):
        r1, r2, r3, r4 = st.columns(4)
        sfx = "_fine" if fine else ""   # 15-minute data gets its own rules: a single storm drains in hours to days
        skip = r1.number_input("Days skipped after the peak", 0.0, 10.0, 0.5 if fine else 2.0, step=0.25 if fine else 1.0, key="rec_skip" + sfx,
                               help="The first days after a peak are quickflow draining off, not base flow.")
        min_len = r2.number_input("Shortest segment (days)", 0.5 if fine else 3.0, 30.0, 1.0 if fine else 5.0, step=0.5 if fine else 1.0,
                                  key="rec_min_len" + sfx)
        min_r2 = r3.slider("Smallest r² of the exponential fit", 0.0, 0.99, 0.8, key="rec_min_r2")
        skip_flagged = r4.checkbox("Stop at ice or estimated days", value=True, key="rec_flags",
                                   help="USGS flags days whose flow was estimated or affected by ice; those are not measured recessions.")
    table, series = recession.segments(window, months, skip=float(skip), min_len=float(min_len), min_r2=float(min_r2),
                                       skip_flagged=skip_flagged, tol=0.005 if fine else 0.0)
    if table.empty:
        st.warning("No recession segments match these rules. Pick a longer time span, drag across a storm and its fall, "
                   "or shorten the segments in the rules.")
        return
    mrc = recession.master_curve(series)

    tab_k, tab_bn = st.tabs(["Segments, k and master curve", "-dQ/dt against Q"])
    with tab_k:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Segments", f"{len(table)}")
        m2.metric("Median k of segments", _days(table['k_days'].median()),
                  help="Q = Q0·exp(-t/k), fitted to each segment; k is the time for flow to fall to 37% of where it started.")
        if mrc:
            m3.metric("Master curve k", _days(mrc['k_days']), help="Pooled from every recession day of every segment.")
            m4.metric("Daily recession constant K", f"{mrc['daily_K']:.3f}", help="Tomorrow's flow over today's: K = exp(-1/k).")
        left, right = st.columns(2, gap="large")
        with left:
            fig = go.Figure(go.Scatter(x=window["date"], y=window["value"], mode="lines", line=dict(width=1, color="rgba(120,120,120,0.6)"),
                                       name="Hourly mean flow" if fine else "Daily mean flow",
                                       hovertemplate="%{x|%b %d, %Y %H:%M}<br>%{y:,.0f} cfs<extra></extra>"))
            xs, ys = [], []
            for part in series:
                xs += list(part.index) + [None]
                ys += list(part.to_numpy()) + [None]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=2.2, color=ORANGE), name="Recession segments",
                                     hovertemplate="%{x|%b %d, %Y}<br>%{y:,.0f} cfs<extra></extra>"))
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), title="Where the segments are", yaxis_type="log",
                              xaxis_title="Date", yaxis_title="Discharge (cfs, log scale)", legend=dict(orientation="h", x=0, y=-0.28))
            st.plotly_chart(fig, width="stretch", config={"displaylogo": False}, key="rec_segments_chart")
        with right:
            fig = go.Figure()
            if mrc:
                for i, part in enumerate(mrc["shifted"]):
                    fig.add_trace(go.Scatter(x=part["t"], y=part["q"], mode="lines", line=dict(width=1, color="rgba(42,120,214,0.45)"),
                                             name="Segments, slid in time", showlegend=i == 0, legendgroup="seg",
                                             hovertemplate=f"{part['start'].iloc[0]:%b %d, %Y}<br>%{{y:,.0f}} cfs<extra></extra>"))
                t_end = max(float(p["t"].iloc[-1]) for p in mrc["shifted"])
                tt = np.linspace(0, t_end, 100)
                fig.add_trace(go.Scatter(x=tt, y=mrc["q_top"] * np.exp(-tt / mrc["k_days"]), mode="lines",
                                         line=dict(width=2.5, dash="dash", color=ORANGE), name=f"Master curve, k = {_days(mrc['k_days'])}",
                                         hoverinfo="skip"))
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), title="Master recession curve (strip matching)",
                              yaxis_type="log", xaxis_title="Days along the master curve", yaxis_title="Discharge (cfs, log scale)",
                              legend=dict(orientation="h", x=0, y=-0.28))
            st.plotly_chart(fig, width="stretch", config={"displaylogo": False}, key="rec_mrc_chart")
        left, right = st.columns(2, gap="large")
        with left:
            fig = go.Figure(go.Scatter(x=table["start"], y=table["k_days"], mode="markers", marker=dict(size=7, color=BLUE),
                                       customdata=table[["days", "r2"]],
                                       hovertemplate="%{x|%b %d, %Y}<br>k %{y:.0f} days<br>%{customdata[0]} days, r² %{customdata[1]:.2f}<extra></extra>"))
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), title="k of each segment", xaxis_title="Segment start",
                              yaxis_title="k (days)")
            st.plotly_chart(fig, width="stretch", config={"displaylogo": False}, key="rec_k_chart")
        with right:
            st.markdown("**k by season**")
            st.dataframe(recession.by_season(table).round(1), hide_index=True, width="stretch")
            st.caption("k is the time for flow to fall to 37% (1/e) of where it started; half-life = k × ln 2. "
                       "Longer k means a larger, slower groundwater store.")
            st.download_button("Download the segments (CSV)", table.to_csv(index=False, date_format="%Y-%m-%d"),
                               f"USGS_{site_no}_recession_segments.csv", "text/csv", key="rec_csv")
    with tab_bn:
        pts = recession.dqdt_points(series)
        fit = recession.fit_power_law(pts)
        fig = go.Figure(go.Scatter(x=pts["q"], y=pts["dqdt"], mode="markers", marker=dict(size=4, color="rgba(42,120,214,0.4)"),
                                   name="Recession days", customdata=pts["date"],
                                   hovertemplate="%{customdata|%b %d, %Y}<br>Q %{x:,.1f} cfs<br>-dQ/dt %{y:,.2f} cfs/day<extra></extra>"))
        if fit:
            fx, fy = _loglog_line(fit["all"], pts["q"])
            fig.add_trace(go.Scatter(x=fx, y=fy, mode="lines", line=dict(width=2, color="#555555"),
                                     name=f"All days: b = {fit['all']['b']:.2f}", hoverinfo="skip"))
            if fit["envelope"]:
                env = fit["envelope_points"]
                fig.add_trace(go.Scatter(x=env["q"], y=env["dqdt"], mode="markers", marker=dict(size=9, color=ORANGE, symbol="diamond"),
                                         name="Lower envelope (5th percentile per bin)",
                                         hovertemplate="Q %{x:,.1f} cfs<br>-dQ/dt %{y:,.3f} cfs/day<extra></extra>"))
                fx, fy = _loglog_line(fit["envelope"], pts["q"])
                fig.add_trace(go.Scatter(x=fx, y=fy, mode="lines", line=dict(width=2.5, dash="dash", color=ORANGE),
                                         name=f"Envelope: b = {fit['envelope']['b']:.2f}", hoverinfo="skip"))
        fig.update_layout(height=470, margin=dict(l=10, r=10, t=10, b=10), xaxis_type="log", yaxis_type="log",
                          xaxis_title="Q, mean of two days (cfs)", yaxis_title="-dQ/dt, one day's drop (cfs per day)",
                          legend=dict(orientation="h", x=0, y=1.1))
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False}, key="rec_bn_chart")
        for line in recession.describe_power_law(fit):
            st.markdown("- " + line)
        st.caption("Brutsaert and Nieber (1977): on log-log axes, -dQ/dt = a·Q^b is a straight line. b = 1 is a linear reservoir "
                   "with k = 1/a. The lower envelope keeps the slowest recessions, least disturbed by rain or evapotranspiration. "
                   "USGS daily flows are rounded, so very low flows step in jumps and their -dQ/dt is coarse.")
        if len(pts):
            st.download_button("Download the -dQ/dt points (CSV)", pts.to_csv(index=False, date_format="%Y-%m-%d"),
                               f"USGS_{site_no}_dQdt.csv", "text/csv", key="rec_bn_csv")


# ---------------------------------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------------------------------

init_state()
sidebar()

st.title("🌊 River Explorer")
st.markdown("Name a river, a town, or a storm you remember, and I'll find the real stream gage that measured it.")

col_in, col_btn = st.columns([5, 1], vertical_alignment="bottom")
col_in.text_input("What do you want to explore?", key="query_text", max_chars=finder.MAX_TEXT, on_change=queue_search,
                  placeholder="Try: Cedar River in Iowa, or the creek that flooded Montpelier in 2023")
col_btn.button("Find gages", on_click=queue_search, type="primary", width="stretch")
cols = st.columns(len(EXAMPLES))
for col, example in zip(cols, EXAMPLES):
    col.button(example, key=f"ex_{example}", on_click=use_example, args=(example,), width="stretch")

with st.expander("Or browse the gallery: river personalities", expanded=st.session_state.selected is None):
    st.caption("Each card is a real gage and a real stretch of time. Look at the shape first, then work out why it looks that way.")
    st.button("Surprise me", on_click=surprise_scene, type="primary")
    for row in (gallery.SCENES[:4], gallery.SCENES[4:]):
        for col, sc in zip(st.columns(4), row):
            with col.container(border=True):
                st.markdown(f"**{sc.title}**")
                st.caption(sc.teaser)
                st.button("Explore", key=f"scene_{sc.id}", on_click=pick_scene, args=(sc.id,), width="stretch")

if st.session_state.queued:
    st.session_state.queued = False
    if st.session_state.query_text.strip():
        run_search()

show_results()

if st.session_state.selected:
    chart_window = hydrograph(st.session_state.selected)
    conductance_panel(st.session_state.selected, chart_window)
    recession_panel(st.session_state.selected, chart_window)
