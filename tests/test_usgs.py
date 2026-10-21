from datetime import date

import pandas as pd
import pytest

import usgs


# ---------------------------------------------------------------- pure helpers

def test_name_terms_drops_generic_words_and_unsafe_characters():
    assert usgs.name_terms("Charles River") == ["CHARLES"]
    assert usgs.name_terms("river") == []
    assert usgs.name_terms("Blue Earth River") == ["BLUE", "EARTH"]
    for term in usgs.name_terms("O'Brien'; DROP TABLE x -- \" creek"):
        assert not set("'\"%;") & set(term)


def test_bbox_contains_point_and_clamps_radius():
    w, s, e, n = usgs.bbox_around(42.36, -71.06, 30)
    assert w < -71.06 < e and s < 42.36 < n
    assert (n - s) == pytest.approx(2 * 30 / 111, rel=1e-6)
    w2, s2, e2, n2 = usgs.bbox_around(42.36, -71.06, 100000)
    assert (n2 - s2) == pytest.approx(2 * 150 / 111, rel=1e-6)
    with pytest.raises(ValueError):
        usgs.bbox_around(95, 0)


def test_state_lookup():
    assert usgs.state_full_name("ma") == "Massachusetts"
    assert usgs.state_full_name("New Hampshire") == "New Hampshire"
    assert usgs.state_full_name("Atlantis") is None
    assert usgs.state_abbr("Iowa") == "IA"


def test_pretty_name_expands_usgs_shorthand():
    assert usgs.pretty_name("MYSTIC R INLET TO LOWER MYSTIC LAKE NR MEDFORD, MA") == \
        "Mystic River Inlet to Lower Mystic Lake near Medford, MA"
    assert usgs.pretty_name("CONCORD R BELOW R MEADOW BROOK, AT LOWELL, MA") == \
        "Concord River below River Meadow Brook, at Lowell, MA"
    assert usgs.pretty_name("Cedar River at Cedar Rapids, IA") == "Cedar River at Cedar Rapids, IA"
    assert usgs.pretty_name("") == ""


def test_site_id():
    assert usgs.site_id("05464500") == "USGS-05464500"
    assert usgs.site_id("USGS-05464500") == "USGS-05464500"
    for bad in ("", "abc", "123", "05464500; drop"):
        with pytest.raises(ValueError):
            usgs.site_id(bad)


def test_site_timezone():
    assert usgs.site_timezone({"time_zone": "CST"}) == "America/Chicago"
    assert usgs.site_timezone({"time_zone": "MST", "uses_dst": "N"}) == "America/Phoenix"
    assert usgs.site_timezone({}) == "UTC"


# ---------------------------------------------------------------- parsing canned API output

def feat(**props):
    return {"type": "Feature", "properties": props, "geometry": {"type": "Point", "coordinates": [-71.2, 42.3]}}


def test_daily_frame_parses_numbers_flags_and_duplicates():
    df = usgs.daily_frame([
        feat(time="2020-01-02", value="12.5", approval_status="Approved", qualifier=None),
        feat(time="2020-01-01", value="10", approval_status="Provisional", qualifier=["ESTIMATED", "ICE"]),
        feat(time="2020-01-03", value="", approval_status="Approved"),          # no number: dropped
        feat(time="2020-01-04", value="NaN", approval_status="Approved"),       # not finite: dropped
        feat(time="2020-01-02", value="13.0", approval_status="Approved"),      # duplicate day: last wins
    ])
    assert list(df["date"].dt.strftime("%Y-%m-%d")) == ["2020-01-01", "2020-01-02"]
    assert list(df["value"]) == [10.0, 13.0]
    assert list(df["approved"]) == [False, True]
    assert list(df["flags"]) == ["ESTIMATED,ICE", ""]


def test_daily_stats_frame_spreads_mean_max_min_into_columns():
    df = usgs.daily_stats_frame([
        feat(time="2020-01-01", value="10", statistic_id="00003", approval_status="Approved", qualifier=None),
        feat(time="2020-01-01", value="14", statistic_id="00001", approval_status="Approved", qualifier=None),
        feat(time="2020-01-01", value="8", statistic_id="00002", approval_status="Approved", qualifier=None),
        feat(time="2020-01-02", value="11", statistic_id="00003", approval_status="Provisional", qualifier=["ICE"]),
        feat(time="2020-01-02", value="99", statistic_id="00008", approval_status="Provisional"),   # median: not used
        feat(time="2020-01-03", value="7", statistic_id="00002", approval_status="Approved", qualifier=["ESTIMATED"]),
    ])
    assert list(df.columns) == ["date", "mean", "max", "min", "approved", "flags"]
    assert list(df["date"].dt.strftime("%Y-%m-%d")) == ["2020-01-01", "2020-01-02", "2020-01-03"]
    assert df.loc[0, ["mean", "max", "min"]].tolist() == [10.0, 14.0, 8.0]
    assert pd.isna(df.loc[1, "max"]) and df.loc[1, "flags"] == "ICE" and not df.loc[1, "approved"]
    assert pd.isna(df.loc[2, "mean"]) and df.loc[2, "min"] == 7.0      # a day with only a minimum keeps it
    assert usgs.daily_stats_frame([]).empty


def test_fetch_daily_stats_asks_for_every_statistic_in_one_query(monkeypatch):
    seen = {}

    def fake_features(collection, params, max_rows=None, page=None):
        seen.update(params, collection=collection)
        return iter([feat(time="2020-07-01", value="512", statistic_id="00003", approval_status="Approved")])

    monkeypatch.setattr(usgs, "_features", fake_features)
    df = usgs.fetch_daily_stats("01104500", "2020-07-01", "2020-07-31", parameter=usgs.CONDUCTANCE)
    assert seen["collection"] == "daily" and "statistic_id" not in seen and "statistic_id" in seen["properties"]
    assert seen["parameter_code"] == "00095" and seen["time"] == "2020-07-01/2020-07-31"
    assert df.attrs["unit"] == "µS/cm" and df["mean"].tolist() == [512.0]
    monkeypatch.setattr(usgs, "_features", lambda *a, **k: iter([]))
    with pytest.raises(usgs.USGSError):
        usgs.fetch_daily_stats("01104500")


def test_continuous_frame_is_utc_and_labels_unit():
    df = usgs.continuous_frame([
        feat(time="2026-10-15T12:30:00+00:00", value="181", approval_status="Provisional", unit_of_measure="ft^3/s"),
        feat(time="2026-10-15T12:15:00+00:00", value="180", approval_status="Provisional", unit_of_measure="ft^3/s"),
    ])
    assert str(df["time"].dt.tz) == "UTC"
    assert list(df["value"]) == [180.0, 181.0]
    assert df.attrs["unit"] == "cfs"


def test_gage_rows_merge_duplicates_and_flag_active():
    today = date(2026, 10, 17)
    rows = usgs.gage_rows([
        feat(monitoring_location_id="USGS-01104500", monitoring_location_name="CHARLES RIVER AT WALTHAM, MA",
             state_name="Massachusetts", drainage_area=251.0, begin="1931-08-04T04:00:00+00:00",
             end="2026-10-15T04:00:00+00:00"),
        feat(monitoring_location_id="USGS-01104500", monitoring_location_name="CHARLES RIVER AT WALTHAM, MA",
             state_name="Massachusetts", drainage_area=251.0, begin="1990-01-01T00:00:00+00:00",
             end="2026-10-10T00:00:00+00:00"),
        feat(monitoring_location_id="USGS-01104615", monitoring_location_name="OLD GAGE", state_name="Massachusetts",
             begin="1999-08-19T04:00:00+00:00", end="2001-09-29T04:00:00+00:00"),
        feat(monitoring_location_id="IN033-395818085061301", monitoring_location_name="NOT USGS",
             begin="2000-01-01T00:00:00+00:00", end="2026-10-01T00:00:00+00:00"),
    ], today=today)
    by = {r["site_no"]: r for r in rows}
    assert set(by) == {"01104500", "01104615"}                       # other agencies dropped, duplicates merged
    assert by["01104500"]["begin"] == date(1931, 8, 4) and by["01104500"]["end"] == date(2026, 10, 15)
    assert by["01104500"]["active"] is True and by["01104615"]["active"] is False
    assert by["01104500"]["label"] == "Charles River at Waltham, MA"
    assert by["01104500"]["lat"] == 42.3


def test_site_from_feature():
    info = usgs.site_from_feature(feat(id="USGS-05464500", monitoring_location_name="Cedar River at Cedar Rapids, IA",
                                       state_name="Iowa", drainage_area=6510.0, time_zone_abbreviation="CST"))
    assert info["site_no"] == "05464500" and info["state"] == "Iowa" and info["drainage_area_mi2"] == 6510.0
    assert usgs.site_timezone(info) == "America/Chicago"


# ---------------------------------------------------------------- paging and query building (no network)

def test_features_follow_next_links_and_stop_at_max_rows(monkeypatch):
    pages = {
        "first": {"features": [{"n": 1}, {"n": 2}], "links": [{"rel": "next", "href": "second"}]},
        "second": {"features": [{"n": 3}, {"n": 4}], "links": [{"rel": "next", "href": "third"}]},
        "third": {"features": [{"n": 5}], "links": []},
    }
    calls = []

    def fake_get(url, params=None):
        calls.append((url, params))
        return pages["first" if params else url]

    monkeypatch.setattr(usgs, "_get_json", fake_get)
    assert [f["n"] for f in usgs._features("daily", {"a": "b"})] == [1, 2, 3, 4, 5]
    assert calls[0][1]["limit"] == usgs.PAGE and calls[0][1]["a"] == "b" and calls[1][1] is None
    assert [f["n"] for f in usgs._features("daily", {}, max_rows=3)] == [1, 2, 3]


def test_find_gages_builds_a_safe_filter(monkeypatch):
    seen = {}
    monkeypatch.setattr(usgs, "_features", lambda collection, params, max_rows=None, page=None: seen.update(
        collection=collection, params=params) or [])
    usgs.find_gages(name="Charles River", state="ma")
    assert seen["collection"] == "combined-metadata"
    assert seen["params"]["state_name"] == "Massachusetts"
    assert seen["params"]["filter"] == "casei(monitoring_location_name) LIKE casei('%charles%')"
    assert seen["params"]["parameter_code"] == "00060" and seen["params"]["computation_identifier"] == "Mean"
    usgs.find_gages(name="x'; DROP", state=None, bbox=(-72, 42, -71, 43))
    assert seen["params"]["filter"].count("'") == 2 * len(usgs.name_terms("x'; DROP"))  # only our own quotes
    assert seen["params"]["bbox"] == "-72.00000,42.00000,-71.00000,43.00000"


def test_find_gages_refuses_a_nationwide_sweep_and_unknown_states():
    with pytest.raises(ValueError):
        usgs.find_gages()
    with pytest.raises(ValueError):
        usgs.find_gages(name="charles", state="Atlantis")


def test_rate_limit_fails_fast_with_a_readable_wait(monkeypatch):
    class Resp:
        status_code = 429
        headers = {"Retry-After": "499"}

    class Sess:
        def get(self, url, params=None, timeout=None):
            return Resp()

    monkeypatch.setattr(usgs, "_session", lambda: Sess())
    with pytest.raises(usgs.USGSRateLimit) as e:
        usgs.fetch_site_info("05464500")
    assert e.value.retry_after == 499 and "9 minutes" in str(e.value) and isinstance(e.value, usgs.USGSError)
    assert usgs.USGSRateLimit(None).retry_after is None


def test_session_never_sleeps_on_retry_after():
    adapter = usgs._session().get_adapter("https://api.waterdata.usgs.gov")
    assert 429 not in adapter.max_retries.status_forcelist and not adapter.max_retries.respect_retry_after_header


# ---------------------------------------------------------------- live USGS

@pytest.mark.live
def test_live_cedar_rapids_full_history():
    df = usgs.fetch_daily("05464500")
    assert len(df) >= 45000
    assert df["date"].iloc[0] == pd.Timestamp("1902-10-01")
    assert (pd.Timestamp.now() - df["date"].iloc[-1]).days < 10
    assert df["approved"].any() and df["value"].min() > 0
    assert df.attrs["unit"] == "cfs"


@pytest.mark.live
def test_live_paging_matches_a_single_request():
    window = {"monitoring_location_id": "USGS-05464500", "parameter_code": "00060", "statistic_id": "00003",
              "time": "2000-01-01/2010-12-31", "properties": "time,value,approval_status,qualifier"}
    paged = usgs.daily_frame(usgs._features("daily", window, page=2000))
    whole = usgs.fetch_daily("05464500", "2000-01-01", "2010-12-31")
    assert len(paged) == len(whole) == 4018
    assert paged["value"].tolist() == whole["value"].tolist()


@pytest.mark.live
def test_live_continuous_last_days():
    df = usgs.fetch_continuous("05464500", days=3)
    assert len(df) > 200 and str(df["time"].dt.tz) == "UTC" and df.attrs["unit"] == "cfs"
    stage = usgs.fetch_continuous("05464500", days=1, parameter=usgs.GAGE_HEIGHT)
    assert stage.attrs["unit"] == "ft"


@pytest.mark.live
def test_live_site_info_and_unknown_site():
    info = usgs.fetch_site_info("05464500")
    assert "Cedar" in info["name"] and info["state"] == "Iowa"
    assert info["drainage_area_mi2"] == pytest.approx(6510, rel=0.01)
    assert info["lat"] == pytest.approx(41.97, abs=0.05)
    with pytest.raises(usgs.USGSError):
        usgs.fetch_site_info("12345678")  # USGS has no such site (99999999 does exist: a Kentucky calibration site)


@pytest.mark.live
def test_live_find_gages_by_name_state_and_box():
    rows = usgs.find_gages(name="charles", state="MA")
    sites = {r["site_no"]: r for r in rows}
    assert "01104500" in sites and sites["01104500"]["active"] and sites["01104500"]["record_years"] > 90
    boston = usgs.find_gages(bbox=usgs.bbox_around(42.36, -71.06, 30))
    assert len(boston) >= 10 and all(r["lat"] and r["lon"] for r in boston)
    assert usgs.find_gages(name="zzzzqqqq", state="MA") == []
    # Newer USGS names are Mixed Case ("Cedar River at Cedar Rapids, IA"); older ones are ALL CAPS. Both must match.
    cedar = {r["site_no"]: r for r in usgs.find_gages(name="Cedar River", state="Iowa")}
    assert "05464500" in cedar and cedar["05464500"]["record_years"] > 120
    by_id = usgs.find_gages(site_no="05464500")
    assert [r["site_no"] for r in by_id] == ["05464500"]


@pytest.mark.live
def test_live_daily_stats_mean_matches_the_daily_mean():
    stats = usgs.fetch_daily_stats("05464500", "2020-01-01", "2020-12-31")
    mean = usgs.fetch_daily("05464500", "2020-01-01", "2020-12-31")
    assert stats["mean"].dropna().tolist() == mean["value"].tolist()
    both = stats.dropna(subset=["max", "min"])
    assert ((both["max"] >= both["mean"]) & (both["mean"] >= both["min"])).all()


@pytest.mark.live
def test_live_conductance_gages_near_boston():
    rows = usgs.find_gages(bbox=usgs.bbox_around(42.36, -71.06, 50), parameter=usgs.CONDUCTANCE, active_days=3650)
    assert rows, "expected at least one gage with daily specific conductance near Boston"
    best = max(rows, key=lambda r: r["record_years"])
    sc = usgs.fetch_daily(best["site_no"], "2020-06-01", "2020-11-30", parameter=usgs.CONDUCTANCE)
    assert sc.attrs["unit"] == "µS/cm" and 20 < sc["value"].median() < 20000
