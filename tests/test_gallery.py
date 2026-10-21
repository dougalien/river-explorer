import random
import re

import pandas as pd
import pytest

import gallery
import usgs

CAUSE_WORDS = ("dam", "snow", "monsoon", "flood", "storm", "hurricane", "rain", "aquifer", "spring-fed", "tide", "melt", "release")


def names_a_cause(text: str) -> str | None:
    """The first cause word that appears as a whole word (or plural) in the text."""
    for word in CAUSE_WORDS:
        if re.search(r"\b" + re.escape(word) + r"s?\b", text.lower()):
            return word
    return None


def test_every_scene_is_well_formed():
    ids = [s.id for s in gallery.SCENES]
    assert len(ids) == len(set(ids)) >= 8
    for s in gallery.SCENES:
        assert usgs.site_id(s.site_no)
        assert pd.Timestamp(s.start) < pd.Timestamp(s.end)
        assert s.resolution in ("daily", "15min")
        if s.resolution == "15min":
            assert (pd.Timestamp(s.end) - pd.Timestamp(s.start)).days <= 30      # 15-minute windows stay small
        assert len(s.cause) > 40 and s.title and s.teaser


def test_the_cause_word_check_can_actually_fail():
    assert names_a_cause("A river below a dam. Two weeks.") == "dam"
    assert names_a_cause("Flooded streets after the storms") == "storm"
    assert names_a_cause("A creek that drains rooftops and parking lots.") is None      # 'drains' must not trip 'rain'


def test_titles_and_teasers_never_name_the_cause():
    for s in gallery.SCENES:
        assert names_a_cause(f"{s.title} {s.teaser}") is None, f"{s.id} gives the answer away"


def test_lookup_and_surprise():
    assert gallery.scene_for("05464500").id == "great_flood" and gallery.scene_for("00000000") is None
    rng = random.Random(1)
    picks = {gallery.surprise(exclude="snowmelt", rng=rng).id for _ in range(60)}
    assert "snowmelt" not in picks and len(picks) >= 5


@pytest.mark.live
def test_live_every_gallery_gage_has_data_in_its_window():
    for s in gallery.SCENES:
        info = usgs.fetch_site_info(s.site_no)
        assert info["site_no"] == s.site_no
        if s.resolution == "15min":
            df = usgs.fetch_continuous(s.site_no, s.start, s.end)
            assert len(df) > 96 * 7
        else:
            df = usgs.fetch_daily(s.site_no, s.start, s.end)
            assert len(df) >= 0.9 * ((pd.Timestamp(s.end) - pd.Timestamp(s.start)).days + 1)
