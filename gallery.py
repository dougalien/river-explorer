"""A hand-picked gallery of river "personalities": each one is a real gage and a stretch of time that shows a different pattern.

The title and teaser describe what you will SEE and never name the cause, so the guide still has something to ask.
`cause` is the answer (shown in the instructor's view and by the guide only at its explanation step).
Every entry is checked against live USGS by `scripts/verify_gallery.py` and by the live test in tests/test_gallery.py.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Scene:
    id: str
    title: str
    teaser: str
    site_no: str
    start: str            # YYYY-MM-DD
    end: str
    resolution: str       # "daily" or "15min"
    cause: str

    @property
    def window_text(self) -> str:
        return f"{self.start} to {self.end}"


SCENES: tuple[Scene, ...] = (
    Scene("flashy_urban", "A small city stream", "A creek that drains rooftops and parking lots. Two winter months.",
          "01104460", "2023-12-01", "2024-01-31", "daily",
          "A small, mostly paved watershed: rain runs off pavement and storm drains straight into the creek, so it rises and falls within a day or two."),
    Scene("snowmelt", "A river fed by mountains", "March through September in the Rockies.",
          "09095500", "2023-03-01", "2023-09-30", "daily",
          "Mountain snowpack melts in late spring, so the river swells slowly over weeks, peaks around May, and recedes through summer."),
    Scene("hydropeaking", "A river with a daily rhythm", "Two weeks of 15-minute readings in a desert canyon.",
          "09380000", "2026-10-01", "2026-10-14", "15min",
          "A big dam upstream releases more water in the daytime, when electricity demand is high, and less at night and on weekends."),
    Scene("great_flood", "One extraordinary spring", "Two months in an Iowa spring and early summer.",
          "05464500", "2008-05-20", "2008-07-10", "daily",
          "Heavy June 2008 rain on already-wet ground sent the Cedar River into a major flood that put much of Cedar Rapids under water."),
    Scene("monsoon_wash", "A dry riverbed", "Summer in the Arizona desert.",
          "09485700", "2021-06-15", "2021-09-30", "daily",
          "Desert streams are dry most of the year; summer monsoon thunderstorms send short, violent flash floods down the wash, and then it dries up again."),
    Scene("spring_fed", "A river that hardly changes", "Two years in central Texas.",
          "08169000", "2022-01-01", "2023-12-31", "daily",
          "The river is fed by springs from a huge underground aquifer, which changes slowly, so the flow stays steady and only sags in a long drought."),
    Scene("vermont_2023", "Five weeks in a mountain valley", "Late June to the end of July in Vermont.",
          "04286000", "2023-06-25", "2023-07-31", "daily",
          "Days of heavy rain in July 2023 on already-saturated mountain soils sent the Winooski River far above its normal flow and flooded Montpelier."),
    Scene("big_slow_river", "A giant in no hurry", "Six months on a river that drains a good part of a continent.",
          "07010000", "2019-03-01", "2019-08-31", "daily",
          "The Mississippi drains a huge area, so rain and snowmelt from many states add up slowly: floods build over weeks and last for months."),
)
BY_ID = {s.id: s for s in SCENES}


def scene_for(site_no: str) -> Scene | None:
    return next((s for s in SCENES if s.site_no == site_no), None)


def surprise(exclude: str | None = None, rng: random.Random | None = None) -> Scene:
    pool = [s for s in SCENES if s.id != exclude]
    return (rng or random).choice(pool)
