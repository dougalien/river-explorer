"""Checks every gallery gage against live USGS: it exists, has data in its window, and looks like its personality.

    python scripts/verify_gallery.py            # about 20 requests
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

import notes  # noqa: E402
import usgs  # noqa: E402

CANDIDATES = [
    # id, site, start, end, resolution
    ("flashy_urban", "01104460", "2023-12-01", "2024-01-31", "daily"),
    ("snowmelt", "09095500", "2023-03-01", "2023-09-30", "daily"),
    ("hydropeaking", "09380000", "2026-10-01", "2026-10-15", "15min"),
    ("great_flood", "05464500", "2008-05-20", "2008-07-10", "daily"),
    ("monsoon_wash", "09482500", "2021-06-15", "2021-09-30", "daily"),
    ("monsoon_wash_b", "09485700", "2021-06-15", "2021-09-30", "daily"),
    ("spring_fed", "08169000", "2022-01-01", "2023-12-31", "daily"),
    ("vermont_2023", "04286000", "2023-06-25", "2023-07-31", "daily"),
    ("big_slow_river", "07010000", "2019-03-01", "2019-08-31", "daily"),
]


def main() -> None:
    for pid, site, start, end, res in CANDIDATES:
        try:
            info = usgs.fetch_site_info(site)
            if res == "15min":
                df = usgs.fetch_continuous(site, start, end)
                s = pd.Series(df["value"].to_numpy(), index=df["time"]).sort_index()
                daily = s.resample("D").mean()
                first = s.iloc[:96 * 3]
                print(f"{pid:15} {site} {info['label'][:48]:48} n={len(s)} min={s.min():,.0f} max={s.max():,.0f} "
                      f"swing/day={(first.groupby(first.index.date).max() - first.groupby(first.index.date).min()).mean():,.0f}")
            else:
                df = usgs.fetch_daily(site, start, end)
                s = df.set_index("date")["value"]
                ev = notes.find_events(s.asfreq("D"), window=min(30, max(len(s) // 3, 3)))
                top = max(ev, key=lambda e: e.peak_q) if ev else None
                print(f"{pid:15} {site} {info['label'][:48]:48} n={len(s)} min={s.min():,.0f} max={s.max():,.0f} "
                      f"flash={notes.flashiness(s.asfreq('D')) or 0:.2f} peak={(top.peak_date.date(), round(top.peak_q)) if top else None}")
        except usgs.USGSError as exc:
            print(f"{pid:15} {site} FAILED: {exc}")


if __name__ == "__main__":
    main()
