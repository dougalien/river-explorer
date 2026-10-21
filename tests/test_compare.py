import numpy as np
import pandas as pd
import pytest

import compare


def storm_series(base, peak, half_life, start="2024-01-01", days=60, at=20):
    dates = pd.date_range(start, periods=days)
    q = np.full(days, float(base))
    t = np.arange(days - at)
    q[at:] += (peak - base) * np.exp(-np.log(2) / half_life * t)
    return pd.Series(q, index=dates)


def test_per_square_mile_makes_a_creek_and_a_river_comparable():
    creek = storm_series(5, 200, 2)        # 20 mi² creek
    river = storm_series(2000, 6000, 12)   # 2,000 mi² river
    a = compare.normalize(creek, "per_sq_mi", 20)
    b = compare.normalize(river, "per_sq_mi", 2000)
    assert a.max() == pytest.approx(10.0) and b.max() == pytest.approx(3.0)      # the small watershed yields more per mile
    assert creek.max() / river.max() < 0.05                                       # raw flow hides it


def test_shape_mode_scales_to_each_gages_own_peak():
    s = compare.normalize(storm_series(5, 200, 2), "shape", None)
    assert s.max() == pytest.approx(1.0) and s.min() > 0
    assert compare.normalize(storm_series(5, 200, 2), "actual", None).max() == 200


def test_modes_refuse_what_they_cannot_do():
    s = storm_series(5, 200, 2)
    for area in (None, 0, -3):
        with pytest.raises(ValueError, match="drainage"):
            compare.normalize(s, "per_sq_mi", area)
    with pytest.raises(ValueError, match="no flow"):
        compare.normalize(pd.Series([0.0, 0.0]), "shape", None)
    with pytest.raises(ValueError):
        compare.normalize(s, "bogus", 1)


def test_daily_window_turns_15_minute_values_into_daily_means():
    x = pd.date_range("2026-10-01", periods=96 * 3, freq="15min")
    out = compare.daily_window(np.r_[np.full(96, 10.0), np.full(96, 20.0), np.full(96, 30.0)], x)
    assert list(out) == [10.0, 20.0, 30.0]


def test_comparison_says_which_is_flashier_and_which_yields_more():
    creek = storm_series(5, 200, 2)
    river = storm_series(2000, 6000, 12)
    facts = compare.comparison_facts(creek, 20, river, 2000)
    assert facts["a"]["flashiness"] > facts["b"]["flashiness"]
    assert facts["a"]["half_fall_days"] < facts["b"]["half_fall_days"]
    lines = compare.describe("Creek", "River", facts)
    text = " ".join(lines)
    assert "100x difference in size" in text and "Creek is 3.3x higher than River" in text and "Creek responds faster" in text


def test_missing_pieces_just_drop_those_lines():
    flat = pd.Series(np.full(60, 5.0), index=pd.date_range("2024-01-01", periods=60))
    facts = compare.comparison_facts(flat, None, flat, None)
    assert compare.describe("A", "B", facts) == [] or all("mi²" not in l for l in compare.describe("A", "B", facts))
