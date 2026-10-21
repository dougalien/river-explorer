import json

import pandas as pd

import notes
import report
import tutor
from tutor import CONCEPTS, Tutor, TutorContext

META = {"student": "Ada Lovelace", "site_label": "Cedar River at Cedar Rapids, IA", "site_no": "05464500",
        "view": "Last 30 days", "scene_title": None, "compare_on": True}

GOOD = {
    "axes": "The chart shows river flow in cfs over time, about a month of days.",
    "biggest_event": "The biggest rise is a peak in early October, a big spike.",
    "rise_vs_fall": "It rises fast, then falls back down more slowly, a slow recession.",
    "why_slow": "Rain runs off the surface fast, but water that soaked into the ground drains slowly as groundwater.",
    "normal": "The line is well above the normal band, so it is high for this date.",
    "drivers": "Probably heavy rain or a storm, because the rise was sudden.",
    "predict": "It would keep falling slowly because the earlier peaks receded gradually.",
}


def ctx():
    ev = notes.Event(pd.Timestamp("2026-10-05"), 26_600.0, 2_630.0, 10.1, 7, 3)
    return TutorContext("Cedar River", "Last 30 days", pd.Timestamp("2026-09-17"), pd.Timestamp("2026-10-16"), [ev],
                        percentile=94.0, compare_on=True, latest_flow=7_430.0, half_life=10.4)


def run(answers):
    t, c = Tutor(client=None), ctx()
    t.start(c)
    for a in answers:
        t.turn(c, a) if a != "STUCK" else t.turn(c, "", hint=True)
    return t


def test_a_perfect_session_scores_full_marks():
    t = run([GOOD[c.id] for c in CONCEPTS])
    s = report.summary(t)
    assert s["points"] == 14 and s["out_of"] == 14 and s["on_their_own"] == 7 and s["finished"] and s["stuck_presses"] == 0
    assert s["grading"] == "keyword matching" and s["wording"] == "built-in text"
    assert [r["status"] for r in report.step_rows(t)] == ["own"] * 7


def test_hints_and_explanations_lower_the_marks_in_a_readable_way():
    # axes on their own; biggest_event after a nudge; rise_vs_fall explained after three misses; why_slow reached but unanswered
    t = run([GOOD["axes"], "hmm", GOOD["biggest_event"], "idk", "idk", "STUCK", "anything", "STUCK"])
    rows = {r["id"]: r for r in report.step_rows(t)}
    assert rows["axes"]["status"] == "own" and rows["axes"]["points"] == 2
    assert rows["biggest_event"]["status"] == "own_after_hints" and rows["biggest_event"]["points"] == 1
    assert rows["biggest_event"]["highest_hint_level"] == 1
    assert rows["rise_vs_fall"]["status"] == "explained" and rows["rise_vs_fall"]["points"] == 0
    assert rows["rise_vs_fall"]["stuck_presses"] >= 1
    assert rows["predict"]["status"] == "not_reached" and rows["predict"]["points"] is None
    s = report.summary(t)
    assert s["points"] == 3 and not s["finished"] and s["stuck_presses"] == 2


def test_markdown_has_the_evidence_and_survives_awkward_student_text():
    nasty = "| pipe | and **bold** <script>alert(1)</script>\nsecond line"
    t = run([GOOD["axes"], nasty])
    md = report.build_markdown(META, t)
    assert "Ada Lovelace" in md and "Cedar River at Cedar Rapids, IA (USGS 05464500)" in md and "Suggested score" in md
    assert "> The chart shows river flow in cfs" in md                          # their words, verbatim, quoted
    assert "> | pipe | and **bold** <script>alert(1)</script>" in md and "> second line" in md   # quoted, so inert in a viewer
    table = [l for l in md.splitlines() if l.startswith("| ")]
    assert all(l.count("|") == table[0].count("|") for l in table)               # no stray pipes broke the table
    assert "stopped after" in md and "Full transcript" in md


def test_json_roundtrips_and_has_what_a_gradebook_needs():
    t = run([GOOD["axes"], "idk", "idk", "idk"])
    data = json.loads(report.to_json_text({**META, "scene_title": "A dry riverbed"}, t))
    assert data["student"] == "Ada Lovelace" and data["scene_title"] == "A dry riverbed"
    assert data["summary"]["points"] == 2 and len(data["steps"]) == 7 and len(data["turns"]) == 4
    assert data["turns"][1]["grade"]["asks_for_answer"] is True and data["transcript"][0]["role"] == "assistant"


def test_a_session_with_no_answers_still_reports():
    t = Tutor(client=None)
    t.start(ctx())
    md = report.build_markdown({**META, "student": ""}, t)
    assert "(name not given)" in md and "0 of 14" in md and "Not reached" in md


def test_file_names_are_safe():
    stem = report.file_stem({**META, "student": "../Ada Lovelace?!"})
    assert "/" not in stem and "?" not in stem and stem.startswith("river-guide-ada-lovelace")
    assert report.file_stem({**META, "student": ""}).startswith("river-guide-student-")
