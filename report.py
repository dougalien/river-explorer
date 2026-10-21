"""The session report: what a student did with the guide, in a form an instructor can read, grade and file.

The rubric is deliberately plain so it can be explained to a student: each of the seven steps is worth up to 2 points.
  2  worked out on their own, no hints
  1  worked out after a nudge or a pointed question
  0  needed the guide's explanation (they still took part)
  -  not reached
The total is a suggestion for the instructor, who has the transcript to overrule it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from tutor import CONCEPTS, Tutor

POINTS = {"own": 2, "own_after_hints": 1, "explained": 0}
LABEL = {"own": "On their own", "own_after_hints": "After hints", "explained": "Explained by the guide", "not_reached": "Not reached"}
STEP_NAMES = {
    "axes": "Read the axes and units",
    "biggest_event": "Find the biggest event",
    "rise_vs_fall": "Compare the rise with the fall",
    "why_slow": "Explain the slow fall",
    "normal": "Compare with normal for the date",
    "drivers": "Suggest causes",
    "predict": "Predict the next week",
}


def step_rows(t: Tutor) -> list[dict]:
    rows = []
    for c in CONCEPTS:
        if c.id in t.done:
            status = "own" if t.peak_level.get(c.id, 0) == 0 else "own_after_hints"
        elif c.id in t.explained_ids:
            status = "explained"
        else:
            status = "not_reached"
        answers = [e["student"] for e in t.log if e["concept"] == c.id and e["student"]]
        hints = sum(1 for e in t.log if e["concept"] == c.id and e["stuck_button"])
        rows.append({"id": c.id, "step": STEP_NAMES[c.id], "status": status, "label": LABEL[status],
                     "points": POINTS.get(status), "highest_hint_level": t.peak_level.get(c.id, 0),
                     "stuck_presses": hints, "answers": answers})
    return rows


def summary(t: Tutor) -> dict:
    rows = step_rows(t)
    reached = [r for r in rows if r["status"] != "not_reached"]
    return {
        "points": sum(r["points"] for r in reached), "out_of": 2 * len(CONCEPTS),
        "steps_reached": len(reached), "steps_total": len(CONCEPTS),
        "on_their_own": sum(r["status"] == "own" for r in rows),
        "stuck_presses": sum(1 for e in t.log if e["stuck_button"]),
        "finished": t.finished,
        "wording": "AI-phrased" if any(e["ai_wording"] for e in t.log) else "built-in text",
        "grading": "AI" if any((e["grade"] or {}).get("by") == "ai" for e in t.log) else "keyword matching",
    }


def build_json(meta: dict, t: Tutor) -> dict:
    """meta: student, site_label, site_no, view, scene_title (or None), compare_on."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {"report": "River Explorer guided session", "generated_at": now, "started_at": t.started_at or None, **meta,
            "summary": summary(t), "steps": step_rows(t), "transcript": t.history, "turns": t.log,
            "rubric": "2 on their own, 1 after hints, 0 explained by the guide; a suggestion for the instructor"}


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _quote(text: str) -> str:
    return "\n".join("> " + line for line in str(text).splitlines() or [""])


def build_markdown(meta: dict, t: Tutor) -> str:
    data = build_json(meta, t)
    s = data["summary"]
    name = meta.get("student") or "(name not given)"
    status = "finished all steps" if s["finished"] else "stopped after %d of %d steps" % (s["steps_reached"], s["steps_total"])
    lines = [
        "# River Explorer: guided session report", "",
        f"- **Student:** {_cell(name)}",
        f"- **Gage:** {_cell(meta['site_label'])} (USGS {meta['site_no']})",
    ]
    if meta.get("scene_title"):
        lines.append(f"- **Gallery scene:** {_cell(meta['scene_title'])}")
    lines += [
        f"- **Chart:** {_cell(meta['view'])}; comparison with normal {'on' if meta.get('compare_on') else 'off'}",
        f"- **Started:** {data['started_at'] or 'unknown'} (UTC); report made {data['generated_at']}",
        f"- **Status:** {status}",
        "", f"## Suggested score: {s['points']} of {s['out_of']}", "",
        f"{s['on_their_own']} of {s['steps_total']} steps done entirely on their own; the \"I'm stuck\" button was pressed {s['stuck_presses']} time(s). "
        f"Guide wording: {s['wording']}; answers graded by {s['grading']}.", "",
        "| Step | Result | Points | Highest hint | Stuck presses |", "|---|---|---|---|---|",
    ]
    for r in data["steps"]:
        lines.append(f"| {r['step']} | {r['label']} | {'' if r['points'] is None else r['points']} | "
                     f"{['none', 'nudge', 'pointed question', 'explanation'][r['highest_hint_level']] if r['status'] != 'not_reached' else ''} | {r['stuck_presses']} |")
    lines += ["", "Rubric: 2 points for a step worked out on their own, 1 after hints, 0 if the guide had to explain it (they still took part). "
              "It is a suggestion; the transcript below is the evidence.", "", "## What the student said, step by step", ""]
    for r in data["steps"]:
        if r["answers"]:
            lines += [f"**{r['step']}**", ""] + [_quote(a) + "\n" for a in r["answers"]]
    lines += ["## Full transcript", ""]
    for m in data["transcript"]:
        who = "Student" if m["role"] == "student" else "Guide"
        lines += [f"**{who}:**", _quote(m["text"]), ""]
    return "\n".join(lines).rstrip() + "\n"


def to_json_text(meta: dict, t: Tutor) -> str:
    return json.dumps(build_json(meta, t), indent=2, default=str)


def file_stem(meta: dict) -> str:
    safe = "".join(ch if ch.isalnum() else "-" for ch in (meta.get("student") or "student")).strip("-").lower() or "student"
    return f"river-guide-{safe}-{meta['site_no']}-{datetime.now(timezone.utc):%Y%m%d}"
