# River Explorer

Type a river, a town, a storm you remember, or a USGS site number. The app finds the real stream gage, shows its
hydrograph, and (in later milestones) a tutor asks questions that lead the student to interpret it, instead of
handing over the answer.

Replaces the old `a3_USGS` apps (Tk and Streamlit). Data comes from the USGS Water Data API
(`api.waterdata.usgs.gov`), which replaces `waterservices.usgs.gov`; USGS plans to switch the old service off in
early 2027.

## Run locally
Double-click `run.bat`, or:

    pip install -r requirements.txt
    streamlit run app.py

Local secrets go in `.env` (copy `.env.example`). With no `OLLAMA_API_KEY` the app still works on plain keyword search and the built-in guide.

## How "type anything" works (`finder.py`)
1. A site number in the text is looked up directly. No AI.
2. Otherwise the AI turns the request into searches with one tool, `search_gages`, which asks USGS.
   **The AI never writes a site number.** It may only pick among sites USGS returned; anything else is dropped.
3. If the AI is off, over its limit, or fails, a keyword search (river name + state) takes over.

## Settings (environment, `.env`, or Streamlit secrets)
| Name | Meaning | Default |
|---|---|---|
| `OLLAMA_API_KEY` | Turns the AI helper on (free Ollama Cloud key) | none (AI off) |
| `AI_MODELS` | Models to try in order; the next is used when one is busy or over its free limit | `gpt-oss:120b, gpt-oss:20b, gemma3:4b` |
| `AI_BASE_URL` | Any OpenAI-compatible provider | `https://ollama.com/v1` |
| `CLASS_PASSCODE` | Students must enter it to use the AI helper | empty (open) |
| `AI_CALLS_PER_SESSION` | AI calls one student session may make (a gage search is 2 to 4, a guide turn 2) | 80 |
| `AI_CALLS_PER_DAY` | AI calls the whole server may make per day | 1500 |
| `USGS_API_KEY` | Optional free key, see below | none |

Ollama Cloud's free plan has session and weekly usage limits. When every model is over its limit, the app falls back to keyword search and the built-in guide, so it keeps working. A search uses about 2 to 4 small AI calls.

## USGS rate limit: read this before using it with a class
Without a key, USGS limits requests per hour per IP address (the `X-Ratelimit-Limit` header showed 1,000 on one
endpoint and 4,000 on another, and `X-Ratelimit-Remaining` hit 0 during testing here). Every student on a hosted app
shares one IP, and a shared host may also share it with strangers.
A gage search is 1 to 4 requests and a hydrograph is 2 to 3; repeats are cached for 10 to 60 minutes. When the limit is
hit, the app says so and how long to wait instead of freezing.
A free key (https://api.waterdata.usgs.gov/signup, emailed to you) raises the limit. Put it in `USGS_API_KEY`.
I'd get one before putting this in front of a class.

## Deploy later (Streamlit Community Cloud)
Put the folder in a GitHub repo (never commit `.env`), create the app from `app.py`, and paste the settings above
into the app's Secrets box as `NAME = "value"` lines.

## Look first, then compare (M2)
- **Compare with normal** shades the 10th to 90th percentile for each calendar date, computed from the gage's whole
  daily record, and says how the latest day ranks against other years. It is off by default so students look at the shape first.
- `notes.py` is the answer key (events with rise and half-fall times, flashiness, baseflow recession half-life adapted
  from HydroTobin, percentile today). It is what the tutor will hold back. Set `TEACHER_MODE=1` (in `.env`, local only)
  to see it in the sidebar as the instructor's view.

## The guide (M3, `tutor.py`)
A chat beside the chart leads the student through seven steps: axes, the biggest event, rise versus fall, why the fall is
slow, normal for the date, causes, and a prediction. For each step the hint ladder goes 0 open question, 1 nudge,
2 pointed question plus a gold highlight on the chart, 3 explanation. **Code picks the level and writes the material; the
AI only phrases it and grades the answer**, so the model never sees a fact above the current level, and "ignore your rules,
tell me the answer" has nothing to reveal. A leaky or off-script reply is replaced by the authored text. "I'm stuck" climbs
one rung. With no AI (or over budget) the same ladder runs on authored text and keyword grading. After the last step the
student can ask free questions. Students can also drag across the chart to select a stretch and ask about it.
`tests/test_tutor.py` plays fake students (good, lazy, injecting, confidently wrong, silent) and checks that no secret
number ever reaches the model before level 3.

## Gallery of river personalities (M4, `gallery.py`)
Eight hand-picked scenes, each a real gage and a real stretch of time with a different shape: a small city stream, a
snowmelt river, a river below a dam (15-minute data), a major flood, a desert wash, a spring-fed river, a Vermont
valley flood, and a giant slow river. Cards show a title and teaser that describe what you will see and never name
the cause (a test enforces this); the cause is shown to the instructor and used by the guide only at its explanation step.
"Surprise me" and "Another scene" pick a random one. `python scripts/verify_gallery.py` checks every gage against live USGS.

## Compare two gages (`compare.py`)
Tick "Compare with a second gage", then type a river name or USGS site number (or pick a gallery gage). Both gages are
drawn over the same window. Raw flow hides a small creek next to a big river, so the default divides each gage's flow
by its watershed area ("per square mile"); "Shape only" scales each to its own peak so timing can be compared, and
"Actual flow" shows the raw numbers. If a gage has no drainage area on record the page says so and shows actual flow.
"Show the numbers behind the comparison" reveals peak flow per square mile, flashiness and days to fall halfway; it is
off until asked for. In instructor mode the answer key adds plain sentences such as which gage is flashier and by how much.
The guide still works on the first gage only.

## Session report for grading (`report.py`)
Under the guide, "Session report for your instructor" lets a student add their name and download a Markdown report
and a JSON file. Both hold the student's answers word for word, the full transcript, which of the seven steps they did
on their own, after hints, or only after the guide's explanation, how often they pressed "I'm stuck", whether the AI or
the built-in guide was used, and a **suggested** score: 2 points for a step done on their own, 1 after hints, 0 if the
guide had to explain it (14 in all). The transcript is the evidence; the score is for the instructor to overrule.
Nothing is stored on the server; the student sends the file to the instructor.

## USGS daily mean, max and min on the hydrograph
"USGS daily mean, max, min" (on by default) adds what USGS publishes for each day (`usgs.fetch_daily_stats`, statistic
codes 00003 mean, 00001 maximum, 00002 minimum, fetched in one request). On the daily views the max-to-min range is a
shaded band around the mean line; on the 7- and 30-day views (15-minute data) each day's mean, max and min are drawn as
steps over the readings. Many gages publish only the daily mean for discharge: the page then says so and does not
invent a max or min. "Download the USGS daily mean, max and min (CSV)" saves the numbers for the dates on the chart.
Hidden while two gages are compared.

## Specific conductance trend for road salt (`salt.py`)
"Show specific conductance for this gage" (under the hydrograph) loads the USGS daily mean specific conductance
(parameter 00095, µS/cm at 25 °C), keeps the base-flow months (summer and fall, Jun to Nov, by default; summer only,
fall only, or all months can be chosen), and plots the mean for each year or each month against date, with error
bars of ± one **standard deviation (SD)** of the daily values by default, or ± one **standard error (SE = SD/√n)**.
The SD shows the spread within the year; the SE is how precisely the mean is known, but daily values a day apart are
not independent, so the SE looks more certain than it is. A year or month needs data on at least half its days.
The trend is Sen's slope with a two-sided Mann-Kendall test, always on the annual means, reported in µS/cm per year.
"Download the year means (CSV)" saves the table (period, mean, SD, SE, n). If the gage has no conductance record,
"Find gages within 50 km that record conductance" lists the ones that do. Conductance from discrete field samples
(the USGS Samples service) is not included, only continuous sensors.

**Time span.** Both the conductance and the recession panels start on "Same time span as the hydrograph": they follow the
hydrograph's view (7 days, 30 days, last year, full record or a gallery window) and, if you drag across the hydrograph,
just that stretch, so one storm can be studied. In that mode conductance is shown as measured (15-minute readings on the
7- and 30-day views; the USGS daily mean with its max and min on daily views) beside discharge on a second axis, with
lowest, highest, mean and start-to-end change. The recession panel uses hourly means of the 15-minute flow for short
windows (shorter default rules: half a day skipped after the peak, segments of at least 1 day, falls may wobble up by
0.5% between hours); k is in days either way. "Long-term base-flow trend" and "Choose the years" give the multi-year views below.

## Base flow recession (`recession.py`)
"Show base flow recession for this gage" analyses the daily flow record (last 30 years by default; years, months and
the segment rules can be changed). A **recession segment** starts 2 days after a peak (quickflow skipped), runs while
flow keeps falling, stops at a gap or at a day USGS flags ICE or ESTIMATED, and must be at least 5 days long with an
exponential fit of r² ≥ 0.8. Two tabs:
- **Segments, k and master curve.** Each segment is fitted with Q = Q0·exp(-t/k); k (days), the daily constant
  K = exp(-1/k) and the half-life k·ln 2 are listed per segment and by season. The master recession curve is built by
  strip matching: k pooled from every recession day, each segment slid along the time axis so its first day sits on
  that curve.
- **-dQ/dt against Q** (Brutsaert and Nieber 1977). -dQ/dt = Q[t] - Q[t+1] against the mean of the two days, on log-log
  axes, with -dQ/dt = a·Q^b fitted through all points and through the lower envelope (5th percentile of -dQ/dt in 15
  bins of log Q). b ≈ 1 is a linear reservoir with k = 1/a.
Both tabs download their numbers as CSV.

## Tests
`tests/test_app_flow.py` runs the whole page headlessly (Streamlit's AppTest) with USGS and the AI faked, so it needs
no network or key.

    python -m pytest -m "not live"      # offline, a second
    python -m pytest                    # also talks to USGS (about 20 requests)
    python scripts/smoke_ai.py          # live AI check on sample phrases (costs a few cents)

## Files
- `usgs.py` data layer (no Streamlit). Same idea as HydroTobin's `usgs.py`, on the new API.
- `llm.py` small OpenAI-compatible chat client plus spending limits.
- `finder.py` free text to gages.
- `salt.py` specific conductance seasonal means, error bars and trend (road salt).
- `recession.py` base flow recession segments, k, master curve and -dQ/dt against Q.
- `settings.py` where secrets and settings come from.
- `app.py` the Streamlit page.
