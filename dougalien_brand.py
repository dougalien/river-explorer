"""Dougalien Systems look for the Geo-Tools Streamlit apps.

Colors come from the Dougalien Systems logo (sand, slate, rust) and match
dougalien.com. The base palette lives in .streamlit/config.toml; this module
adds the logo, a strata stripe under the header and a few light CSS touches.
It only changes appearance, never app behavior.

Usage, right after st.set_page_config(...):

    from dougalien_brand import apply_branding
    apply_branding()
"""

from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).resolve().parent / "assets"
LOGO = ASSETS / "dougalien_systems_logo.png"
MARK = ASSETS / "dougalien_mark.png"
SITE_URL = "https://dougalien.com"

PAPER = "#F4F4F1"
SHEET = "#FFFFFF"
INK = "#1B1E20"
MUTED = "#586770"
RULE = "#D6DBDA"
SLATE = "#3E5C6B"
RUST = "#B4532A"
SAND = "#D6B27C"
SAND_TINT = "#EFE8DA"

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Familjen+Grotesk:wght@500;600;700&display=swap');

/* Header bar: paper background with the logo's strata as a stripe along the bottom */
[data-testid="stHeader"] {{
  background: {PAPER};
}}
[data-testid="stHeader"]::after {{
  content: "";
  position: absolute; left: 0; right: 0; bottom: 0; height: 5px;
  background: linear-gradient(90deg,
    {SAND} 0 22%, {SLATE} 22% 34%, {SAND} 34% 62%,
    {RUST} 62% 70%, {SAND} 70% 88%, {SLATE} 88% 100%);
  pointer-events: none;
}}

/* Headings in the dougalien.com display face */
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
  font-family: "Familjen Grotesk", "Helvetica Neue", Arial, sans-serif;
  letter-spacing: -0.01em;
}}
.stApp h1 {{ color: {INK}; font-weight: 700; }}
.stApp h2, .stApp h3 {{ color: {SLATE}; font-weight: 600; }}

/* Sidebar: soft sand panel with a rule on the edge */
[data-testid="stSidebar"] {{
  background: {SAND_TINT};
  border-right: 1px solid {RULE};
}}

/* Links, metrics, expanders, tabs, captions */
.stApp a {{ color: {SLATE}; }}
.stApp a:hover {{ color: {RUST}; }}
[data-testid="stMetricValue"] {{ color: {SLATE}; }}
[data-testid="stExpander"] details {{
  border-color: {RULE};
  background: {SHEET};
}}
.stTabs [data-baseweb="tab-list"] {{ border-bottom: 1px solid {RULE}; }}
.stTabs [aria-selected="true"] {{ color: {SLATE}; }}
[data-testid="stCaptionContainer"] {{ color: {MUTED}; }}

/* Buttons: slate fill for primary (from the theme), rust edge on hover for the rest */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
  border-radius: 4px;
  font-weight: 600;
}}
.stButton > button[kind="secondary"]:hover,
.stDownloadButton > button[kind="secondary"]:hover {{
  border-color: {RUST};
  color: {RUST};
}}

/* Quotes pick up the rust accent */
.stApp blockquote {{ border-left: 3px solid {RUST}; }}
</style>
"""


def _logo():
    if not hasattr(st, "logo") or not LOGO.exists():
        return
    # No icon_image, so apps without a sidebar show the full lockup in the header too
    try:
        st.logo(str(LOGO), size="large", link=SITE_URL)
    except TypeError:  # Streamlit without the size option
        st.logo(str(LOGO), link=SITE_URL)


def apply_branding():
    """Add the Dougalien Systems logo and brand styling to the current page."""
    _logo()
    if hasattr(st, "html"):
        st.html(_CSS)
    else:
        st.markdown(_CSS, unsafe_allow_html=True)


def page_icon(default="🪨"):
    """Logo mark for st.set_page_config(page_icon=...), or the emoji if the file is missing."""
    return str(MARK) if MARK.exists() else default
