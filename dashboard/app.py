"""
FSL Thesis Dashboard — Navigation Router
Few-Shot Learning Paradigms for Bird Species Classification under Data Scarcity
Ghulam Mustafa | FAU Erlangen-Nürnberg | MSc AI
"""

import os
import streamlit as st

_DASH = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="FSL Bird Classification — Thesis Dashboard",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load CSS so sidebar/nav styles apply before any page renders ───────────────
_CSS = os.path.join(_DASH, "assets", "style.css")
if os.path.exists(_CSS):
    with open(_CSS) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Global dataset filter (top-right settings button) ─────────────────────────
if "ds_filter" not in st.session_state:
    st.session_state["ds_filter"] = "CUB-200"

_LABELS = {"Birds-20": "🦅 Birds-20", "CUB-200": "🌿 CUB-200", "All": "🌐 All"}
_curr   = st.session_state["ds_filter"]

_, _col_gear = st.columns([14, 1])
with _col_gear:
    with st.popover("⚙️", width="stretch"):
        st.markdown(
            "<div style='font-size:13px;font-weight:700;color:#003865;"
            "margin-bottom:6px'>Dataset filter</div>",
            unsafe_allow_html=True,
        )
        st.radio(
            "Show",
            ["Birds-20", "CUB-200", "All"],
            format_func=lambda x: _LABELS[x],
            key="ds_filter",
            label_visibility="collapsed",
        )
        st.markdown(
            f"<div style='font-size:11px;color:#6B7280;margin-top:4px'>"
            f"Showing: <b>{_LABELS[_curr]}</b></div>",
            unsafe_allow_html=True,
        )

# ── Navigation — paths are relative to this file's directory ──────────────────
def _page(filename, **kw):
    return st.Page(os.path.join(_DASH, "pages", filename), **kw)

pg = st.navigation(
    {
        "": [
            _page("00_home.py", title="Home", icon="🏠", default=True),
        ],
        "Results Analysis": [
            _page("01_overview.py",  title="Method Ranking",      icon="📊"),
            _page("02_kshot.py",     title="K-Shot Efficiency",   icon="📈"),
            _page("03_backbone.py",  title="Backbone Comparison", icon="🔬"),
            _page("04_distance.py",  title="Distance Ablation",   icon="📐"),
        ],
        "Deep Dives": [
            _page("05_perclass.py",       title="Per-Class Heatmap",  icon="🐦"),
            _page("06_crossbenchmark.py", title="Cross-Benchmark",    icon="🌍"),
            _page("07_literature.py",     title="Literature",         icon="📚"),
        ],
    }
)
pg.run()
