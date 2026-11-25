"""
Page 1 — Method Ranking Overview
Grouped bar chart of all methods at a chosen K-shot and split.
"""

import os, sys
_DASH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DASH not in sys.path:
    sys.path.insert(0, _DASH)

import streamlit as st
import pandas as pd

from data.loader import (
    get_fsl_results, filter_results, METHOD_ORDER, BACKBONE_COLOR, METHOD_COLORS
)
from components.charts import bar_chart
from components.filters import shot_filter, split_filter, get_global_dataset, ci_toggle
from components.cards import metric_cards_row, finding_box, section_header

_CSS = os.path.join(_DASH, "assets", "style.css")
if os.path.exists(_CSS):
    with open(_CSS) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def _render(dataset, split_val, shot, show_ci):
    df = get_fsl_results()
    sub = filter_results(df, split=split_val, shot=shot, dataset=dataset)

    # One row per method, preferring Euclidean > Cosine > Manhattan > TTA.
    # (Plain sort + drop_duplicates instead of groupby.apply, which behaves
    # differently in pandas 3: the grouping column is no longer passed in.)
    pref = {"Euclidean": 0, "Cosine": 1, "Manhattan": 2, "Euclidean+TTA": 3}
    sub = (
        sub.assign(_pref=sub["distance"].map(pref).fillna(99))
           .sort_values(["method", "_pref"], kind="stable")
           .drop_duplicates(subset=["method"])
           .drop(columns="_pref")
           .reset_index(drop=True)
    )

    order = [m for m in METHOD_ORDER if m in sub["method"].values]
    sub["method"] = pd.Categorical(sub["method"], categories=order, ordered=True)
    sub = sub.sort_values("method")

    if sub.empty:
        st.warning(f"No results found for {dataset} / {split_val} / {shot}-shot.")
        return

    # ── Metric summary row ─────────────────────────────────────────────────────
    best      = sub.loc[sub["accuracy"].idxmax()]
    worst     = sub.loc[sub["accuracy"].idxmin()]
    ncc_sub   = sub[sub["method"].str.startswith("NCC")]
    proto_sub = sub[sub["method"].str.startswith("ProtoNet") & ~sub["method"].str.contains("TTA")]

    metric_cards_row([
        {"label": "Best Method",   "value": f"{best['accuracy']:.2f}%",
         "sub": f"{best['method']} ({best['backbone']})",    "color": "#003865"},
        {"label": "Worst Method",  "value": f"{worst['accuracy']:.2f}%",
         "sub": f"{worst['method']} ({worst['backbone']})",  "color": "#ae2012"},
        {"label": "NCC Best",      "value": f"{ncc_sub['accuracy'].max():.2f}%" if not ncc_sub.empty else "—",
         "sub": "No episodic training",    "color": "#FF6B35"},
        {"label": "ProtoNet Best", "value": f"{proto_sub['accuracy'].max():.2f}%" if not proto_sub.empty else "—",
         "sub": "With episodic training",  "color": "#1e6bb8"},
    ])
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main bar chart ─────────────────────────────────────────────────────────
    section_header(
        f"Accuracy at {shot}-shot — {dataset} {split_val}",
        "Grouped by backbone · distance = best available per method · 600 episodes"
    )

    error_col = "ci" if show_ci else None
    fig = bar_chart(
        sub, x="method", y="accuracy",
        color="backbone",
        error_y=error_col,
        title=None,
        xlab="Method",
        ylab="Accuracy (%)",
        color_map=BACKBONE_COLOR,
        category_orders={"method": order, "backbone": ["ResNet-50", "DeiT-S"]},
        custom_data=["method", "backbone", "split", "k_shot", "ci"],
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Accuracy: %{y:.2f}% ± %{customdata[4]:.2f}%<br>"
            "Backbone: %{customdata[1]}<br>"
            "<extra></extra>"
        )
    )
    fig.update_layout(height=420, xaxis_tickangle=-25)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Error bars show ±95% Wilson CI over 600 episodes. "
        "Multiple distance results per method: Euclidean shown where available."
    )

    # ── Key insight callout ────────────────────────────────────────────────────
    if not ncc_sub.empty and not proto_sub.empty:
        ncc_deit  = sub[sub["method"] == "NCC (DeiT-S)"]["accuracy"].max() \
                    if "NCC (DeiT-S)" in sub["method"].values else None
        proto_r50 = sub[sub["method"] == "ProtoNet"]["accuracy"].max() \
                    if "ProtoNet" in sub["method"].values else None
        if ncc_deit is not None and proto_r50 is not None:
            pp = ncc_deit - proto_r50
            finding_box(
                f"<b>Backbone > Algorithm:</b> NCC/DeiT-S (no episodic training) achieves "
                f"<b>{ncc_deit:.2f}%</b> vs ProtoNet/R50 at <b>{proto_r50:.2f}%</b> "
                f"(<b>{pp:+.2f} pp</b>). A stronger backbone beats a fancier training algorithm.",
                icon="💡", style="info",
            )

    # ── Distance breakdown per method ──────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("All Distance Variants", "Including Euclidean, Manhattan, Cosine per method")

    full_sub = filter_results(df, split=split_val, shot=shot, dataset=dataset)
    full_sub = full_sub[full_sub["method"].isin(order)].copy()
    full_sub["method"] = pd.Categorical(full_sub["method"], categories=order, ordered=True)
    full_sub = full_sub.sort_values(["method", "distance"])

    if not full_sub.empty:
        fig2 = bar_chart(
            full_sub, x="method", y="accuracy",
            color="distance",
            error_y=error_col,
            title=None,
            xlab="Method",
            ylab="Accuracy (%)",
            category_orders={"method": order,
                             "distance": ["Euclidean", "Manhattan", "Cosine", "Euclidean+TTA"]},
            color_map={"Euclidean": "#003865", "Manhattan": "#FF6B35",
                       "Cosine": "#0a9396", "Euclidean+TTA": "#94d2bd"},
        )
        fig2.update_layout(height=400, xaxis_tickangle=-25)
        st.plotly_chart(fig2, width="stretch")

    # ── Sortable results table ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Full Results Table", "Click column headers to sort")

    display_cols = ["method", "backbone", "distance", "k_shot", "accuracy", "ci", "f1", "auc"]
    avail = [c for c in display_cols if c in full_sub.columns]
    tbl = full_sub[avail].copy()
    tbl = tbl.rename(columns={
        "method": "Method", "backbone": "Backbone", "distance": "Distance",
        "k_shot": "K", "accuracy": "Acc (%)", "ci": "±CI95", "f1": "F1", "auc": "AUC"
    })
    tbl["Acc (%)"] = tbl["Acc (%)"].round(2)
    tbl["±CI95"]   = tbl["±CI95"].round(3)
    tbl["F1"]      = tbl["F1"].round(4)
    tbl["AUC"]     = tbl["AUC"].round(4)
    tbl["K"]       = tbl["K"].astype("Int64")
    st.dataframe(
        tbl.sort_values("Acc (%)", ascending=False).reset_index(drop=True),
        width="stretch",
        height=350,
    )


# ── Sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.markdown("## Filters")
ds = get_global_dataset()

if ds != "All":
    split = split_filter(key="ov_split", dataset=ds,
                         default="B15/N5" if ds == "Birds-20" else "B150/N50")
shot    = shot_filter(key="ov_shot", default=5)
show_ci = ci_toggle(key="ov_ci")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#003865,#1e4e8c);border-radius:10px;
            padding:20px 24px;color:white;margin-bottom:16px">
  <h2 style="margin:0;font-size:20px">📊 Method Ranking Overview</h2>
  <p style="margin:4px 0 0;opacity:.85;font-size:13px">
    All FSL methods compared at selected K-shot and split — error bars = 95% Wilson CI
  </p>
</div>
""", unsafe_allow_html=True)

# ── Render ─────────────────────────────────────────────────────────────────────
if ds == "All":
    _tabs = st.tabs(["🦅 Birds-20 (B15/N5)", "🌿 CUB-200 (B150/N50)"])
    with _tabs[0]:
        _render("Birds-20", "B15/N5", shot, show_ci)
    with _tabs[1]:
        _render("CUB-200", "B150/N50", shot, show_ci)
else:
    _render(ds, split, shot, show_ci)
