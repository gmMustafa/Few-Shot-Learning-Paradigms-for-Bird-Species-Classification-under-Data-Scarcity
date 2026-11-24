"""
Page 3 — Backbone Comparison (ResNet-50 vs DeiT-S)
Scatter and bar charts showing the backbone advantage.
"""

import os, sys
_DASH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DASH not in sys.path:
    sys.path.insert(0, _DASH)

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from data.loader import get_fsl_results, filter_results, METHOD_ORDER, BACKBONE_COLOR
from components.charts import (
    scatter_with_diagonal, bar_chart, _apply_base, _BASE_LAYOUT,
    FAU_BLUE, FAU_ORANGE, PALETTE,
)
from components.filters import shot_filter, get_global_dataset, ci_toggle
from components.cards import finding_box, section_header, metric_cards_row

_CSS = os.path.join(_DASH, "assets", "style.css")
if os.path.exists(_CSS):
    with open(_CSS) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

PAIRED = {
    "NCC":      ("NCC (R50)",  "NCC (DeiT-S)"),
    "ProtoNet": ("ProtoNet",   "ProtoNet (DeiT-S)"),
}


def _render(dataset, shot, show_ci):
    df = get_fsl_results()
    splits = ["B15/N5", "B10/N10"] if dataset == "Birds-20" else ["B100/N50", "B150/N50"]

    def _get_acc(method, spl, sh, dist):
        sub = filter_results(df, method=method, split=spl, shot=sh,
                             dataset=dataset, distance=dist)
        if sub.empty:
            return np.nan, np.nan
        row = sub.iloc[0]
        return row["accuracy"], row["ci"]

    pair_rows = []
    for pair_name, (r50_m, deit_m) in PAIRED.items():
        for spl in splits:
            for sh in [1, 3, 5]:
                a_r50,  ci_r50  = _get_acc(r50_m,  spl, sh, "Euclidean")
                a_deit, ci_deit = _get_acc(deit_m, spl, sh, "Euclidean")
                pair_rows.append({
                    "pair":        pair_name,
                    "split":       spl,
                    "k_shot":      sh,
                    "r50_acc":     a_r50,
                    "deit_acc":    a_deit,
                    "r50_ci":      ci_r50,
                    "deit_ci":     ci_deit,
                    "deit_gain":   a_deit - a_r50
                                   if not (np.isnan(a_r50) or np.isnan(a_deit)) else np.nan,
                    "label_short": f"{pair_name}/{sh}s",
                })

    pair_df = pd.DataFrame(pair_rows).dropna(subset=["r50_acc", "deit_acc"])

    if pair_df.empty:
        st.warning(f"No paired R50/DeiT-S data for {dataset}.")
        return

    # ── Metric cards ───────────────────────────────────────────────────────────
    ncc_gain   = pair_df[pair_df["pair"] == "NCC"]["deit_gain"].mean()
    proto_gain = pair_df[pair_df["pair"] == "ProtoNet"]["deit_gain"].mean()
    shot1_ncc  = pair_df[(pair_df["pair"] == "NCC") & (pair_df["k_shot"] == 1)]["deit_gain"].mean()
    shot1_prot = pair_df[(pair_df["pair"] == "ProtoNet") & (pair_df["k_shot"] == 1)]["deit_gain"].mean()

    metric_cards_row([
        {"label": "DeiT-S avg gain (NCC)",           "value": f"+{ncc_gain:.2f} pp",
         "sub": "vs ResNet-50 NCC, all shots/splits", "color": "#FF6B35"},
        {"label": "DeiT-S avg gain (ProtoNet)",      "value": f"+{proto_gain:.2f} pp",
         "sub": "vs ResNet-50 ProtoNet",              "color": "#003865"},
        {"label": "Backbone effect at 1-shot (NCC)", "value": f"+{shot1_ncc:.2f} pp",
         "sub": "NCC DeiT-S vs NCC R50",              "color": "#FF6B35"},
        {"label": "Backbone effect at 1-shot (Proto)", "value": f"+{shot1_prot:.2f} pp",
         "sub": "ProtoNet DeiT-S vs ProtoNet R50",    "color": "#003865"},
    ])
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Scatter ────────────────────────────────────────────────────────────────
    section_header(
        f"ResNet-50 vs DeiT-S Accuracy Scatter — {dataset}",
        "Points above diagonal = DeiT-S wins. Each point = method/split/shot combo."
    )

    shot_df = pair_df[pair_df["k_shot"] == shot].copy()
    shot_df["label"] = shot_df.apply(lambda r: f"{r['pair']} {r['split']}", axis=1)

    if shot_df.empty:
        st.warning("No paired data for selected shot count.")
    else:
        fig = scatter_with_diagonal(
            shot_df,
            x="r50_acc", y="deit_acc",
            label_col="label",
            color="pair",
            title=None,
            xlab="ResNet-50 Accuracy (%)",
            ylab="DeiT-S Accuracy (%)",
            color_map={"NCC": FAU_ORANGE, "ProtoNet": FAU_BLUE},
            add_trendline=False,
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"At {shot}-shot. Points above the dashed diagonal = DeiT-S outperforms ResNet-50."
        )

    # ── Bar: DeiT-S gain ───────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section_header(
        "DeiT-S Accuracy Gain over ResNet-50",
        "Positive = DeiT-S wins. Split by method and K-shot."
    )

    gain_df = pair_df.copy()
    gain_df["k_shot_str"] = gain_df["k_shot"].apply(lambda x: f"{int(x)}-shot")

    fig2 = bar_chart(
        gain_df, x="k_shot_str", y="deit_gain",
        color="pair",
        title=None,
        xlab="K-Shot",
        ylab="DeiT-S gain (pp)",
        color_map={"NCC": FAU_ORANGE, "ProtoNet": FAU_BLUE},
        category_orders={"k_shot_str": ["1-shot", "3-shot", "5-shot"],
                         "pair": ["NCC", "ProtoNet"]},
        barmode="group",
    )
    fig2.add_hline(y=0, line_dash="dash", line_color="#6B7280", line_width=1)
    fig2.update_layout(yaxis=dict(range=None), height=380)
    st.plotly_chart(fig2, width="stretch")

    # ── Table ──────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Paired Accuracy Table", "R50 vs DeiT-S side by side")

    tbl = pair_df[["pair", "split", "k_shot", "r50_acc", "r50_ci",
                   "deit_acc", "deit_ci", "deit_gain"]].copy()
    tbl = tbl.rename(columns={
        "pair": "Method", "split": "Split", "k_shot": "K",
        "r50_acc": "R50 Acc (%)", "r50_ci": "R50 ±CI",
        "deit_acc": "DeiT-S Acc (%)", "deit_ci": "DeiT-S ±CI",
        "deit_gain": "DeiT-S Gain (pp)",
    })
    tbl = tbl.round(2)
    tbl["K"] = tbl["K"].astype(int)
    st.dataframe(
        tbl.sort_values(["Method", "Split", "K"]).reset_index(drop=True),
        width="stretch", height=320,
    )

    # ── Insight ────────────────────────────────────────────────────────────────
    finding_box(
        "<b>Why DeiT-S wins more for NCC than ProtoNet:</b> "
        "DeiT-S produces a near-isotropic feature space (CLS token from self-attention), "
        "making Euclidean distance a reliable proxy for class proximity. "
        "ResNet-50's avgpool features are anisotropic — distances are distorted across dimensions. "
        "NCC relies entirely on geometry; ProtoNet partly compensates via episodic optimisation. "
        "Hence the backbone swap benefits NCC disproportionately.",
        icon="📐", style="info",
    )


# ── Sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.markdown("## Filters")
ds      = get_global_dataset()
shot    = shot_filter(key="bb_shot", default=1)
show_ci = ci_toggle(key="bb_ci")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#003865,#1e4e8c);border-radius:10px;
            padding:20px 24px;color:white;margin-bottom:16px">
  <h2 style="margin:0;font-size:20px">🔬 Backbone Comparison: ResNet-50 vs DeiT-S</h2>
  <p style="margin:4px 0 0;opacity:.85;font-size:13px">
    How much does the backbone architecture matter vs the FSL algorithm?
  </p>
</div>
""", unsafe_allow_html=True)

# ── Render ─────────────────────────────────────────────────────────────────────
if ds == "All":
    _tabs = st.tabs(["🦅 Birds-20", "🌿 CUB-200"])
    with _tabs[0]:
        _render("Birds-20", shot, show_ci)
    with _tabs[1]:
        _render("CUB-200", shot, show_ci)
else:
    _render(ds, shot, show_ci)
