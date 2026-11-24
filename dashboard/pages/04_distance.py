"""
Page 4 — Distance Metric Ablation
Euclidean vs Manhattan for NCC and ProtoNet across both backbones.
"""

import os, sys
_DASH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DASH not in sys.path:
    sys.path.insert(0, _DASH)

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from data.loader import get_fsl_results, filter_results, BACKBONE_COLOR
from components.charts import (
    bar_chart, _apply_base, _BASE_LAYOUT, FAU_BLUE, FAU_ORANGE, PALETTE
)
from components.filters import shot_filter, get_global_dataset
from components.cards import finding_box, caveat_box, section_header, metric_cards_row

_CSS = os.path.join(_DASH, "assets", "style.css")
if os.path.exists(_CSS):
    with open(_CSS) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

DIST_METHODS = {
    "NCC (R50)":         "ResNet-50",
    "NCC (DeiT-S)":      "DeiT-S",
    "ProtoNet":          "ResNet-50",
    "ProtoNet (DeiT-S)": "DeiT-S",
}


def _build_gap_df(split_sel, dataset, shot, df):
    rows = []
    for method, backbone in DIST_METHODS.items():
        for dist in ["Euclidean", "Manhattan"]:
            sub = filter_results(df, method=method, split=split_sel,
                                 shot=shot, dataset=dataset, distance=dist)
            if not sub.empty:
                rows.append({
                    "method":   method,
                    "backbone": backbone,
                    "distance": dist,
                    "accuracy": sub.iloc[0]["accuracy"],
                    "ci":       sub.iloc[0]["ci"],
                })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _render_split(split_sel, dataset, shot, df):
    gap_df = _build_gap_df(split_sel, dataset, shot, df)
    if gap_df.empty:
        st.warning(f"No data for {dataset} / {split_sel} / {shot}-shot.")
        return

    pivot = gap_df.pivot_table(
        index="method", columns="distance", values="accuracy"
    ).reindex(columns=["Euclidean", "Manhattan"])
    pivot["Gap (Euc−Man)"] = pivot["Euclidean"] - pivot["Manhattan"]
    pivot["backbone"] = pivot.index.map(DIST_METHODS)

    section_header(
        f"Euclidean vs Manhattan — {split_sel} · {shot}-shot",
        "Annotations show Euclidean−Manhattan gap"
    )

    fig = bar_chart(
        gap_df, x="method", y="accuracy",
        color="distance",
        error_y="ci",
        title=None,
        xlab="Method",
        ylab="Accuracy (%)",
        color_map={"Euclidean": FAU_BLUE, "Manhattan": FAU_ORANGE},
        category_orders={
            "distance": ["Euclidean", "Manhattan"],
            "method": list(DIST_METHODS.keys()),
        },
    )
    for method in pivot.index:
        gap = pivot.loc[method, "Gap (Euc−Man)"]
        euc = pivot.loc[method, "Euclidean"]
        if not np.isnan(gap) and abs(gap) > 0.05:
            color = "red" if gap > 1.5 else "#003865"
            fig.add_annotation(
                x=method, y=euc + 2.5,
                text=f"Δ{gap:+.1f}pp",
                showarrow=False,
                font=dict(size=11, color=color),
            )
    fig.update_layout(height=430, xaxis_tickangle=-20)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Error bars = ±95% Wilson CI. "
        "Δ annotation shows Euclidean − Manhattan gap."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Gap Size by Backbone",
                   "R50 shows large distance sensitivity; DeiT-S does not")

    r50_gaps  = pivot[pivot["backbone"] == "ResNet-50"]["Gap (Euc−Man)"].dropna()
    deit_gaps = pivot[pivot["backbone"] == "DeiT-S"]["Gap (Euc−Man)"].dropna()

    c1, c2 = st.columns(2)
    with c1:
        metric_cards_row([{
            "label": "ResNet-50 avg Euc−Man gap",
            "value": f"{r50_gaps.mean():+.2f} pp",
            "sub":   f"range [{r50_gaps.min():+.1f}, {r50_gaps.max():+.1f}]",
            "color": FAU_BLUE,
        }])
    with c2:
        metric_cards_row([{
            "label": "DeiT-S avg Euc−Man gap",
            "value": f"{deit_gaps.mean():+.2f} pp" if not deit_gaps.empty else "—",
            "sub":   "near-zero: metric choice irrelevant",
            "color": FAU_ORANGE,
        }])

    st.markdown("<br>", unsafe_allow_html=True)
    tbl = pivot.reset_index().rename(columns={"method": "Method"})
    tbl = tbl[["Method", "backbone", "Euclidean", "Manhattan", "Gap (Euc−Man)"]].round(3)
    st.dataframe(tbl, width="stretch", height=220)


def _render(dataset, shot, df):
    splits = ["B15/N5", "B10/N10"] if dataset == "Birds-20" else ["B100/N50", "B150/N50"]
    tab_labels = splits + ["Both"]
    tabs = st.tabs(tab_labels)

    for i, spl in enumerate(splits):
        with tabs[i]:
            _render_split(spl, dataset, shot, df)

    with tabs[-1]:
        all_rows = []
        for spl in splits:
            d = _build_gap_df(spl, dataset, shot, df)
            if not d.empty:
                d["split"] = spl
                all_rows.append(d)
        if all_rows:
            combined = pd.concat(all_rows)
            combined["label"] = combined["method"] + " · " + combined["split"]
            fig = bar_chart(
                combined, x="label", y="accuracy",
                color="distance",
                error_y="ci",
                title=None,
                xlab="Method · Split",
                ylab="Accuracy (%)",
                color_map={"Euclidean": FAU_BLUE, "Manhattan": FAU_ORANGE},
                category_orders={"distance": ["Euclidean", "Manhattan"]},
            )
            fig.update_layout(height=460, xaxis_tickangle=-35)
            st.plotly_chart(fig, width="stretch")


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## Filters")
ds   = get_global_dataset()
shot = shot_filter(key="dist_shot", default=5)

# ── Load ───────────────────────────────────────────────────────────────────────
df = get_fsl_results()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#003865,#1e4e8c);border-radius:10px;
            padding:20px 24px;color:white;margin-bottom:16px">
  <h2 style="margin:0;font-size:20px">📐 Distance Metric Ablation</h2>
  <p style="margin:4px 0 0;opacity:.85;font-size:13px">
    Euclidean vs Manhattan distance — when does the choice matter?
  </p>
</div>
""", unsafe_allow_html=True)

# ── Render ─────────────────────────────────────────────────────────────────────
if ds == "All":
    _tabs = st.tabs(["🦅 Birds-20", "🌿 CUB-200"])
    with _tabs[0]:
        _render("Birds-20", shot, df)
    with _tabs[1]:
        _render("CUB-200", shot, df)
else:
    _render(ds, shot, df)

# ── Key insight callouts ───────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
finding_box(
    "<b>Distance metric choice is a ResNet-50 concern only.</b> "
    "Euclidean consistently outperforms Manhattan by ≈8 pp for ResNet-50. "
    "This reflects the anisotropic geometry of avgpool features — dimensions with "
    "larger variance dominate L2 distance calculation in a way that is "
    "better aligned with class boundaries.",
    icon="📐", style="info",
)
finding_box(
    "<b>With DeiT-S, Euclidean and Manhattan are interchangeable.</b> "
    "The CLS token from ViT self-attention produces an approximately isotropic "
    "feature distribution — all dimensions contribute equally. "
    "L2 and L1 distances converge, making the choice irrelevant for DeiT-S-based models.",
    icon="✅", style="success",
)
caveat_box(
    "ProtoNet Manhattan results use the same episodic training as Euclidean "
    "(distance is applied only at inference). The gap therefore reflects "
    "purely the geometric properties of the learned feature space."
)
