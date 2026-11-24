"""
Page 5 — Per-Class Difficulty Heatmap
Which novel species are hardest / easiest across all methods and shots?
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

from data.loader import (
    get_fsl_results, load_perclass_data,
    get_perclass_birds20, get_perclass_cub,
    METHOD_ORDER, BACKBONE_COLOR
)
from components.charts import heatmap, bar_chart, FAU_BLUE, FAU_ORANGE, _BASE_LAYOUT
from components.filters import shot_filter, get_global_dataset
from components.cards import finding_box, section_header, metric_cards_row, caveat_box

_CSS = os.path.join(_DASH, "assets", "style.css")
if os.path.exists(_CSS):
    with open(_CSS) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## Filters")
ds   = get_global_dataset()
shot = shot_filter(key="pc_shot", default=5)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#003865,#1e4e8c);border-radius:10px;
            padding:20px 24px;color:white;margin-bottom:16px">
  <h2 style="margin:0;font-size:20px">🐦 Per-Class Difficulty Heatmap</h2>
  <p style="margin:4px 0 0;opacity:.85;font-size:13px">
    Which novel species are hardest to classify across all methods and shots?
  </p>
</div>
""", unsafe_allow_html=True)

# ── Load per-class data ────────────────────────────────────────────────────────
pc_df = load_perclass_data()

if pc_df.empty:
    st.error("Per-class data not found. Run evaluations to generate summary.json files.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
def _render_birds20(shot):
    pivot = get_perclass_birds20(split="B15/N5", shot=shot, distance="Euclidean")

    if pivot.empty:
        st.warning("No per-class data for Birds-20 B15/N5. Check that summary.json files exist.")
        return

    mean_per_species = pivot.mean(axis=0).sort_values()
    pivot = pivot[mean_per_species.index]

    section_header("Species Difficulty Ranking",
                   "Average accuracy across all methods — hardest on left")

    species_avg = pivot.mean(axis=0).reset_index()
    species_avg.columns = ["species", "avg_acc"]
    species_avg = species_avg.sort_values("avg_acc")

    colors = []
    for _, row in species_avg.iterrows():
        if row["avg_acc"] < 80:
            colors.append("#ae2012")
        elif row["avg_acc"] > 97:
            colors.append("#0a9396")
        else:
            colors.append(FAU_BLUE)

    fig_bar = go.Figure(go.Bar(
        x=species_avg["species"],
        y=species_avg["avg_acc"],
        marker_color=colors,
        text=[f"{v:.1f}%" for v in species_avg["avg_acc"]],
        textposition="outside",
    ))
    fig_bar.update_layout(**_BASE_LAYOUT, height=380)
    fig_bar.update_layout(
        yaxis=dict(range=[0, 110], title="Avg Accuracy (%)"),
        xaxis=dict(title="Novel Species"),
    )
    st.plotly_chart(fig_bar, width="stretch")

    hardest_sp = species_avg.iloc[0]
    easiest_sp = species_avg.iloc[-1]

    c1, c2 = st.columns(2)
    with c1:
        finding_box(
            f"<b>Hardest: {hardest_sp['species']}</b> — "
            f"avg {hardest_sp['avg_acc']:.1f}% across all methods. "
            "American Kestrel exhibits strong sexual dimorphism "
            "(males: rufous/blue; females: brown-streaked), causing high "
            "intra-class visual variance.",
            icon="🔴", style="danger",
        )
    with c2:
        finding_box(
            f"<b>Easiest: {easiest_sp['species']}</b> — "
            f"avg {easiest_sp['avg_acc']:.1f}% across all methods. "
            "American Goldfinch has highly distinctive bright yellow plumage "
            "in breeding males — a near-unique colour signature.",
            icon="🟢", style="success",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    section_header(
        f"Per-Class Accuracy Heatmap — {shot}-shot · Euclidean",
        "Rows = methods · Columns = novel species (hardest → easiest)"
    )

    fig_heat = heatmap(
        pivot, title=None, colorscale="RdYlGn", zmin=0, zmax=100,
        xlab="Novel Species", ylab="Method",
    )
    fig_heat.update_layout(height=max(280, 60 * len(pivot) + 100))
    st.plotly_chart(fig_heat, width="stretch")
    st.caption(
        "Cell values = per-class mean accuracy (%) over 600 episodes. "
        "Red = hard, Green = easy."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Method Comparison per Species")

    melt = pivot.reset_index().melt(id_vars="method", var_name="species",
                                     value_name="accuracy")
    melt = melt[melt["method"].isin([m for m in METHOD_ORDER if m in pivot.index])]
    melt["method"] = pd.Categorical(
        melt["method"],
        categories=[m for m in METHOD_ORDER if m in pivot.index],
        ordered=True,
    )
    melt = melt.sort_values("method")

    fig3 = bar_chart(
        melt, x="species", y="accuracy",
        color="method", title=None,
        xlab="Novel Species", ylab="Accuracy (%)",
    )
    fig3.update_layout(height=420)
    st.plotly_chart(fig3, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
def _render_cub(shot):
    cub_split_sel = st.radio(
        "CUB Split", ["B100/N50", "B150/N50"], horizontal=True, key="pc_cub_split"
    )

    pivot_cub = get_perclass_cub(split=cub_split_sel, shot=shot,
                                  distance="Euclidean", top_n=5)

    if pivot_cub.empty:
        st.warning("No CUB per-class data for selected filters.")
    else:
        n_species = len(pivot_cub.columns)

        st.markdown("<br>", unsafe_allow_html=True)
        section_header(
            f"Top-5 Hardest + Top-5 Easiest CUB Species — {cub_split_sel} · {shot}-shot",
            "Sorted hardest (left) → easiest (right)"
        )

        sp_avg = pivot_cub.mean(axis=0).reset_index()
        sp_avg.columns = ["species", "avg_acc"]

        colors_cub = []
        for _, row in sp_avg.iterrows():
            if row["avg_acc"] < 70:
                colors_cub.append("#ae2012")
            elif row["avg_acc"] > 90:
                colors_cub.append("#0a9396")
            else:
                colors_cub.append(FAU_BLUE)

        fig_sp = go.Figure(go.Bar(
            x=sp_avg["species"],
            y=sp_avg["avg_acc"],
            marker_color=colors_cub,
            text=[f"{v:.1f}%" for v in sp_avg["avg_acc"]],
            textposition="outside",
        ))
        fig_sp.update_layout(**_BASE_LAYOUT, height=400)
        fig_sp.update_layout(
            yaxis=dict(range=[0, 110], title="Avg Accuracy (%)"),
            xaxis=dict(title="Species", tickangle=-30),
        )
        st.plotly_chart(fig_sp, width="stretch")

        hard_row = sp_avg.iloc[0]
        easy_row = sp_avg.iloc[-1]

        finding_box(
            f"<b>Hardest CUB species: {hard_row['species']}</b> "
            f"({hard_row['avg_acc']:.1f}% avg). "
            "Fine-grained warblers and vireos with subtle plumage differences "
            "are the main failure mode — taxonomically similar yet visually overlapping.",
            icon="🔴", style="danger",
        )
        finding_box(
            f"<b>Easiest CUB species: {easy_row['species']}</b> "
            f"({easy_row['avg_acc']:.1f}% avg). "
            "Distinctive woodpeckers and cactus wrens with unique markings "
            "or habitat-specific appearances are reliably classified.",
            icon="🟢", style="success",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Per-Class Accuracy Heatmap (CUB Top-10 Species)")

        fig_heat_cub = heatmap(
            pivot_cub, title=None, colorscale="RdYlGn", zmin=0, zmax=100,
            xlab="Species (hardest → easiest)", ylab="Method",
        )
        fig_heat_cub.update_layout(height=max(280, 60 * len(pivot_cub) + 120))
        st.plotly_chart(fig_heat_cub, width="stretch")

    caveat_box(
        "CUB per-class accuracy is averaged over 600 5-way episodes. "
        "Each episode randomly samples 5 novel classes, so per-class frequency "
        "may vary slightly; only classes appearing in ≥50 episodes are shown."
    )


# ── Dataset-aware rendering ────────────────────────────────────────────────────
if ds == "Birds-20":
    _render_birds20(shot)

elif ds == "CUB-200":
    _render_cub(shot)

else:  # All
    tab_b20, tab_cub = st.tabs(["🦅 Birds-20 B15/N5", "🌿 CUB-200"])
    with tab_b20:
        _render_birds20(shot)
    with tab_cub:
        _render_cub(shot)
