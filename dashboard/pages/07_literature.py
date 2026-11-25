"""
Page 7 — Literature Comparison
Our CUB B100/N50 results vs published SOTA.
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
from components.charts import bar_chart, _BASE_LAYOUT, FAU_BLUE, FAU_ORANGE, PALETTE
from components.cards import finding_box, caveat_box, section_header, metric_cards_row

_CSS = os.path.join(_DASH, "assets", "style.css")
if os.path.exists(_CSS):
    with open(_CSS) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#003865,#1e4e8c);border-radius:10px;
            padding:20px 24px;color:white;margin-bottom:16px">
  <h2 style="margin:0;font-size:20px">📚 Literature Comparison</h2>
  <p style="margin:4px 0 0;opacity:.85;font-size:13px">
    CUB-200 B100/N50 · 5-way K-shot results vs published methods
  </p>
</div>
""", unsafe_allow_html=True)

# ── Disclaimer banner ──────────────────────────────────────────────────────────
st.markdown("""
<div style="
  background:#FFFBEB;border:1px solid #FCD34D;border-radius:8px;
  padding:14px 18px;font-size:13px;margin-bottom:16px;
">
  ⚠️ <b>Important caveat:</b> Direct comparison is approximate.
  Backbone capacity, pre-training data source, data augmentation, and implementation
  details differ across studies. These numbers contextualise rather than supersede
  purpose-built comparisons.
</div>
""", unsafe_allow_html=True)

# ── Published CUB-200 5-way results, copied from the cited papers ─────────────
# Not part of the submitted thesis. Sources:
#   Chen et al. 2019 (ICLR), "A Closer Look at Few-Shot Classification":
#       Table 2 (Conv-4) and Table A5 (ResNet-18); ProtoNet is their re-implementation.
#       (Snell et al. 2017 reported only zero-shot results on CUB.)
#   Ye et al. 2020 (CVPR), FEAT: Table 8, ConvNet backbone.
#   Zhang et al., DeepEMD (arXiv:2003.06777, extended version): Table (c) CUB, DeepEMD-FCN.
LITERATURE = [
    {
        "Reference":      "ProtoNet / Conv-4 [Chen 2019]",
        "Backbone":       "Conv-4",
        "Training":       "Episodic",
        "1-shot (%)":     50.46,
        "5-shot (%)":     76.39,
        "Source":         "published",
        "Note":           "Re-implementation by Chen et al. (Table 2)",
    },
    {
        "Reference":      "ProtoNet / ResNet-18 [Chen 2019]",
        "Backbone":       "ResNet-18",
        "Training":       "Episodic",
        "1-shot (%)":     72.99,
        "5-shot (%)":     86.64,
        "Source":         "published",
        "Note":           "Re-implementation by Chen et al. (Table A5)",
    },
    {
        "Reference":      "Baseline++ [Chen 2019]",
        "Backbone":       "ResNet-18",
        "Training":       "Supervised + cosine classifier",
        "1-shot (%)":     67.02,
        "5-shot (%)":     83.58,
        "Source":         "published",
        "Note":           "Strong fine-tuning baseline (Table A5)",
    },
    {
        "Reference":      "FEAT [Ye 2020]",
        "Backbone":       "ConvNet",
        "Training":       "Episodic + transformer",
        "1-shot (%)":     68.87,
        "5-shot (%)":     82.90,
        "Source":         "published",
        "Note":           "Embedding adaptation with a transformer (Table 8)",
    },
    {
        "Reference":      "DeepEMD [Zhang 2020]",
        "Backbone":       "ResNet-12",
        "Training":       "Earth Mover's Distance",
        "1-shot (%)":     77.14,
        "5-shot (%)":     88.98,
        "Source":         "published",
        "Note":           "DeepEMD-FCN, optimal transport matching",
    },
]

# ── Our results from actual data ─────────────────────────────────────────────
df = get_fsl_results()

def _get(method, shot):
    sub = filter_results(df, method=method, split="B100/N50",
                         shot=shot, dataset="CUB-200", distance="Euclidean")
    if sub.empty:
        return np.nan, np.nan
    row = sub.iloc[0]
    return row["accuracy"], row["ci"]

our_results = []
for method_name, backbone_label, training_label in [
    ("ProtoNet",    "ResNet-50", "Episodic (NeurIPS 2017 protocol)"),
    ("NCC (R50)",   "ResNet-50", "No episodic training (NCC)"),
    ("NCC (DeiT-S)","DeiT-S",   "No episodic training (NCC)"),
]:
    a1, ci1 = _get(method_name, 1)
    a5, ci5 = _get(method_name, 5)
    our_results.append({
        "Reference":  f"Ours — {method_name}",
        "Backbone":   backbone_label,
        "Training":   training_label,
        "1-shot (%)": round(a1, 2) if not np.isnan(a1) else None,
        "5-shot (%)": round(a5, 2) if not np.isnan(a5) else None,
        "Source":     "ours",
        "Note":       "ImageNet-pretrained, 600 episodes, Wilson CI",
    })

lit_df  = pd.DataFrame(LITERATURE)
ours_df = pd.DataFrame(our_results)
full_df = pd.concat([lit_df, ours_df], ignore_index=True)

# ── Metric highlight cards ─────────────────────────────────────────────────────
best_ours_1 = ours_df["1-shot (%)"].max()
best_ours_5 = ours_df["5-shot (%)"].max()
best_pub_5  = lit_df["5-shot (%)"].max()
best_pub_1  = lit_df["1-shot (%)"].max()

metric_cards_row([
    {"label": "Our best 1-shot (CUB B100/N50)", "value": f"{best_ours_1:.2f}%",
     "sub": "NCC / DeiT-S",  "color": FAU_ORANGE},
    {"label": "Our best 5-shot (CUB B100/N50)", "value": f"{best_ours_5:.2f}%",
     "sub": "NCC / DeiT-S",  "color": FAU_ORANGE},
    {"label": "Best published 5-shot",          "value": f"{best_pub_5:.2f}%",
     "sub": "DeepEMD [Zhang 2020]", "color": "#6B7280"},
    {"label": "Our 5-shot vs DeepEMD",
     "value": f"{best_ours_5 - best_pub_5:+.2f} pp",
     "sub": "positive = we win", "color": "#0a9396"},
])
st.markdown("<br>", unsafe_allow_html=True)

# ── Combined bar chart ─────────────────────────────────────────────────────────
section_header("1-shot and 5-shot Results — Published vs Ours",
               "CUB-200 5-way evaluation · our rows highlighted in orange")

melt = full_df.melt(
    id_vars=["Reference","Backbone","Source"],
    value_vars=["1-shot (%)","5-shot (%)"],
    var_name="shot_label",
    value_name="accuracy",
).dropna(subset=["accuracy"])

color_map = {}
for ref in full_df["Reference"].unique():
    src = full_df[full_df["Reference"]==ref]["Source"].iloc[0]
    color_map[ref] = FAU_ORANGE if src == "ours" else FAU_BLUE

# Order: literature first (chronological), then ours
order_refs = [r["Reference"] for r in LITERATURE] + [r["Reference"] for r in our_results]

melt["Reference"] = pd.Categorical(melt["Reference"], categories=order_refs, ordered=True)
melt = melt.sort_values(["shot_label","Reference"])

fig = go.Figure()
refs_seen: set = set()
for shot_label in ["1-shot (%)", "5-shot (%)"]:
    for ref in order_refs:
        sub = melt[(melt["Reference"]==ref) & (melt["shot_label"]==shot_label)]
        if sub.empty:
            continue
        src = full_df[full_df["Reference"]==ref]["Source"].iloc[0]
        color = FAU_ORANGE if src == "ours" else FAU_BLUE
        show  = ref not in refs_seen
        fig.add_trace(go.Bar(
            name=ref,
            x=[shot_label],
            y=[sub["accuracy"].iloc[0]],
            offsetgroup=ref,
            marker_color=color,
            marker_line_color="white" if src == "ours" else color,
            marker_line_width=2 if src == "ours" else 0,
            opacity=1.0 if src == "ours" else 0.75,
            showlegend=show,
            legendgroup=ref,
        ))
        refs_seen.add(ref)

fig.update_layout(barmode="group", **_BASE_LAYOUT, height=450)
fig.update_layout(
    xaxis=dict(title="Shot count"),
    yaxis=dict(title="Accuracy (%)", range=[40, 100]),
    legend=dict(orientation="v", x=1.01, y=1),
)
st.plotly_chart(fig, width="stretch")
st.caption(
    "Orange bars = our results. Blue bars = published methods. "
    "Our results use ImageNet-pretrained ResNet-50 or DeiT-S; "
    "published methods use task-specific training on CUB training split."
)

# ── Results table ──────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
section_header("Full Comparison Table")

# Reorder: literature first, then ours
tbl = full_df[["Reference","Backbone","Training","1-shot (%)","5-shot (%)","Note"]].copy()
tbl = tbl.set_index("Reference")

# Style ours rows
def _style_row(row):
    src = full_df[full_df["Reference"]==row.name]["Source"].iloc[0]
    if src == "ours":
        return ["background-color: #EFF6FF; font-weight: 600"] * len(row)
    return [""] * len(row)

styled = tbl.style.apply(_style_row, axis=1).format({
    "1-shot (%)": lambda x: f"{x:.2f}%" if pd.notnull(x) else "—",
    "5-shot (%)": lambda x: f"{x:.2f}%" if pd.notnull(x) else "—",
})

st.dataframe(styled, width="stretch", height=320)

# ── Contextual insights ────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

ncc_r50_5 = ours_df[ours_df["Reference"]=="Ours — NCC (R50)"]["5-shot (%)"].iloc[0]
bl_5 = lit_df[lit_df["Reference"]=="Baseline++ [Chen 2019]"]["5-shot (%)"].iloc[0]
ncc_deit_5 = ours_df[ours_df["Reference"]=="Ours — NCC (DeiT-S)"]["5-shot (%)"].iloc[0]
deep_5 = lit_df[lit_df["Reference"]=="DeepEMD [Zhang 2020]"]["5-shot (%)"].iloc[0]

finding_box(
    f"<b>NCC/R50 is consistent with Baseline++</b> at similar backbone scale: "
    f"{ncc_r50_5:.2f}% vs {bl_5:.2f}% at 5-shot. This is notable because NCC requires "
    "zero episodic training — the ResNet-50 feature extractor alone provides "
    "competitive performance.",
    icon="📌", style="info",
)
finding_box(
    f"<b>NCC/DeiT-S ({ncc_deit_5:.2f}%) is on par with a task-specific method</b> "
    f"(DeepEMD {deep_5:.2f}%) without any episodic training, specialised architecture "
    "or optimal transport solver. The stronger ImageNet-pretrained backbone does most of the work.",
    icon="🏆", style="success",
)
proto_r50_5 = ours_df[ours_df["Reference"]=="Ours — ProtoNet"]["5-shot (%)"]
proto_r50_txt = f"{proto_r50_5.iloc[0]:.2f}%" if len(proto_r50_5) else "our ProtoNet/R50"
finding_box(
    f"<b>Our ProtoNet/R50 ({proto_r50_txt} at 5-shot) beats ProtoNet with a Conv-4 backbone "
    "(76.39%) but trails Chen et al.'s ProtoNet with ResNet-18 (86.64%).</b> "
    "Pretraining, image resolution and the class split differ, so the gap is indicative only.",
    icon="💡", style="info",
)
