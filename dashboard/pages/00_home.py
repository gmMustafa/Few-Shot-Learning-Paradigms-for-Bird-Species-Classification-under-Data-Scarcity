"""
Home / Landing Page
Few-Shot Learning Paradigms for Bird Species Classification under Data Scarcity
Ghulam Mustafa | MSc AI Thesis | Pattern Recognition Lab (Informatik 5), FAU Erlangen-Nürnberg
"""

import os, sys
_DASH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DASH not in sys.path:
    sys.path.insert(0, _DASH)

import streamlit as st

_CSS = os.path.join(_DASH, "assets", "style.css")
if os.path.exists(_CSS):
    with open(_CSS) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from data.loader import load_master, get_fsl_results, filter_results

df     = get_fsl_results()
master = load_master()

best_row    = df.loc[df["accuracy"].idxmax()]
best_acc    = f"{best_row['accuracy']:.2f}%"
best_method = f"{best_row['method']} / {best_row['backbone']}"

cub_rows = df[df["dataset"] == "CUB-200"]
best_cub = f"{cub_rows['accuracy'].max():.2f}%" if not cub_rows.empty else "—"

n_configs = df.drop_duplicates(
    subset=["method","backbone","dataset","split","k_shot","distance"]
).shape[0]

# ── Header banner ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#003865 0%,#1a5fa8 100%);
            border-radius:12px;padding:28px 32px;margin-bottom:20px">
  <div style="font-size:11px;color:rgba(255,255,255,0.7);letter-spacing:1px;
              text-transform:uppercase;font-weight:600;margin-bottom:6px">
    MSc AI Thesis · Pattern Recognition Lab · FAU Erlangen-Nürnberg · 2025–2026
  </div>
  <h1 style="margin:0;font-size:24px;font-weight:800;color:#ffffff;letter-spacing:-.3px">
    🐦 Few-Shot Learning for Bird Species Classification
  </h1>
  <p style="margin:8px 0 0;color:rgba(255,255,255,0.85);font-size:14px">
    Ghulam Mustafa &nbsp;·&nbsp; Interactive results dashboard
  </p>
</div>
""", unsafe_allow_html=True)

# ── Metric cards ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

def _card(col, label, value, sub, border_color):
    col.markdown(f"""
<div style="background:#ffffff;border-left:4px solid {border_color};border-radius:8px;
            padding:16px 18px;box-shadow:0 1px 4px rgba(0,0,0,0.08)">
  <div style="font-size:11px;color:#6B7280;font-weight:700;letter-spacing:.6px;
              text-transform:uppercase">{label}</div>
  <div style="font-size:26px;font-weight:800;color:{border_color};margin:6px 0 4px">{value}</div>
  <div style="font-size:12px;color:#64748b">{sub}</div>
</div>""", unsafe_allow_html=True)

_card(c1, "Best Accuracy (Birds-20)", best_acc,    best_method,           "#003865")
_card(c2, "Best CUB-200 Result",      best_cub,    "NCC/DeiT-S · B150/N50 · 5-shot", "#FF6B35")
_card(c3, "Unique Experiment Configs",str(n_configs),"method × backbone × split × shot × distance","#0a9396")
_card(c4, "Episodes per Config",      "600",       "5-way K-shot · Wilson 95% CI",    "#ae2012")

st.markdown("<br>", unsafe_allow_html=True)

# ── Core findings ──────────────────────────────────────────────────────────────
st.markdown("""
<h3 style="color:#003865;border-bottom:2px solid #003865;padding-bottom:6px;margin-bottom:16px">
  🔑 Core Findings
</h3>
""", unsafe_allow_html=True)

findings = [
    ("#EFF6FF", "#1e40af", "#BFDBFE", "💡",
     "<b>Finding 1 — Backbone dominates algorithm.</b> NCC (training-free nearest-centroid) "
     "equals or outperforms episodically trained ProtoNet in 8 of 12 backbone/split/shot "
     "combinations. A stronger backbone beats a fancier training algorithm."),
    ("#FFFBEB", "#92400e", "#FDE68A", "📉",
     "<b>Finding 2 — FCE hurts MatchingNet.</b> Adding full-context embedding (bidirectional LSTM) "
     "reduces accuracy by −3.6 to −7.9 pp. Sequence context designed for Mini-ImageNet "
     "does not transfer to fine-grained bird domains."),
    ("#EFF6FF", "#1e40af", "#BFDBFE", "📐",
     "<b>Finding 3 — Distance gap is backbone-specific.</b> Euclidean outperforms Manhattan "
     "by ≈8 pp for ResNet-50 (anisotropic features), but the gap collapses to <0.1 pp "
     "for DeiT-S (near-isotropic CLS token)."),
    ("#F0FDF4", "#166534", "#BBF7D0", "💰",
     "<b>Finding 4 — 3-shot is the cost-effective threshold.</b> The average 3→5-shot gain "
     "is <1.2 pp. Collecting 3 field photos per species captures ≥99% of 5-shot "
     "performance at 40% lower data cost."),
    ("#EFF6FF", "#1e40af", "#BFDBFE", "📊",
     "<b>Finding 5 — Rankings preserved across benchmarks.</b> The 8–14 pp drop from "
     "Birds-20 to CUB-200 (10× more novel classes, field photography) does not "
     "alter method ordering."),
]

for bg, text_color, border, icon, text in findings:
    st.markdown(f"""
<div style="background:{bg};border:1px solid {border};border-radius:8px;
            padding:12px 16px;margin:8px 0;font-size:14px;line-height:1.6;color:{text_color}">
  {icon}&nbsp; {text}
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Navigation guide ───────────────────────────────────────────────────────────
st.markdown("""
<h3 style="color:#003865;border-bottom:2px solid #003865;padding-bottom:6px;margin-bottom:16px">
  🗺️ Dashboard Pages
</h3>
""", unsafe_allow_html=True)

pages = [
    ("📊", "01 · Method Ranking Overview",       "All methods ranked by accuracy at chosen K-shot and split."),
    ("📈", "02 · K-Shot Efficiency",             "Accuracy vs K (1,3,5) — visualising diminishing returns."),
    ("🔬", "03 · Backbone Comparison",           "ResNet-50 vs DeiT-S — quantifying the backbone advantage."),
    ("📐", "04 · Distance Metric Ablation",      "Euclidean vs Manhattan — when the metric choice matters."),
    ("🐦", "05 · Per-Class Difficulty Heatmap",  "Which species are hardest? Kestrel vs Goldfinch."),
    ("🌍", "06 · Cross-Benchmark Comparison",    "Birds-20 B15/N5 vs CUB-200 B150/N50 — generalisation test."),
    ("📚", "07 · Literature Comparison",         "Our results vs published SOTA on CUB B100/N50."),
]

cols = st.columns(2)
for i, (icon, title, desc) in enumerate(pages):
    with cols[i % 2]:
        st.markdown(f"""
<div style="background:#ffffff;border-left:4px solid #003865;border-radius:8px;
            padding:12px 16px;margin:6px 0;box-shadow:0 1px 3px rgba(0,0,0,0.06)">
  <div style="font-weight:700;font-size:14px;color:#003865">{icon} {title}</div>
  <div style="font-size:12px;color:#64748b;margin-top:3px">{desc}</div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Datasets section ───────────────────────────────────────────────────────────
st.markdown("""
<h3 style="color:#003865;border-bottom:2px solid #003865;padding-bottom:6px;margin-bottom:16px">
  🗂️ Datasets
</h3>
""", unsafe_allow_html=True)

_THESIS_DIR = os.path.dirname(_DASH)   # repository root

col_b20, col_cub = st.columns(2)

with col_b20:
    st.markdown("""
<div style="background:#ffffff;border-left:4px solid #003865;border-radius:8px;
            padding:16px 18px;box-shadow:0 1px 4px rgba(0,0,0,0.08);margin-bottom:12px">
  <div style="font-size:13px;font-weight:700;color:#003865;margin-bottom:8px">
    🦅 Birds-20
  </div>
  <table style="font-size:12px;color:#1e293b;width:100%;border-collapse:collapse">
    <tr><td style="padding:2px 8px 2px 0;color:#6B7280;white-space:nowrap">Classes</td>
        <td><b>20 bird species</b> (curated subset)</td></tr>
    <tr><td style="padding:2px 8px 2px 0;color:#6B7280">Split</td>
        <td>15 base + 5 novel &nbsp;|&nbsp; 10 base + 10 novel</td></tr>
    <tr><td style="padding:2px 8px 2px 0;color:#6B7280">Train</td>
        <td>~130–190 images / class (~3,150 total)</td></tr>
    <tr><td style="padding:2px 8px 2px 0;color:#6B7280">Val / Test</td>
        <td>5 images / class (100 each) — deliberately scarce</td></tr>
    <tr><td style="padding:2px 8px 2px 0;color:#6B7280">Novel classes</td>
        <td>Bittern · Coot · Flamingo · Goldfinch · Kestrel</td></tr>
    <tr><td style="padding:2px 8px 2px 0;color:#6B7280">Source</td>
        <td>Subset of CUB-200-2011, studio photography</td></tr>
    <tr><td style="padding:2px 8px 2px 0;color:#6B7280">Purpose</td>
        <td>Controlled FSL benchmark, high base-class accuracy possible</td></tr>
  </table>
</div>
""", unsafe_allow_html=True)

    # Sample grid image — Birds-20
    _B20_SAMPLE = os.path.join(_THESIS_DIR, "notebooks", "fsl_all_classes_sample_grid.png")
    if os.path.exists(_B20_SAMPLE):
        st.image(_B20_SAMPLE, caption="Birds-20 · sample images", width="stretch")

with col_cub:
    st.markdown("""
<div style="background:#ffffff;border-left:4px solid #FF6B35;border-radius:8px;
            padding:16px 18px;box-shadow:0 1px 4px rgba(0,0,0,0.08);margin-bottom:12px">
  <div style="font-size:13px;font-weight:700;color:#FF6B35;margin-bottom:8px">
    🌿 CUB-200-2011
  </div>
  <table style="font-size:12px;color:#1e293b;width:100%;border-collapse:collapse">
    <tr><td style="padding:2px 8px 2px 0;color:#6B7280;white-space:nowrap">Classes</td>
        <td><b>200 North American bird species</b></td></tr>
    <tr><td style="padding:2px 8px 2px 0;color:#6B7280">Split</td>
        <td>100 base + 50 novel &nbsp;|&nbsp; 150 base + 50 novel</td></tr>
    <tr><td style="padding:2px 8px 2px 0;color:#6B7280">Train</td>
        <td>~30 images / class (~5,994 total in base)</td></tr>
    <tr><td style="padding:2px 8px 2px 0;color:#6B7280">Images</td>
        <td>11,788 total, field photography with natural backgrounds</td></tr>
    <tr><td style="padding:2px 8px 2px 0;color:#6B7280">Challenge</td>
        <td>Fine-grained: 50 novel classes, subtle inter-class differences</td></tr>
    <tr><td style="padding:2px 8px 2px 0;color:#6B7280">Source</td>
        <td>Caltech-UCSD Birds-200-2011 (Wah et al., 2011)</td></tr>
    <tr><td style="padding:2px 8px 2px 0;color:#6B7280">Purpose</td>
        <td>Stress-test: 10× more novel classes, realistic field conditions</td></tr>
  </table>
</div>
""", unsafe_allow_html=True)

    # Sample grid image — CUB-200
    _CUB_SAMPLE = os.path.join(_THESIS_DIR, "notebooks", "cub_sample_images.png")
    if os.path.exists(_CUB_SAMPLE):
        st.image(_CUB_SAMPLE, caption="CUB-200 · sample images", width="stretch")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:8px;
            padding:12px 16px;font-size:13px;color:#92400e">
  ⚠️ All FSL results use 5-way K-shot evaluation over <b>600 episodes</b> with
  95% Wilson confidence intervals. Direct comparison with literature is approximate —
  backbone capacity, pre-training source, and augmentation strategies differ.
</div>""", unsafe_allow_html=True)

# ── Sidebar summary ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="text-align:center;padding:12px 0">
  <div style="font-size:32px">🐦</div>
  <div style="font-weight:700;font-size:15px;color:#003865">FSL Dashboard</div>
  <div style="font-size:11px;color:#6B7280;margin-top:2px">Ghulam Mustafa · FAU · 2026</div>
</div>
<hr style="border:none;border-top:1px solid #CBD5E1;margin:12px 0">
""", unsafe_allow_html=True)
    st.metric("FSL result rows", len(df))
    st.metric("Unique configs",  n_configs)
    st.metric("Datasets", 2)
    st.markdown("<hr style='border:none;border-top:1px solid #CBD5E1'>", unsafe_allow_html=True)
    st.caption("Use the sidebar in each page to filter results.")
