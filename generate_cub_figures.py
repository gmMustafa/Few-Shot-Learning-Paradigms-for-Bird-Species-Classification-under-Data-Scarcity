"""Generate 4 CUB figures for LaTeX thesis."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import shutil

# Paths
BASE = r".\Few-Shot-Learning-Paradigms-for-Bird-Species-Classification-under-Data-Scarcity"
CSV_PATH = os.path.join(BASE, "results", "cub", "analysis", "cub_summary.csv")
OUT_DIR = os.path.join(BASE, "latex", "figures")

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH, index_col=0)
df.columns = ["Split", "Method", "KShot", "Acc", "CI", "F1", "Prec", "Rec", "AUC"]

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

COLORS = {
    "NCC/R50 (Euc)":        "#1f77b4",
    "NCC/R50 (Man)":        "#aec7e8",
    "NCC/DeiT-S (Euc)":    "#d62728",
    "NCC/DeiT-S (Man)":    "#ff9896",
    "ProtoNet/R50 (Euc)":  "#2ca02c",
    "ProtoNet/R50 (Man)":  "#98df8a",
    "ProtoNet/DeiT-S (Euc)":"#ff7f0e",
    "ProtoNet/DeiT-S (Man)":"#ffbb78",
    "MatchingNet":          "#9467bd",
    "MatchingNet+FCE":      "#c5b0d5",
    "RelationNet":          "#8c564b",
}

# ── Figure 1: K-Shot Trend (NCC & ProtoNet best methods, both splits) ──────────
methods_trend = [
    "NCC/DeiT-S (Euc)", "NCC/R50 (Euc)",
    "ProtoNet/DeiT-S (Euc)", "ProtoNet/R50 (Euc)",
    "MatchingNet", "RelationNet",
]
shots = [1, 3, 5]
splits = ["B100", "B150"]
split_labels = {"B100": "B100/N50", "B150": "B150/N50"}

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
for ax, split in zip(axes, splits):
    for method in methods_trend:
        rows = df[(df["Split"] == split) & (df["Method"] == method)].sort_values("KShot")
        if rows.empty:
            continue
        accs = rows["Acc"].values
        ax.plot(shots, accs, marker="o", linewidth=1.8, color=COLORS.get(method, "gray"),
                label=method)
    ax.set_xticks(shots)
    ax.set_xlabel("K (Support Shots)")
    ax.set_ylabel("5-way Accuracy (%)")
    ax.set_title(f"CUB-200-2011 {split_labels[split]}")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(40, 96)

fig.suptitle("CUB-200-2011: K-Shot Accuracy Trends by Method and Split", fontsize=13, y=1.01)
fig.tight_layout()
out1 = os.path.join(OUT_DIR, "cub_kshot_trend.png")
fig.savefig(out1, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"Saved: {out1}")

# ── Figure 2: 5-Shot Bar Comparison (both splits) ─────────────────────────────
all_methods = list(df["Method"].unique())
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, split in zip(axes, splits):
    sub = df[(df["Split"] == split) & (df["KShot"] == 5)].copy()
    sub = sub.set_index("Method").reindex(all_methods).dropna()
    colors = [COLORS.get(m, "gray") for m in sub.index]
    bars = ax.barh(sub.index, sub["Acc"], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title(f"5-Shot: {split_labels[split]}")
    ax.set_xlim(40, 100)
    ax.axvline(x=90, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    for bar, val in zip(bars, sub["Acc"]):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8.5)
    ax.grid(axis="x", alpha=0.3)

fig.suptitle("CUB-200-2011: 5-Shot Method Comparison (Euclidean unless noted)", fontsize=13)
fig.tight_layout()
out2 = os.path.join(OUT_DIR, "cub_comparison_bar.png")
fig.savefig(out2, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"Saved: {out2}")

# ── Figure 3: CUB vs Birds-20 Cross-Benchmark Comparison ──────────────────────
# Birds-20 best results (B15/N5, Euclidean, 5-shot) - from thesis
birds20_data = {
    "NCC/DeiT-S":    99.99,
    "NCC/R50":       98.14,
    "ProtoNet/DeiT-S": 99.20,
    "ProtoNet/R50":  99.80,
    "MatchingNet":   70.87,
    "RelationNet":   78.75,
}
# CUB B150/N50 Euclidean 5-shot
cub_b150_5shot_euc = {
    "NCC/DeiT-S":    90.57,
    "NCC/R50":       89.97,
    "ProtoNet/DeiT-S": 87.65,
    "ProtoNet/R50":  85.59,
    "MatchingNet":   60.86,
    "RelationNet":   59.54,
}
method_labels = list(birds20_data.keys())
x = np.arange(len(method_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 5))
bars1 = ax.bar(x - width/2, [birds20_data[m] for m in method_labels],
               width, label="Birds-20 B15/N5", color="#1f77b4", alpha=0.85)
bars2 = ax.bar(x + width/2, [cub_b150_5shot_euc[m] for m in method_labels],
               width, label="CUB-200-2011 B150/N50", color="#ff7f0e", alpha=0.85)

for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.1f}",
            ha="center", va="bottom", fontsize=8)
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.1f}",
            ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(method_labels, rotation=15, ha="right")
ax.set_ylabel("5-way 5-shot Accuracy (%)")
ax.set_title("Cross-Benchmark Comparison: Birds-20 vs CUB-200-2011 (5-shot, Euclidean)")
ax.legend()
ax.set_ylim(50, 108)
ax.axhline(y=90, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="90% line")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
out3 = os.path.join(OUT_DIR, "cub_vs_birds20_comparison.png")
fig.savefig(out3, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"Saved: {out3}")

# ── Figure 4: Distance Metric Comparison (Euc vs Man, both splits, NCC & Proto) ─
metric_methods = [
    ("NCC/R50", "NCC/R50 (Euc)", "NCC/R50 (Man)"),
    ("NCC/DeiT-S", "NCC/DeiT-S (Euc)", "NCC/DeiT-S (Man)"),
    ("ProtoNet/R50", "ProtoNet/R50 (Euc)", "ProtoNet/R50 (Man)"),
    ("ProtoNet/DeiT-S", "ProtoNet/DeiT-S (Euc)", "ProtoNet/DeiT-S (Man)"),
]
birds20_metric = {
    ("NCC/R50", "Euc"):          98.14,
    ("NCC/R50", "Man"):          84.00,
    ("NCC/DeiT-S", "Euc"):      99.99,
    ("NCC/DeiT-S", "Man"):      99.98,
    ("ProtoNet/R50", "Euc"):    99.80,
    ("ProtoNet/R50", "Man"):    85.95,
    ("ProtoNet/DeiT-S", "Euc"): 99.20,
    ("ProtoNet/DeiT-S", "Man"): 99.15,
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
for ax, kshot in zip(axes, [1, 3, 5]):
    x = np.arange(len(metric_methods))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for i, (label, euc_key, man_key) in enumerate(metric_methods):
        for split_i, (split, split_label, color_offset) in enumerate([
            ("B100", "B100/N50", 0), ("B150", "B150/N50", 1)
        ]):
            for metric_i, (metric_key, linestyle, hatch) in enumerate([
                ("Euc", "-", ""), ("Man", "--", "//")
            ]):
                m_key = f"{label} ({metric_key})"
                row = df[(df["Split"] == split) & (df["Method"] == m_key) & (df["KShot"] == kshot)]
                if row.empty:
                    continue
                acc = row["Acc"].values[0]
                bar_x = x[i] + (split_i * 2 + metric_i - 1.5) * width
                color = "#1f77b4" if "R50" in label else "#d62728"
                alpha = 0.9 if split == "B150" else 0.5
                ax.bar(bar_x, acc, width * 0.9, color=color, alpha=alpha,
                       hatch=hatch, edgecolor="gray", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metric_methods], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{kshot}-shot")
    ax.set_ylim(60, 100)
    ax.grid(axis="y", alpha=0.3)

# Legend
patch_b100_euc = mpatches.Patch(facecolor="#1f77b4", alpha=0.5, label="NCC/R50 B100 Euc")
patch_b150_euc = mpatches.Patch(facecolor="#1f77b4", alpha=0.9, label="NCC/R50 B150 Euc")
patch_b100_man = mpatches.Patch(facecolor="#1f77b4", alpha=0.5, hatch="//", label="B100 Man")
patch_b150_man = mpatches.Patch(facecolor="#1f77b4", alpha=0.9, hatch="//", label="B150 Man")
patch_deit    = mpatches.Patch(facecolor="#d62728", alpha=0.9, label="DeiT-S methods")
fig.legend(handles=[patch_b100_euc, patch_b150_euc, patch_b100_man, patch_b150_man, patch_deit],
           loc="upper center", ncol=5, fontsize=8.5, bbox_to_anchor=(0.5, 1.04))
fig.suptitle("CUB-200-2011: Euclidean vs Manhattan Distance Comparison", fontsize=13, y=1.08)
fig.tight_layout()
out4 = os.path.join(OUT_DIR, "cub_distance_metric_comparison.png")
fig.savefig(out4, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"Saved: {out4}")

print("\nAll 4 CUB figures generated successfully.")
