"""
plots/kshot_trend_cub.py — CUB-200-2011 K-shot trend plot.

Two subplots side by side:
  Left  = b100_n50 (Base-100 / Novel-50, standard benchmark split)
  Right = b150_n50 (Base-150 / Novel-50, richer-base split)

For each subplot:
  X = number of shots (1, 3, 5)
  Y = mean accuracy (%)
  One line per FSL method, error bars = 95% CI.
  Horizontal dashed line = K=5 fine-tune supervised baseline.
  Horizontal line        = random chance (5-way = 20%).

Usage:
    python plots/kshot_trend_cub.py
Output:
    results/cub/comparison/kshot_trend_cub.png
"""
import os
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

plt.rcParams['figure.facecolor']  = 'white'
plt.rcParams['axes.facecolor']    = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES  = os.path.join(ROOT, 'results', 'cub')
OUT  = os.path.join(RES, 'comparison')
os.makedirs(OUT, exist_ok=True)


def load(path):
    full = os.path.join(RES, path)
    if not os.path.exists(full):
        return None
    with open(full) as f:
        return json.load(f)


def get_acc_ci(d):
    """Return (mean_acc_pct, ci95_pct) or (None, None) if missing."""
    if d is None:
        return None, None
    return float(d['mean_accuracy_pct']), float(d['ci95_pct'])


def get_baseline_acc(d):
    """Return accuracy % from a supervised baseline summary.json, or None."""
    if d is None:
        return None
    raw = d.get('accuracy') or d.get('test_accuracy')
    return float(raw) * 100 if raw is not None else None


# ---------------------------------------------------------------------------
# Load all FSL results for both splits
# ---------------------------------------------------------------------------
shots = [1, 3, 5]

methods_b100 = {
    # ResNet-50 NCC
    'NCC/R50 (Euclidean)': {
        k: load(f'baseline/resnet50_base100/novel_ncc/{k}shot_euclidean/summary.json')
        for k in shots
    },
    'NCC/R50 (Manhattan)': {
        k: load(f'baseline/resnet50_base100/novel_ncc/{k}shot_manhattan/summary.json')
        for k in shots
    },
    # DeiT-S NCC
    'NCC/DeiT-S (Euclidean)': {
        k: load(f'vit/baseline/base100/novel_ncc/{k}shot_euclidean/summary.json')
        for k in shots
    },
    'NCC/DeiT-S (Manhattan)': {
        k: load(f'vit/baseline/base100/novel_ncc/{k}shot_manhattan/summary.json')
        for k in shots
    },
    # ResNet-50 FSL
    'ProtoNet/R50 (Euclidean)': {
        k: load(f'proto_net/base100_novel50/{k}shot_euclidean/summary.json')
        for k in shots
    },
    'ProtoNet/R50 (Manhattan)': {
        k: load(f'proto_net/base100_novel50/{k}shot_manhattan/summary.json')
        for k in shots
    },
    'MatchingNet/R50': {
        k: load(f'matching_net/base100_novel50/{k}shot/summary.json')
        for k in shots
    },
    'MatchingNet+FCE/R50': {
        k: load(f'matching_net_fce/base100_novel50/{k}shot/summary.json')
        for k in shots
    },
    'RelationNet/R50': {
        k: load(f'relation_net/base100_novel50/{k}shot/summary.json')
        for k in shots
    },
    # DeiT-S ProtoNet
    'ProtoNet/DeiT-S (Euclidean)': {
        k: load(f'vit/proto_net/base100_novel50/{k}shot_euclidean/summary.json')
        for k in shots
    },
    'ProtoNet/DeiT-S (Manhattan)': {
        k: load(f'vit/proto_net/base100_novel50/{k}shot_manhattan/summary.json')
        for k in shots
    },
}

methods_b150 = {
    # ResNet-50 NCC
    'NCC/R50 (Euclidean)': {
        k: load(f'baseline/resnet50_base150/novel_ncc/{k}shot_euclidean/summary.json')
        for k in shots
    },
    'NCC/R50 (Manhattan)': {
        k: load(f'baseline/resnet50_base150/novel_ncc/{k}shot_manhattan/summary.json')
        for k in shots
    },
    # DeiT-S NCC
    'NCC/DeiT-S (Euclidean)': {
        k: load(f'vit/baseline/base150/novel_ncc/{k}shot_euclidean/summary.json')
        for k in shots
    },
    'NCC/DeiT-S (Manhattan)': {
        k: load(f'vit/baseline/base150/novel_ncc/{k}shot_manhattan/summary.json')
        for k in shots
    },
    # ResNet-50 FSL
    'ProtoNet/R50 (Euclidean)': {
        k: load(f'proto_net/base150_novel50/{k}shot_euclidean/summary.json')
        for k in shots
    },
    'ProtoNet/R50 (Manhattan)': {
        k: load(f'proto_net/base150_novel50/{k}shot_manhattan/summary.json')
        for k in shots
    },
    'MatchingNet/R50': {
        k: load(f'matching_net/base150_novel50/{k}shot/summary.json')
        for k in shots
    },
    'MatchingNet+FCE/R50': {
        k: load(f'matching_net_fce/base150_novel50/{k}shot/summary.json')
        for k in shots
    },
    'RelationNet/R50': {
        k: load(f'relation_net/base150_novel50/{k}shot/summary.json')
        for k in shots
    },
    # DeiT-S ProtoNet
    'ProtoNet/DeiT-S (Euclidean)': {
        k: load(f'vit/proto_net/base150_novel50/{k}shot_euclidean/summary.json')
        for k in shots
    },
    'ProtoNet/DeiT-S (Manhattan)': {
        k: load(f'vit/proto_net/base150_novel50/{k}shot_manhattan/summary.json')
        for k in shots
    },
}

# K=5 fine-tune supervised baselines (horizontal reference lines)
kshot5_b100_acc = get_baseline_acc(load('baseline/resnet50_base100/novel_kshot_5/summary.json'))
kshot5_b150_acc = get_baseline_acc(load('baseline/resnet50_base150/novel_kshot_5/summary.json'))

# ---------------------------------------------------------------------------
# Style — identical palette/markers to Birds20 kshot_trend.py
# ---------------------------------------------------------------------------
colors = [
    '#fd8d3c', '#d94801',        # R50 NCC (Euclidean / Manhattan)
    '#9467bd', '#c5b0d5',        # DeiT-S NCC
    '#1f77b4', '#aec7e8',        # R50 ProtoNet
    '#2ca02c', '#98df8a',        # R50 MatchingNet / FCE
    '#d62728',                   # R50 RelationNet
    '#17becf', '#9edae5',        # DeiT-S ProtoNet
]
markers    = ['o', 's', 'o', 's', 'o', 's', '^', '^', 'D', 'o', 's']
linestyles = ['--', ':', '--', ':', '-', '-.', '-', '--', '--', '-', '-.']


def plot_subplot(ax, methods_dict, kshot5_acc, title):
    any_data = False
    for (method_name, data_by_k), color, marker, ls in zip(
            methods_dict.items(), colors, markers, linestyles):

        xs, ys, errs = [], [], []
        for k in shots:
            acc, ci = get_acc_ci(data_by_k[k])
            if acc is not None:
                xs.append(k)
                ys.append(acc)
                errs.append(ci)

        if not xs:
            continue

        any_data = True
        ax.errorbar(xs, ys, yerr=errs, label=method_name,
                    color=color, marker=marker, linestyle=ls,
                    linewidth=2, markersize=7, capsize=5, capthick=1.5,
                    elinewidth=1.2)

    # K=5 supervised fine-tune reference line
    if kshot5_acc is not None:
        ax.axhline(y=kshot5_acc, color='steelblue', linestyle=':', linewidth=1.8,
                   label=f'K=5 Fine-tune Supervised ({kshot5_acc:.1f}%)')

    # Random chance
    ax.axhline(y=20, color='lightgray', linestyle='--', linewidth=1,
               label='Random chance (5-way, 20%)')

    ax.set_xticks(shots)
    ax.set_xticklabels([f'{k}-shot' for k in shots], fontsize=11)
    ax.set_xlabel('Number of Support Examples per Class (K)', fontsize=11)
    ax.set_ylabel('Mean Accuracy (%)', fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_title(title, fontsize=11)
    ax.legend(loc='lower right', fontsize=7.5, ncol=2)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    if not any_data:
        ax.text(0.5, 0.5, 'No results available yet.\nRun evaluate.py first.',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='gray')


# ---------------------------------------------------------------------------
# Draw figure with two side-by-side subplots
# ---------------------------------------------------------------------------
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20, 7), sharey=True)
fig.suptitle(
    'CUB-200-2011: Few-Shot Accuracy vs Number of Shots\n'
    '(5-way, 50 novel classes, 600 episodes, 95% CI error bars)',
    fontsize=13
)

plot_subplot(
    ax_left, methods_b100, kshot5_b100_acc,
    title='b100_n50 — Standard Split\n(Base-100 / Novel-50)'
)
plot_subplot(
    ax_right, methods_b150, kshot5_b150_acc,
    title='b150_n50 — Richer-Base Split\n(Base-150 / Novel-50)'
)

# Remove duplicate y-label on right subplot (shared axis)
ax_right.set_ylabel('')

plt.tight_layout()
out_path = os.path.join(OUT, 'kshot_trend_cub.png')
plt.savefig(out_path, dpi=150)
plt.close()
print(f"Saved: results/cub/comparison/kshot_trend_cub.png")
