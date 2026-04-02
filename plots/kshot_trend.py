"""
plots/kshot_trend.py — K-shot trend plot.

Line chart: X = number of shots (1, 3, 5)
            Y = mean accuracy (%)
One line per FSL method. Error bars = 95% CI.
Horizontal dashed line = K=5 fine-tune supervised baseline (direct comparator).

Usage:
    python plots/kshot_trend.py
Output:
    results/birds20/comparison/kshot_trend.png
"""
import os
import json
import math
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor']   = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES  = os.path.join(ROOT, 'results', 'birds20')
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


# ---------------------------------------------------------------------------
# Load all FSL results
# ---------------------------------------------------------------------------
shots = [1, 3, 5]

methods = {
    # ResNet-50 NCC baselines
    'NCC/R50 (Euclidean)':       {k: load(f'baseline/resnet50/base15/novel_ncc/{k}shot_euclidean/summary.json') for k in shots},
    'NCC/R50 (Manhattan)':       {k: load(f'baseline/resnet50/base15/novel_ncc/{k}shot_manhattan/summary.json') for k in shots},
    # DeiT-S NCC baselines
    'NCC/DeiT-S (Euclidean)':    {k: load(f'vit/baseline/base15/novel_ncc/{k}shot_euclidean/summary.json') for k in shots},
    'NCC/DeiT-S (Manhattan)':    {k: load(f'vit/baseline/base15/novel_ncc/{k}shot_manhattan/summary.json') for k in shots},
    # ResNet-50 FSL methods
    'ProtoNet/R50 (Euclidean)':  {k: load(f'proto_net/base15_novel5/{k}shot_euclidean/summary.json') for k in shots},
    'ProtoNet/R50 (Manhattan)':  {k: load(f'proto_net/base15_novel5/{k}shot_manhattan/summary.json') for k in shots},
    'MatchingNet/R50':           {k: load(f'matching_net/base15_novel5/{k}shot/summary.json') for k in shots},
    'MatchingNet+FCE/R50':       {k: load(f'matching_net_fce/base15_novel5/{k}shot/summary.json') for k in shots},
    'RelationNet/R50':           {k: load(f'relation_net/base15_novel5/{k}shot/summary.json') for k in shots},
    # DeiT-S FSL methods
    'ProtoNet/DeiT-S (Euclidean)':{k: load(f'vit/proto_net/base15_novel5/{k}shot_euclidean/summary.json') for k in shots},
    'ProtoNet/DeiT-S (Manhattan)':{k: load(f'vit/proto_net/base15_novel5/{k}shot_manhattan/summary.json') for k in shots},
}

# K=5 fine-tune supervised baseline (direct comparator — fixed horizontal line)
kshot5_sup = load('baseline/resnet50/base15/novel_kshot_5/summary.json')
kshot5_acc = float(kshot5_sup.get('accuracy') or kshot5_sup.get('test_accuracy')) * 100 if kshot5_sup else None

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
colors = [
    '#fd8d3c', '#d94801',        # R50 NCC
    '#9467bd', '#c5b0d5',        # DeiT-S NCC
    '#1f77b4', '#aec7e8',        # R50 ProtoNet
    '#2ca02c', '#98df8a',        # R50 MatchingNet / FCE
    '#d62728',                   # R50 RelationNet
    '#17becf', '#9edae5',        # DeiT-S ProtoNet
]
markers    = ['o', 's', 'o', 's', 'o', 's', '^', '^', 'D', 'o', 's']
linestyles = ['--', ':', '--', ':', '-', '-.', '-', '--', '--', '-', '-.']

fig, ax = plt.subplots(figsize=(12, 6))

any_data = False

for (method_name, data_by_k), color, marker, ls in zip(
        methods.items(), colors, markers, linestyles):

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

# Horizontal dashed line: K=5 supervised fine-tune
if kshot5_acc is not None:
    ax.axhline(y=kshot5_acc, color='gray', linestyle=':', linewidth=1.8,
               label=f'K=5 Fine-tune Supervised ({kshot5_acc:.1f}%)')

# Random chance reference
ax.axhline(y=20, color='lightgray', linestyle='--', linewidth=1,
           label='Random chance (5-way, 20%)')

ax.set_xticks(shots)
ax.set_xticklabels([f'{k}-shot' for k in shots], fontsize=11)
ax.set_xlabel('Number of Support Examples per Class (K)', fontsize=12)
ax.set_ylabel('Mean Accuracy (%)', fontsize=12)
ax.set_ylim(0, 105)
ax.set_title('Few-Shot Accuracy vs Number of Shots — B15/N5\n'
             '(5-way, 5 novel bird species, 600 episodes, 95% CI error bars)',
             fontsize=11)
ax.legend(loc='lower right', fontsize=8, ncol=2)
ax.grid(axis='y', linestyle='--', alpha=0.4)

if not any_data:
    ax.text(0.5, 0.5, 'No results available yet.\nRun evaluate.py first.',
            ha='center', va='center', transform=ax.transAxes, fontsize=12,
            color='gray')

plt.tight_layout()
out_path = os.path.join(OUT, 'kshot_trend.png')
plt.savefig(out_path, dpi=150)
plt.close()
print(f"Saved: {out_path}")
