"""
plots/kshot_trend.py — K-shot trend plot.

Line chart: X = number of shots (1, 3, 5)
            Y = mean accuracy (%)
One line per FSL method. Error bars = 95% CI.
Horizontal dashed line = K=5 fine-tune supervised baseline (direct comparator).

Usage:
    python plots/kshot_trend.py
Output:
    results/comparison/kshot_trend.png
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
RES  = os.path.join(ROOT, 'results')
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
    # NCC baselines — frozen backbone, no episodic training
    'NCC (Euclidean)':      {k: load(f'baseline/resnet50/base15/novel_ncc/{k}shot_euclidean/summary.json') for k in shots},
    'NCC (Manhattan)':      {k: load(f'baseline/resnet50/base15/novel_ncc/{k}shot_manhattan/summary.json') for k in shots},
    # FSL methods
    'ProtoNet (Euclidean)': {k: load(f'proto_net/base15_novel5/{k}shot_euclidean/summary.json') for k in shots},
    'ProtoNet (Manhattan)': {k: load(f'proto_net/base15_novel5/{k}shot_manhattan/summary.json') for k in shots},
    'MatchingNet (Cosine)': {k: load(f'matching_net/base15_novel5/{k}shot/summary.json') for k in shots},
    'RelationNet (Learned)':{k: load(f'relation_net/base15_novel5/{k}shot/summary.json') for k in shots},
}

# K=5 fine-tune supervised baseline (direct comparator — fixed horizontal line)
kshot5_sup = load('baseline/resnet50/base15/novel_kshot_5/summary.json')
kshot5_acc = float(kshot5_sup.get('accuracy') or kshot5_sup.get('test_accuracy')) * 100 if kshot5_sup else None

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
colors     = ['#fd8d3c', '#d94801', '#1f77b4', '#aec7e8', '#2ca02c', '#d62728']
markers    = ['o', 's',  'o', 's',  '^', 'D']
linestyles = ['--', ':',  '-', '-.',  '-', '--']

fig, ax = plt.subplots(figsize=(9, 5))

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
ax.set_title('Few-Shot Accuracy vs Number of Shots\n'
             '(5-way, 5 novel bird species, 600 episodes, 95% CI error bars)',
             fontsize=11)
ax.legend(loc='lower right', fontsize=9)
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
