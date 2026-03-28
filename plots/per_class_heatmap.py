"""
plots/per_class_heatmap.py — Per-class accuracy heatmap.

Rows    = FSL methods
Columns = 5 novel bird species
Values  = per-class mean accuracy (%) across all 600 episodes

One heatmap generated per k-shot (1, 3, 5).
Saved as: results/comparison/per_class_heatmap_{k}shot.png

Usage:
    python plots/per_class_heatmap.py
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES  = os.path.join(ROOT, 'results')
OUT  = os.path.join(RES, 'comparison')
os.makedirs(OUT, exist_ok=True)

NOVEL_CLASSES = [
    'AMERICAN BITTERN',
    'AMERICAN COOT',
    'AMERICAN FLAMINGO',
    'AMERICAN GOLDFINCH',
    'AMERICAN KESTREL',
]
# Short names for axis labels
SHORT_NAMES = [c.replace('AMERICAN ', 'AM. ') for c in NOVEL_CLASSES]


def load(path):
    full = os.path.join(RES, path)
    if not os.path.exists(full):
        return None
    with open(full) as f:
        return json.load(f)


def get_per_class(d):
    """Return list of per-class acc (%) in NOVEL_CLASSES order, or None."""
    if d is None:
        return None
    pca = d.get('per_class_accuracy_pct', {})
    return [pca.get(cls, float('nan')) for cls in NOVEL_CLASSES]


def build_heatmap(k_shot):
    # Each method: (label, path_to_summary)
    method_defs = [
        ('ProtoNet (Euclidean)', load(f'proto_net/base15_novel5/{k_shot}shot_euclidean/summary.json')),
        ('ProtoNet (Manhattan)', load(f'proto_net/base15_novel5/{k_shot}shot_manhattan/summary.json')),
        ('MatchingNet (Cosine)', load(f'matching_net/base15_novel5/{k_shot}shot/summary.json')),
        ('RelationNet (Learned)',load(f'relation_net/base15_novel5/{k_shot}shot/summary.json')),
    ]

    matrix = []
    row_labels = []
    any_data = False

    for method_name, d in method_defs:
        pca = get_per_class(d)
        if pca is None:
            pca = [float('nan')] * len(NOVEL_CLASSES)
        else:
            any_data = True
        matrix.append(pca)
        row_labels.append(method_name)

    matrix = np.array(matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 4))

    if not any_data:
        ax.text(0.5, 0.5, f'No {k_shot}-shot results available yet.',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='gray')
        ax.set_title(f'Per-Class Accuracy — {k_shot}-Shot')
    else:
        mask = np.isnan(matrix)
        sns.heatmap(matrix, annot=True, fmt='.1f', cmap='YlOrRd',
                    xticklabels=SHORT_NAMES, yticklabels=row_labels,
                    ax=ax, vmin=0, vmax=100,
                    linewidths=0.5, linecolor='white',
                    annot_kws={'size': 10},
                    mask=mask)
        ax.set_xlabel('Novel Class', fontsize=11)
        ax.set_ylabel('Method', fontsize=11)
        ax.set_title(f'Per-Class Accuracy (%) — 5-way {k_shot}-Shot\n'
                     f'(5 novel bird species, 600 evaluation episodes)',
                     fontsize=11)
        ax.tick_params(axis='x', labelrotation=15)
        ax.tick_params(axis='y', labelrotation=0)

    plt.tight_layout()
    out_path = os.path.join(OUT, f'per_class_heatmap_{k_shot}shot.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    for k in [1, 3, 5]:
        build_heatmap(k)
