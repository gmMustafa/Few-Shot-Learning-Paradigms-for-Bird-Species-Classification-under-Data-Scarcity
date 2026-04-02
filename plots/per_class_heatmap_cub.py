"""
plots/per_class_heatmap_cub.py — CUB-200-2011 per-class accuracy heatmap.

CUB has 50 novel classes — too many to display all at once.
Strategy: show only the 15 hardest classes (lowest mean accuracy across all
methods and shots that have data), sorted hardest -> easiest left to right.

Two splits are shown separately:
  b100_n50 : Base-100 / Novel-50 (standard benchmark split)
  b150_n50 : Base-150 / Novel-50 (richer-base split)

One heatmap per k-shot per split = 6 total PNG files.

File names:
    results/cub/comparison/per_class_heatmap_cub_1shot_b100.png
    results/cub/comparison/per_class_heatmap_cub_1shot_b150.png
    results/cub/comparison/per_class_heatmap_cub_3shot_b100.png
    results/cub/comparison/per_class_heatmap_cub_3shot_b150.png
    results/cub/comparison/per_class_heatmap_cub_5shot_b100.png
    results/cub/comparison/per_class_heatmap_cub_5shot_b150.png

Rows    = FSL methods (11 methods)
Columns = 15 hardest novel classes

Usage:
    python plots/per_class_heatmap_cub.py
"""
import os
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.facecolor']  = 'white'
plt.rcParams['axes.facecolor']    = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES  = os.path.join(ROOT, 'results', 'cub')
OUT  = os.path.join(RES, 'comparison')
os.makedirs(OUT, exist_ok=True)

N_HARDEST = 15   # number of hardest classes to display


def load(path):
    full = os.path.join(RES, path)
    if not os.path.exists(full):
        return None
    with open(full) as f:
        return json.load(f)


def get_per_class(d):
    """Return per_class_accuracy_pct dict, or {} if missing."""
    if d is None:
        return {}
    return d.get('per_class_accuracy_pct', {})


def abbrev_class(name):
    """
    Abbreviate a CUB class name to at most 12 characters.
    Strips common long suffixes (_Warbler, _Wren, _Sparrow, etc.)
    that are shared across many classes, then truncates.
    """
    suffixes = [
        '_Warbler', '_Wren', '_Sparrow', '_Flycatcher', '_Woodpecker',
        '_Thrush', '_Vireo', '_Finch', '_Bunting', '_Grosbeak',
        '_Swallow', '_Tern', '_Gull', '_Heron', '_Cormorant',
        '_Tanager', '_Oriole', '_Hawk', '_Crow', '_Jay',
    ]
    short = name
    for sfx in suffixes:
        if name.endswith(sfx):
            short = name[:-len(sfx)]
            break
    # Final truncation to 12 chars
    return short[:12]


def method_paths_for_split(split_tag, k):
    """
    Return list of (method_label, summary_dict) for a given split and k-shot.
    split_tag is either 'b100' or 'b150'.
    """
    if split_tag == 'b100':
        base_r50   = 'resnet50_base100'
        base_fsl   = 'base100_novel50'
        base_vit   = 'base100'
    else:
        base_r50   = 'resnet50_base150'
        base_fsl   = 'base150_novel50'
        base_vit   = 'base150'

    return [
        # ResNet-50 NCC
        ('NCC/R50 (Euclidean)',
         load(f'baseline/{base_r50}/novel_ncc/{k}shot_euclidean/summary.json')),
        ('NCC/R50 (Manhattan)',
         load(f'baseline/{base_r50}/novel_ncc/{k}shot_manhattan/summary.json')),
        # DeiT-S NCC
        ('NCC/DeiT-S (Euclidean)',
         load(f'vit/baseline/{base_vit}/novel_ncc/{k}shot_euclidean/summary.json')),
        ('NCC/DeiT-S (Manhattan)',
         load(f'vit/baseline/{base_vit}/novel_ncc/{k}shot_manhattan/summary.json')),
        # ResNet-50 FSL
        ('ProtoNet/R50 (Euclidean)',
         load(f'proto_net/{base_fsl}/{k}shot_euclidean/summary.json')),
        ('ProtoNet/R50 (Manhattan)',
         load(f'proto_net/{base_fsl}/{k}shot_manhattan/summary.json')),
        ('MatchingNet/R50',
         load(f'matching_net/{base_fsl}/{k}shot/summary.json')),
        ('MatchingNet+FCE/R50',
         load(f'matching_net_fce/{base_fsl}/{k}shot/summary.json')),
        ('RelationNet/R50',
         load(f'relation_net/{base_fsl}/{k}shot/summary.json')),
        # DeiT-S ProtoNet
        ('ProtoNet/DeiT-S (Euclidean)',
         load(f'vit/proto_net/{base_fsl}/{k}shot_euclidean/summary.json')),
        ('ProtoNet/DeiT-S (Manhattan)',
         load(f'vit/proto_net/{base_fsl}/{k}shot_manhattan/summary.json')),
    ]


def collect_all_class_accs(split_tag):
    """
    Collect per-class accuracies across all shots and all methods for a split.
    Returns a dict: class_name -> list of acc values (ignoring NaN).
    """
    all_accs = {}
    for k in [1, 3, 5]:
        for _label, d in method_paths_for_split(split_tag, k):
            pca = get_per_class(d)
            for cls, acc in pca.items():
                if cls not in all_accs:
                    all_accs[cls] = []
                if acc is not None and not (isinstance(acc, float) and np.isnan(acc)):
                    all_accs[cls].append(float(acc))
    return all_accs


def find_hardest_classes(split_tag, n=N_HARDEST):
    """
    Return the n class names with the lowest mean accuracy across all
    methods and shots, sorted hardest -> easiest (ascending mean acc).
    """
    all_accs = collect_all_class_accs(split_tag)
    if not all_accs:
        return []
    means = {cls: (np.mean(vals) if vals else 100.0) for cls, vals in all_accs.items()}
    sorted_classes = sorted(means.keys(), key=lambda c: means[c])
    return sorted_classes[:n]


def build_heatmap(k_shot, split_tag):
    """
    Build and save one heatmap for (k_shot, split_tag).
    split_tag: 'b100' or 'b150'
    """
    hardest = find_hardest_classes(split_tag, n=N_HARDEST)
    if not hardest:
        # No data available; create a placeholder
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.text(0.5, 0.5, f'No {k_shot}-shot results available yet for {split_tag}.',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='gray')
        ax.set_title(f'Per-Class Accuracy — {k_shot}-Shot — {split_tag}')
        plt.tight_layout()
        out_path = os.path.join(OUT, f'per_class_heatmap_cub_{k_shot}shot_{split_tag}.png')
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: results/cub/comparison/per_class_heatmap_cub_{k_shot}shot_{split_tag}.png")
        return

    short_names = [abbrev_class(c) for c in hardest]

    method_defs = method_paths_for_split(split_tag, k_shot)

    matrix     = []
    row_labels = []
    any_data   = False

    for method_name, d in method_defs:
        pca = get_per_class(d)
        if pca:
            any_data = True
            row = [pca.get(cls, float('nan')) for cls in hardest]
        else:
            row = [float('nan')] * len(hardest)
        matrix.append(row)
        row_labels.append(method_name)

    matrix = np.array(matrix, dtype=float)

    split_label = 'b100_n50 (Standard)' if split_tag == 'b100' else 'b150_n50 (Richer-Base)'

    fig, ax = plt.subplots(figsize=(12, 5))

    if not any_data:
        ax.text(0.5, 0.5,
                f'No {k_shot}-shot results available yet for {split_tag}.',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='gray')
        ax.set_title(f'Per-Class Accuracy — {k_shot}-Shot — {split_label}')
    else:
        mask = np.isnan(matrix)
        sns.heatmap(
            matrix,
            annot=True,
            fmt='.1f',
            cmap='YlOrRd',
            xticklabels=short_names,
            yticklabels=row_labels,
            ax=ax,
            vmin=0,
            vmax=100,
            linewidths=0.5,
            linecolor='white',
            annot_kws={'size': 8},
            mask=mask,
        )
        ax.set_xlabel('15 Hardest Novel Classes (hardest left)', fontsize=10)
        ax.set_ylabel('Method', fontsize=10)
        ax.set_title(
            f'Per-Class Accuracy (%) — 5-way {k_shot}-Shot — {split_label}\n'
            f'(15 hardest of 50 novel CUB classes, 600 evaluation episodes)',
            fontsize=11
        )
        ax.tick_params(axis='x', labelrotation=30, labelsize=8)
        ax.tick_params(axis='y', labelrotation=0, labelsize=9)

    plt.tight_layout()
    out_path = os.path.join(OUT, f'per_class_heatmap_cub_{k_shot}shot_{split_tag}.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: results/cub/comparison/per_class_heatmap_cub_{k_shot}shot_{split_tag}.png")


if __name__ == '__main__':
    for k in [1, 3, 5]:
        for split in ['b100', 'b150']:
            build_heatmap(k, split)
