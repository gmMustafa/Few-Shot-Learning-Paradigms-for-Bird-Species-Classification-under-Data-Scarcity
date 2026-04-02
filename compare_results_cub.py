"""
compare_results_cub.py — CUB-200-2011 thesis comparison table generator.

Reads all experiment summary.json files under results/cub/, builds a unified
comparison table, and saves:
  - Console output (formatted table)
  - results/cub/comparison/comparison_table.csv
  - results/cub/comparison/comparison_table.tex  (ready to paste into LaTeX)
  - results/cub/comparison/comparison_bar_chart_b100_n50.png
  - results/cub/comparison/comparison_bar_chart_b150_n50.png
  - results/cub/comparison/comparison_bar_chart_proto_split.png
  - results/cub/comparison/all_in_one_summary.csv

Two splits:
  b100_n50 : Base-100 / Novel-50 (standard FSL benchmark split)
  b150_n50 : Base-150 / Novel-50 (richer-base split)

Missing results (not yet run) are shown as '—' gracefully.

Usage:
    python compare_results_cub.py
"""
import os
import json
import math

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams['figure.facecolor']  = 'white'
plt.rcParams['axes.facecolor']    = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

# ---------------------------------------------------------------------------
# Project root = directory containing this script
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(ROOT, 'results', 'cub')
OUT  = os.path.join(RES, 'comparison')


# ---------------------------------------------------------------------------
# Helper: load a summary.json, return None if missing
# ---------------------------------------------------------------------------
def load(path: str):
    full = os.path.join(RES, path)
    if not os.path.exists(full):
        return None
    with open(full) as f:
        return json.load(f)


def pct(v, scale=1.0):
    """Convert a value to percentage string, handling None."""
    if v is None:
        return '—'
    return f'{float(v) * scale:.2f}'


def fmt_ci(ci, scale=1.0):
    """Format a [low, high] CI list."""
    if ci is None:
        return '—'
    lo, hi = ci
    return f'[{float(lo)*scale:.2f}, {float(hi)*scale:.2f}]'


def fmt_f1(v):
    if v is None:
        return '—'
    return f'{float(v):.4f}'


# ---------------------------------------------------------------------------
# Load all results
# ---------------------------------------------------------------------------

# Supervised upper bounds
full200   = load('baseline/resnet50_full200/summary.json')

# base100 supervised
base100   = load('baseline/resnet50_base100/base_test/summary.json')
kshot5_b100 = load('baseline/resnet50_base100/novel_kshot_5/summary.json')
zeroshot_b100 = load('baseline/resnet50_base100/novel_zeroshot/summary.json')

# base150 supervised
base150   = load('baseline/resnet50_base150/base_test/summary.json')
kshot5_b150 = load('baseline/resnet50_base150/novel_kshot_5/summary.json')

# NCC base100
ncc_b100_1_euc = load('baseline/resnet50_base100/novel_ncc/1shot_euclidean/summary.json')
ncc_b100_1_man = load('baseline/resnet50_base100/novel_ncc/1shot_manhattan/summary.json')
ncc_b100_3_euc = load('baseline/resnet50_base100/novel_ncc/3shot_euclidean/summary.json')
ncc_b100_3_man = load('baseline/resnet50_base100/novel_ncc/3shot_manhattan/summary.json')
ncc_b100_5_euc = load('baseline/resnet50_base100/novel_ncc/5shot_euclidean/summary.json')
ncc_b100_5_man = load('baseline/resnet50_base100/novel_ncc/5shot_manhattan/summary.json')

# NCC base150
ncc_b150_1_euc = load('baseline/resnet50_base150/novel_ncc/1shot_euclidean/summary.json')
ncc_b150_1_man = load('baseline/resnet50_base150/novel_ncc/1shot_manhattan/summary.json')
ncc_b150_3_euc = load('baseline/resnet50_base150/novel_ncc/3shot_euclidean/summary.json')
ncc_b150_3_man = load('baseline/resnet50_base150/novel_ncc/3shot_manhattan/summary.json')
ncc_b150_5_euc = load('baseline/resnet50_base150/novel_ncc/5shot_euclidean/summary.json')
ncc_b150_5_man = load('baseline/resnet50_base150/novel_ncc/5shot_manhattan/summary.json')

# ProtoNet base100_novel50
proto_b100_1_euc = load('proto_net/base100_novel50/1shot_euclidean/summary.json')
proto_b100_1_man = load('proto_net/base100_novel50/1shot_manhattan/summary.json')
proto_b100_3_euc = load('proto_net/base100_novel50/3shot_euclidean/summary.json')
proto_b100_3_man = load('proto_net/base100_novel50/3shot_manhattan/summary.json')
proto_b100_5_euc = load('proto_net/base100_novel50/5shot_euclidean/summary.json')
proto_b100_5_man = load('proto_net/base100_novel50/5shot_manhattan/summary.json')

# ProtoNet base150_novel50
proto_b150_1_euc = load('proto_net/base150_novel50/1shot_euclidean/summary.json')
proto_b150_1_man = load('proto_net/base150_novel50/1shot_manhattan/summary.json')
proto_b150_3_euc = load('proto_net/base150_novel50/3shot_euclidean/summary.json')
proto_b150_3_man = load('proto_net/base150_novel50/3shot_manhattan/summary.json')
proto_b150_5_euc = load('proto_net/base150_novel50/5shot_euclidean/summary.json')
proto_b150_5_man = load('proto_net/base150_novel50/5shot_manhattan/summary.json')

# MatchingNet base100_novel50
match_b100_1 = load('matching_net/base100_novel50/1shot/summary.json')
match_b100_3 = load('matching_net/base100_novel50/3shot/summary.json')
match_b100_5 = load('matching_net/base100_novel50/5shot/summary.json')

# MatchingNet base150_novel50
match_b150_1 = load('matching_net/base150_novel50/1shot/summary.json')
match_b150_3 = load('matching_net/base150_novel50/3shot/summary.json')
match_b150_5 = load('matching_net/base150_novel50/5shot/summary.json')

# MatchingNet+FCE base100_novel50
match_fce_b100_1 = load('matching_net_fce/base100_novel50/1shot/summary.json')
match_fce_b100_3 = load('matching_net_fce/base100_novel50/3shot/summary.json')
match_fce_b100_5 = load('matching_net_fce/base100_novel50/5shot/summary.json')

# MatchingNet+FCE base150_novel50
match_fce_b150_1 = load('matching_net_fce/base150_novel50/1shot/summary.json')
match_fce_b150_3 = load('matching_net_fce/base150_novel50/3shot/summary.json')
match_fce_b150_5 = load('matching_net_fce/base150_novel50/5shot/summary.json')

# RelationNet base100_novel50
rel_b100_1 = load('relation_net/base100_novel50/1shot/summary.json')
rel_b100_3 = load('relation_net/base100_novel50/3shot/summary.json')
rel_b100_5 = load('relation_net/base100_novel50/5shot/summary.json')

# RelationNet base150_novel50
rel_b150_1 = load('relation_net/base150_novel50/1shot/summary.json')
rel_b150_3 = load('relation_net/base150_novel50/3shot/summary.json')
rel_b150_5 = load('relation_net/base150_novel50/5shot/summary.json')

# DeiT-S NCC base100
vit_ncc_b100_1_euc = load('vit/baseline/base100/novel_ncc/1shot_euclidean/summary.json')
vit_ncc_b100_1_man = load('vit/baseline/base100/novel_ncc/1shot_manhattan/summary.json')
vit_ncc_b100_3_euc = load('vit/baseline/base100/novel_ncc/3shot_euclidean/summary.json')
vit_ncc_b100_3_man = load('vit/baseline/base100/novel_ncc/3shot_manhattan/summary.json')
vit_ncc_b100_5_euc = load('vit/baseline/base100/novel_ncc/5shot_euclidean/summary.json')
vit_ncc_b100_5_man = load('vit/baseline/base100/novel_ncc/5shot_manhattan/summary.json')

# DeiT-S NCC base150
vit_ncc_b150_1_euc = load('vit/baseline/base150/novel_ncc/1shot_euclidean/summary.json')
vit_ncc_b150_1_man = load('vit/baseline/base150/novel_ncc/1shot_manhattan/summary.json')
vit_ncc_b150_3_euc = load('vit/baseline/base150/novel_ncc/3shot_euclidean/summary.json')
vit_ncc_b150_3_man = load('vit/baseline/base150/novel_ncc/3shot_manhattan/summary.json')
vit_ncc_b150_5_euc = load('vit/baseline/base150/novel_ncc/5shot_euclidean/summary.json')
vit_ncc_b150_5_man = load('vit/baseline/base150/novel_ncc/5shot_manhattan/summary.json')

# DeiT-S ProtoNet base100_novel50
vit_proto_b100_1_euc = load('vit/proto_net/base100_novel50/1shot_euclidean/summary.json')
vit_proto_b100_1_man = load('vit/proto_net/base100_novel50/1shot_manhattan/summary.json')
vit_proto_b100_3_euc = load('vit/proto_net/base100_novel50/3shot_euclidean/summary.json')
vit_proto_b100_3_man = load('vit/proto_net/base100_novel50/3shot_manhattan/summary.json')
vit_proto_b100_5_euc = load('vit/proto_net/base100_novel50/5shot_euclidean/summary.json')
vit_proto_b100_5_man = load('vit/proto_net/base100_novel50/5shot_manhattan/summary.json')

# DeiT-S ProtoNet base150_novel50
vit_proto_b150_1_euc = load('vit/proto_net/base150_novel50/1shot_euclidean/summary.json')
vit_proto_b150_1_man = load('vit/proto_net/base150_novel50/1shot_manhattan/summary.json')
vit_proto_b150_3_euc = load('vit/proto_net/base150_novel50/3shot_euclidean/summary.json')
vit_proto_b150_3_man = load('vit/proto_net/base150_novel50/3shot_manhattan/summary.json')
vit_proto_b150_5_euc = load('vit/proto_net/base150_novel50/5shot_euclidean/summary.json')
vit_proto_b150_5_man = load('vit/proto_net/base150_novel50/5shot_manhattan/summary.json')


# ---------------------------------------------------------------------------
# Normalise fields into a unified dict per experiment
# Baselines report fraction (0.99); FSL reports percentage (99.0)
# We store everything as percentage floats (or None)
# ---------------------------------------------------------------------------

def from_baseline(d, label, group, note, split='b100_n50'):
    """Parse a supervised baseline summary.json."""
    if d is None:
        return dict(split=split, group=group, method=label, setup='—', acc=None, ci='—',
                    ci_lo=None, ci_hi=None, f1=None, precision=None, recall=None,
                    auc=None, note=note)
    raw_acc = d.get('accuracy') or d.get('test_accuracy')
    acc = float(raw_acc) * 100
    ci  = d.get('accuracy_ci_95') or d.get('test_accuracy_ci_95')
    ci_str = fmt_ci(ci, scale=100) if ci else '—'
    f1        = d.get('macro_f1')
    precision = d.get('macro_precision')
    recall    = d.get('macro_recall')
    auc       = d.get('auc_ovr')
    return dict(split=split, group=group, method=label,
                setup=note,
                acc=acc, ci=ci_str,
                ci_lo=float(ci[0])*100 if ci else None,
                ci_hi=float(ci[1])*100 if ci else None,
                f1=float(f1) if f1 is not None else None,
                precision=float(precision) if precision is not None else None,
                recall=float(recall) if recall is not None else None,
                auc=float(auc) if auc is not None else None,
                note='Reference')


def from_fsl(d, label, group, k_shot, split='b100_n50', n_way=5):
    """Parse an FSL summary.json."""
    setup = f'{n_way}-way {k_shot}-shot'
    if d is None:
        return dict(split=split, group=group, method=label, setup=setup, acc=None, ci='—',
                    ci_lo=None, ci_hi=None, f1=None, precision=None, recall=None,
                    auc=None, note='Direct')
    acc   = float(d['mean_accuracy_pct'])
    ci95  = float(d['ci95_pct'])
    wci   = d.get('wilson_ci_95')
    ci_str = fmt_ci(wci, scale=1.0) if wci else f'[{acc-ci95:.2f}, {acc+ci95:.2f}]'
    ci_lo  = float(wci[0]) if wci else acc - ci95
    ci_hi  = float(wci[1]) if wci else acc + ci95
    precision = d.get('macro_precision')
    recall    = d.get('macro_recall')
    auc       = d.get('auc_ovr')
    return dict(split=split, group=group, method=label, setup=setup,
                acc=acc, ci=ci_str, ci_lo=ci_lo, ci_hi=ci_hi,
                f1=float(d['macro_f1']) if d.get('macro_f1') is not None else None,
                precision=float(precision) if precision is not None else None,
                recall=float(recall) if recall is not None else None,
                auc=float(auc) if auc is not None else None,
                note='Direct')


rows = [
    # --- Supervised upper bounds (reference only) ---
    from_baseline(full200,      'Full-200 Baseline',  'Supervised (Reference)',
                  '200-way, full data', split='b100_n50'),
    from_baseline(base100,      'Base-100 Baseline',  'Supervised (Reference)',
                  '100-way, full data', split='b100_n50'),

    # --- Low-data supervised (direct comparator) ---
    from_baseline(kshot5_b100,  'K=5 Fine-tune',      'Low-Data Supervised',
                  '5-way, 5-shot fine-tune', split='b100_n50'),

    # --- Zero-shot reference ---
    dict(split='b100_n50', group='Zero-Shot Reference', method='Zero-Shot Transfer',
         setup='5-way, 0-shot', acc=0.0, ci='[0.0, 0.0]',
         ci_lo=0.0, ci_hi=0.0, f1=None, precision=None, recall=None, auc=None,
         note='0% by design (100-class head cannot output novel labels)'),

    # --- NCC base100 ---
    from_fsl(ncc_b100_1_euc, 'NCC (Euclidean)', 'NCC 1-Shot', k_shot=1, split='b100_n50'),
    from_fsl(ncc_b100_1_man, 'NCC (Manhattan)', 'NCC 1-Shot', k_shot=1, split='b100_n50'),
    from_fsl(ncc_b100_3_euc, 'NCC (Euclidean)', 'NCC 3-Shot', k_shot=3, split='b100_n50'),
    from_fsl(ncc_b100_3_man, 'NCC (Manhattan)', 'NCC 3-Shot', k_shot=3, split='b100_n50'),
    from_fsl(ncc_b100_5_euc, 'NCC (Euclidean)', 'NCC 5-Shot', k_shot=5, split='b100_n50'),
    from_fsl(ncc_b100_5_man, 'NCC (Manhattan)', 'NCC 5-Shot', k_shot=5, split='b100_n50'),

    # --- DeiT-S NCC base100 ---
    from_fsl(vit_ncc_b100_1_euc, 'DeiT-S NCC (Euclidean)', 'NCC 1-Shot', k_shot=1, split='b100_n50'),
    from_fsl(vit_ncc_b100_1_man, 'DeiT-S NCC (Manhattan)', 'NCC 1-Shot', k_shot=1, split='b100_n50'),
    from_fsl(vit_ncc_b100_3_euc, 'DeiT-S NCC (Euclidean)', 'NCC 3-Shot', k_shot=3, split='b100_n50'),
    from_fsl(vit_ncc_b100_3_man, 'DeiT-S NCC (Manhattan)', 'NCC 3-Shot', k_shot=3, split='b100_n50'),
    from_fsl(vit_ncc_b100_5_euc, 'DeiT-S NCC (Euclidean)', 'NCC 5-Shot', k_shot=5, split='b100_n50'),
    from_fsl(vit_ncc_b100_5_man, 'DeiT-S NCC (Manhattan)', 'NCC 5-Shot', k_shot=5, split='b100_n50'),

    # --- FSL 1-shot base100 ---
    from_fsl(proto_b100_1_euc,    'ProtoNet (Euclidean)',    'FSL 1-Shot', k_shot=1, split='b100_n50'),
    from_fsl(proto_b100_1_man,    'ProtoNet (Manhattan)',    'FSL 1-Shot', k_shot=1, split='b100_n50'),
    from_fsl(match_b100_1,        'MatchingNet (Cosine)',    'FSL 1-Shot', k_shot=1, split='b100_n50'),
    from_fsl(match_fce_b100_1,    'MatchingNet+FCE (Cosine)','FSL 1-Shot', k_shot=1, split='b100_n50'),
    from_fsl(rel_b100_1,          'RelationNet (Learned)',   'FSL 1-Shot', k_shot=1, split='b100_n50'),
    from_fsl(vit_proto_b100_1_euc,'DeiT-S ProtoNet (Euclidean)','FSL 1-Shot', k_shot=1, split='b100_n50'),
    from_fsl(vit_proto_b100_1_man,'DeiT-S ProtoNet (Manhattan)','FSL 1-Shot', k_shot=1, split='b100_n50'),

    # --- FSL 3-shot base100 ---
    from_fsl(proto_b100_3_euc,    'ProtoNet (Euclidean)',    'FSL 3-Shot', k_shot=3, split='b100_n50'),
    from_fsl(proto_b100_3_man,    'ProtoNet (Manhattan)',    'FSL 3-Shot', k_shot=3, split='b100_n50'),
    from_fsl(match_b100_3,        'MatchingNet (Cosine)',    'FSL 3-Shot', k_shot=3, split='b100_n50'),
    from_fsl(match_fce_b100_3,    'MatchingNet+FCE (Cosine)','FSL 3-Shot', k_shot=3, split='b100_n50'),
    from_fsl(rel_b100_3,          'RelationNet (Learned)',   'FSL 3-Shot', k_shot=3, split='b100_n50'),
    from_fsl(vit_proto_b100_3_euc,'DeiT-S ProtoNet (Euclidean)','FSL 3-Shot', k_shot=3, split='b100_n50'),
    from_fsl(vit_proto_b100_3_man,'DeiT-S ProtoNet (Manhattan)','FSL 3-Shot', k_shot=3, split='b100_n50'),

    # --- FSL 5-shot base100 ---
    from_fsl(proto_b100_5_euc,    'ProtoNet (Euclidean)',    'FSL 5-Shot', k_shot=5, split='b100_n50'),
    from_fsl(proto_b100_5_man,    'ProtoNet (Manhattan)',    'FSL 5-Shot', k_shot=5, split='b100_n50'),
    from_fsl(match_b100_5,        'MatchingNet (Cosine)',    'FSL 5-Shot', k_shot=5, split='b100_n50'),
    from_fsl(match_fce_b100_5,    'MatchingNet+FCE (Cosine)','FSL 5-Shot', k_shot=5, split='b100_n50'),
    from_fsl(rel_b100_5,          'RelationNet (Learned)',   'FSL 5-Shot', k_shot=5, split='b100_n50'),
    from_fsl(vit_proto_b100_5_euc,'DeiT-S ProtoNet (Euclidean)','FSL 5-Shot', k_shot=5, split='b100_n50'),
    from_fsl(vit_proto_b100_5_man,'DeiT-S ProtoNet (Manhattan)','FSL 5-Shot', k_shot=5, split='b100_n50'),

    # =========================================================================
    # b150_n50 split
    # =========================================================================
    from_baseline(base150,      'Base-150 Baseline',  'Supervised (Reference)',
                  '150-way, full data', split='b150_n50'),
    from_baseline(kshot5_b150,  'K=5 Fine-tune',      'Low-Data Supervised',
                  '5-way, 5-shot fine-tune', split='b150_n50'),

    # --- NCC base150 ---
    from_fsl(ncc_b150_1_euc, 'NCC (Euclidean)', 'NCC 1-Shot', k_shot=1, split='b150_n50'),
    from_fsl(ncc_b150_1_man, 'NCC (Manhattan)', 'NCC 1-Shot', k_shot=1, split='b150_n50'),
    from_fsl(ncc_b150_3_euc, 'NCC (Euclidean)', 'NCC 3-Shot', k_shot=3, split='b150_n50'),
    from_fsl(ncc_b150_3_man, 'NCC (Manhattan)', 'NCC 3-Shot', k_shot=3, split='b150_n50'),
    from_fsl(ncc_b150_5_euc, 'NCC (Euclidean)', 'NCC 5-Shot', k_shot=5, split='b150_n50'),
    from_fsl(ncc_b150_5_man, 'NCC (Manhattan)', 'NCC 5-Shot', k_shot=5, split='b150_n50'),

    # --- DeiT-S NCC base150 ---
    from_fsl(vit_ncc_b150_1_euc, 'DeiT-S NCC (Euclidean)', 'NCC 1-Shot', k_shot=1, split='b150_n50'),
    from_fsl(vit_ncc_b150_1_man, 'DeiT-S NCC (Manhattan)', 'NCC 1-Shot', k_shot=1, split='b150_n50'),
    from_fsl(vit_ncc_b150_3_euc, 'DeiT-S NCC (Euclidean)', 'NCC 3-Shot', k_shot=3, split='b150_n50'),
    from_fsl(vit_ncc_b150_3_man, 'DeiT-S NCC (Manhattan)', 'NCC 3-Shot', k_shot=3, split='b150_n50'),
    from_fsl(vit_ncc_b150_5_euc, 'DeiT-S NCC (Euclidean)', 'NCC 5-Shot', k_shot=5, split='b150_n50'),
    from_fsl(vit_ncc_b150_5_man, 'DeiT-S NCC (Manhattan)', 'NCC 5-Shot', k_shot=5, split='b150_n50'),

    # --- FSL 1-shot base150 ---
    from_fsl(proto_b150_1_euc,    'ProtoNet (Euclidean)',    'FSL 1-Shot', k_shot=1, split='b150_n50'),
    from_fsl(proto_b150_1_man,    'ProtoNet (Manhattan)',    'FSL 1-Shot', k_shot=1, split='b150_n50'),
    from_fsl(match_b150_1,        'MatchingNet (Cosine)',    'FSL 1-Shot', k_shot=1, split='b150_n50'),
    from_fsl(match_fce_b150_1,    'MatchingNet+FCE (Cosine)','FSL 1-Shot', k_shot=1, split='b150_n50'),
    from_fsl(rel_b150_1,          'RelationNet (Learned)',   'FSL 1-Shot', k_shot=1, split='b150_n50'),
    from_fsl(vit_proto_b150_1_euc,'DeiT-S ProtoNet (Euclidean)','FSL 1-Shot', k_shot=1, split='b150_n50'),
    from_fsl(vit_proto_b150_1_man,'DeiT-S ProtoNet (Manhattan)','FSL 1-Shot', k_shot=1, split='b150_n50'),

    # --- FSL 3-shot base150 ---
    from_fsl(proto_b150_3_euc,    'ProtoNet (Euclidean)',    'FSL 3-Shot', k_shot=3, split='b150_n50'),
    from_fsl(proto_b150_3_man,    'ProtoNet (Manhattan)',    'FSL 3-Shot', k_shot=3, split='b150_n50'),
    from_fsl(match_b150_3,        'MatchingNet (Cosine)',    'FSL 3-Shot', k_shot=3, split='b150_n50'),
    from_fsl(match_fce_b150_3,    'MatchingNet+FCE (Cosine)','FSL 3-Shot', k_shot=3, split='b150_n50'),
    from_fsl(rel_b150_3,          'RelationNet (Learned)',   'FSL 3-Shot', k_shot=3, split='b150_n50'),
    from_fsl(vit_proto_b150_3_euc,'DeiT-S ProtoNet (Euclidean)','FSL 3-Shot', k_shot=3, split='b150_n50'),
    from_fsl(vit_proto_b150_3_man,'DeiT-S ProtoNet (Manhattan)','FSL 3-Shot', k_shot=3, split='b150_n50'),

    # --- FSL 5-shot base150 ---
    from_fsl(proto_b150_5_euc,    'ProtoNet (Euclidean)',    'FSL 5-Shot', k_shot=5, split='b150_n50'),
    from_fsl(proto_b150_5_man,    'ProtoNet (Manhattan)',    'FSL 5-Shot', k_shot=5, split='b150_n50'),
    from_fsl(match_b150_5,        'MatchingNet (Cosine)',    'FSL 5-Shot', k_shot=5, split='b150_n50'),
    from_fsl(match_fce_b150_5,    'MatchingNet+FCE (Cosine)','FSL 5-Shot', k_shot=5, split='b150_n50'),
    from_fsl(rel_b150_5,          'RelationNet (Learned)',   'FSL 5-Shot', k_shot=5, split='b150_n50'),
    from_fsl(vit_proto_b150_5_euc,'DeiT-S ProtoNet (Euclidean)','FSL 5-Shot', k_shot=5, split='b150_n50'),
    from_fsl(vit_proto_b150_5_man,'DeiT-S ProtoNet (Manhattan)','FSL 5-Shot', k_shot=5, split='b150_n50'),
]

df = pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------

def fmt_acc(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '—'
    return f'{v:.2f}%'

def fmt_f1_cell(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '—'
    return f'{v:.4f}'

print('\n' + '=' * 150)
print('CUB-200-2011 THESIS COMPARISON TABLE')
print('=' * 150)
print(f"{'Split':<12} {'Group':<24} {'Method':<28} {'Setup':<18} {'Accuracy':>10} {'95% CI':>22} {'Macro F1':>10} {'Precision':>10} {'Recall':>10} {'AUC-OVR':>10}")
print('-' * 150)

current_split = None
current_group = None
for _, r in df.iterrows():
    if r['split'] != current_split:
        if current_split is not None:
            print()
        current_split = r['split']
        current_group = None
        print(f"\n[{r['split']}]")
    if r['group'] != current_group:
        current_group = r['group']
        print(f"  <{r['group']}>")

    acc_str  = fmt_acc(r['acc'])
    f1_str   = fmt_f1_cell(r['f1'])
    prec_str = fmt_f1_cell(r.get('precision'))
    rec_str  = fmt_f1_cell(r.get('recall'))
    auc_str  = fmt_f1_cell(r.get('auc'))
    print(f"    {r['method']:<28} {r['setup']:<18} {acc_str:>10} {r['ci']:>22} {f1_str:>10} {prec_str:>10} {rec_str:>10} {auc_str:>10}")

print('=' * 150)
print('\nNotes:')
print('  b100_n50 : Base-100 / Novel-50 (standard FSL benchmark split)')
print('  b150_n50 : Base-150 / Novel-50 (richer-base split)')
print('  Reference: Full-data supervised results. NOT directly comparable to few-shot methods.')
print('  Direct   : Evaluated on novel classes with same K images.')
print()


# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------

os.makedirs(OUT, exist_ok=True)

csv_df = df[['split', 'group', 'method', 'setup', 'acc', 'ci', 'f1', 'precision', 'recall', 'auc', 'note']].copy()
_fmt = lambda v: f'{v:.4f}' if v is not None and not (isinstance(v, float) and math.isnan(v)) else '—'
csv_df['acc'] = csv_df['acc'].apply(
    lambda v: f'{v:.2f}' if v is not None and not (isinstance(v, float) and math.isnan(v)) else '—'
)
csv_df['f1']        = csv_df['f1'].apply(_fmt)
csv_df['precision'] = csv_df['precision'].apply(_fmt)
csv_df['recall']    = csv_df['recall'].apply(_fmt)
csv_df['auc']       = csv_df['auc'].apply(_fmt)
csv_df.to_csv(os.path.join(OUT, 'comparison_table.csv'), index=False)
print(f"Saved: results/cub/comparison/comparison_table.csv")


# ---------------------------------------------------------------------------
# Save LaTeX table
# ---------------------------------------------------------------------------

def tex_escape(s):
    return str(s).replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')

def acc_tex(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '---'
    return f'{v:.2f}\\%'

def f1_tex(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '---'
    return f'{v:.4f}'

def ci_tex(s):
    return s if s != '—' else '---'

def pr_tex(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '---'
    return f'{v:.4f}'

lines = [
    r'\begin{table}[htbp]',
    r'\centering',
    r'\caption{Comparison of methods on CUB-200-2011 novel class evaluation. '
    r'Includes both the base100/novel50 and base150/novel50 setups for all FSL methods '
    r'(ProtoNet, MatchingNet, MatchingNet+FCE, RelationNet) and NCC baselines.}',
    r'\label{tab:cub_results_comparison}',
    r'\begin{tabular}{lllccccc}',
    r'\hline',
    r'\textbf{Split} & \textbf{Method} & \textbf{Accuracy (\%)} & \textbf{95\% CI} & \textbf{Macro F1} & \textbf{Precision} & \textbf{Recall} & \textbf{AUC-OVR} \\',
    r'\hline',
]

current_split = None
for _, r in df.iterrows():
    if r['split'] != current_split:
        if current_split is not None:
            lines.append(r'\hline')
        current_split = r['split']
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{tex_escape(r['split'])}}}}}" + r' \\')

    method_cell = tex_escape(str(r['method']))
    acc_cell    = acc_tex(r['acc'])
    ci_cell     = ci_tex(r['ci']).replace('%', r'\%')
    f1_cell     = f1_tex(r['f1'])
    prec_cell   = pr_tex(r.get('precision'))
    rec_cell    = pr_tex(r.get('recall'))
    auc_cell    = pr_tex(r.get('auc'))
    lines.append(f"{tex_escape(r['split'])} & {method_cell} & {acc_cell} & {ci_cell} & {f1_cell} & {prec_cell} & {rec_cell} & {auc_cell} \\\\")

lines += [
    r'\hline',
    r'\end{tabular}',
    r'\end{table}',
]

with open(os.path.join(OUT, 'comparison_table.tex'), 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f"Saved: results/cub/comparison/comparison_table.tex")


# ---------------------------------------------------------------------------
# Bar chart helpers
# ---------------------------------------------------------------------------

def half_ci_width(row):
    if row['ci_lo'] is not None and row['ci_hi'] is not None:
        return (float(row['ci_hi']) - float(row['ci_lo'])) / 2.0
    return 0.0


color_map = {
    'Supervised (Reference)': '#9ecae1',
    'Low-Data Supervised':    '#2171b5',
    'Zero-Shot Reference':    '#bdbdbd',
    'NCC 1-Shot':             '#fdd0a2',
    'NCC 3-Shot':             '#fd8d3c',
    'NCC 5-Shot':             '#d94801',
    'FSL 1-Shot':             '#c7e9c0',
    'FSL 3-Shot':             '#74c476',
    'FSL 5-Shot':             '#238b45',
}


def _build_ordered_rows(plot_df, ordered_pairs):
    rows = []
    for method, setup in ordered_pairs:
        match = plot_df[(plot_df['method'] == method) & (plot_df['setup'] == setup)]
        if len(match) == 0:
            rows.append({'method': method, 'setup': setup, 'acc': np.nan, 'err': 0.0, 'group': None})
        else:
            r = match.iloc[0]
            rows.append({'method': r['method'], 'setup': r['setup'],
                         'acc': float(r['acc']), 'err': half_ci_width(r), 'group': r['group']})
    return pd.DataFrame(rows)


def _short_label_main(method, setup, split):
    """Short x-axis label for the main bar chart."""
    if method == 'Full-200 Baseline':  return 'Full-200'
    if method == 'Base-100 Baseline':  return 'Base-100'
    if method == 'Base-150 Baseline':  return 'Base-150'
    if method == 'K=5 Fine-tune':      return 'Fine-tune'
    if method == 'Zero-Shot Transfer': return 'Zero-shot'
    shot = setup.replace('5-way ', '').replace('-shot', 's')
    if method == 'NCC (Euclidean)':               return f'NCC-E\n{shot}'
    if method == 'NCC (Manhattan)':               return f'NCC-M\n{shot}'
    if method == 'ProtoNet (Euclidean)':          return f'Proto-E\n{shot}'
    if method == 'ProtoNet (Manhattan)':          return f'Proto-M\n{shot}'
    if method == 'MatchingNet (Cosine)':          return f'Match\n{shot}'
    if method == 'MatchingNet+FCE (Cosine)':      return f'Match+FCE\n{shot}'
    if method == 'RelationNet (Learned)':         return f'RelNet\n{shot}'
    return method


def _draw_bar_chart(chart_df, x_labels, title, out_path, separators, kshot5_acc=None):
    x_vals   = chart_df['acc'].tolist()
    x_errs   = chart_df['err'].tolist()
    x_colors = [color_map.get(g, '#aaaaaa') if g is not None else '#eeeeee'
                for g in chart_df['group']]

    fig, ax = plt.subplots(figsize=(18, 6))
    x = np.arange(len(x_labels))
    bars = ax.bar(x, x_vals, color=x_colors, edgecolor='white', linewidth=0.8)

    ax.errorbar(x, x_vals, yerr=x_errs, fmt='none',
                color='black', capsize=4, capthick=1.1, linewidth=1.1)

    for bar, val in zip(bars, x_vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.8,
                    f'{val:.1f}%', ha='center', va='bottom',
                    fontsize=7, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_title(title, fontsize=12)
    ax.axhline(y=20, color='gray', linestyle='--', linewidth=0.9, alpha=0.6,
               label='Random chance (5-way, 20%)')

    if kshot5_acc is not None:
        ax.axhline(y=kshot5_acc, color='steelblue', linestyle=':', linewidth=1.5,
                   label=f'K=5 Fine-tune ({kshot5_acc:.1f}%)')

    for xpos in separators:
        ax.axvline(x=xpos, color='lightgray', linestyle='-', linewidth=0.8)

    group_patches = [
        mpatches.Patch(color='#9ecae1', label='Supervised reference'),
        mpatches.Patch(color='#2171b5', label='Low-data supervised'),
        mpatches.Patch(color='#bdbdbd', label='Zero-shot'),
        mpatches.Patch(color='#fdd0a2', label='NCC / 1-shot'),
        mpatches.Patch(color='#fd8d3c', label='NCC / 3-shot'),
        mpatches.Patch(color='#d94801', label='NCC / 5-shot'),
        mpatches.Patch(color='#c7e9c0', label='FSL / 1-shot'),
        mpatches.Patch(color='#74c476', label='FSL / 3-shot'),
        mpatches.Patch(color='#238b45', label='FSL / 5-shot'),
    ]
    extra_lines = [
        plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=0.9,
                   label='Random chance (5-way)'),
    ]
    if kshot5_acc is not None:
        extra_lines.append(
            plt.Line2D([0], [0], color='steelblue', linestyle=':', linewidth=1.5,
                       label=f'K=5 Fine-tune ({kshot5_acc:.1f}%)')
        )
    ax.legend(handles=group_patches + extra_lines, loc='upper right', fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main bar chart — b100_n50 (Standard split)
# ---------------------------------------------------------------------------

def save_b100_main_chart(df, out_dir):
    """Main comparison chart for the b100_n50 setup."""
    plot_df = df[
        (df['split'] == 'b100_n50') &
        (df['acc'].notna())
    ].copy()

    keep_methods = {
        'Full-200 Baseline',
        'Base-100 Baseline',
        'K=5 Fine-tune',
        'Zero-Shot Transfer',
        'NCC (Euclidean)',
        'NCC (Manhattan)',
        'ProtoNet (Euclidean)',
        'ProtoNet (Manhattan)',
        'MatchingNet (Cosine)',
        'MatchingNet+FCE (Cosine)',
        'RelationNet (Learned)',
    }
    plot_df = plot_df[plot_df['method'].isin(keep_methods)].copy()

    # Build k=5 fine-tune acc for horizontal reference line
    ft_row = df[(df['split'] == 'b100_n50') & (df['method'] == 'K=5 Fine-tune')]
    kshot5_acc = float(ft_row.iloc[0]['acc']) if not ft_row.empty and ft_row.iloc[0]['acc'] is not None else None

    ordered_pairs = [
        ('Full-200 Baseline',       '200-way, full data'),
        ('Base-100 Baseline',       '100-way, full data'),
        ('K=5 Fine-tune',           '5-way, 5-shot fine-tune'),
        ('Zero-Shot Transfer',      '5-way, 0-shot'),

        ('NCC (Euclidean)',          '5-way 1-shot'),
        ('NCC (Manhattan)',          '5-way 1-shot'),
        ('ProtoNet (Euclidean)',     '5-way 1-shot'),
        ('ProtoNet (Manhattan)',     '5-way 1-shot'),
        ('MatchingNet (Cosine)',     '5-way 1-shot'),
        ('MatchingNet+FCE (Cosine)','5-way 1-shot'),
        ('RelationNet (Learned)',    '5-way 1-shot'),

        ('NCC (Euclidean)',          '5-way 3-shot'),
        ('NCC (Manhattan)',          '5-way 3-shot'),
        ('ProtoNet (Euclidean)',     '5-way 3-shot'),
        ('ProtoNet (Manhattan)',     '5-way 3-shot'),
        ('MatchingNet (Cosine)',     '5-way 3-shot'),
        ('MatchingNet+FCE (Cosine)','5-way 3-shot'),
        ('RelationNet (Learned)',    '5-way 3-shot'),

        ('NCC (Euclidean)',          '5-way 5-shot'),
        ('NCC (Manhattan)',          '5-way 5-shot'),
        ('ProtoNet (Euclidean)',     '5-way 5-shot'),
        ('ProtoNet (Manhattan)',     '5-way 5-shot'),
        ('MatchingNet (Cosine)',     '5-way 5-shot'),
        ('MatchingNet+FCE (Cosine)','5-way 5-shot'),
        ('RelationNet (Learned)',    '5-way 5-shot'),
    ]

    chart_df = _build_ordered_rows(plot_df, ordered_pairs)
    x_labels = [_short_label_main(r['method'], r['setup'], 'b100_n50') for _, r in chart_df.iterrows()]

    out_path = os.path.join(out_dir, 'comparison_bar_chart_b100_n50.png')
    _draw_bar_chart(
        chart_df, x_labels,
        title='CUB-200-2011 Method Comparison — b100_n50 (Standard Split)\n'
              '(5-way, 50 novel classes, 600 episodes, 95% CI error bars)',
        out_path=out_path,
        separators=[3.5, 10.5, 17.5],
        kshot5_acc=kshot5_acc,
    )
    print("Saved: results/cub/comparison/comparison_bar_chart_b100_n50.png")


# ---------------------------------------------------------------------------
# Main bar chart — b150_n50 (Richer-base split)
# ---------------------------------------------------------------------------

def save_b150_main_chart(df, out_dir):
    """Main comparison chart for the b150_n50 setup."""
    plot_df = df[
        (df['split'] == 'b150_n50') &
        (df['acc'].notna())
    ].copy()

    keep_methods = {
        'Base-150 Baseline',
        'K=5 Fine-tune',
        'NCC (Euclidean)',
        'NCC (Manhattan)',
        'ProtoNet (Euclidean)',
        'ProtoNet (Manhattan)',
        'MatchingNet (Cosine)',
        'MatchingNet+FCE (Cosine)',
        'RelationNet (Learned)',
    }
    plot_df = plot_df[plot_df['method'].isin(keep_methods)].copy()

    ft_row = df[(df['split'] == 'b150_n50') & (df['method'] == 'K=5 Fine-tune')]
    kshot5_acc = float(ft_row.iloc[0]['acc']) if not ft_row.empty and ft_row.iloc[0]['acc'] is not None else None

    ordered_pairs = [
        ('Base-150 Baseline',       '150-way, full data'),
        ('K=5 Fine-tune',           '5-way, 5-shot fine-tune'),

        ('NCC (Euclidean)',          '5-way 1-shot'),
        ('NCC (Manhattan)',          '5-way 1-shot'),
        ('ProtoNet (Euclidean)',     '5-way 1-shot'),
        ('ProtoNet (Manhattan)',     '5-way 1-shot'),
        ('MatchingNet (Cosine)',     '5-way 1-shot'),
        ('MatchingNet+FCE (Cosine)','5-way 1-shot'),
        ('RelationNet (Learned)',    '5-way 1-shot'),

        ('NCC (Euclidean)',          '5-way 3-shot'),
        ('NCC (Manhattan)',          '5-way 3-shot'),
        ('ProtoNet (Euclidean)',     '5-way 3-shot'),
        ('ProtoNet (Manhattan)',     '5-way 3-shot'),
        ('MatchingNet (Cosine)',     '5-way 3-shot'),
        ('MatchingNet+FCE (Cosine)','5-way 3-shot'),
        ('RelationNet (Learned)',    '5-way 3-shot'),

        ('NCC (Euclidean)',          '5-way 5-shot'),
        ('NCC (Manhattan)',          '5-way 5-shot'),
        ('ProtoNet (Euclidean)',     '5-way 5-shot'),
        ('ProtoNet (Manhattan)',     '5-way 5-shot'),
        ('MatchingNet (Cosine)',     '5-way 5-shot'),
        ('MatchingNet+FCE (Cosine)','5-way 5-shot'),
        ('RelationNet (Learned)',    '5-way 5-shot'),
    ]

    chart_df = _build_ordered_rows(plot_df, ordered_pairs)
    x_labels = [_short_label_main(r['method'], r['setup'], 'b150_n50') for _, r in chart_df.iterrows()]

    out_path = os.path.join(out_dir, 'comparison_bar_chart_b150_n50.png')
    _draw_bar_chart(
        chart_df, x_labels,
        title='CUB-200-2011 Method Comparison — b150_n50 (Richer-Base Split)\n'
              '(5-way, 50 novel classes, 600 episodes, 95% CI error bars)',
        out_path=out_path,
        separators=[1.5, 8.5, 15.5],
        kshot5_acc=kshot5_acc,
    )
    print("Saved: results/cub/comparison/comparison_bar_chart_b150_n50.png")


# ---------------------------------------------------------------------------
# ProtoNet split comparison chart
# ---------------------------------------------------------------------------

def save_proto_split_chart(df, out_dir):
    """Focused ProtoNet comparison between b100_n50 and b150_n50."""
    plot_df = df[
        (df['method'].isin(['ProtoNet (Euclidean)', 'ProtoNet (Manhattan)'])) &
        (df['split'].isin(['b100_n50', 'b150_n50'])) &
        (df['acc'].notna())
    ].copy()

    ordered_pairs = [
        ('b100_n50', 'ProtoNet (Euclidean)', '5-way 1-shot'),
        ('b150_n50', 'ProtoNet (Euclidean)', '5-way 1-shot'),
        ('b100_n50', 'ProtoNet (Manhattan)', '5-way 1-shot'),
        ('b150_n50', 'ProtoNet (Manhattan)', '5-way 1-shot'),

        ('b100_n50', 'ProtoNet (Euclidean)', '5-way 3-shot'),
        ('b150_n50', 'ProtoNet (Euclidean)', '5-way 3-shot'),
        ('b100_n50', 'ProtoNet (Manhattan)', '5-way 3-shot'),
        ('b150_n50', 'ProtoNet (Manhattan)', '5-way 3-shot'),

        ('b100_n50', 'ProtoNet (Euclidean)', '5-way 5-shot'),
        ('b150_n50', 'ProtoNet (Euclidean)', '5-way 5-shot'),
        ('b100_n50', 'ProtoNet (Manhattan)', '5-way 5-shot'),
        ('b150_n50', 'ProtoNet (Manhattan)', '5-way 5-shot'),
    ]

    rows = []
    for split, method, setup in ordered_pairs:
        match = plot_df[
            (plot_df['split'] == split) &
            (plot_df['method'] == method) &
            (plot_df['setup'] == setup)
        ]
        if len(match) == 0:
            rows.append({'split': split, 'method': method, 'setup': setup,
                         'acc': np.nan, 'err': 0.0})
        else:
            r = match.iloc[0]
            rows.append({'split': split, 'method': r['method'], 'setup': r['setup'],
                         'acc': float(r['acc']), 'err': half_ci_width(r)})

    chart_df = pd.DataFrame(rows)

    split_colors = {
        'b100_n50': '#5DA5DA',
        'b150_n50': '#7B6FD0',
    }
    metric_hatch = {
        'ProtoNet (Euclidean)': '',
        'ProtoNet (Manhattan)': '//',
    }

    def short_label(split, method, setup):
        shot = setup.replace('5-way ', '').replace('-shot', 's')
        metric = 'E' if 'Euclidean' in method else 'M'
        split_short = '100/50' if split == 'b100_n50' else '150/50'
        return f'{split_short}\nProto-{metric}\n{shot}'

    x_labels = [short_label(r['split'], r['method'], r['setup']) for _, r in chart_df.iterrows()]
    x_vals   = chart_df['acc'].tolist()
    x_errs   = chart_df['err'].tolist()

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(x_labels))
    bars = ax.bar(
        x, x_vals,
        color=[split_colors[r['split']] for _, r in chart_df.iterrows()],
        edgecolor='white', linewidth=0.8
    )

    for bar, (_, r) in zip(bars, chart_df.iterrows()):
        bar.set_hatch(metric_hatch[r['method']])

    ax.errorbar(x, x_vals, yerr=x_errs, fmt='none',
                color='black', capsize=4, capthick=1.1, linewidth=1.1)

    for bar, val in zip(bars, x_vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.8,
                    f'{val:.1f}%', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_ylim(0, 108)
    ax.set_title('CUB-200-2011 ProtoNet Split Comparison — b100_n50 vs b150_n50\n'
                 '(5-way, 50 novel classes, 600 episodes)',
                 fontsize=12)
    ax.axhline(y=20, color='gray', linestyle='--', linewidth=0.9, alpha=0.6)

    for xpos in [3.5, 7.5]:
        ax.axvline(x=xpos, color='lightgray', linestyle='-', linewidth=0.8)

    legend_handles = [
        mpatches.Patch(color=split_colors['b100_n50'], label='b100_n50 (Standard)'),
        mpatches.Patch(color=split_colors['b150_n50'], label='b150_n50 (Richer-base)'),
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='', label='Euclidean'),
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='//', label='Manhattan'),
        plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=0.9,
                   label='Random chance (5-way)'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'comparison_bar_chart_proto_split.png'), dpi=150)
    plt.close()
    print("Saved: results/cub/comparison/comparison_bar_chart_proto_split.png")


# Generate bar charts
save_b100_main_chart(df, OUT)
save_b150_main_chart(df, OUT)
save_proto_split_chart(df, OUT)


# ---------------------------------------------------------------------------
# All-in-One Summary Table (console + CSV)
# ---------------------------------------------------------------------------

print('\n' + '=' * 105)
print('ALL-IN-ONE SUMMARY TABLE')
print('=' * 105)
print(f"{'Split':<12} {'Method':<30} {'K-Shot':<10} {'Acc (%)':>9} {'95% CI':>22} {'F1':>8} {'Prec':>8} {'Rec':>8} {'AUC':>8}")
print('-' * 105)

for split in ['b100_n50', 'b150_n50']:
    split_df = df[df['split'] == split].copy()
    print(f"\n  [{split}]")
    for _, r in split_df.iterrows():
        acc_s = (f"{r['acc']:.2f}%" if r['acc'] is not None
                 and not (isinstance(r['acc'], float) and math.isnan(r['acc'])) else '—')
        f1_s  = (f"{r['f1']:.4f}" if r['f1'] is not None
                 and not (isinstance(r['f1'], float) and math.isnan(r['f1'])) else '—')
        pr_s  = (f"{r['precision']:.4f}" if r.get('precision') is not None
                 and not (isinstance(r.get('precision'), float) and math.isnan(r.get('precision'))) else '—')
        rc_s  = (f"{r['recall']:.4f}" if r.get('recall') is not None
                 and not (isinstance(r.get('recall'), float) and math.isnan(r.get('recall'))) else '—')
        auc_s = (f"{r['auc']:.4f}" if r.get('auc') is not None
                 and not (isinstance(r.get('auc'), float) and math.isnan(r.get('auc'))) else '—')
        print(f"  {split:<12} {r['method']:<30} {r['setup']:<10} {acc_s:>9} {r['ci']:>22} {f1_s:>8} {pr_s:>8} {rc_s:>8} {auc_s:>8}")

print('=' * 105)

# Save all-in-one CSV
aio_df = df[['split', 'method', 'setup', 'acc', 'ci', 'f1', 'precision', 'recall', 'auc']].copy()
for col in ['acc']:
    aio_df[col] = aio_df[col].apply(
        lambda v: f'{v:.2f}' if v is not None and not (isinstance(v, float) and math.isnan(v)) else '—'
    )
for col in ['f1', 'precision', 'recall', 'auc']:
    aio_df[col] = aio_df[col].apply(
        lambda v: f'{v:.4f}' if v is not None and not (isinstance(v, float) and math.isnan(v)) else '—'
    )
aio_df.to_csv(os.path.join(OUT, 'all_in_one_summary.csv'), index=False)
print(f"Saved: results/cub/comparison/all_in_one_summary.csv")


# ---------------------------------------------------------------------------
# Key Insights
# ---------------------------------------------------------------------------

print('\n' + '=' * 80)
print('KEY INSIGHTS — CUB-200-2011')
print('=' * 80)

# Best/worst FSL method per split
for split in ['b100_n50', 'b150_n50']:
    sdf = df[
        (df['split'] == split) &
        df['acc'].notna() &
        ~df['group'].str.contains('Supervised|Zero', na=False)
    ]
    if sdf.empty:
        continue
    best  = sdf.loc[sdf['acc'].idxmax()]
    worst = sdf.loc[sdf['acc'].idxmin()]
    mean_fsl = sdf['acc'].mean()
    print(f"\n  [{split}]")
    print(f"  Best  FSL method : {best['method']} @ {best['setup']} -> {best['acc']:.2f}%")
    print(f"  Worst FSL method : {worst['method']} @ {worst['setup']} -> {worst['acc']:.2f}%")
    print(f"  Mean  FSL acc    : {mean_fsl:.2f}%")

# Shot count effect
print('\n  [Shot Count Effect — ProtoNet (Euclidean)]')
for split in ['b100_n50', 'b150_n50']:
    for shot in [1, 3, 5]:
        row = df[
            (df['split'] == split) &
            (df['method'] == 'ProtoNet (Euclidean)') &
            (df['setup'] == f'5-way {shot}-shot')
        ]
        if not row.empty and row.iloc[0]['acc'] is not None:
            print(f"  {split} {shot}-shot: {float(row.iloc[0]['acc']):.2f}%")

# Base diversity effect
print('\n  [Base Diversity Effect — b100_n50 vs b150_n50, ProtoNet (Euclidean)]')
for shot in [1, 3, 5]:
    r100 = df[
        (df['split'] == 'b100_n50') &
        (df['method'] == 'ProtoNet (Euclidean)') &
        (df['setup'] == f'5-way {shot}-shot')
    ]
    r150 = df[
        (df['split'] == 'b150_n50') &
        (df['method'] == 'ProtoNet (Euclidean)') &
        (df['setup'] == f'5-way {shot}-shot')
    ]
    if (not r100.empty and not r150.empty
            and r100.iloc[0]['acc'] is not None
            and r150.iloc[0]['acc'] is not None):
        gap = float(r150.iloc[0]['acc']) - float(r100.iloc[0]['acc'])
        print(f"  {shot}-shot gap (b150-b100): {gap:+.2f}%")

print('=' * 80)
print(f"\nAll outputs saved to: results/cub/comparison/")
