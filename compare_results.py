"""
compare_results.py — Thesis comparison table generator.

Reads all experiment summary.json files, builds a unified comparison table,
and saves:
  - Console output (formatted table)
  - results/comparison/comparison_table.csv
  - results/comparison/comparison_table.tex  (ready to paste into LaTeX)
  - results/comparison/comparison_bar_chart.png

Missing results (not yet run) are shown as '—' gracefully.

Usage:
    python compare_results.py
"""
import os
import json
import math

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Project root = directory containing this script
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, 'results')


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

full20   = load('baseline/resnet50/full20/summary.json')
base15   = load('baseline/resnet50/base15/base_test/summary.json')
kshot5   = load('baseline/resnet50/base15/novel_kshot_5/summary.json')
zeroshot = load('baseline/resnet50/base15/novel_zeroshot/summary.json')

# NCC Baseline: base-15 backbone (frozen) + nearest centroid — no episodic training
ncc_1_euc = load('baseline/resnet50/base15/novel_ncc/1shot_euclidean/summary.json')
ncc_1_man = load('baseline/resnet50/base15/novel_ncc/1shot_manhattan/summary.json')
ncc_3_euc = load('baseline/resnet50/base15/novel_ncc/3shot_euclidean/summary.json')
ncc_3_man = load('baseline/resnet50/base15/novel_ncc/3shot_manhattan/summary.json')
ncc_5_euc = load('baseline/resnet50/base15/novel_ncc/5shot_euclidean/summary.json')
ncc_5_man = load('baseline/resnet50/base15/novel_ncc/5shot_manhattan/summary.json')

# ProtoNet: base15 / novel5
proto_1_euc = load('proto_net/1shot_euclidean/summary.json')
proto_1_man = load('proto_net/1shot_manhattan/summary.json')
proto_3_euc = load('proto_net/3shot_euclidean/summary.json')
proto_3_man = load('proto_net/3shot_manhattan/summary.json')
proto_5_euc = load('proto_net/5shot_euclidean/summary.json')
proto_5_man = load('proto_net/5shot_manhattan/summary.json')

# ProtoNet: base10 / novel10
proto_b10n10_1_euc = load('proto_net/base10_novel10/1shot_euclidean/summary.json')
proto_b10n10_1_man = load('proto_net/base10_novel10/1shot_manhattan/summary.json')
proto_b10n10_3_euc = load('proto_net/base10_novel10/3shot_euclidean/summary.json')
proto_b10n10_3_man = load('proto_net/base10_novel10/3shot_manhattan/summary.json')
proto_b10n10_5_euc = load('proto_net/base10_novel10/5shot_euclidean/summary.json')
proto_b10n10_5_man = load('proto_net/base10_novel10/5shot_manhattan/summary.json')

# MatchingNet (cosine) and RelationNet (learned) — distance fixed by architecture
match_1   = load('matching_net/1shot/summary.json')
match_3   = load('matching_net/3shot/summary.json')
match_5   = load('matching_net/5shot/summary.json')
rel_1     = load('relation_net/1shot/summary.json')
rel_3     = load('relation_net/3shot/summary.json')
rel_5     = load('relation_net/5shot/summary.json')


# ---------------------------------------------------------------------------
# Normalise fields into a unified dict per experiment
# Baselines report fraction (0.99); FSL reports percentage (99.0)
# We store everything as percentage floats (or None)
# ---------------------------------------------------------------------------

def from_baseline(d, label, group, note, split='b15_n5'):
    """Parse a supervised baseline summary.json."""
    if d is None:
        return dict(split=split, group=group, method=label, setup='—', acc=None, ci='—',
                    ci_lo=None, ci_hi=None, f1=None, note=note)
    acc = float(d['accuracy']) * 100
    ci  = d.get('accuracy_ci_95')
    ci_str = fmt_ci(ci, scale=100) if ci else '—'
    f1  = d.get('macro_f1')
    return dict(split=split, group=group, method=label,
                setup=note,
                acc=acc, ci=ci_str,
                ci_lo=float(ci[0])*100 if ci else None,
                ci_hi=float(ci[1])*100 if ci else None,
                f1=float(f1) if f1 else None,
                note='Reference')


def from_fsl(d, label, group, k_shot, split='b15_n5', n_way=5):
    """Parse an FSL summary.json."""
    setup = f'{n_way}-way {k_shot}-shot'
    if d is None:
        return dict(split=split, group=group, method=label, setup=setup, acc=None, ci='—',
                    ci_lo=None, ci_hi=None, f1=None, note='Direct')
    acc   = float(d['mean_accuracy_pct'])
    ci95  = float(d['ci95_pct'])
    wci   = d.get('wilson_ci_95')
    ci_str = fmt_ci(wci, scale=1.0) if wci else f'[{acc-ci95:.2f}, {acc+ci95:.2f}]'
    ci_lo  = float(wci[0]) if wci else acc - ci95
    ci_hi  = float(wci[1]) if wci else acc + ci95
    return dict(split=split, group=group, method=label, setup=setup,
                acc=acc, ci=ci_str, ci_lo=ci_lo, ci_hi=ci_hi,
                f1=float(d['macro_f1']) if d.get('macro_f1') is not None else None,
                note='Direct')


rows = [
    # --- Supervised upper bounds (reference only) ---
    from_baseline(full20,   'Full-20 Baseline',  'Supervised (Reference)',
                  '20-way, full data', split='b15_n5'),
    from_baseline(base15,   'Base-15 Baseline',  'Supervised (Reference)',
                  '15-way, full data', split='b15_n5'),

    # --- Low-data supervised (direct comparator) ---
    from_baseline(kshot5,   'K=5 Fine-tune',     'Low-Data Supervised',
                  '5-way, 5-shot fine-tune', split='b15_n5'),

    # --- Zero-shot reference (always 0%) ---
    dict(split='b15_n5', group='Zero-Shot Reference', method='Zero-Shot Transfer',
         setup='5-way, 0-shot', acc=0.0, ci='[0.0, 0.0]',
         ci_lo=0.0, ci_hi=0.0, f1=None,
         note='0% by design (15-class head cannot output novel labels)'),

    # --- NCC Baseline (direct comparator to ProtoNet — same K, same distance) ---
    from_fsl(ncc_1_euc, 'NCC (Euclidean)', 'NCC 1-Shot', k_shot=1, split='b15_n5'),
    from_fsl(ncc_1_man, 'NCC (Manhattan)', 'NCC 1-Shot', k_shot=1, split='b15_n5'),
    from_fsl(ncc_3_euc, 'NCC (Euclidean)', 'NCC 3-Shot', k_shot=3, split='b15_n5'),
    from_fsl(ncc_3_man, 'NCC (Manhattan)', 'NCC 3-Shot', k_shot=3, split='b15_n5'),
    from_fsl(ncc_5_euc, 'NCC (Euclidean)', 'NCC 5-Shot', k_shot=5, split='b15_n5'),
    from_fsl(ncc_5_man, 'NCC (Manhattan)', 'NCC 5-Shot', k_shot=5, split='b15_n5'),

    # --- FSL 1-shot ---
    from_fsl(proto_1_euc, 'ProtoNet (Euclidean)', 'FSL 1-Shot', k_shot=1, split='b15_n5'),
    from_fsl(proto_1_man, 'ProtoNet (Manhattan)', 'FSL 1-Shot', k_shot=1, split='b15_n5'),
    from_fsl(match_1,     'MatchingNet (Cosine)', 'FSL 1-Shot', k_shot=1, split='b15_n5'),
    from_fsl(rel_1,       'RelationNet (Learned)','FSL 1-Shot', k_shot=1, split='b15_n5'),

    # --- FSL 3-shot ---
    from_fsl(proto_3_euc, 'ProtoNet (Euclidean)', 'FSL 3-Shot', k_shot=3, split='b15_n5'),
    from_fsl(proto_3_man, 'ProtoNet (Manhattan)', 'FSL 3-Shot', k_shot=3, split='b15_n5'),
    from_fsl(match_3,     'MatchingNet (Cosine)', 'FSL 3-Shot', k_shot=3, split='b15_n5'),
    from_fsl(rel_3,       'RelationNet (Learned)','FSL 3-Shot', k_shot=3, split='b15_n5'),

    # --- FSL 5-shot ---
    from_fsl(proto_5_euc, 'ProtoNet (Euclidean)', 'FSL 5-Shot', k_shot=5, split='b15_n5'),
    from_fsl(proto_5_man, 'ProtoNet (Manhattan)', 'FSL 5-Shot', k_shot=5, split='b15_n5'),
    from_fsl(match_5,     'MatchingNet (Cosine)', 'FSL 5-Shot', k_shot=5, split='b15_n5'),
    from_fsl(rel_5,       'RelationNet (Learned)','FSL 5-Shot', k_shot=5, split='b15_n5'),

    # --- ProtoNet only: base10 / novel10 ---
    from_fsl(proto_b10n10_1_euc, 'ProtoNet (Euclidean)', 'ProtoNet b10_n10', k_shot=1, split='b10_n10'),
    from_fsl(proto_b10n10_1_man, 'ProtoNet (Manhattan)', 'ProtoNet b10_n10', k_shot=1, split='b10_n10'),
    from_fsl(proto_b10n10_3_euc, 'ProtoNet (Euclidean)', 'ProtoNet b10_n10', k_shot=3, split='b10_n10'),
    from_fsl(proto_b10n10_3_man, 'ProtoNet (Manhattan)', 'ProtoNet b10_n10', k_shot=3, split='b10_n10'),
    from_fsl(proto_b10n10_5_euc, 'ProtoNet (Euclidean)', 'ProtoNet b10_n10', k_shot=5, split='b10_n10'),
    from_fsl(proto_b10n10_5_man, 'ProtoNet (Manhattan)', 'ProtoNet b10_n10', k_shot=5, split='b10_n10'),
]

df = pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------

DISPLAY_COLS = ['split', 'group', 'method', 'setup', 'acc', 'ci', 'f1', 'note']

def fmt_acc(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '—'
    return f'{v:.2f}%'

def fmt_f1_cell(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '—'
    return f'{v:.4f}'

print('\n' + '=' * 110)
print('THESIS COMPARISON TABLE')
print('=' * 110)
print(f"{'Split':<10} {'Group':<24} {'Method':<24} {'Setup':<18} {'Accuracy':>10} {'95% CI':>22} {'Macro F1':>10}")
print('-' * 110)

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

    acc_str = fmt_acc(r['acc'])
    f1_str  = fmt_f1_cell(r['f1'])
    print(f"    {r['method']:<24} {r['setup']:<18} {acc_str:>10} {r['ci']:>22} {f1_str:>10}")

print('=' * 110)
print('\nNotes:')
print('  b15_n5      : original 15 base / 5 novel setup')
print('  b10_n10     : new 10 base / 10 novel setup (ProtoNet only for now)')
print('  Reference   : Full-data supervised results. NOT directly comparable to few-shot methods.')
print('  Direct      : Evaluated on novel classes with same K images.')
print()


# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------

out_dir = os.path.join(RES, 'comparison')
os.makedirs(out_dir, exist_ok=True)

csv_df = df[['split', 'group', 'method', 'setup', 'acc', 'ci', 'f1', 'note']].copy()
csv_df['acc'] = csv_df['acc'].apply(
    lambda v: f'{v:.2f}' if v is not None and not (isinstance(v, float) and math.isnan(v)) else '—'
)
csv_df['f1'] = csv_df['f1'].apply(
    lambda v: f'{v:.4f}' if v is not None and not (isinstance(v, float) and math.isnan(v)) else '—'
)
csv_df.to_csv(os.path.join(out_dir, 'comparison_table.csv'), index=False)
print(f"Saved: results/comparison/comparison_table.csv")


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

lines = [
    r'\begin{table}[htbp]',
    r'\centering',
    r'\caption{Comparison of methods on Birds-20 novel class evaluation. '
    r'Includes the original base15/novel5 setup and ProtoNet-only results for the new base10/novel10 setup.}',
    r'\label{tab:results_comparison}',
    r'\begin{tabular}{lllcc}',
    r'\hline',
    r'\textbf{Split} & \textbf{Method} & \textbf{Accuracy (\%)} & \textbf{95\% CI} & \textbf{Macro F1} \\',
    r'\hline',
]

current_split = None
for _, r in df.iterrows():
    if r['split'] != current_split:
        if current_split is not None:
            lines.append(r'\hline')
        current_split = r['split']
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{tex_escape(r['split'])}}}}}" + r' \\')

    method_cell = tex_escape(str(r['method']))
    acc_cell    = acc_tex(r['acc'])
    ci_cell     = ci_tex(r['ci']).replace('%', r'\%')
    f1_cell     = f1_tex(r['f1'])
    lines.append(f"{tex_escape(r['split'])} & {method_cell} & {acc_cell} & {ci_cell} & {f1_cell} \\\\")

lines += [
    r'\hline',
    r'\end{tabular}',
    r'\end{table}',
]

with open(os.path.join(out_dir, 'comparison_table.tex'), 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f"Saved: results/comparison/comparison_table.tex")


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------

# plot_df = df[df['acc'].notna()].copy()

# group_colors = {
#     'Supervised (Reference)': '#9ecae1',
#     'Low-Data Supervised':    '#2171b5',
#     'Zero-Shot Reference':    '#bdbdbd',
#     'NCC 1-Shot':             '#fdd0a2',
#     'NCC 3-Shot':             '#fd8d3c',
#     'NCC 5-Shot':             '#d94801',
#     'FSL 1-Shot':             '#c7e9c0',
#     'FSL 3-Shot':             '#74c476',
#     'FSL 5-Shot':             '#238b45',
#     'ProtoNet b10_n10':       '#756bb1',
# }

# split_hatches = {
#     'b15_n5': '',
#     'b10_n10': '//',
# }

# fig, ax = plt.subplots(figsize=(15, 7))

# x_labels = []
# x_accs   = []
# x_errs   = []
# x_colors = []
# x_hatch  = []

# for _, r in plot_df.iterrows():
#     label = f"{r['method']}\n({r['setup']}, {r['split']})"
#     acc   = float(r['acc'])

#     if r['ci_lo'] is not None and r['ci_hi'] is not None:
#         err = (float(r['ci_hi']) - float(r['ci_lo'])) / 2
#     else:
#         err = 0

#     x_labels.append(label)
#     x_accs.append(acc)
#     x_errs.append(err)
#     x_colors.append(group_colors.get(r['group'], '#aaaaaa'))
#     x_hatch.append(split_hatches.get(r['split'], ''))

# x = np.arange(len(x_labels))
# bars = ax.bar(x, x_accs, color=x_colors, edgecolor='white', linewidth=0.8)

# for bar, hatch in zip(bars, x_hatch):
#     bar.set_hatch(hatch)

# ax.errorbar(x, x_accs, yerr=x_errs, fmt='none', color='black',
#             capsize=4, capthick=1.2, linewidth=1.2)

# for bar, acc in zip(bars, x_accs):
#     ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
#             f'{acc:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

# ax.set_xticks(x)
# ax.set_xticklabels(x_labels, fontsize=8, rotation=25, ha='right')
# ax.set_ylabel('Accuracy (%)', fontsize=11)
# ax.set_ylim(0, 115)
# ax.set_title('Method Comparison — Birds-20 Novel Class Evaluation\n'
#              '(includes ProtoNet base10/novel10 results)',
#              fontsize=11)
# ax.axhline(y=20, color='gray', linestyle='--', linewidth=0.8, alpha=0.5,
#            label='Random chance (5-way)')

# patches = [
#     mpatches.Patch(color=c, label=g) for g, c in group_colors.items()
#     if g in plot_df['group'].values
# ]
# split_patches = [
#     mpatches.Patch(facecolor='white', edgecolor='black', hatch=h, label=s)
#     for s, h in split_hatches.items() if s in plot_df['split'].values
# ]

# ax.legend(
#     handles=patches + split_patches + [
#         plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=0.8,
#                    label='Random chance (5-way)')
#     ],
#     loc='upper right',
#     fontsize=8
# )

# plt.tight_layout()
# plt.savefig(os.path.join(out_dir, 'comparison_bar_chart.png'), dpi=150)
# plt.close()
# print(f"Saved: results/comparison/comparison_bar_chart.png")

# print(f"\nAll outputs in: results/comparison/")


# ---------------------------------------------------------------------------
# Clean charts
#   1) Main comparison for b15_n5
#   2) ProtoNet split comparison: b15_n5 vs b10_n10
# ---------------------------------------------------------------------------

def half_ci_width(row):
    if row['ci_lo'] is not None and row['ci_hi'] is not None:
        return (float(row['ci_hi']) - float(row['ci_lo'])) / 2.0
    return 0.0


def save_b15_main_chart(df, out_dir):
    """
    Main comparison chart for the original b15_n5 setup only.
    Includes the key methods, grouped by shot setting.
    """
    plot_df = df[
        (df['split'] == 'b15_n5') &
        (df['acc'].notna())
    ].copy()

    # Keep the most important comparisons only
    keep_methods = {
        'Full-20 Baseline',
        'Base-15 Baseline',
        'K=5 Fine-tune',
        'Zero-Shot Transfer',
        'NCC (Euclidean)',
        'NCC (Manhattan)',
        'ProtoNet (Euclidean)',
        'ProtoNet (Manhattan)',
        'MatchingNet (Cosine)',
        'RelationNet (Learned)',
    }
    plot_df = plot_df[plot_df['method'].isin(keep_methods)].copy()

    # Explicit order
    ordered_pairs = [
        ('Full-20 Baseline', '20-way, full data'),
        ('Base-15 Baseline', '15-way, full data'),
        ('K=5 Fine-tune', '5-way, 5-shot fine-tune'),
        ('Zero-Shot Transfer', '5-way, 0-shot'),

        ('NCC (Euclidean)', '5-way 1-shot'),
        ('NCC (Manhattan)', '5-way 1-shot'),
        ('ProtoNet (Euclidean)', '5-way 1-shot'),
        ('ProtoNet (Manhattan)', '5-way 1-shot'),
        ('MatchingNet (Cosine)', '5-way 1-shot'),
        ('RelationNet (Learned)', '5-way 1-shot'),

        ('NCC (Euclidean)', '5-way 3-shot'),
        ('NCC (Manhattan)', '5-way 3-shot'),
        ('ProtoNet (Euclidean)', '5-way 3-shot'),
        ('ProtoNet (Manhattan)', '5-way 3-shot'),
        ('MatchingNet (Cosine)', '5-way 3-shot'),
        ('RelationNet (Learned)', '5-way 3-shot'),

        ('NCC (Euclidean)', '5-way 5-shot'),
        ('NCC (Manhattan)', '5-way 5-shot'),
        ('ProtoNet (Euclidean)', '5-way 5-shot'),
        ('ProtoNet (Manhattan)', '5-way 5-shot'),
        ('MatchingNet (Cosine)', '5-way 5-shot'),
        ('RelationNet (Learned)', '5-way 5-shot'),
    ]

    rows = []
    for method, setup in ordered_pairs:
        match = plot_df[(plot_df['method'] == method) & (plot_df['setup'] == setup)]
        if len(match) == 0:
            rows.append({
                'method': method,
                'setup': setup,
                'acc': np.nan,
                'err': 0.0,
                'group': None,
            })
        else:
            r = match.iloc[0]
            rows.append({
                'method': r['method'],
                'setup': r['setup'],
                'acc': float(r['acc']),
                'err': half_ci_width(r),
                'group': r['group'],
            })

    chart_df = pd.DataFrame(rows)

    color_map = {
        'Supervised (Reference)': '#9ecae1',
        'Low-Data Supervised': '#2171b5',
        'Zero-Shot Reference': '#bdbdbd',
        'NCC 1-Shot': '#fdd0a2',
        'NCC 3-Shot': '#fd8d3c',
        'NCC 5-Shot': '#d94801',
        'FSL 1-Shot': '#c7e9c0',
        'FSL 3-Shot': '#74c476',
        'FSL 5-Shot': '#238b45',
    }

    def short_label(method, setup):
        if method == 'Full-20 Baseline':
            return 'Full-20'
        if method == 'Base-15 Baseline':
            return 'Base-15'
        if method == 'K=5 Fine-tune':
            return 'Fine-tune'
        if method == 'Zero-Shot Transfer':
            return 'Zero-shot'

        shot = setup.replace('5-way ', '').replace('-shot', 's')
        if method == 'NCC (Euclidean)':
            return f'NCC-E\n{shot}'
        if method == 'NCC (Manhattan)':
            return f'NCC-M\n{shot}'
        if method == 'ProtoNet (Euclidean)':
            return f'Proto-E\n{shot}'
        if method == 'ProtoNet (Manhattan)':
            return f'Proto-M\n{shot}'
        if method == 'MatchingNet (Cosine)':
            return f'Match\n{shot}'
        if method == 'RelationNet (Learned)':
            return f'RelNet\n{shot}'
        return method

    x_labels = [short_label(r['method'], r['setup']) for _, r in chart_df.iterrows()]
    x_vals = chart_df['acc'].tolist()
    x_errs = chart_df['err'].tolist()
    x_colors = [color_map.get(g, '#aaaaaa') if g is not None else '#eeeeee'
                for g in chart_df['group']]

    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(x_labels))
    bars = ax.bar(x, x_vals, color=x_colors, edgecolor='white', linewidth=0.8)

    ax.errorbar(
        x, x_vals, yerr=x_errs, fmt='none',
        color='black', capsize=4, capthick=1.1, linewidth=1.1
    )

    for bar, val in zip(bars, x_vals):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.8,
                f'{val:.1f}%',
                ha='center', va='bottom',
                fontsize=8, fontweight='bold'
            )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_ylim(0, 112)
    ax.set_title(
        'Birds-20 Method Comparison — b15_n5\n'
        '(clean view of baselines, NCC, and few-shot models)',
        fontsize=12
    )
    ax.axhline(y=20, color='gray', linestyle='--', linewidth=0.9, alpha=0.6)

    # vertical separators
    for xpos in [3.5, 9.5, 15.5]:
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
    ax.legend(
        handles=group_patches + [
            plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=0.9,
                       label='Random chance (5-way)')
        ],
        loc='upper right',
        fontsize=8
    )

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'comparison_bar_chart_b15_n5.png'), dpi=150)
    plt.close()
    print("Saved: results/comparison/comparison_bar_chart_b15_n5.png")


def save_proto_split_chart(df, out_dir):
    """
    Focused ProtoNet comparison between b15_n5 and b10_n10.
    """
    plot_df = df[
        (df['method'].isin(['ProtoNet (Euclidean)', 'ProtoNet (Manhattan)'])) &
        (df['split'].isin(['b15_n5', 'b10_n10'])) &
        (df['acc'].notna())
    ].copy()

    ordered_pairs = [
        ('b15_n5', 'ProtoNet (Euclidean)', '5-way 1-shot'),
        ('b10_n10', 'ProtoNet (Euclidean)', '5-way 1-shot'),
        ('b15_n5', 'ProtoNet (Manhattan)', '5-way 1-shot'),
        ('b10_n10', 'ProtoNet (Manhattan)', '5-way 1-shot'),

        ('b15_n5', 'ProtoNet (Euclidean)', '5-way 3-shot'),
        ('b10_n10', 'ProtoNet (Euclidean)', '5-way 3-shot'),
        ('b15_n5', 'ProtoNet (Manhattan)', '5-way 3-shot'),
        ('b10_n10', 'ProtoNet (Manhattan)', '5-way 3-shot'),

        ('b15_n5', 'ProtoNet (Euclidean)', '5-way 5-shot'),
        ('b10_n10', 'ProtoNet (Euclidean)', '5-way 5-shot'),
        ('b15_n5', 'ProtoNet (Manhattan)', '5-way 5-shot'),
        ('b10_n10', 'ProtoNet (Manhattan)', '5-way 5-shot'),
    ]

    rows = []
    for split, method, setup in ordered_pairs:
        match = plot_df[
            (plot_df['split'] == split) &
            (plot_df['method'] == method) &
            (plot_df['setup'] == setup)
        ]
        if len(match) == 0:
            rows.append({
                'split': split,
                'method': method,
                'setup': setup,
                'acc': np.nan,
                'err': 0.0,
            })
        else:
            r = match.iloc[0]
            rows.append({
                'split': split,
                'method': r['method'],
                'setup': r['setup'],
                'acc': float(r['acc']),
                'err': half_ci_width(r),
            })

    chart_df = pd.DataFrame(rows)

    split_colors = {
        'b15_n5': '#5DA5DA',
        'b10_n10': '#7B6FD0',
    }
    metric_hatch = {
        'ProtoNet (Euclidean)': '',
        'ProtoNet (Manhattan)': '//',
    }

    def short_label(split, method, setup):
        shot = setup.replace('5-way ', '').replace('-shot', 's')
        metric = 'E' if 'Euclidean' in method else 'M'
        split_short = '15/5' if split == 'b15_n5' else '10/10'
        return f'{split_short}\nProto-{metric}\n{shot}'

    x_labels = [short_label(r['split'], r['method'], r['setup']) for _, r in chart_df.iterrows()]
    x_vals = chart_df['acc'].tolist()
    x_errs = chart_df['err'].tolist()

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(x_labels))
    bars = ax.bar(
        x,
        x_vals,
        color=[split_colors[r['split']] for _, r in chart_df.iterrows()],
        edgecolor='white',
        linewidth=0.8
    )

    for bar, (_, r) in zip(bars, chart_df.iterrows()):
        bar.set_hatch(metric_hatch[r['method']])

    ax.errorbar(
        x, x_vals, yerr=x_errs, fmt='none',
        color='black', capsize=4, capthick=1.1, linewidth=1.1
    )

    for bar, val in zip(bars, x_vals):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.8,
                f'{val:.1f}%',
                ha='center', va='bottom',
                fontsize=8, fontweight='bold'
            )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_ylim(0, 108)
    ax.set_title(
        'ProtoNet Split Comparison — b15_n5 vs b10_n10',
        fontsize=12
    )
    ax.axhline(y=20, color='gray', linestyle='--', linewidth=0.9, alpha=0.6)

    # separators between 1/3/5 shot groups
    for xpos in [3.5, 7.5]:
        ax.axvline(x=xpos, color='lightgray', linestyle='-', linewidth=0.8)

    legend_handles = [
        mpatches.Patch(color=split_colors['b15_n5'], label='b15_n5'),
        mpatches.Patch(color=split_colors['b10_n10'], label='b10_n10'),
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='', label='Euclidean'),
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='//', label='Manhattan'),
        plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=0.9,
                   label='Random chance (5-way)'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'comparison_bar_chart_proto_split.png'), dpi=150)
    plt.close()
    print("Saved: results/comparison/comparison_bar_chart_proto_split.png")


# Generate the two cleaner plots
save_b15_main_chart(df, out_dir)
save_proto_split_chart(df, out_dir)