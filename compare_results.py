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
RES  = os.path.join(ROOT, 'results')


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

# ProtoNet: two distance metrics × three k-shots (ablation)
# NCC Baseline: base-15 backbone (frozen) + nearest centroid — no episodic training
ncc_1_euc = load('baseline/resnet50/base15/novel_ncc/1shot_euclidean/summary.json')
ncc_1_man = load('baseline/resnet50/base15/novel_ncc/1shot_manhattan/summary.json')
ncc_3_euc = load('baseline/resnet50/base15/novel_ncc/3shot_euclidean/summary.json')
ncc_3_man = load('baseline/resnet50/base15/novel_ncc/3shot_manhattan/summary.json')
ncc_5_euc = load('baseline/resnet50/base15/novel_ncc/5shot_euclidean/summary.json')
ncc_5_man = load('baseline/resnet50/base15/novel_ncc/5shot_manhattan/summary.json')

# ProtoNet: two distance metrics × three k-shots (ablation)
proto_1_euc = load('proto_net/1shot_euclidean/summary.json')
proto_1_man = load('proto_net/1shot_manhattan/summary.json')
proto_3_euc = load('proto_net/3shot_euclidean/summary.json')
proto_3_man = load('proto_net/3shot_manhattan/summary.json')
proto_5_euc = load('proto_net/5shot_euclidean/summary.json')
proto_5_man = load('proto_net/5shot_manhattan/summary.json')
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

def from_baseline(d, label, group, note):
    """Parse a supervised baseline summary.json."""
    if d is None:
        return dict(group=group, method=label, setup='—', acc=None, ci='—',
                    f1=None, note=note)
    acc = float(d['accuracy']) * 100
    ci  = d.get('accuracy_ci_95')
    ci_str = fmt_ci(ci, scale=100) if ci else '—'
    f1  = d.get('macro_f1')
    return dict(group=group, method=label,
                setup=note,
                acc=acc, ci=ci_str,
                ci_lo=float(ci[0])*100 if ci else None,
                ci_hi=float(ci[1])*100 if ci else None,
                f1=float(f1) if f1 else None,
                note='Reference')


def from_fsl(d, label, group, k_shot):
    """Parse an FSL summary.json."""
    setup = f'5-way {k_shot}-shot'
    if d is None:
        return dict(group=group, method=label, setup=setup, acc=None, ci='—',
                    ci_lo=None, ci_hi=None, f1=None, note='Direct')
    acc   = float(d['mean_accuracy_pct'])
    ci95  = float(d['ci95_pct'])
    wci   = d.get('wilson_ci_95')
    ci_str = fmt_ci(wci, scale=1.0) if wci else f'[{acc-ci95:.2f}, {acc+ci95:.2f}]'
    ci_lo  = float(wci[0]) if wci else acc - ci95
    ci_hi  = float(wci[1]) if wci else acc + ci95
    return dict(group=group, method=label, setup=setup,
                acc=acc, ci=ci_str, ci_lo=ci_lo, ci_hi=ci_hi,
                f1=None,   # episodic eval doesn't produce a single macro-F1
                note='Direct')


rows = [
    # --- Supervised upper bounds (reference only) ---
    from_baseline(full20,   'Full-20 Baseline',  'Supervised (Reference)',
                  '20-way, full data'),
    from_baseline(base15,   'Base-15 Baseline',  'Supervised (Reference)',
                  '15-way, full data'),

    # --- Low-data supervised (direct comparator) ---
    from_baseline(kshot5,   'K=5 Fine-tune',     'Low-Data Supervised',
                  '5-way, 5-shot fine-tune'),

    # --- Zero-shot reference (always 0%) ---
    dict(group='Zero-Shot Reference', method='Zero-Shot Transfer',
         setup='5-way, 0-shot', acc=0.0, ci='[0.0, 0.0]',
         ci_lo=0.0, ci_hi=0.0, f1=None,
         note='0% by design (15-class head cannot output novel labels)'),

    # --- NCC Baseline (direct comparator to ProtoNet — same K, same distance) ---
    from_fsl(ncc_1_euc, 'NCC (Euclidean)', 'NCC 1-Shot', k_shot=1),
    from_fsl(ncc_1_man, 'NCC (Manhattan)', 'NCC 1-Shot', k_shot=1),
    from_fsl(ncc_3_euc, 'NCC (Euclidean)', 'NCC 3-Shot', k_shot=3),
    from_fsl(ncc_3_man, 'NCC (Manhattan)', 'NCC 3-Shot', k_shot=3),
    from_fsl(ncc_5_euc, 'NCC (Euclidean)', 'NCC 5-Shot', k_shot=5),
    from_fsl(ncc_5_man, 'NCC (Manhattan)', 'NCC 5-Shot', k_shot=5),

    # --- FSL 1-shot ---
    from_fsl(proto_1_euc, 'ProtoNet (Euclidean)', 'FSL 1-Shot', k_shot=1),
    from_fsl(proto_1_man, 'ProtoNet (Manhattan)', 'FSL 1-Shot', k_shot=1),
    from_fsl(match_1,     'MatchingNet (Cosine)', 'FSL 1-Shot', k_shot=1),
    from_fsl(rel_1,       'RelationNet (Learned)','FSL 1-Shot', k_shot=1),

    # --- FSL 3-shot ---
    from_fsl(proto_3_euc, 'ProtoNet (Euclidean)', 'FSL 3-Shot', k_shot=3),
    from_fsl(proto_3_man, 'ProtoNet (Manhattan)', 'FSL 3-Shot', k_shot=3),
    from_fsl(match_3,     'MatchingNet (Cosine)', 'FSL 3-Shot', k_shot=3),
    from_fsl(rel_3,       'RelationNet (Learned)','FSL 3-Shot', k_shot=3),

    # --- FSL 5-shot ---
    from_fsl(proto_5_euc, 'ProtoNet (Euclidean)', 'FSL 5-Shot', k_shot=5),
    from_fsl(proto_5_man, 'ProtoNet (Manhattan)', 'FSL 5-Shot', k_shot=5),
    from_fsl(match_5,     'MatchingNet (Cosine)', 'FSL 5-Shot', k_shot=5),
    from_fsl(rel_5,       'RelationNet (Learned)','FSL 5-Shot', k_shot=5),
]

df = pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------

DISPLAY_COLS = ['group', 'method', 'setup', 'acc', 'ci', 'f1', 'note']

def fmt_acc(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '—'
    return f'{v:.2f}%'

def fmt_f1_cell(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '—'
    return f'{v:.4f}'

print('\n' + '=' * 100)
print('THESIS COMPARISON TABLE  —  Novel Class Evaluation (5 Novel Classes)')
print('=' * 100)
print(f"{'Group':<28} {'Method':<22} {'Setup':<25} {'Accuracy':>10} {'95% CI':>22} {'Macro F1':>10}")
print('-' * 100)

current_group = None
for _, r in df.iterrows():
    if r['group'] != current_group:
        if current_group is not None:
            print()
        current_group = r['group']
        print(f"  [{r['group']}]")
    acc_str = fmt_acc(r['acc'])
    f1_str  = fmt_f1_cell(r['f1'])
    print(f"  {'':4}{r['method']:<22} {r['setup']:<25} {acc_str:>10} {r['ci']:>22} {f1_str:>10}")

print('=' * 100)
print('\nNotes:')
print('  Reference  : Full-data supervised results. NOT directly comparable to few-shot methods.')
print('  Direct     : Evaluated on same 5 novel classes with same K images — directly comparable.')
print('  Zero-Shot  : Model trained on 15 base classes; novel class images fed to 15-class head.')
print('  Epochs     : Debug runs (1 epoch). Real results require full training on GPU.')
print()


# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------

out_dir = os.path.join(RES, 'comparison')
os.makedirs(out_dir, exist_ok=True)

csv_df = df[['group', 'method', 'setup', 'acc', 'ci', 'f1', 'note']].copy()
csv_df['acc'] = csv_df['acc'].apply(lambda v: f'{v:.2f}' if v is not None else '—')
csv_df['f1']  = csv_df['f1'].apply(lambda v: f'{v:.4f}' if v is not None else '—')
csv_df.to_csv(os.path.join(out_dir, 'comparison_table.csv'), index=False)
print(f"Saved: results/comparison/comparison_table.csv")


# ---------------------------------------------------------------------------
# Save LaTeX table
# ---------------------------------------------------------------------------

def tex_escape(s):
    return s.replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')

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
    r'\caption{Comparison of Methods on Novel Bird Species Classification (5 Novel Classes). '
    r'Reference results use full training data and are not directly comparable to few-shot methods. '
    r'Direct results use the same K images per novel class as the FSL methods.}',
    r'\label{tab:results_comparison}',
    r'\begin{tabular}{llccc}',
    r'\hline',
    r'\textbf{Group} & \textbf{Method} & \textbf{Accuracy (\%)} & \textbf{95\% CI} & \textbf{Macro F1} \\',
    r'\hline',
]

current_group = None
for _, r in df.iterrows():
    if r['group'] != current_group:
        if current_group is not None:
            lines.append(r'\hline')
        current_group = r['group']
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{tex_escape(r['group'])}}}}}" + r' \\')

    group_cell = ''   # don't repeat group name per row
    method_cell = tex_escape(str(r['method']))
    acc_cell    = acc_tex(r['acc'])
    ci_cell     = ci_tex(r['ci']).replace('%', r'\%')
    f1_cell     = f1_tex(r['f1'])
    lines.append(f"  & {method_cell} & {acc_cell} & {ci_cell} & {f1_cell} \\\\")

lines += [
    r'\hline',
    r'\end{tabular}',
    r'\end{table}',
]

with open(os.path.join(out_dir, 'comparison_table.tex'), 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f"Saved: results/comparison/comparison_table.tex")


# ---------------------------------------------------------------------------
# Bar chart — only rows with actual results and that are "Direct" or FSL
# ---------------------------------------------------------------------------

plot_df = df[df['acc'].notna()].copy()

# Assign colours by group
group_colors = {
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

fig, ax = plt.subplots(figsize=(13, 6))

x_labels = []
x_accs   = []
x_errs   = []
x_colors = []

for _, r in plot_df.iterrows():
    label = f"{r['method']}\n({r['setup']})"
    acc   = float(r['acc'])
    # error bar = half-width of CI if available
    if r['ci_lo'] is not None and r['ci_hi'] is not None:
        err = (float(r['ci_hi']) - float(r['ci_lo'])) / 2
    else:
        err = 0
    x_labels.append(label)
    x_accs.append(acc)
    x_errs.append(err)
    x_colors.append(group_colors.get(r['group'], '#aaaaaa'))

x = np.arange(len(x_labels))
bars = ax.bar(x, x_accs, color=x_colors, edgecolor='white', linewidth=0.8)
ax.errorbar(x, x_accs, yerr=x_errs, fmt='none', color='black',
            capsize=4, capthick=1.2, linewidth=1.2)

# Annotate each bar
for bar, acc in zip(bars, x_accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=8.5, rotation=20, ha='right')
ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_ylim(0, 115)
ax.set_title('Method Comparison — Novel Bird Species Classification\n'
             '(error bars = 95% CI; debug results shown until full training is run)',
             fontsize=11)
ax.axhline(y=20, color='gray', linestyle='--', linewidth=0.8, alpha=0.5,
           label='Random chance (5-way)')

# Legend for groups
patches = [mpatches.Patch(color=c, label=g) for g, c in group_colors.items()
           if g in plot_df['group'].values]
ax.legend(handles=patches + [
    plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=0.8,
               label='Random chance (5-way)')
], loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'comparison_bar_chart.png'), dpi=150)
plt.close()
print(f"Saved: results/comparison/comparison_bar_chart.png")

print(f"\nAll outputs in: results/comparison/")
