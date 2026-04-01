"""
Build all updated notebooks with new experiment results.
Run from project root: python notebooks/_build_notebooks.py
"""
import json, os

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source if isinstance(source, list) else [source]}

def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source if isinstance(source, list) else [source]}

def nb(cells):
    return {"nbformat": 4, "nbformat_minor": 4,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.10.0"}},
            "cells": cells}

HERE = os.path.dirname(os.path.abspath(__file__))

# ===========================================================================
# SHARED SETUP CODE
# ===========================================================================
SETUP_15 = """
%matplotlib inline
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi']=120; plt.rcParams['savefig.dpi']=150
plt.rcParams['figure.facecolor']='white'; plt.rcParams['axes.facecolor']='white'

NB_DIR=os.getcwd()
ROOT=os.path.normpath(os.path.join(NB_DIR,'..'))
RES=os.path.join(ROOT,'results')
ANALYSIS=os.path.join(RES,'analysis')
os.makedirs(ANALYSIS,exist_ok=True)

PATHS={
    'full20_json'              :f'{RES}/baseline/resnet50/full20/summary.json',
    'base15_json'              :f'{RES}/baseline/resnet50/base15/base_test/summary.json',
    'kshot5_json'              :f'{RES}/baseline/resnet50/base15/novel_kshot_5/summary.json',
    'ncc_1shot_euc_json'       :f'{RES}/baseline/resnet50/base15/novel_ncc/1shot_euclidean/summary.json',
    'ncc_1shot_man_json'       :f'{RES}/baseline/resnet50/base15/novel_ncc/1shot_manhattan/summary.json',
    'ncc_3shot_euc_json'       :f'{RES}/baseline/resnet50/base15/novel_ncc/3shot_euclidean/summary.json',
    'ncc_3shot_man_json'       :f'{RES}/baseline/resnet50/base15/novel_ncc/3shot_manhattan/summary.json',
    'ncc_5shot_euc_json'       :f'{RES}/baseline/resnet50/base15/novel_ncc/5shot_euclidean/summary.json',
    'ncc_5shot_man_json'       :f'{RES}/baseline/resnet50/base15/novel_ncc/5shot_manhattan/summary.json',
    'proto_1shot_euc_json'     :f'{RES}/proto_net/base15_novel5/1shot_euclidean/summary.json',
    'proto_1shot_man_json'     :f'{RES}/proto_net/base15_novel5/1shot_manhattan/summary.json',
    'proto_3shot_euc_json'     :f'{RES}/proto_net/base15_novel5/3shot_euclidean/summary.json',
    'proto_3shot_man_json'     :f'{RES}/proto_net/base15_novel5/3shot_manhattan/summary.json',
    'proto_5shot_euc_json'     :f'{RES}/proto_net/base15_novel5/5shot_euclidean/summary.json',
    'proto_5shot_man_json'     :f'{RES}/proto_net/base15_novel5/5shot_manhattan/summary.json',
    'proto_5shot_euc_tta8_json':f'{RES}/proto_net/base15_novel5/5shot_euclidean_tta8/summary.json',
    'match_1shot_json'         :f'{RES}/matching_net/base15_novel5/1shot/summary.json',
    'match_3shot_json'         :f'{RES}/matching_net/base15_novel5/3shot/summary.json',
    'match_5shot_json'         :f'{RES}/matching_net/base15_novel5/5shot/summary.json',
    'match_fce_1shot_json'     :f'{RES}/matching_net_fce/base15_novel5/1shot/summary.json',
    'match_fce_3shot_json'     :f'{RES}/matching_net_fce/base15_novel5/3shot/summary.json',
    'match_fce_5shot_json'     :f'{RES}/matching_net_fce/base15_novel5/5shot/summary.json',
    'rel_1shot_json'           :f'{RES}/relation_net/base15_novel5/1shot/summary.json',
    'rel_3shot_json'           :f'{RES}/relation_net/base15_novel5/3shot/summary.json',
    'rel_5shot_json'           :f'{RES}/relation_net/base15_novel5/5shot/summary.json',
    'vit_ncc_1shot_euc_json'   :f'{RES}/vit/baseline/base15/novel_ncc/1shot_euclidean/summary.json',
    'vit_ncc_1shot_man_json'   :f'{RES}/vit/baseline/base15/novel_ncc/1shot_manhattan/summary.json',
    'vit_ncc_3shot_euc_json'   :f'{RES}/vit/baseline/base15/novel_ncc/3shot_euclidean/summary.json',
    'vit_ncc_3shot_man_json'   :f'{RES}/vit/baseline/base15/novel_ncc/3shot_manhattan/summary.json',
    'vit_ncc_5shot_euc_json'   :f'{RES}/vit/baseline/base15/novel_ncc/5shot_euclidean/summary.json',
    'vit_ncc_5shot_man_json'   :f'{RES}/vit/baseline/base15/novel_ncc/5shot_manhattan/summary.json',
    'vit_proto_1shot_euc_json' :f'{RES}/vit/proto_net/base15_novel5/1shot_euclidean/summary.json',
    'vit_proto_1shot_man_json' :f'{RES}/vit/proto_net/base15_novel5/1shot_manhattan/summary.json',
    'vit_proto_3shot_euc_json' :f'{RES}/vit/proto_net/base15_novel5/3shot_euclidean/summary.json',
    'vit_proto_3shot_man_json' :f'{RES}/vit/proto_net/base15_novel5/3shot_manhattan/summary.json',
    'vit_proto_5shot_euc_json' :f'{RES}/vit/proto_net/base15_novel5/5shot_euclidean/summary.json',
    'vit_proto_5shot_man_json' :f'{RES}/vit/proto_net/base15_novel5/5shot_manhattan/summary.json',
}

R={}
for k,p in PATHS.items():
    if os.path.exists(p):
        with open(p) as f: R[k]=json.load(f)
    else:
        R[k]=None; print(f'  [missing] {k}')
present=sum(1 for v in R.values() if v is not None)
print(f'Loaded {present}/{len(R)} files.')

def acc(k): return float(R[k]['mean_accuracy_pct']) if R.get(k) else float('nan')
def ci(k):  return float(R[k]['ci95_pct'])          if R.get(k) else 0.0
def f1(k):  return float(R[k]['macro_f1'])          if R.get(k) and R[k].get('macro_f1') else float('nan')
SHOTS=[1,3,5]
NOVEL=['AMERICAN BITTERN','AMERICAN COOT','AMERICAN FLAMINGO','AMERICAN GOLDFINCH','AMERICAN KESTREL']
SHORT=[c.replace('AMERICAN ','AM. ') for c in NOVEL]
"""

SETUP_10 = """
%matplotlib inline
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi']=120; plt.rcParams['savefig.dpi']=150
plt.rcParams['figure.facecolor']='white'; plt.rcParams['axes.facecolor']='white'

NB_DIR=os.getcwd()
ROOT=os.path.normpath(os.path.join(NB_DIR,'..'))
RES=os.path.join(ROOT,'results')
ANALYSIS=os.path.join(RES,'analysis')
os.makedirs(ANALYSIS,exist_ok=True)

PATHS={
    'base10_json'               :f'{RES}/baseline/resnet50/base10/base_test/summary.json',
    'kshot5_b10_json'           :f'{RES}/baseline/resnet50/base10/novel_kshot_5/summary.json',
    'ncc_1shot_euc_json'        :f'{RES}/baseline/resnet50/base10/novel_ncc/1shot_euclidean/summary.json',
    'ncc_1shot_man_json'        :f'{RES}/baseline/resnet50/base10/novel_ncc/1shot_manhattan/summary.json',
    'ncc_3shot_euc_json'        :f'{RES}/baseline/resnet50/base10/novel_ncc/3shot_euclidean/summary.json',
    'ncc_3shot_man_json'        :f'{RES}/baseline/resnet50/base10/novel_ncc/3shot_manhattan/summary.json',
    'ncc_5shot_euc_json'        :f'{RES}/baseline/resnet50/base10/novel_ncc/5shot_euclidean/summary.json',
    'ncc_5shot_man_json'        :f'{RES}/baseline/resnet50/base10/novel_ncc/5shot_manhattan/summary.json',
    'proto_1shot_euc_json'      :f'{RES}/proto_net/base10_novel10/1shot_euclidean/summary.json',
    'proto_1shot_man_json'      :f'{RES}/proto_net/base10_novel10/1shot_manhattan/summary.json',
    'proto_3shot_euc_json'      :f'{RES}/proto_net/base10_novel10/3shot_euclidean/summary.json',
    'proto_3shot_man_json'      :f'{RES}/proto_net/base10_novel10/3shot_manhattan/summary.json',
    'proto_5shot_euc_json'      :f'{RES}/proto_net/base10_novel10/5shot_euclidean/summary.json',
    'proto_5shot_man_json'      :f'{RES}/proto_net/base10_novel10/5shot_manhattan/summary.json',
    'proto_5shot_euc_tta8_json' :f'{RES}/proto_net/base10_novel10/5shot_euclidean_tta8/summary.json',
    'match_1shot_json'          :f'{RES}/matching_net/base10_novel10/1shot/summary.json',
    'match_3shot_json'          :f'{RES}/matching_net/base10_novel10/3shot/summary.json',
    'match_5shot_json'          :f'{RES}/matching_net/base10_novel10/5shot/summary.json',
    'match_fce_1shot_json'      :f'{RES}/matching_net_fce/base10_novel10/1shot/summary.json',
    'match_fce_3shot_json'      :f'{RES}/matching_net_fce/base10_novel10/3shot/summary.json',
    'match_fce_5shot_json'      :f'{RES}/matching_net_fce/base10_novel10/5shot/summary.json',
    'rel_1shot_json'            :f'{RES}/relation_net/base10_novel10/1shot/summary.json',
    'rel_3shot_json'            :f'{RES}/relation_net/base10_novel10/3shot/summary.json',
    'rel_5shot_json'            :f'{RES}/relation_net/base10_novel10/5shot/summary.json',
    'vit_ncc_1shot_euc_json'    :f'{RES}/vit/baseline/base10/novel_ncc/1shot_euclidean/summary.json',
    'vit_ncc_1shot_man_json'    :f'{RES}/vit/baseline/base10/novel_ncc/1shot_manhattan/summary.json',
    'vit_ncc_3shot_euc_json'    :f'{RES}/vit/baseline/base10/novel_ncc/3shot_euclidean/summary.json',
    'vit_ncc_3shot_man_json'    :f'{RES}/vit/baseline/base10/novel_ncc/3shot_manhattan/summary.json',
    'vit_ncc_5shot_euc_json'    :f'{RES}/vit/baseline/base10/novel_ncc/5shot_euclidean/summary.json',
    'vit_ncc_5shot_man_json'    :f'{RES}/vit/baseline/base10/novel_ncc/5shot_manhattan/summary.json',
    'vit_proto_1shot_euc_json'  :f'{RES}/vit/proto_net/base10_novel10/1shot_euclidean/summary.json',
    'vit_proto_1shot_man_json'  :f'{RES}/vit/proto_net/base10_novel10/1shot_manhattan/summary.json',
    'vit_proto_3shot_euc_json'  :f'{RES}/vit/proto_net/base10_novel10/3shot_euclidean/summary.json',
    'vit_proto_3shot_man_json'  :f'{RES}/vit/proto_net/base10_novel10/3shot_manhattan/summary.json',
    'vit_proto_5shot_euc_json'  :f'{RES}/vit/proto_net/base10_novel10/5shot_euclidean/summary.json',
    'vit_proto_5shot_man_json'  :f'{RES}/vit/proto_net/base10_novel10/5shot_manhattan/summary.json',
}

R={}
for k,p in PATHS.items():
    if os.path.exists(p):
        with open(p) as f: R[k]=json.load(f)
    else:
        R[k]=None; print(f'  [missing] {k}')
present=sum(1 for v in R.values() if v is not None)
print(f'Loaded {present}/{len(R)} files.')

def acc(k): return float(R[k]['mean_accuracy_pct']) if R.get(k) else float('nan')
def ci(k):  return float(R[k]['ci95_pct'])          if R.get(k) else 0.0
def f1(k):  return float(R[k]['macro_f1'])          if R.get(k) and R[k].get('macro_f1') else float('nan')
SHOTS=[1,3,5]
NOVEL=['ALBERTS TOWHEE','ALEXANDRINE PARAKEET','ALPINE CHOUGH','ALTAMIRA YELLOWTHROAT',
       'AMERICAN AVOCET','AMERICAN BITTERN','AMERICAN COOT','AMERICAN FLAMINGO',
       'AMERICAN GOLDFINCH','AMERICAN KESTREL']
SHORT=[c.replace('AMERICAN ','AM.').replace('ALEXANDRINE ','ALEX.').replace('ALTAMIRA ','ALTA.').replace('ALBERTS ','ALB.').replace('ALPINE ','ALP.') for c in NOVEL]
"""

METHODS_CODE = """
method_configs=[
    ('NCC/R50 (Euc)',         'ncc_{s}shot_euc_json',         '#fd8d3c','o','--'),
    ('NCC/R50 (Man)',         'ncc_{s}shot_man_json',         '#d94801','s',':'),
    ('NCC/DeiT-S (Euc)',      'vit_ncc_{s}shot_euc_json',     '#9467bd','o','--'),
    ('NCC/DeiT-S (Man)',      'vit_ncc_{s}shot_man_json',     '#c5b0d5','s',':'),
    ('ProtoNet/R50 (Euc)',    'proto_{s}shot_euc_json',       '#1f77b4','o','-'),
    ('ProtoNet/R50 (Man)',    'proto_{s}shot_man_json',       '#aec7e8','s','-.'),
    ('ProtoNet/DeiT-S (Euc)','vit_proto_{s}shot_euc_json',   '#17becf','o','-'),
    ('ProtoNet/DeiT-S (Man)','vit_proto_{s}shot_man_json',   '#9edae5','s','-.'),
    ('MatchingNet',           'match_{s}shot_json',           '#2ca02c','^','-'),
    ('MatchingNet+FCE',       'match_fce_{s}shot_json',       '#98df8a','^','--'),
    ('RelationNet',           'rel_{s}shot_json',             '#d62728','D','--'),
]
"""

KSHOT_PLOT = """
fig,ax=plt.subplots(figsize=(12,6))
for label,tmpl,color,marker,ls in method_configs:
    xs,ys,es=[],[],[]
    for s in SHOTS:
        k=tmpl.format(s=s)
        if R.get(k): xs.append(s); ys.append(acc(k)); es.append(ci(k))
    if xs: ax.errorbar(xs,ys,yerr=es,label=label,color=color,marker=marker,linestyle=ls,linewidth=2,markersize=7,capsize=4)
ax.axhline(20,color='lightgray',linestyle='--',linewidth=1,label='Random (20%)')
ax.set_xticks(SHOTS); ax.set_xticklabels([f'{k}-shot' for k in SHOTS],fontsize=11)
ax.set_xlabel('K (support per class)',fontsize=12); ax.set_ylabel('Accuracy (%)',fontsize=12)
ax.set_ylim(0,108); ax.legend(loc='lower right',fontsize=8,ncol=2)
ax.grid(axis='y',linestyle='--',alpha=0.4)
"""

HEATMAP_CODE = """
for shot in SHOTS:
    defs=[(label,R.get(tmpl.format(s=shot))) for label,tmpl,*_ in method_configs]
    matrix,row_labels,any_data=[],[],False
    for name,d in defs:
        if d and 'per_class_accuracy_pct' in d:
            pca=d['per_class_accuracy_pct']
            matrix.append([pca.get(c,float('nan')) for c in NOVEL]); any_data=True
        else: matrix.append([float('nan')]*len(NOVEL))
        row_labels.append(name)
    if not any_data: continue
    matrix=np.array(matrix,dtype=float)
    fig,ax=plt.subplots(figsize=(max(10,len(NOVEL)*1.2),len(method_configs)*0.55+1.5))
    sns.heatmap(matrix,annot=True,fmt='.1f',cmap='YlOrRd',
                xticklabels=SHORT,yticklabels=row_labels,
                ax=ax,vmin=0,vmax=100,linewidths=0.5,linecolor='white',
                annot_kws={'size':8},mask=np.isnan(matrix))
    ax.set_title(f'Per-Class Accuracy (%) — {shot}-Shot',fontsize=11)
    ax.tick_params(axis='x',labelrotation=20,labelsize=8); ax.tick_params(axis='y',labelrotation=0)
    plt.tight_layout(); plt.show()
"""

SUMMARY_CODE_15 = """
rows=[]
methods=[
    ('NCC/R50 (Euc)','ncc_{s}shot_euc_json'),
    ('NCC/R50 (Man)','ncc_{s}shot_man_json'),
    ('NCC/DeiT-S (Euc)','vit_ncc_{s}shot_euc_json'),
    ('NCC/DeiT-S (Man)','vit_ncc_{s}shot_man_json'),
    ('ProtoNet/R50 (Euc)','proto_{s}shot_euc_json'),
    ('ProtoNet/R50 (Man)','proto_{s}shot_man_json'),
    ('ProtoNet/DeiT-S (Euc)','vit_proto_{s}shot_euc_json'),
    ('ProtoNet/DeiT-S (Man)','vit_proto_{s}shot_man_json'),
    ('MatchingNet','match_{s}shot_json'),
    ('MatchingNet+FCE','match_fce_{s}shot_json'),
    ('RelationNet','rel_{s}shot_json'),
]
for s in SHOTS:
    for label,tmpl in methods:
        k=tmpl.format(s=s); d=R.get(k)
        rows.append({'Method':label,'K-Shot':s,
                     'Accuracy(%)':f'{acc(k):.2f}' if d else '—',
                     '95% CI':f'+-{ci(k):.2f}' if d else '—',
                     'F1':f'{f1(k):.4f}' if d and not np.isnan(f1(k)) else '—',
                     'Precision':f"{float(d['macro_precision']):.4f}" if d and d.get('macro_precision') else '—',
                     'Recall':f"{float(d['macro_recall']):.4f}" if d and d.get('macro_recall') else '—',
                     'AUC':f"{float(d['auc_ovr']):.4f}" if d and d.get('auc_ovr') else '—'})
k='proto_5shot_euc_tta8_json'; d=R.get(k)
rows.append({'Method':'ProtoNet/R50+TTA8 (Euc)','K-Shot':5,
             'Accuracy(%)':f'{acc(k):.2f}' if d else '—','95% CI':f'+-{ci(k):.2f}' if d else '—',
             'F1':f'{f1(k):.4f}' if d and not np.isnan(f1(k)) else '—',
             'Precision':f"{float(d['macro_precision']):.4f}" if d and d.get('macro_precision') else '—',
             'Recall':f"{float(d['macro_recall']):.4f}" if d and d.get('macro_recall') else '—',
             'AUC':f"{float(d['auc_ovr']):.4f}" if d and d.get('auc_ovr') else '—'})
df_s=pd.DataFrame(rows); df_s.index=range(1,len(df_s)+1)
display(df_s)
df_s.to_csv(f'{ANALYSIS}/b15n5_summary.csv')
print(f'Saved to {ANALYSIS}/b15n5_summary.csv')
"""

INSIGHTS_15 = """
print('='*70)
print('  KEY FINDINGS — B15/N5')
print('='*70)

for algo,r50t,vitr in [('NCC','ncc_{s}shot_euc_json','vit_ncc_{s}shot_euc_json'),
                        ('ProtoNet','proto_{s}shot_euc_json','vit_proto_{s}shot_euc_json')]:
    print(f'\\n{algo} — DeiT-S vs R50 lift (Euclidean):')
    for s in SHOTS:
        r=acc(r50t.format(s=s)); v=acc(vitr.format(s=s))
        print(f'  {s}-shot: DeiT-S={v:.2f}%  R50={r:.2f}%  delta={v-r:+.2f}pp')

print('\\nMatchingNet FCE effect:')
for s in SHOTS:
    v=acc(f'match_{s}shot_json'); fce=acc(f'match_fce_{s}shot_json')
    print(f'  {s}-shot: FCE={fce:.2f}%  vanilla={v:.2f}%  delta={fce-v:+.2f}pp')

print(f'\\nTTA (5-shot Euc): {acc("proto_5shot_euc_tta8_json"):.2f}% vs {acc("proto_5shot_euc_json"):.2f}%  delta={acc("proto_5shot_euc_tta8_json")-acc("proto_5shot_euc_json"):+.2f}pp')

print('\\nRanking at 5-shot Euclidean:')
ranking=[('NCC/DeiT-S','vit_ncc_5shot_euc_json'),
         ('ProtoNet/DeiT-S','vit_proto_5shot_euc_json'),
         ('ProtoNet/R50+TTA8','proto_5shot_euc_tta8_json'),
         ('NCC/R50','ncc_5shot_euc_json'),
         ('ProtoNet/R50','proto_5shot_euc_json'),
         ('RelationNet','rel_5shot_json'),
         ('MatchingNet','match_5shot_json'),
         ('MatchingNet+FCE','match_fce_5shot_json')]
for i,(name,key) in enumerate(sorted(ranking,key=lambda x:acc(x[1]),reverse=True),1):
    print(f'  {i}. {name}: {acc(key):.2f}%')
"""

INSIGHTS_10 = """
print('='*70)
print('  KEY FINDINGS — B10/N10')
print('='*70)

for algo,r50t,vitr in [('NCC','ncc_{s}shot_euc_json','vit_ncc_{s}shot_euc_json'),
                        ('ProtoNet','proto_{s}shot_euc_json','vit_proto_{s}shot_euc_json')]:
    print(f'\\n{algo} — DeiT-S vs R50 lift (Euclidean):')
    for s in SHOTS:
        r=acc(r50t.format(s=s)); v=acc(vitr.format(s=s))
        print(f'  {s}-shot: DeiT-S={v:.2f}%  R50={r:.2f}%  delta={v-r:+.2f}pp')

print('\\nMatchingNet FCE effect (B10/N10):')
for s in SHOTS:
    v=acc(f'match_{s}shot_json'); fce=acc(f'match_fce_{s}shot_json')
    print(f'  {s}-shot: FCE={fce:.2f}%  vanilla={v:.2f}%  delta={fce-v:+.2f}pp')

print(f'\\nTTA (5-shot Euc): {acc("proto_5shot_euc_tta8_json"):.2f}% vs {acc("proto_5shot_euc_json"):.2f}%  delta={acc("proto_5shot_euc_tta8_json")-acc("proto_5shot_euc_json"):+.2f}pp')

print('\\nHardest novel class (lowest avg accuracy across all methods at 5-shot):')
class_accs={c:[] for c in NOVEL}
for k in ['ncc_5shot_euc_json','vit_ncc_5shot_euc_json','proto_5shot_euc_json','vit_proto_5shot_euc_json']:
    d=R.get(k)
    if d and 'per_class_accuracy_pct' in d:
        for c in NOVEL: class_accs[c].append(d['per_class_accuracy_pct'].get(c,float('nan')))
avg={c:np.nanmean(vs) for c,vs in class_accs.items() if vs}
for c,a in sorted(avg.items(),key=lambda x:x[1]):
    print(f'  {c}: {a:.1f}%')
"""

COMPARISON_SETUP = """
%matplotlib inline
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from IPython.display import display, Markdown
warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi']=120; plt.rcParams['savefig.dpi']=150
plt.rcParams['figure.facecolor']='white'; plt.rcParams['axes.facecolor']='white'

NB_DIR=os.getcwd()
ROOT=os.path.normpath(os.path.join(NB_DIR,'..'))
RES=os.path.join(ROOT,'results')
ANALYSIS=os.path.join(RES,'analysis')
os.makedirs(ANALYSIS,exist_ok=True)

def load(rel):
    p=os.path.join(RES,rel)
    if os.path.exists(p):
        with open(p) as f: return json.load(f)
    return None

SHOTS=[1,3,5]

# B15/N5
B15={
    'ncc_euc'    :{s:load(f'baseline/resnet50/base15/novel_ncc/{s}shot_euclidean/summary.json') for s in SHOTS},
    'ncc_man'    :{s:load(f'baseline/resnet50/base15/novel_ncc/{s}shot_manhattan/summary.json') for s in SHOTS},
    'vit_ncc_euc':{s:load(f'vit/baseline/base15/novel_ncc/{s}shot_euclidean/summary.json') for s in SHOTS},
    'vit_ncc_man':{s:load(f'vit/baseline/base15/novel_ncc/{s}shot_manhattan/summary.json') for s in SHOTS},
    'proto_euc'  :{s:load(f'proto_net/base15_novel5/{s}shot_euclidean/summary.json') for s in SHOTS},
    'proto_man'  :{s:load(f'proto_net/base15_novel5/{s}shot_manhattan/summary.json') for s in SHOTS},
    'vit_proto_euc':{s:load(f'vit/proto_net/base15_novel5/{s}shot_euclidean/summary.json') for s in SHOTS},
    'vit_proto_man':{s:load(f'vit/proto_net/base15_novel5/{s}shot_manhattan/summary.json') for s in SHOTS},
    'match'      :{s:load(f'matching_net/base15_novel5/{s}shot/summary.json') for s in SHOTS},
    'match_fce'  :{s:load(f'matching_net_fce/base15_novel5/{s}shot/summary.json') for s in SHOTS},
    'rel'        :{s:load(f'relation_net/base15_novel5/{s}shot/summary.json') for s in SHOTS},
    'proto_tta8' :{5:load(f'proto_net/base15_novel5/5shot_euclidean_tta8/summary.json')},
}

# B10/N10
B10={
    'ncc_euc'    :{s:load(f'baseline/resnet50/base10/novel_ncc/{s}shot_euclidean/summary.json') for s in SHOTS},
    'ncc_man'    :{s:load(f'baseline/resnet50/base10/novel_ncc/{s}shot_manhattan/summary.json') for s in SHOTS},
    'vit_ncc_euc':{s:load(f'vit/baseline/base10/novel_ncc/{s}shot_euclidean/summary.json') for s in SHOTS},
    'vit_ncc_man':{s:load(f'vit/baseline/base10/novel_ncc/{s}shot_manhattan/summary.json') for s in SHOTS},
    'proto_euc'  :{s:load(f'proto_net/base10_novel10/{s}shot_euclidean/summary.json') for s in SHOTS},
    'proto_man'  :{s:load(f'proto_net/base10_novel10/{s}shot_manhattan/summary.json') for s in SHOTS},
    'vit_proto_euc':{s:load(f'vit/proto_net/base10_novel10/{s}shot_euclidean/summary.json') for s in SHOTS},
    'vit_proto_man':{s:load(f'vit/proto_net/base10_novel10/{s}shot_manhattan/summary.json') for s in SHOTS},
    'match'      :{s:load(f'matching_net/base10_novel10/{s}shot/summary.json') for s in SHOTS},
    'match_fce'  :{s:load(f'matching_net_fce/base10_novel10/{s}shot/summary.json') for s in SHOTS},
    'rel'        :{s:load(f'relation_net/base10_novel10/{s}shot/summary.json') for s in SHOTS},
    'proto_tta8' :{5:load(f'proto_net/base10_novel10/5shot_euclidean_tta8/summary.json')},
}

def fsl_acc(d): return float(d['mean_accuracy_pct']) if d else float('nan')
def fsl_ci(d):  return float(d['ci95_pct'])          if d else 0.0

METHOD_MAP=[
    ('ncc_euc',    'NCC (Euc)',         '#fd8d3c','o','--'),
    ('ncc_man',    'NCC (Man)',         '#d94801','s',':'),
    ('vit_ncc_euc','NCC/DeiT-S (Euc)', '#9467bd','o','--'),
    ('vit_ncc_man','NCC/DeiT-S (Man)', '#c5b0d5','s',':'),
    ('proto_euc',  'Proto (Euc)',       '#1f77b4','o','-'),
    ('proto_man',  'Proto (Man)',       '#aec7e8','s','-.'),
    ('vit_proto_euc','Proto/DeiT-S (Euc)','#17becf','o','-'),
    ('vit_proto_man','Proto/DeiT-S (Man)','#9edae5','s','-.'),
    ('match',      'MatchingNet',       '#2ca02c','^','-'),
    ('match_fce',  'MatchingNet+FCE',   '#98df8a','^','--'),
    ('rel',        'RelationNet',       '#d62728','D','--'),
]
print('Setup complete.')
"""

COMPARISON_KSHOT = """
# K-shot trend: B15 vs B10 for each method
fig,axes=plt.subplots(3,4,figsize=(20,14))
axes=axes.flatten()
for ax_i,(mkey,mlabel,color,marker,ls) in enumerate(METHOD_MAP):
    if ax_i>=len(axes): break
    ax=axes[ax_i]
    b15_a=[fsl_acc(B15[mkey].get(s)) for s in SHOTS]
    b10_a=[fsl_acc(B10[mkey].get(s)) for s in SHOTS]
    b15_c=[fsl_ci(B15[mkey].get(s)) for s in SHOTS]
    b10_c=[fsl_ci(B10[mkey].get(s)) for s in SHOTS]
    ax.errorbar(SHOTS,b15_a,yerr=b15_c,label='B15/N5',color=color,marker=marker,linestyle='-',linewidth=2,markersize=7,capsize=4)
    ax.errorbar(SHOTS,b10_a,yerr=b10_c,label='B10/N10',color=color,marker=marker,linestyle='--',linewidth=2,markersize=7,capsize=4,alpha=0.7)
    ax.set_xticks(SHOTS); ax.set_xticklabels([f'{k}s' for k in SHOTS])
    ax.set_ylim(0,105); ax.set_title(mlabel,fontsize=9); ax.legend(fontsize=7); ax.grid(axis='y',alpha=0.3)
for ax in axes[len(METHOD_MAP):]: ax.set_visible(False)
fig.suptitle('K-Shot Trend: B15/N5 (solid) vs B10/N10 (dashed)',fontsize=13)
plt.tight_layout()
plt.savefig(f'{ANALYSIS}/comparison_kshot_b15_vs_b10.png',dpi=150); plt.show()
"""

COMPARISON_DELTA = """
# Delta: B10 - B15 accuracy (shows degradation from fewer base classes)
fig,axes=plt.subplots(1,3,figsize=(16,5),sharey=True)
for ax,shot in zip(axes,SHOTS):
    labels,deltas,colors=[],[],[]
    for mkey,mlabel,color,*_ in METHOD_MAP:
        a15=fsl_acc(B15[mkey].get(shot)); a10=fsl_acc(B10[mkey].get(shot))
        if not (np.isnan(a15) or np.isnan(a10)):
            labels.append(mlabel); deltas.append(a10-a15); colors.append(color)
    x=np.arange(len(labels))
    bars=ax.bar(x,deltas,color=colors,edgecolor='white')
    ax.axhline(0,color='black',linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels,rotation=30,ha='right',fontsize=8)
    ax.set_title(f'{shot}-Shot',fontsize=11)
    ax.set_ylabel('B10/N10 - B15/N5 (pp)' if shot==1 else '')
    ax.grid(axis='y',alpha=0.3)
fig.suptitle('Accuracy Delta: B10/N10 minus B15/N5\\n(negative = B15 is better)',fontsize=12)
plt.tight_layout()
plt.savefig(f'{ANALYSIS}/comparison_delta_b10_minus_b15.png',dpi=150); plt.show()
"""

COMPARISON_BACKBONE = """
# Backbone comparison: DeiT-S vs ResNet-50, both splits
fig,axes=plt.subplots(2,2,figsize=(14,10))
for row,split_label,split_data in [(0,'B15/N5',B15),(1,'B10/N10',B10)]:
    for col,(algo,r50key,vitkey) in enumerate([('NCC','ncc_euc','vit_ncc_euc'),
                                               ('ProtoNet','proto_euc','vit_proto_euc')]):
        ax=axes[row][col]
        r50a=[fsl_acc(split_data[r50key].get(s)) for s in SHOTS]
        vita=[fsl_acc(split_data[vitkey].get(s)) for s in SHOTS]
        r50c=[fsl_ci(split_data[r50key].get(s)) for s in SHOTS]
        vitc=[fsl_ci(split_data[vitkey].get(s)) for s in SHOTS]
        ax.errorbar(SHOTS,r50a,yerr=r50c,label='ResNet-50',color='#1f77b4',marker='o',linewidth=2,markersize=8,capsize=5)
        ax.errorbar(SHOTS,vita,yerr=vitc,label='DeiT-S',   color='#9467bd',marker='s',linewidth=2,markersize=8,capsize=5)
        for s,r,v in zip(SHOTS,r50a,vita):
            if not np.isnan(r) and not np.isnan(v):
                ax.annotate(f'{v-r:+.1f}pp',(s,max(r,v)+0.3),ha='center',fontsize=8,color='#9467bd')
        ax.set_xticks(SHOTS); ax.set_xticklabels([f'{k}-shot' for k in SHOTS])
        ax.set_ylim(60,105); ax.set_ylabel('Accuracy (%)'); ax.legend(fontsize=9)
        ax.set_title(f'{algo} Euclidean — {split_label}'); ax.grid(axis='y',alpha=0.3)
fig.suptitle('Backbone Ablation: DeiT-S vs ResNet-50',fontsize=13)
plt.tight_layout()
plt.savefig(f'{ANALYSIS}/comparison_backbone_ablation.png',dpi=150); plt.show()
"""

COMPARISON_FCE = """
# FCE effect: both splits
fig,axes=plt.subplots(1,2,figsize=(13,5))
for ax,(split_label_str,split_data) in zip(axes,[('B15/N5',B15),('B10/N10',B10)]):
    x=np.arange(len(SHOTS)); w=0.35
    v_a=[fsl_acc(split_data['match'].get(s)) for s in SHOTS]
    f_a=[fsl_acc(split_data['match_fce'].get(s)) for s in SHOTS]
    v_c=[fsl_ci(split_data['match'].get(s)) for s in SHOTS]
    f_c=[fsl_ci(split_data['match_fce'].get(s)) for s in SHOTS]
    ax.bar(x-w/2,v_a,w,label='Vanilla',color='#2ca02c',yerr=v_c,capsize=4)
    ax.bar(x+w/2,f_a,w,label='+FCE',   color='#98df8a',yerr=f_c,capsize=4)
    ax.set_xticks(x); ax.set_xticklabels([f'{s}-shot' for s in SHOTS])
    ax.set_ylim(0,80); ax.set_ylabel('Accuracy (%)'); ax.legend()
    ax.set_title(f'MatchingNet: Vanilla vs FCE — {split_label_str}'); ax.grid(axis='y',alpha=0.3)
    deltas=[fsl_acc(split_data['match_fce'].get(s))-fsl_acc(split_data['match'].get(s)) for s in SHOTS]
    for xi,d in zip(x,deltas):
        if not np.isnan(d): ax.text(xi+w/2,fsl_acc(split_data['match_fce'].get(SHOTS[xi]))+1,f'{d:+.1f}',ha='center',fontsize=9,fontweight='bold',color='#d62728')
fig.suptitle('MatchingNet FCE Effect Comparison',fontsize=12)
plt.tight_layout()
plt.savefig(f'{ANALYSIS}/comparison_fce_both_splits.png',dpi=150); plt.show()
"""

COMPARISON_SUMMARY = """
# Master summary table — best result per method per split
rows=[]
for mkey,mlabel,*_ in METHOD_MAP:
    for split_label,split_data in [('B15/N5',B15),('B10/N10',B10)]:
        best_shot=max(SHOTS,key=lambda s: fsl_acc(split_data[mkey].get(s)) if split_data[mkey].get(s) else -1)
        d=split_data[mkey].get(best_shot)
        rows.append({
            'Method':mlabel,'Split':split_label,
            'Best K-Shot':best_shot,
            'Best Accuracy':f'{fsl_acc(d):.2f}%' if d else '—',
            '95% CI':f'+-{fsl_ci(d):.2f}' if d else '—',
            'F1':f'{float(d["macro_f1"]):.4f}' if d and d.get("macro_f1") else '—',
        })
df_master=pd.DataFrame(rows)
display(df_master.pivot(index='Method',columns='Split',values='Best Accuracy'))
display(df_master)
df_master.to_csv(f'{ANALYSIS}/master_comparison.csv'); print(f'Saved master_comparison.csv')
"""

RQANALYSIS = """
print('='*70)
print('RESEARCH QUESTION ANALYSIS — ALL EXPERIMENTS')
print('='*70)

# RQ1: Does episodic training add value beyond a frozen backbone? (ProtoNet vs NCC)
print('\\nRQ1: ProtoNet vs NCC (same backbone, same backbone)')
for split_label,split_data in [('B15/N5',B15),('B10/N10',B10)]:
    print(f'  {split_label}:')
    for s in SHOTS:
        p=fsl_acc(split_data['proto_euc'].get(s)); n=fsl_acc(split_data['ncc_euc'].get(s))
        print(f'    {s}-shot: ProtoNet={p:.2f}%  NCC={n:.2f}%  gain={p-n:+.2f}pp')

# RQ2: DeiT-S vs ResNet-50 backbone impact
print('\\nRQ2: DeiT-S backbone lift over ResNet-50 (NCC, Euclidean)')
for split_label,split_data in [('B15/N5',B15),('B10/N10',B10)]:
    print(f'  {split_label}:')
    for s in SHOTS:
        r=fsl_acc(split_data['ncc_euc'].get(s)); v=fsl_acc(split_data['vit_ncc_euc'].get(s))
        print(f'    {s}-shot: DeiT-S={v:.2f}%  R50={r:.2f}%  delta={v-r:+.2f}pp')

# RQ3: FCE effect
print('\\nRQ3: MatchingNet FCE effect')
for split_label,split_data in [('B15/N5',B15),('B10/N10',B10)]:
    print(f'  {split_label}:')
    for s in SHOTS:
        v=fsl_acc(split_data['match'].get(s)); f=fsl_acc(split_data['match_fce'].get(s))
        print(f'    {s}-shot: FCE={f:.2f}%  vanilla={v:.2f}%  delta={f-v:+.2f}pp')

# RQ4: Base class reduction impact
print('\\nRQ4: B15->B10 degradation (ProtoNet Euclidean)')
for s in SHOTS:
    a15=fsl_acc(B15['proto_euc'].get(s)); a10=fsl_acc(B10['proto_euc'].get(s))
    print(f'  {s}-shot: B15={a15:.2f}%  B10={a10:.2f}%  drop={a10-a15:+.2f}pp')

# RQ5: TTA benefit
print('\\nRQ5: TTA effect (5-shot Euclidean)')
for split_label,split_data in [('B15/N5',B15),('B10/N10',B10)]:
    base=fsl_acc(split_data['proto_euc'].get(5)); tta=fsl_acc(split_data['proto_tta8'].get(5))
    print(f'  {split_label}: TTA={tta:.2f}%  base={base:.2f}%  delta={tta-base:+.2f}pp')
"""

# ===========================================================================
# BUILD results15b5n.ipynb
# ===========================================================================
cells_15 = [
    md(["# Few-Shot Learning — B15/N5 Results\n",
        "\n**Split:** 15 base / 5 novel | **Protocol:** 5-way K-shot, 600 episodes, K∈{1,3,5}\n",
        "\n**Backbones:** ResNet-50 · DeiT-S | **Methods:** NCC · ProtoNet · MatchingNet · MatchingNet+FCE · RelationNet · ProtoNet+TTA\n",
        "\n**Novel (5):** AMERICAN BITTERN, AMERICAN COOT, AMERICAN FLAMINGO, AMERICAN GOLDFINCH, AMERICAN KESTREL"]),
    code(SETUP_15),
    md("---\n## 1. Supervised Baselines (Reference)"),
    code("""
for label,key,note in [('Full-20 Supervised','full20_json','Upper bound'),
                        ('Base-15 Supervised','base15_json','Base-class eval'),
                        ('K=5 Fine-tune (novel)','kshot5_json','Novel 5-shot finetune')]:
    d=R.get(key)
    if not d: print(f'{label}: [missing]'); continue
    a=float(d.get('accuracy',d.get('test_accuracy',0)))*100
    fm=d.get('macro_f1')
    fm_str=f'{float(fm):.4f}' if fm else '—'
    print(f'{label:<30} acc={a:.2f}%  F1={fm_str}  | {note}')
"""),
    md("---\n## 2. NCC: ResNet-50 vs DeiT-S"),
    code("""
rows=[]
for s in SHOTS:
    for label,tmpl in [('NCC/R50 (Euc)','ncc_{s}shot_euc_json'),('NCC/R50 (Man)','ncc_{s}shot_man_json'),
                        ('NCC/DeiT-S (Euc)','vit_ncc_{s}shot_euc_json'),('NCC/DeiT-S (Man)','vit_ncc_{s}shot_man_json')]:
        k=tmpl.format(s=s); d=R.get(k)
        rows.append({'Model':label,'K-Shot':s,'Accuracy(%)':f'{acc(k):.2f}' if d else '—',
                     '95% CI':f'+-{ci(k):.2f}' if d else '—','F1':f'{f1(k):.4f}' if d and not np.isnan(f1(k)) else '—'})
display(pd.DataFrame(rows))

fig,axes=plt.subplots(1,3,figsize=(15,4),sharey=True)
for ax,shot in zip(axes,SHOTS):
    cfgs=[('R50 Euc',f'ncc_{shot}shot_euc_json','#fd8d3c'),('R50 Man',f'ncc_{shot}shot_man_json','#d94801'),
           ('DeiT-S Euc',f'vit_ncc_{shot}shot_euc_json','#9467bd'),('DeiT-S Man',f'vit_ncc_{shot}shot_man_json','#c5b0d5')]
    ls=[c[0] for c in cfgs]; vs=[acc(c[1]) for c in cfgs]; es=[ci(c[1]) for c in cfgs]; cs=[c[2] for c in cfgs]
    bars=ax.bar(ls,vs,color=cs,edgecolor='white',width=0.6)
    ax.errorbar(range(len(ls)),vs,yerr=es,fmt='none',color='black',capsize=4)
    for bar,v in zip(bars,vs):
        ax.text(bar.get_x()+bar.get_width()/2,v+0.4,f'{v:.1f}%',ha='center',fontsize=8,fontweight='bold')
    ax.set_title(f'{shot}-Shot NCC'); ax.set_ylim(0,108)
    ax.set_ylabel('Accuracy (%)' if shot==1 else ''); ax.tick_params(axis='x',labelrotation=15,labelsize=8)
fig.suptitle('NCC: ResNet-50 vs DeiT-S — B15/N5',fontsize=12)
plt.tight_layout(); plt.savefig(f'{ANALYSIS}/b15n5_ncc_r50_vs_deit.png',dpi=150); plt.show()
"""),
    md("---\n## 3. ProtoNet: ResNet-50 vs DeiT-S"),
    code("""
rows=[]
for s in SHOTS:
    for label,tmpl in [('ProtoNet/R50 (Euc)','proto_{s}shot_euc_json'),('ProtoNet/R50 (Man)','proto_{s}shot_man_json'),
                        ('ProtoNet/DeiT-S (Euc)','vit_proto_{s}shot_euc_json'),('ProtoNet/DeiT-S (Man)','vit_proto_{s}shot_man_json')]:
        k=tmpl.format(s=s); d=R.get(k)
        rows.append({'Model':label,'K-Shot':s,'Accuracy(%)':f'{acc(k):.2f}' if d else '—',
                     '95% CI':f'+-{ci(k):.2f}' if d else '—','F1':f'{f1(k):.4f}' if d and not np.isnan(f1(k)) else '—'})
display(pd.DataFrame(rows))

fig,ax=plt.subplots(figsize=(9,5))
for label,tmpl,color,marker,ls in [('ProtoNet/R50 (Euc)','proto_{s}shot_euc_json','#1f77b4','o','-'),
                                     ('ProtoNet/R50 (Man)','proto_{s}shot_man_json','#aec7e8','s','-.'),
                                     ('ProtoNet/DeiT-S (Euc)','vit_proto_{s}shot_euc_json','#17becf','o','-'),
                                     ('ProtoNet/DeiT-S (Man)','vit_proto_{s}shot_man_json','#9edae5','s','-.')]:
    xs,ys,es=zip(*[(s,acc(tmpl.format(s=s)),ci(tmpl.format(s=s))) for s in SHOTS])
    ax.errorbar(xs,ys,yerr=es,label=label,color=color,marker=marker,linestyle=ls,linewidth=2,markersize=7,capsize=5)
ax.set_xticks(SHOTS); ax.set_xticklabels([f'{k}-shot' for k in SHOTS])
ax.set_ylim(70,105); ax.set_ylabel('Accuracy (%)'); ax.legend(fontsize=9)
ax.set_title('ProtoNet: R50 vs DeiT-S — B15/N5'); ax.grid(axis='y',linestyle='--',alpha=0.4)
plt.tight_layout(); plt.savefig(f'{ANALYSIS}/b15n5_protonet_r50_vs_deit.png',dpi=150); plt.show()
"""),
    md(["---\n## 4. MatchingNet: Vanilla vs FCE (B15/N5)\n\n",
        "FCE adds BiLSTM over support set + attention LSTM for query. Tests whether contextualised embeddings help."]),
    code("""
rows=[]
for s in SHOTS:
    for label,tmpl in [('MatchingNet (vanilla)','match_{s}shot_json'),('MatchingNet+FCE','match_fce_{s}shot_json')]:
        k=tmpl.format(s=s); d=R.get(k)
        rows.append({'Model':label,'K-Shot':s,'Accuracy(%)':f'{acc(k):.2f}' if d else '—',
                     '95% CI':f'+-{ci(k):.2f}' if d else '—','F1':f'{f1(k):.4f}' if d and not np.isnan(f1(k)) else '—'})
display(pd.DataFrame(rows))

fig,axes=plt.subplots(1,2,figsize=(13,5))
x=np.arange(len(SHOTS)); w=0.35
v_a=[acc(f'match_{s}shot_json') for s in SHOTS]; f_a=[acc(f'match_fce_{s}shot_json') for s in SHOTS]
v_c=[ci(f'match_{s}shot_json') for s in SHOTS];  f_c=[ci(f'match_fce_{s}shot_json') for s in SHOTS]
axes[0].bar(x-w/2,v_a,w,label='Vanilla',color='#2ca02c',yerr=v_c,capsize=4)
axes[0].bar(x+w/2,f_a,w,label='+FCE',   color='#98df8a',yerr=f_c,capsize=4)
axes[0].set_xticks(x); axes[0].set_xticklabels([f'{s}-shot' for s in SHOTS])
axes[0].set_ylim(0,85); axes[0].set_ylabel('Accuracy (%)'); axes[0].legend()
axes[0].set_title('MatchingNet: Vanilla vs FCE — B15/N5'); axes[0].grid(axis='y',alpha=0.3)
deltas=[acc(f'match_fce_{s}shot_json')-acc(f'match_{s}shot_json') for s in SHOTS]
axes[1].plot(SHOTS,deltas,'o-',color='#d62728',linewidth=2,markersize=8)
axes[1].axhline(0,color='gray',linestyle='--')
axes[1].set_xticks(SHOTS); axes[1].set_xticklabels([f'{s}-shot' for s in SHOTS])
axes[1].set_ylabel('FCE - Vanilla (pp)'); axes[1].set_title('FCE Delta')
for s,d in zip(SHOTS,deltas): axes[1].text(s,d+0.2,f'{d:+.1f}',ha='center',fontsize=10,fontweight='bold')
axes[1].grid(axis='y',alpha=0.3)
plt.tight_layout(); plt.savefig(f'{ANALYSIS}/b15n5_matchingnet_fce.png',dpi=150); plt.show()
print('FCE effect: '+', '.join([f'{s}-shot: {d:+.2f}pp' for s,d in zip(SHOTS,deltas)]))
"""),
    md("---\n## 5. RelationNet (B15/N5)"),
    code("""
print('RelationNet — B15/N5')
for s in SHOTS:
    k=f'rel_{s}shot_json'
    print(f'  {s}-shot: {acc(k):.2f}% +-{ci(k):.2f}  F1={f1(k):.4f}')
"""),
    md(["---\n## 6. ProtoNet + TTA (Test-Time Augmentation)\n\n",
        "TTA averages 8 augmented views per query image at inference. No retraining."]),
    code("""
base_a=acc('proto_5shot_euc_json'); tta_a=acc('proto_5shot_euc_tta8_json')
base_c=ci('proto_5shot_euc_json');  tta_c=ci('proto_5shot_euc_tta8_json')
delta=tta_a-base_a
print(f'ProtoNet/R50 5-shot Euclidean — TTA (B15/N5)')
print(f'  Without TTA : {base_a:.2f}% +-{base_c:.2f}')
print(f'  With TTA x8 : {tta_a:.2f}% +-{tta_c:.2f}')
print(f'  Delta       : {delta:+.2f} pp')
fig,ax=plt.subplots(figsize=(5,4))
bars=ax.bar(['Without TTA','With TTA x8'],[base_a,tta_a],color=['#1f77b4','#17becf'],width=0.5)
ax.errorbar([0,1],[base_a,tta_a],yerr=[base_c,tta_c],fmt='none',color='black',capsize=5)
for bar,v in zip(bars,[base_a,tta_a]):
    ax.text(bar.get_x()+bar.get_width()/2,v+0.2,f'{v:.2f}%',ha='center',fontweight='bold')
ax.set_ylim(min(base_a,tta_a)-2,max(base_a,tta_a)+3); ax.set_ylabel('Accuracy (%)')
ax.set_title('TTA Effect — ProtoNet 5-shot — B15/N5'); ax.grid(axis='y',alpha=0.3)
plt.tight_layout(); plt.savefig(f'{ANALYSIS}/b15n5_tta.png',dpi=150); plt.show()
"""),
    md("---\n## 7. K-Shot Trend — All Methods"),
    code(METHODS_CODE + KSHOT_PLOT + """
ax.set_title('K-Shot Trend — All Methods — B15/N5')
kd=R.get('kshot5_json')
if kd:
    ref=float(kd.get('test_accuracy',kd.get('accuracy',0)))*100
    ax.axhline(ref,color='gray',linestyle=':',linewidth=1.8,label=f'K=5 Fine-tune ({ref:.1f}%)')
ax.legend(loc='lower right',fontsize=8,ncol=2)
plt.tight_layout(); plt.savefig(f'{ANALYSIS}/b15n5_kshot_all.png',dpi=150); plt.show()
"""),
    md("---\n## 8. Per-Class Accuracy Heatmap"),
    code(METHODS_CODE + HEATMAP_CODE),
    md("---\n## 9. Backbone Ablation: DeiT-S vs ResNet-50"),
    code("""
fig,axes=plt.subplots(1,2,figsize=(14,5))
for ax,(algo,r50t,vitr,title) in zip(axes,[
    ('NCC','ncc_{s}shot_euc_json','vit_ncc_{s}shot_euc_json','NCC Euclidean'),
    ('ProtoNet','proto_{s}shot_euc_json','vit_proto_{s}shot_euc_json','ProtoNet Euclidean'),
]):
    r50a=[acc(r50t.format(s=s)) for s in SHOTS]; vita=[acc(vitr.format(s=s)) for s in SHOTS]
    r50c=[ci(r50t.format(s=s)) for s in SHOTS];  vitc=[ci(vitr.format(s=s)) for s in SHOTS]
    ax.errorbar(SHOTS,r50a,yerr=r50c,label='ResNet-50',color='#1f77b4',marker='o',linewidth=2,markersize=8,capsize=5)
    ax.errorbar(SHOTS,vita,yerr=vitc,label='DeiT-S',   color='#9467bd',marker='s',linewidth=2,markersize=8,capsize=5)
    for s,r,v in zip(SHOTS,r50a,vita):
        if not np.isnan(r) and not np.isnan(v):
            ax.annotate(f'{v-r:+.1f}pp',(s,max(r,v)+0.3),ha='center',fontsize=8,color='#9467bd')
    ax.set_xticks(SHOTS); ax.set_xticklabels([f'{k}-shot' for k in SHOTS])
    ax.set_ylim(60,105); ax.set_ylabel('Accuracy (%)'); ax.set_title(f'{title} — B15/N5'); ax.legend()
    ax.grid(axis='y',alpha=0.3)
fig.suptitle('Backbone Ablation: DeiT-S vs ResNet-50 — B15/N5',fontsize=12)
plt.tight_layout(); plt.savefig(f'{ANALYSIS}/b15n5_backbone_ablation.png',dpi=150); plt.show()
"""),
    md("---\n## 10. Full Summary Table"),
    code(SUMMARY_CODE_15),
    md("---\n## 11. Key Insights"),
    code(INSIGHTS_15),
]

# ===========================================================================
# BUILD results10b10n.ipynb
# ===========================================================================
SUMMARY_CODE_10 = SUMMARY_CODE_15.replace(
    "proto_5shot_euc_tta8_json","proto_5shot_euc_tta8_json").replace(
    "b15n5_summary.csv","b10n10_summary.csv")

cells_10 = [
    md(["# Few-Shot Learning — B10/N10 Results\n",
        "\n**Split:** 10 base / 10 novel | **Protocol:** 5-way K-shot (5 of 10 novel sampled), 600 episodes\n",
        "\n**Backbones:** ResNet-50 · DeiT-S | **Methods:** NCC · ProtoNet · MatchingNet · MatchingNet+FCE · RelationNet · ProtoNet+TTA\n",
        "\n**Novel (10):** ALBERTS TOWHEE, ALEXANDRINE PARAKEET, ALPINE CHOUGH, ALTAMIRA YELLOWTHROAT, AMERICAN AVOCET,\nAMERICAN BITTERN, AMERICAN COOT, AMERICAN FLAMINGO, AMERICAN GOLDFINCH, AMERICAN KESTREL"]),
    code(SETUP_10),
    md("---\n## 1. Supervised Baselines (Reference)"),
    code("""
for label,key,note in [('Base-10 Supervised','base10_json','Base-class eval'),
                        ('K=5 Fine-tune (novel)','kshot5_b10_json','Novel 5-shot finetune')]:
    d=R.get(key)
    if not d: print(f'{label}: [missing]'); continue
    a=float(d.get('accuracy',d.get('test_accuracy',0)))*100
    fm=d.get('macro_f1')
    print(f'{label:<30} acc={a:.2f}%  | {note}')
"""),
    md("---\n## 2. NCC: ResNet-50 vs DeiT-S (B10/N10)"),
    code("""
rows=[]
for s in SHOTS:
    for label,tmpl in [('NCC/R50 (Euc)','ncc_{s}shot_euc_json'),('NCC/R50 (Man)','ncc_{s}shot_man_json'),
                        ('NCC/DeiT-S (Euc)','vit_ncc_{s}shot_euc_json'),('NCC/DeiT-S (Man)','vit_ncc_{s}shot_man_json')]:
        k=tmpl.format(s=s); d=R.get(k)
        rows.append({'Model':label,'K-Shot':s,'Accuracy(%)':f'{acc(k):.2f}' if d else '—',
                     '95% CI':f'+-{ci(k):.2f}' if d else '—','F1':f'{f1(k):.4f}' if d and not np.isnan(f1(k)) else '—'})
display(pd.DataFrame(rows))
fig,axes=plt.subplots(1,3,figsize=(15,4),sharey=True)
for ax,shot in zip(axes,SHOTS):
    cfgs=[('R50 Euc',f'ncc_{shot}shot_euc_json','#fd8d3c'),('R50 Man',f'ncc_{shot}shot_man_json','#d94801'),
           ('DeiT-S Euc',f'vit_ncc_{shot}shot_euc_json','#9467bd'),('DeiT-S Man',f'vit_ncc_{shot}shot_man_json','#c5b0d5')]
    ls=[c[0] for c in cfgs]; vs=[acc(c[1]) for c in cfgs]; es=[ci(c[1]) for c in cfgs]; cs=[c[2] for c in cfgs]
    bars=ax.bar(ls,vs,color=cs,edgecolor='white',width=0.6)
    ax.errorbar(range(len(ls)),vs,yerr=es,fmt='none',color='black',capsize=4)
    for bar,v in zip(bars,vs): ax.text(bar.get_x()+bar.get_width()/2,v+0.4,f'{v:.1f}%',ha='center',fontsize=8,fontweight='bold')
    ax.set_title(f'{shot}-Shot NCC'); ax.set_ylim(0,108)
    ax.set_ylabel('Accuracy (%)' if shot==1 else ''); ax.tick_params(axis='x',labelrotation=15,labelsize=8)
fig.suptitle('NCC: ResNet-50 vs DeiT-S — B10/N10',fontsize=12)
plt.tight_layout(); plt.savefig(f'{ANALYSIS}/b10n10_ncc_r50_vs_deit.png',dpi=150); plt.show()
"""),
    md("---\n## 3. ProtoNet: ResNet-50 vs DeiT-S (B10/N10)"),
    code("""
rows=[]
for s in SHOTS:
    for label,tmpl in [('ProtoNet/R50 (Euc)','proto_{s}shot_euc_json'),('ProtoNet/R50 (Man)','proto_{s}shot_man_json'),
                        ('ProtoNet/DeiT-S (Euc)','vit_proto_{s}shot_euc_json'),('ProtoNet/DeiT-S (Man)','vit_proto_{s}shot_man_json')]:
        k=tmpl.format(s=s); d=R.get(k)
        rows.append({'Model':label,'K-Shot':s,'Accuracy(%)':f'{acc(k):.2f}' if d else '—',
                     '95% CI':f'+-{ci(k):.2f}' if d else '—','F1':f'{f1(k):.4f}' if d and not np.isnan(f1(k)) else '—'})
display(pd.DataFrame(rows))
"""),
    md(["---\n## 4. MatchingNet: Vanilla vs FCE (B10/N10)\n\n",
        "FCE adds BiLSTM support encoding + attention LSTM query encoding."]),
    code("""
rows=[]
for s in SHOTS:
    for label,tmpl in [('MatchingNet (vanilla)','match_{s}shot_json'),('MatchingNet+FCE','match_fce_{s}shot_json')]:
        k=tmpl.format(s=s); d=R.get(k)
        rows.append({'Model':label,'K-Shot':s,'Accuracy(%)':f'{acc(k):.2f}' if d else '—',
                     '95% CI':f'+-{ci(k):.2f}' if d else '—','F1':f'{f1(k):.4f}' if d and not np.isnan(f1(k)) else '—'})
display(pd.DataFrame(rows))
deltas=[acc(f'match_fce_{s}shot_json')-acc(f'match_{s}shot_json') for s in SHOTS]
print('FCE effect: '+', '.join([f'{s}-shot: {d:+.2f}pp' for s,d in zip(SHOTS,deltas)]))
"""),
    md("---\n## 5. RelationNet (B10/N10)"),
    code("""
print('RelationNet — B10/N10')
for s in SHOTS:
    k=f'rel_{s}shot_json'
    print(f'  {s}-shot: {acc(k):.2f}% +-{ci(k):.2f}  F1={f1(k):.4f}')
"""),
    md("---\n## 6. ProtoNet + TTA (B10/N10)"),
    code("""
base_a=acc('proto_5shot_euc_json'); tta_a=acc('proto_5shot_euc_tta8_json')
base_c=ci('proto_5shot_euc_json');  tta_c=ci('proto_5shot_euc_tta8_json')
print(f'Without TTA: {base_a:.2f}% +-{base_c:.2f}')
print(f'With TTA x8: {tta_a:.2f}% +-{tta_c:.2f}')
print(f'Delta: {tta_a-base_a:+.2f} pp')
"""),
    md("---\n## 7. K-Shot Trend — All Methods (B10/N10)"),
    code(METHODS_CODE + KSHOT_PLOT + """
ax.set_title('K-Shot Trend — All Methods — B10/N10\\n(5-way, 10 novel, 600 episodes)')
ax.legend(loc='lower right',fontsize=8,ncol=2)
plt.tight_layout(); plt.savefig(f'{ANALYSIS}/b10n10_kshot_all.png',dpi=150); plt.show()
"""),
    md("---\n## 8. Per-Class Accuracy Heatmap (B10/N10)"),
    code(METHODS_CODE + HEATMAP_CODE),
    md("---\n## 9. Hardest Novel Classes Analysis"),
    code("""
# Average per-class accuracy across all methods at 5-shot euclidean
keys_5shot=['ncc_5shot_euc_json','vit_ncc_5shot_euc_json','proto_5shot_euc_json','vit_proto_5shot_euc_json',
             'match_5shot_json','match_fce_5shot_json','rel_5shot_json']
class_accs={c:[] for c in NOVEL}
for k in keys_5shot:
    d=R.get(k)
    if d and 'per_class_accuracy_pct' in d:
        for c in NOVEL: class_accs[c].append(d['per_class_accuracy_pct'].get(c,float('nan')))
avg_cls={c:np.nanmean(vs) for c,vs in class_accs.items() if vs}
std_cls={c:np.nanstd(vs) for c,vs in class_accs.items() if vs}

fig,ax=plt.subplots(figsize=(12,4))
sorted_cls=sorted(avg_cls.keys(),key=lambda c:avg_cls[c])
colors=['#d62728' if avg_cls[c]<80 else '#ff7f0e' if avg_cls[c]<90 else '#2ca02c' for c in sorted_cls]
x=np.arange(len(sorted_cls))
ax.bar(x,[avg_cls[c] for c in sorted_cls],color=colors,
        yerr=[std_cls[c] for c in sorted_cls],capsize=4,edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels([c.replace('AMERICAN ','AM.').replace('ALEXANDRINE ','ALEX.').replace('ALTAMIRA ','ALTA.').replace('ALBERTS ','ALB.').replace('ALPINE ','ALP.') for c in sorted_cls],rotation=25,ha='right',fontsize=8)
ax.set_ylabel('Avg Accuracy (%) across all methods'); ax.set_ylim(0,105)
ax.set_title('Per-Class Difficulty Ranking — B10/N10 (5-shot, 7 methods)')
ax.axhline(80,color='orange',linestyle='--',alpha=0.5,label='80% threshold')
ax.legend(); ax.grid(axis='y',alpha=0.3)
plt.tight_layout(); plt.savefig(f'{ANALYSIS}/b10n10_class_difficulty.png',dpi=150); plt.show()
print('Hardest novel classes:')
for c in sorted_cls[:3]: print(f'  {c}: {avg_cls[c]:.1f}% avg')
"""),
    md("---\n## 10. Full Summary Table"),
    code(SUMMARY_CODE_10.replace("b15n5_summary.csv","b10n10_summary.csv")),
    md("---\n## 11. Key Insights (B10/N10)"),
    code(INSIGHTS_10),
]

# ===========================================================================
# BUILD result_comparison.ipynb
# ===========================================================================
cells_cmp = [
    md(["# Few-Shot Learning — Cross-Split Comparison: B15/N5 vs B10/N10\n",
        "\nCompares all FSL methods across both dataset splits and both backbones.\n",
        "\nKey questions answered:\n",
        "- Does DeiT-S backbone consistently outperform ResNet-50?\n",
        "- Does FCE improve MatchingNet?\n",
        "- How much does reducing base classes (15→10) hurt performance?\n",
        "- Does TTA help?\n",
        "- Which novel classes are hardest across splits?"]),
    code(COMPARISON_SETUP),
    md("---\n## 1. K-Shot Trend: B15/N5 vs B10/N10 (per method)"),
    code(COMPARISON_KSHOT),
    md("---\n## 2. Accuracy Delta: B10/N10 − B15/N5"),
    code(COMPARISON_DELTA),
    md(["---\n## 3. Backbone Ablation: DeiT-S vs ResNet-50 (Both Splits)\n\n",
        "Direct comparison of backbone impact on the same FSL algorithm."]),
    code(COMPARISON_BACKBONE),
    md("---\n## 4. MatchingNet FCE Effect (Both Splits)"),
    code(COMPARISON_FCE),
    md("---\n## 5. TTA Effect (Both Splits)"),
    code("""
print('ProtoNet TTA Effect (5-shot Euclidean):')
for split_label,split_data in [('B15/N5',B15),('B10/N10',B10)]:
    base=fsl_acc(split_data['proto_euc'].get(5)); tta=fsl_acc(split_data['proto_tta8'].get(5))
    print(f'  {split_label}: base={base:.2f}%  TTA x8={tta:.2f}%  delta={tta-base:+.2f}pp')
"""),
    md("---\n## 6. Master Summary Table"),
    code(COMPARISON_SUMMARY),
    md("---\n## 7. Research Questions — Quantitative Analysis"),
    code(RQANALYSIS),
    md(["---\n## 8. NCC > ProtoNet Reversal at B10/N10\n\n",
        "On B10/N10 at 5-shot, NCC (Euclidean) outperforms ProtoNet — an anomaly absent in B15/N5.\n",
        "Hypothesis: with only 10 base classes the ProtoNet backbone is under-trained, making the simple centroid\n",
        "distance of NCC more robust than the learned metric. The reversal is consistent across distance metrics."]),
    code("""
print('NCC vs ProtoNet (Euclidean) — full shot/split matrix:')
print(f'{"":30} {"B15/N5":>18} {"B10/N10":>18}')
print(f'{"":30} {"NCC":>9} {"Proto":>9} {"NCC":>9} {"Proto":>9}')
print('-'*70)
for s in SHOTS:
    n15=fsl_acc(B15['ncc_euc'].get(s)); p15=fsl_acc(B15['proto_euc'].get(s))
    n10=fsl_acc(B10['ncc_euc'].get(s)); p10=fsl_acc(B10['proto_euc'].get(s))
    rev15='*REV*' if n15>p15 else '     '; rev10='*REV*' if n10>p10 else '     '
    print(f'  {s}-shot:{rev15}           {n15:6.2f}%  {p15:6.2f}%  {n10:6.2f}%  {p10:6.2f}%{rev10}')

print()
print('NCC vs ProtoNet (Manhattan) — reversal check:')
for s in SHOTS:
    n15=fsl_acc(B15['ncc_man'].get(s)); p15=fsl_acc(B15['proto_man'].get(s))
    n10=fsl_acc(B10['ncc_man'].get(s)); p10=fsl_acc(B10['proto_man'].get(s))
    rev10='  <-- REVERSAL' if n10>p10 else ''
    print(f'  {s}-shot B10: NCC={n10:.2f}%  Proto={p10:.2f}%{rev10}')

print()
print('DeiT-S backbone: NCC vs ProtoNet reversal at B10/N10?')
for s in SHOTS:
    n10=fsl_acc(B10['vit_ncc_euc'].get(s)); p10=fsl_acc(B10['vit_proto_euc'].get(s))
    rev10='  <-- reversal persists' if n10>p10 else '  (no reversal with DeiT-S)'
    print(f'  {s}-shot B10 DeiT-S: NCC={n10:.2f}%  Proto={p10:.2f}%{rev10}')
"""),
    md(["---\n## 9. Distance Metric: Euclidean vs Manhattan\n\n",
        "Across all methods and splits, which distance metric wins more consistently?"]),
    code("""
pairs=[('NCC R50','ncc_euc','ncc_man'),('NCC DeiT-S','vit_ncc_euc','vit_ncc_man'),
       ('ProtoNet R50','proto_euc','proto_man'),('ProtoNet DeiT-S','vit_proto_euc','vit_proto_man')]
euc_wins=0; man_wins=0; total=0
print(f'{"Method":<22} {"Split":<8} {"Shot":<6} {"Euc%":>7} {"Man%":>7} {"Winner":<8}')
print('-'*65)
for label,ekey,mkey in pairs:
    for split_label,split_data in [('B15/N5',B15),('B10/N10',B10)]:
        for s in SHOTS:
            e=fsl_acc(split_data[ekey].get(s)); m=fsl_acc(split_data[mkey].get(s))
            if np.isnan(e) or np.isnan(m): continue
            winner='Euc' if e>=m else 'Man'
            if e>=m: euc_wins+=1
            else: man_wins+=1
            total+=1
            print(f'{label:<22} {split_label:<8} {s}-shot  {e:7.2f} {m:7.2f} {winner}')
print()
print(f'Summary: Euclidean wins {euc_wins}/{total} ({100*euc_wins/total:.0f}%)  |  Manhattan wins {man_wins}/{total} ({100*man_wins/total:.0f}%)')
"""),
    md(["---\n## 10. Statistical Significance Tests\n\n",
        "Welch's t-test on episode-level accuracies for key comparisons.\n",
        "Note: episode-level accuracy arrays are not saved by default. These tests use Wilson CI overlap as a proxy."]),
    code("""
print('CI Overlap Tests (non-overlapping 95% CI => p < 0.05 proxy):')
comparisons=[
    ('DeiT-S NCC vs R50 NCC (1-shot,B15)',
     B15['vit_ncc_euc'].get(1), B15['ncc_euc'].get(1)),
    ('DeiT-S ProtoNet vs R50 ProtoNet (1-shot,B15)',
     B15['vit_proto_euc'].get(1), B15['proto_euc'].get(1)),
    ('MatchingNet+FCE vs Vanilla (5-shot,B15)',
     B15['match_fce'].get(5), B15['match'].get(5)),
    ('ProtoNet vs NCC R50 (1-shot,B15)',
     B15['proto_euc'].get(1), B15['ncc_euc'].get(1)),
    ('B15 vs B10 ProtoNet (5-shot Euc)',
     B15['proto_euc'].get(5), B10['proto_euc'].get(5)),
]
for label,d1,d2 in comparisons:
    if not d1 or not d2: print(f'  {label}: missing data'); continue
    a1,c1=fsl_acc(d1),fsl_ci(d1)
    a2,c2=fsl_acc(d2),fsl_ci(d2)
    overlap=(a1-c1<=a2+c2) and (a2-c2<=a1+c1)
    sig='likely NOT significant (CI overlap)' if overlap else 'likely SIGNIFICANT (no CI overlap)'
    print(f'  {label}:')
    print(f'    {a1:.2f}%+-{c1:.2f} vs {a2:.2f}%+-{c2:.2f} => {sig}')
"""),
    md(["---\n## 11. Hardest Novel Classes — Cross-Split Analysis\n\n",
        "Which novel classes are consistently hard, and which degrade most when moving B15→B10?"]),
    code("""
# B15/N5 novel classes
novel_15=['AMERICAN BITTERN','AMERICAN COOT','AMERICAN FLAMINGO','AMERICAN GOLDFINCH','AMERICAN KESTREL']
novel_10=['ALBERTS TOWHEE','ALEXANDRINE PARAKEET','ALPINE CHOUGH','ALTAMIRA YELLOWTHROAT',
          'AMERICAN AVOCET','AMERICAN BITTERN','AMERICAN COOT','AMERICAN FLAMINGO','AMERICAN GOLDFINCH','AMERICAN KESTREL']

# Gather per-class accuracy from 5-shot euclidean methods
class_results_15={c:[] for c in novel_15}
class_results_10={c:[] for c in novel_10}

for mkey in ['ncc_euc','vit_ncc_euc','proto_euc','vit_proto_euc']:
    d=B15[mkey].get(5)
    if d and 'per_class_accuracy_pct' in d:
        for c in novel_15: class_results_15[c].append(d['per_class_accuracy_pct'].get(c,float('nan')))
    d=B10[mkey].get(5)
    if d and 'per_class_accuracy_pct' in d:
        for c in novel_10: class_results_10[c].append(d['per_class_accuracy_pct'].get(c,float('nan')))

# Shared novel classes — ranked by average difficulty
shared=[c for c in novel_15 if c in novel_10]
print('Shared novel classes — 5-shot avg across 4 methods (ranked hardest→easiest):')
print(f'  {"Class":<30} {"B15/N5":>8} {"B10/N10":>8} {"Drop":>7}')
print('  '+'-'*58)
for c in sorted(shared,key=lambda c:np.nanmean(class_results_15[c]+class_results_10[c])):
    a15=np.nanmean(class_results_15[c]); a10=np.nanmean(class_results_10[c])
    drop=a10-a15
    print(f'  {c:<30} {a15:8.1f}% {a10:8.1f}% {drop:+7.1f}pp')

# B10-only novel classes
b10_only=[c for c in novel_10 if c not in novel_15]
if any(class_results_10[c] for c in b10_only):
    print()
    print('B10-only novel classes (5-shot avg across methods):')
    for c in sorted(b10_only,key=lambda c:np.nanmean(class_results_10[c])):
        a10=np.nanmean(class_results_10[c])
        print(f'  {c:<30} B10={a10:.1f}%')
"""),
    md(["---\n## 12. Per-Class B15→B10 Degradation (Heatmap)\n\n",
        "Visualises which shared novel classes suffer most when base classes are reduced from 15 to 10."]),
    code("""
shared=['AMERICAN BITTERN','AMERICAN COOT','AMERICAN FLAMINGO','AMERICAN GOLDFINCH','AMERICAN KESTREL']
methods_for_heatmap=[('NCC/R50','ncc_euc'),('NCC/DeiT-S','vit_ncc_euc'),
                     ('Proto/R50','proto_euc'),('Proto/DeiT-S','vit_proto_euc')]
mat=np.full((len(methods_for_heatmap),len(shared)),np.nan)
for ri,(label,mkey) in enumerate(methods_for_heatmap):
    d15=B15[mkey].get(5); d10=B10[mkey].get(5)
    if d15 and d10 and 'per_class_accuracy_pct' in d15 and 'per_class_accuracy_pct' in d10:
        for ci,c in enumerate(shared):
            a15=d15['per_class_accuracy_pct'].get(c,np.nan)
            a10=d10['per_class_accuracy_pct'].get(c,np.nan)
            mat[ri,ci]=a10-a15

fig,ax=plt.subplots(figsize=(9,4))
mask=np.isnan(mat)
if not mask.all():
    import seaborn as sns
    sns.heatmap(mat,annot=True,fmt='.1f',cmap='RdYlGn',center=0,
                xticklabels=[c.replace('AMERICAN ','AM. ') for c in shared],
                yticklabels=[l for l,_ in methods_for_heatmap],
                ax=ax,vmin=-20,vmax=20,linewidths=0.5,linecolor='white',
                annot_kws={'size':10},mask=mask)
    ax.set_title('Per-Class Accuracy Change: B10/N10 − B15/N5 (pp) at 5-shot\\n(green=better in B10, red=worse)',fontsize=11)
    ax.tick_params(axis='x',labelrotation=15)
    plt.tight_layout()
    plt.savefig(f'{ANALYSIS}/per_class_b10_vs_b15_delta.png',dpi=150)
plt.show()
"""),
    md("---\n## 13. Final Ranking — Best Method per Scenario"),
    code("""
print('='*65)
print('  FINAL RANKINGS')
print('='*65)
scenarios=[
    ('B15/N5 1-shot',[('NCC/DeiT-S','vit_ncc_euc',B15,1),('ProtoNet/DeiT-S','vit_proto_euc',B15,1),
                      ('NCC/R50','ncc_euc',B15,1),('ProtoNet/R50','proto_euc',B15,1),
                      ('RelationNet','rel',B15,1),('MatchingNet','match',B15,1),('MatchingNet+FCE','match_fce',B15,1)]),
    ('B15/N5 5-shot',[('NCC/DeiT-S','vit_ncc_euc',B15,5),('ProtoNet/DeiT-S','vit_proto_euc',B15,5),
                      ('ProtoNet/R50+TTA8','proto_tta8',B15,5),('NCC/R50','ncc_euc',B15,5),
                      ('ProtoNet/R50','proto_euc',B15,5),('RelationNet','rel',B15,5),
                      ('MatchingNet','match',B15,5),('MatchingNet+FCE','match_fce',B15,5)]),
    ('B10/N10 1-shot',[('NCC/DeiT-S','vit_ncc_euc',B10,1),('ProtoNet/DeiT-S','vit_proto_euc',B10,1),
                       ('NCC/R50','ncc_euc',B10,1),('ProtoNet/R50','proto_euc',B10,1),
                       ('RelationNet','rel',B10,1),('MatchingNet','match',B10,1),('MatchingNet+FCE','match_fce',B10,1)]),
    ('B10/N10 5-shot',[('NCC/DeiT-S','vit_ncc_euc',B10,5),('ProtoNet/DeiT-S','vit_proto_euc',B10,5),
                       ('ProtoNet/R50+TTA8','proto_tta8',B10,5),('NCC/R50','ncc_euc',B10,5),
                       ('ProtoNet/R50','proto_euc',B10,5),('RelationNet','rel',B10,5),
                       ('MatchingNet','match',B10,5),('MatchingNet+FCE','match_fce',B10,5)]),
]
def _sort_key(x):
    v=fsl_acc(x[2][x[1]].get(x[3]))
    return v if not np.isnan(v) else float('-inf')
for title,entries in scenarios:
    print(f'\\n{title}:')
    ranked=sorted(entries,key=_sort_key,reverse=True)
    for i,(name,mkey,split_data,shot) in enumerate(ranked,1):
        d=split_data[mkey].get(shot)
        if d:
            print(f'  {i}. {name:<25} {fsl_acc(d):.2f}% +-{fsl_ci(d):.2f}')
        else:
            print(f'  {i}. {name:<25} —')
"""),
]

# ===========================================================================
# WRITE ALL NOTEBOOKS
# ===========================================================================
nb_dir = HERE

notebooks = [
    ('results15b5n.ipynb', cells_15),
    ('results10b10n.ipynb', cells_10),
    ('result_comparison.ipynb', cells_cmp),
]

for filename, cells in notebooks:
    path = os.path.join(nb_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb(cells), f, ensure_ascii=False, indent=1)
    print(f'Written {filename}: {len(cells)} cells')

print('\nAll notebooks updated.')
