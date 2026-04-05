"""
DeiT-S ProtoNet Evaluation — episodic evaluation on novel classes.

Usage:
    python evaluate.py --k-shot 1 --distance euclidean
    python evaluate.py --k-shot 5 --distance manhattan
    python evaluate.py --config ../../../configs/birds20/vit_proto_configs_10b_10n.yaml --k-shot 3 --distance euclidean
    python evaluate.py --debug
"""
import os
import sys
import json
import random
import argparse
import math
from typing import Dict, List, Sequence, Tuple

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['figure.facecolor']  = 'white'
plt.rcParams['axes.facecolor']    = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from utils.dataset   import FilteredDataset, get_eval_transforms
from models.protonet  import ProtoNet
from models.vit       import DeiTSmallEncoder


# ---------------------------------------------------------------------------
# Helpers  (identical to ResNet-50 ProtoNet evaluate.py)
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def wilson_ci(correct: int, total: int) -> Tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.96
    p = correct / total
    den = 1 + z**2 / total
    ctr = (p + z**2 / (2 * total)) / den
    mar = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / den
    return max(0.0, ctr - mar), min(1.0, ctr + mar)


def sanitize_name(name: str) -> str:
    return name.lower().replace(' ', '_').replace('/', '_').replace('-', '_')


def sample_episode_class_indices(rng, total_novel_classes, n_way):
    if n_way > total_novel_classes:
        raise ValueError(f'n_way={n_way} > total novel classes={total_novel_classes}')
    return sorted(rng.sample(range(total_novel_classes), n_way))


def build_episode_support_query(
    novel_train_ds, novel_test_ds,
    selected_class_indices, k_shot, q_query, rng, device,
):
    train_by_class = novel_train_ds.get_class_to_indices()
    test_by_class  = novel_test_ds.get_class_to_indices()

    support_images, support_labels = [], []
    query_images,   query_labels   = [], []

    for epi_label, global_cls_idx in enumerate(selected_class_indices):
        train_cands = train_by_class[global_cls_idx]
        test_cands  = test_by_class[global_cls_idx]

        if len(train_cands) < k_shot:
            raise ValueError(f'Class {global_cls_idx}: {len(train_cands)} train < k_shot={k_shot}')
        if len(test_cands) < q_query:
            raise ValueError(f'Class {global_cls_idx}: {len(test_cands)} test < q_query={q_query}')

        sampled_support = rng.sample(train_cands, k_shot)
        sampled_query   = rng.sample(test_cands, q_query)

        for ds_idx in sampled_support:
            img, _ = novel_train_ds[ds_idx]
            support_images.append(img)
            support_labels.append(epi_label)

        for ds_idx in sampled_query:
            img, _ = novel_test_ds[ds_idx]
            query_images.append(img)
            query_labels.append(epi_label)

    support_tensor = torch.stack(support_images).to(device)
    query_tensor   = torch.stack(query_images).to(device)
    s_label_tensor = torch.tensor(support_labels, dtype=torch.long, device=device)
    q_label_tensor = torch.tensor(query_labels,   dtype=torch.long, device=device)

    return support_tensor, query_tensor, s_label_tensor, q_label_tensor


def save_confusion_matrix(cm, class_names, save_path, title):
    n = len(class_names)
    if n <= 20:
        sz, tick_fs = 10, 8
    elif n <= 50:
        sz, tick_fs = 16, 6
    elif n <= 100:
        sz, tick_fs = 22, 5
    else:
        sz, tick_fs = 28, 4
    fig, ax = plt.subplots(figsize=(sz, sz - 2))
    sns.heatmap(cm, annot=False, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=tick_fs)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=tick_fs)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',   default='../../../configs/birds20/vit_proto_configs.yaml')
    parser.add_argument('--k-shot',   type=int, default=None)
    parser.add_argument('--distance', default=None, choices=['euclidean', 'manhattan'])
    parser.add_argument('--debug',    action='store_true')
    args = parser.parse_args()

    config_path = os.path.normpath(os.path.join(SCRIPT_DIR, args.config))
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    seed          = cfg['seed']
    image_size    = cfg['image_size']
    n_way         = cfg['eval']['n_way']
    k_shot        = args.k_shot if args.k_shot is not None else cfg['eval']['k_shot']
    q_query       = cfg['eval'].get('q_query', 5)
    distance      = args.distance if args.distance is not None else cfg['eval'].get('distance', 'euclidean')
    num_episodes  = 5 if args.debug else cfg['eval']['episodes']
    novel_classes = cfg['novel_classes']

    set_seed(seed)
    rng    = random.Random(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 60)
    print('DEVICE INFO')
    print(f'  CUDA available : {torch.cuda.is_available()}')
    print(f'  Device         : {device}')
    if torch.cuda.is_available():
        print(f'  GPU name       : {torch.cuda.get_device_name(0)}')
        print(f'  GPU memory     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print('EVAL CONFIG')
    print(f'  Backbone       : deit_small_patch16_224 (384-dim)')
    print(f'  Config file    : {config_path}')
    print(f'  Novel classes  : {len(novel_classes)}')
    print(f'  n_way          : {n_way}')
    print(f'  k_shot         : {k_shot}')
    print(f'  q_query        : {q_query}')
    print(f'  Episodes       : {num_episodes}')
    print(f'  Distance       : {distance}')
    print('=' * 60)

    data_train   = os.path.normpath(os.path.join(SCRIPT_DIR, cfg['data']['train']))
    data_test    = os.path.normpath(os.path.join(SCRIPT_DIR, cfg['data']['test']))
    ckpt_dir     = os.path.normpath(os.path.join(SCRIPT_DIR, cfg['output']['ckpt_dir']))
    results_base = os.path.normpath(os.path.join(SCRIPT_DIR, cfg['output']['results_dir']))
    results_dir  = os.path.join(results_base, f'{k_shot}shot_{distance}')
    os.makedirs(results_dir, exist_ok=True)

    transform = get_eval_transforms(image_size)

    novel_train_ds = FilteredDataset(data_train, novel_classes, transform=transform)
    novel_test_ds  = FilteredDataset(data_test,  novel_classes, transform=transform)

    print(f'Novel train samples : {len(novel_train_ds)}')
    print(f'Novel test samples  : {len(novel_test_ds)}')

    test_by_class = novel_test_ds.get_class_to_indices()
    for idx, cls_name in enumerate(novel_classes):
        n_test = len(test_by_class[idx])
        if n_test < q_query:
            raise ValueError(f"Class '{cls_name}' has {n_test} test images < q_query={q_query}")

    ckpt_path = os.path.join(ckpt_dir, cfg['output']['ckpt_name'])

    # Auto-detect pretrained weights to avoid HuggingFace download on HPC.
    # The ProtoNet checkpoint already contains the fine-tuned weights, so we
    # build the architecture (pretrained=False) and let load_state_dict restore
    # everything from the checkpoint — no internet needed.
    _pretrained_candidates = [
        os.path.join(PROJECT_ROOT, 'pretrained_weights', 'deit_small_patch16_224.pth'),
        os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'deit_small_patch16_224.pth')),
    ]
    _weights_path = next((p for p in _pretrained_candidates if os.path.isfile(p)), None)
    backbone = DeiTSmallEncoder(pretrained=(_weights_path is None), weights_path=_weights_path)
    if _weights_path:
        print(f'[auto] DeiT-S init weights: {_weights_path}')
    model     = ProtoNet(encoder=backbone).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Loaded checkpoint   : {ckpt_path} (trained to epoch {ckpt["epoch"]})')

    episode_accuracies: List[float] = []
    all_truths, all_preds, all_probs = [], [], []

    episode_class_counter = {cls: 0 for cls in novel_classes}
    per_class_correct     = {cls: 0 for cls in novel_classes}
    per_class_total       = {cls: 0 for cls in novel_classes}

    global_cm = np.zeros((len(novel_classes), len(novel_classes)), dtype=float)
    first_episode_class_indices: List[int] = []

    with torch.no_grad():
        for ep in tqdm(range(num_episodes), desc=f'{n_way}-way {k_shot}-shot eval'):
            selected_class_indices = sample_episode_class_indices(
                rng=rng, total_novel_classes=len(novel_classes), n_way=n_way)
            if ep == 0:
                first_episode_class_indices = selected_class_indices.copy()

            for cls_name in [novel_classes[i] for i in selected_class_indices]:
                episode_class_counter[cls_name] += 1

            support_images, query_images, support_labels, query_labels = build_episode_support_query(
                novel_train_ds=novel_train_ds,
                novel_test_ds=novel_test_ds,
                selected_class_indices=selected_class_indices,
                k_shot=k_shot, q_query=q_query, rng=rng, device=device,
            )

            logits      = model(support_images, support_labels, query_images, distance=distance)
            probs_local = torch.softmax(logits, dim=1).cpu().numpy()
            preds_local = logits.argmax(dim=1).cpu().numpy()
            truths_local = query_labels.cpu().numpy()

            acc = float((preds_local == truths_local).mean())
            episode_accuracies.append(acc)

            all_truths.extend(truths_local.tolist())
            all_preds.extend(preds_local.tolist())
            all_probs.extend(probs_local.tolist())

            for t_local, p_local in zip(truths_local, preds_local):
                t_global  = selected_class_indices[int(t_local)]
                p_global  = selected_class_indices[int(p_local)]
                true_name = novel_classes[t_global]
                per_class_total[true_name]   += 1
                per_class_correct[true_name] += int(t_local == p_local)
                global_cm[t_global, p_global] += 1

    mean_acc = float(np.mean(episode_accuracies)) * 100.0
    std_acc  = float(np.std(episode_accuracies))  * 100.0
    ci95     = 1.96 * std_acc / math.sqrt(len(episode_accuracies))

    total_correct = sum(per_class_correct.values())
    total_queries = sum(per_class_total.values())
    ci_low, ci_high = wilson_ci(total_correct, total_queries)

    per_class_acc = {
        cls: (100.0 * per_class_correct[cls] / per_class_total[cls])
             if per_class_total[cls] > 0 else 0.0
        for cls in novel_classes
    }

    macro_f1        = f1_score(all_truths, all_preds, average='macro', zero_division=0)
    macro_precision = precision_score(all_truths, all_preds, average='macro', zero_division=0)
    macro_recall    = recall_score(all_truths, all_preds, average='macro', zero_division=0)
    try:
        auc_ovr = float(roc_auc_score(all_truths, all_probs, multi_class='ovr', average='macro'))
    except Exception:
        auc_ovr = None

    print(f'\n{n_way}-way {k_shot}-shot | {len(episode_accuracies)} episodes')
    print(f'Mean accuracy  : {mean_acc:.2f}% ± {ci95:.2f}% (95% CI)')
    print(f'Std            : {std_acc:.2f}%')
    print(f'Wilson CI      : [{ci_low*100:.2f}%, {ci_high*100:.2f}%]')
    print(f'Macro F1       : {macro_f1:.4f}')
    print(f'Macro Precision: {macro_precision:.4f}')
    print(f'Macro Recall   : {macro_recall:.4f}')
    print(f'AUC-OVR (macro): {auc_ovr:.4f}' if auc_ovr is not None else 'AUC-OVR (macro): N/A')
    print('\nPer-class accuracy:')
    for cls in novel_classes:
        print(f'  {cls:<30s}: {per_class_acc[cls]:6.2f}% '
              f'| appeared in {episode_class_counter[cls]:4d} episodes '
              f'| queries={per_class_total[cls]:5d}')

    summary = {
        'experiment':        cfg['experiment']['name'],
        'backbone':          'deit_small_patch16_224',
        'embedding_dim':     384,
        'seed':              seed,
        'n_way':             n_way,
        'k_shot':            k_shot,
        'q_query':           q_query,
        'distance':          distance,
        'num_episodes':      len(episode_accuracies),
        'novel_classes':     novel_classes,
        'checkpoint':        ckpt_path,
        'epochs_trained':    ckpt['epoch'],
        'mean_accuracy_pct': round(mean_acc, 4),
        'std_accuracy_pct':  round(std_acc, 4),
        'ci95_pct':          round(ci95, 4),
        'wilson_ci_95':      [round(ci_low * 100, 4), round(ci_high * 100, 4)],
        'macro_f1':          round(float(macro_f1), 4),
        'macro_precision':   round(float(macro_precision), 4),
        'macro_recall':      round(float(macro_recall), 4),
        'auc_ovr':           round(auc_ovr, 4) if auc_ovr is not None else None,
        'per_class_accuracy_pct': {cls: round(acc, 4) for cls, acc in per_class_acc.items()},
        'per_class_episode_frequency': episode_class_counter,
        'per_class_query_total': per_class_total,
    }

    with open(os.path.join(results_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    txt_lines = [
        f'DeiT-S ProtoNet — episodic novel-class evaluation ({distance})',
        '=' * 60,
        f'Episodes       : {len(episode_accuracies)}',
        f'n_way          : {n_way}',
        f'k_shot         : {k_shot}',
        f'q_query        : {q_query}',
        f'Distance       : {distance}',
        f'Novel classes  : {", ".join(novel_classes)}',
        f'Checkpoint     : {ckpt_path}',
        '',
        f'Mean accuracy  : {mean_acc:.2f}% ± {ci95:.2f}% (95% CI)',
        f'Std            : {std_acc:.2f}%',
        f'Wilson CI 95%  : [{ci_low*100:.2f}%, {ci_high*100:.2f}%]',
        f'Macro F1       : {macro_f1:.4f}',
        f'Macro Precision: {macro_precision:.4f}',
        f'Macro Recall   : {macro_recall:.4f}',
        f'AUC-OVR (macro): {auc_ovr:.4f}' if auc_ovr is not None else 'AUC-OVR (macro): N/A',
        '',
        'Per-class accuracy:',
    ] + [
        f'  {cls:<30s}: {per_class_acc[cls]:6.2f}% | appeared={episode_class_counter[cls]:4d} | queries={per_class_total[cls]:5d}'
        for cls in novel_classes
    ]

    with open(os.path.join(results_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(txt_lines) + '\n')

    pd.DataFrame({
        'episode':  list(range(1, len(episode_accuracies) + 1)),
        'accuracy': episode_accuracies,
    }).to_csv(os.path.join(results_dir, 'episode_accuracies.csv'), index=False)

    cm_avg = global_cm / len(episode_accuracies)
    save_confusion_matrix(
        cm=cm_avg,
        class_names=[c.replace('AMERICAN ', 'AM. ') for c in novel_classes],
        save_path=os.path.join(results_dir, 'confusion_matrix.png'),
        title='DeiT-S ProtoNet — Aggregated Confusion Matrix',
    )

    print(f'\nResults saved to: {results_dir}')


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
