# evaluate_novel_ncc.py — NCC evaluation with DeiT-S backbone
# Works for both:
#   - vit_baseline_b15_config.yaml  (B15/N5)
#   - vit_baseline_b10_config.yaml  (B10/N10)
#
# Loads the fine-tuned DeiT-S checkpoint, strips the classification head,
# and runs episodic NCC evaluation on novel classes.
#
# Usage:
#   python evaluate_novel_ncc.py --k-shot 1 --distance euclidean
#   python evaluate_novel_ncc.py --config ../../../configs/vit_baseline_b10_config.yaml --k-shot 5 --distance manhattan

import os
import sys
import json
import math
import yaml
import random
import argparse

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['figure.facecolor']  = 'white'
plt.rcParams['axes.facecolor']    = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

from torchvision import transforms, datasets
from torch.utils.data import Subset
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.vit import build_deit_classifier


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def wilson_ci(correct: int, total: int, alpha: float = 0.05):
    if total == 0:
        return 0.0, 0.0
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    p = correct / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def filter_by_classes(imagefolder_ds: datasets.ImageFolder, allowed_classnames: list):
    missing = [c for c in allowed_classnames if c not in imagefolder_ds.class_to_idx]
    if missing:
        raise ValueError(f'Classes missing from dataset folders: {missing}')

    old_c2i    = imagefolder_ds.class_to_idx
    new_c2i    = {c: i for i, c in enumerate(allowed_classnames)}
    allowed_ids = {old_c2i[c] for c in allowed_classnames}
    old_i2c    = {v: k for k, v in old_c2i.items()}

    indices = [i for i, (_, y) in enumerate(imagefolder_ds.samples) if y in allowed_ids]

    def target_transform(old_y: int) -> int:
        return new_c2i[old_i2c[int(old_y)]]

    return indices, new_c2i, target_transform


def extract_embeddings(model, subset, device):
    model.eval()
    embeddings, labels = [], []

    loader = torch.utils.data.DataLoader(
        subset, batch_size=32, shuffle=False,
        num_workers=0, pin_memory=(device.type == 'cuda'),
    )
    with torch.inference_mode():
        for images, lbls in loader:
            feats = model(images.to(device)).cpu().numpy()
            embeddings.append(feats)
            labels.extend(lbls.numpy())

    return np.vstack(embeddings), np.array(labels)


def pairwise_distance(query, prototypes, distance='euclidean'):
    if distance == 'euclidean':
        return np.sqrt(((query[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2))
    elif distance == 'manhattan':
        return np.abs(query[:, None, :] - prototypes[None, :, :]).sum(axis=2)
    else:
        raise ValueError(f'Unsupported distance: {distance}')


def sample_episode(
    train_embeddings, train_labels,
    test_embeddings, test_labels,
    selected_global_classes,
    k_shot, q_query, rng,
):
    support_x, support_y = [], []
    query_x, query_y     = [], []

    for epi_label, global_cls in enumerate(selected_global_classes):
        train_idx = np.where(train_labels == global_cls)[0]
        test_idx  = np.where(test_labels  == global_cls)[0]

        if len(train_idx) < k_shot:
            raise ValueError(
                f'Class {global_cls} has only {len(train_idx)} train samples '
                f'but k_shot={k_shot}')
        if len(test_idx) < q_query:
            raise ValueError(
                f'Class {global_cls} has only {len(test_idx)} test samples '
                f'but q_query={q_query}')

        chosen_support = rng.sample(train_idx.tolist(), k_shot)
        chosen_query   = rng.sample(test_idx.tolist(), q_query)

        support_x.append(train_embeddings[chosen_support])
        support_y.extend([epi_label] * k_shot)

        query_x.append(test_embeddings[chosen_query])
        query_y.extend([epi_label] * q_query)

    return (
        np.vstack(support_x), np.array(support_y),
        np.vstack(query_x),   np.array(query_y),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='../../../configs/vit_baseline_b15_config.yaml',
        help='Path to config yaml',
    )
    parser.add_argument('--k-shot',  type=int, required=True)
    parser.add_argument('--distance', choices=['euclidean', 'manhattan'], default='euclidean')
    parser.add_argument('--episodes', type=int, default=600)
    parser.add_argument('--n-way',   type=int, default=5)
    parser.add_argument('--q-query', type=int, default=5)
    parser.add_argument('--debug',   action='store_true')
    args = parser.parse_args()

    if args.debug:
        args.episodes = 5

    config_path = os.path.normpath(os.path.join(SCRIPT_DIR, args.config))
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    seed         = int(cfg['train']['seed'])
    base_classes = cfg['base_classes']
    novel_classes = cfg['novel_classes']
    image_size   = int(cfg['image_size'])
    n_way        = int(args.n_way)
    k_shot       = int(args.k_shot)
    q_query      = int(args.q_query)
    episodes     = int(args.episodes)
    distance     = args.distance

    if n_way > len(novel_classes):
        raise ValueError(
            f'n_way={n_way} exceeds number of novel classes={len(novel_classes)}')

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
    print('CONFIG')
    print(f'  Experiment     : {cfg["experiment"]["name"]}')
    print(f'  Config file    : {config_path}')
    print(f'  Base classes   : {len(base_classes)}')
    print(f'  Novel classes  : {len(novel_classes)}')
    print(f'  Eval           : {n_way}-way {k_shot}-shot | q_query={q_query} | episodes={episodes}')
    print(f'  Distance       : {distance}')
    print('=' * 60)

    train_dir  = os.path.normpath(os.path.join(SCRIPT_DIR, cfg['data_train']))
    test_dir   = os.path.normpath(os.path.join(SCRIPT_DIR, cfg['data_test']))
    ckpt_dir   = os.path.normpath(os.path.join(SCRIPT_DIR, cfg['output']['ckpt_dir']))
    results_dir = os.path.normpath(
        os.path.join(SCRIPT_DIR, cfg['output']['results_dir'],
                     'novel_ncc', f'{k_shot}shot_{distance}'))
    os.makedirs(results_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, cfg['output']['save_best_name'])
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_full = datasets.ImageFolder(train_dir, transform=test_transform)
    test_full  = datasets.ImageFolder(test_dir,  transform=test_transform)

    train_idx, novel_c2i, train_tf = filter_by_classes(train_full, novel_classes)
    test_idx, _,          test_tf  = filter_by_classes(test_full,  novel_classes)

    train_full.target_transform = train_tf
    test_full.target_transform  = test_tf

    novel_train_subset = Subset(train_full, train_idx)
    novel_test_subset  = Subset(test_full,  test_idx)

    # Load fine-tuned DeiT-S classifier, then strip the head for embedding extraction
    _ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
    _state     = _ckpt['model_state_dict'] if isinstance(_ckpt, dict) and 'model_state_dict' in _ckpt else _ckpt
    num_base   = len(base_classes)
    full_model = build_deit_classifier(num_classes=num_base, pretrained=False)
    full_model.load_state_dict(_state)
    full_model.head = nn.Identity()   # strip classification head → 384-dim output
    model = full_model.to(device)
    model.eval()
    print(f'Loaded checkpoint: {ckpt_path} (epoch {_ckpt.get("epoch", "?")})')

    train_embeddings, train_labels = extract_embeddings(model, novel_train_subset, device)
    test_embeddings,  test_labels  = extract_embeddings(model, novel_test_subset,  device)

    print(f'Novel train pool : {len(train_labels)}')
    print(f'Novel test pool  : {len(test_labels)}')
    print(f'Embedding dim    : {train_embeddings.shape[1]}')

    episode_accuracies = []
    all_truths, all_preds, all_probs = [], [], []

    global_cm       = np.zeros((len(novel_classes), len(novel_classes)), dtype=float)
    episode_freq    = {cls: 0 for cls in novel_classes}
    per_class_correct = {cls: 0 for cls in novel_classes}
    per_class_total   = {cls: 0 for cls in novel_classes}

    for ep in range(episodes):
        selected_global = sorted(rng.sample(range(len(novel_classes)), n_way))
        for cls_name in [novel_classes[i] for i in selected_global]:
            episode_freq[cls_name] += 1

        support_x, support_y, query_x, query_y = sample_episode(
            train_embeddings, train_labels,
            test_embeddings,  test_labels,
            selected_global, k_shot, q_query, rng,
        )

        prototypes = np.vstack([
            support_x[support_y == c].mean(axis=0) for c in range(n_way)
        ])

        dists       = pairwise_distance(query_x, prototypes, distance=distance)
        preds_local = np.argmin(dists, axis=1)
        neg_dists   = -dists
        exp_d       = np.exp(neg_dists - neg_dists.max(axis=1, keepdims=True))
        probs_local = exp_d / exp_d.sum(axis=1, keepdims=True)

        acc = float((preds_local == query_y).mean())
        episode_accuracies.append(acc)

        all_truths.extend(query_y.tolist())
        all_preds.extend(preds_local.tolist())
        all_probs.extend(probs_local.tolist())

        for t_local, p_local in zip(query_y, preds_local):
            t_global  = selected_global[int(t_local)]
            p_global  = selected_global[int(p_local)]
            true_name = novel_classes[t_global]
            per_class_total[true_name]   += 1
            per_class_correct[true_name] += int(t_local == p_local)
            global_cm[t_global, p_global] += 1

    mean_acc = float(np.mean(episode_accuracies)) * 100.0
    std_acc  = float(np.std(episode_accuracies))  * 100.0
    ci95     = 1.96 * std_acc / math.sqrt(len(episode_accuracies))

    total_correct = sum(per_class_correct.values())
    total_queries = sum(per_class_total.values())
    ci_lo, ci_hi  = wilson_ci(total_correct, total_queries)

    macro_f1        = f1_score(all_truths, all_preds, average='macro', zero_division=0)
    macro_precision = precision_score(all_truths, all_preds, average='macro', zero_division=0)
    macro_recall    = recall_score(all_truths, all_preds, average='macro', zero_division=0)
    try:
        auc_ovr = float(roc_auc_score(all_truths, all_probs, multi_class='ovr', average='macro'))
    except Exception:
        auc_ovr = None

    per_class_acc = {
        cls: (100.0 * per_class_correct[cls] / per_class_total[cls])
             if per_class_total[cls] > 0 else 0.0
        for cls in novel_classes
    }

    summary = {
        'experiment':        cfg['experiment']['name'],
        'backbone':          'deit_small_patch16_224',
        'evaluation':        'novel_ncc',
        'split':             f'base{len(base_classes)}_novel{len(novel_classes)}',
        'n_way':             n_way,
        'k_shot':            k_shot,
        'q_query':           q_query,
        'distance':          distance,
        'num_episodes':      episodes,
        'mean_accuracy_pct': round(mean_acc, 6),
        'std_accuracy_pct':  round(std_acc, 6),
        'ci95_pct':          round(ci95, 6),
        'wilson_ci_95':      [round(ci_lo * 100, 6), round(ci_hi * 100, 6)],
        'macro_f1':          round(float(macro_f1), 6),
        'macro_precision':   round(float(macro_precision), 6),
        'macro_recall':      round(float(macro_recall), 6),
        'auc_ovr':           round(auc_ovr, 6) if auc_ovr is not None else None,
        'base_classes':      base_classes,
        'novel_classes':     novel_classes,
        'per_class_accuracy_pct': {k: round(v, 6) for k, v in per_class_acc.items()},
        'checkpoint':        ckpt_path,
    }

    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame({
        'episode':  list(range(1, episodes + 1)),
        'accuracy': episode_accuracies,
    }).to_csv(os.path.join(results_dir, 'episode_accuracies.csv'), index=False)

    pd.DataFrame(global_cm, index=novel_classes, columns=novel_classes).to_csv(
        os.path.join(results_dir, 'confusion_matrix.csv'))

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(global_cm / episodes, cmap='Blues')
    ax.set_xticks(np.arange(len(novel_classes)))
    ax.set_yticks(np.arange(len(novel_classes)))
    ax.set_xticklabels(novel_classes, rotation=90, fontsize=8)
    ax.set_yticklabels(novel_classes, fontsize=8)
    ax.set_title(f'DeiT-S NCC Confusion Matrix — {cfg["experiment"]["name"]} ({distance})')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()

    print(f'Mean accuracy : {mean_acc:.2f}% ± {ci95:.2f}%')
    print(f'Wilson 95% CI : [{ci_lo * 100:.2f}%, {ci_hi * 100:.2f}%]')
    print(f'Macro F1      : {macro_f1:.4f}')
    print(f'Macro Precision: {macro_precision:.4f}')
    print(f'Macro Recall  : {macro_recall:.4f}')
    print(f'AUC-OVR (macro): {auc_ovr:.4f}' if auc_ovr is not None else 'AUC-OVR (macro): N/A')
    print('Per-class accuracy:')
    for cls in novel_classes:
        print(f'  {cls:<30s}: {per_class_acc[cls]:6.2f}% | appeared={episode_freq[cls]:4d} | queries={per_class_total[cls]:5d}')
    print(f'Results saved : {results_dir}')


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
