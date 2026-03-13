"""
evaluate_novel_ncc.py — Nearest Centroid Classifier (NCC) on novel classes.

Uses the base-15 trained ResNet-50 backbone as a FROZEN feature extractor.
No additional training — only the existing base-15 checkpoint is needed.

Classification protocol (identical to FSL evaluate.py for fair comparison):
  - Support : K images randomly sampled from novel TRAIN set per class
  - Query   : all 5 novel TEST images per class (fixed — same as FSL evaluation)
  - Episodes: 600 random episodes (different support selections)
  - Distance: euclidean (L2) or manhattan (L1) — same ablation as ProtoNet

Why NCC matters for the thesis:
  Comparing NCC vs ProtoNet (same distance, same K, same backbone family) directly
  isolates the effect of episodic training. If ProtoNet beats NCC, it is because
  episodic training shaped a better embedding space — not because of the metric.
  This is the key insight from Chen et al. (2019), "A Closer Look at Few-Shot
  Classification."

Usage:
    python evaluate_novel_ncc.py --k-shot 1 --distance euclidean
    python evaluate_novel_ncc.py --k-shot 3 --distance euclidean
    python evaluate_novel_ncc.py --k-shot 5 --distance manhattan
    python evaluate_novel_ncc.py --debug           # 5 episodes smoke test
"""
import os
import sys
import json
import random
import argparse
import math

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from utils.dataset import FilteredDataset, get_eval_transforms

# Reuse the same config as the base-15 baseline for class lists and paths
CONFIG_PATH = os.path.normpath(
    os.path.join(SCRIPT_DIR, '..', '..', '..', 'configs', 'baseline_resnet_50_b15_config.yaml'))
CKPT_PATH = os.path.join(SCRIPT_DIR, 'checkpoints', 'baseline', 'base15', 'resnet50_base15_best.pth')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def wilson_ci(correct: int, total: int):
    if total == 0:
        return 0.0, 0.0
    z   = 1.96
    p   = correct / total
    den = 1 + z**2 / total
    ctr = (p + z**2 / (2 * total)) / den
    mar = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / den
    return max(0.0, ctr - mar), min(1.0, ctr + mar)


def compute_centroid(embeddings):
    """Mean of a set of embeddings. embeddings: [K, D] → [D]"""
    return embeddings.mean(dim=0)


def nearest_centroid_predict(query_emb, centroids, distance: str):
    """
    query_emb:  [N*Q, D]
    centroids:  [n_way, D]
    Returns:    predicted class indices [N*Q]
    """
    nq   = query_emb.size(0)
    n    = centroids.size(0)
    q_exp = query_emb.unsqueeze(1).expand(nq, n, -1)
    c_exp = centroids.unsqueeze(0).expand(nq, n, -1)

    if distance == 'euclidean':
        dists = torch.norm(q_exp - c_exp, dim=2)       # [nq, n]
    elif distance == 'manhattan':
        dists = torch.abs(q_exp - c_exp).sum(dim=2)    # [nq, n]
    else:
        raise ValueError(f"Unknown distance: {distance}")

    return dists.argmin(dim=1)   # [nq]


def build_query_tensors(novel_test_ds, novel_classes, device):
    by_class = novel_test_ds.get_class_to_indices()
    images, labels = [], []
    for cls_idx in range(len(novel_classes)):
        for i in by_class[cls_idx]:
            img, _ = novel_test_ds[i]
            images.append(img)
            labels.append(cls_idx)
    return torch.stack(images).to(device), torch.tensor(labels, dtype=torch.long).to(device)


def build_support_tensors(novel_train_ds, novel_classes, k_shot, rng, device):
    by_class = novel_train_ds.get_class_to_indices()
    images, labels = [], []
    for cls_idx in range(len(novel_classes)):
        for i in rng.sample(by_class[cls_idx], k_shot):
            img, _ = novel_train_ds[i]
            images.append(img)
            labels.append(cls_idx)
    return torch.stack(images).to(device), torch.tensor(labels, dtype=torch.long).to(device)


def save_confusion_matrix(cm, class_names, save_path, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k-shot', type=int, default=1,
                        help='Number of support images per novel class (1, 3, or 5)')
    parser.add_argument('--distance', default='euclidean',
                        choices=['euclidean', 'manhattan'],
                        help='Distance metric for centroid classification')
    parser.add_argument('--episodes', type=int, default=600,
                        help='Number of evaluation episodes')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true',
                        help='Smoke test: 5 episodes only')
    args = parser.parse_args()

    k_shot       = args.k_shot
    distance     = args.distance
    num_episodes = 5 if args.debug else args.episodes
    seed         = args.seed

    set_seed(seed)
    device = torch.device('cpu')

    # Load config for class lists and paths
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    novel_classes = cfg['novel_classes']
    base_classes  = cfg['base_classes']
    image_size    = cfg.get('image_size', 224)
    n_way         = len(novel_classes)   # always 5

    # Resolve data paths relative to script dir
    data_train = os.path.normpath(os.path.join(SCRIPT_DIR, cfg['data_train']))
    data_test  = os.path.normpath(os.path.join(SCRIPT_DIR, cfg['data_test']))

    results_base = os.path.normpath(
        os.path.join(PROJECT_ROOT, 'results', 'baseline', 'resnet50', 'base15', 'novel_ncc'))
    results_dir  = os.path.join(results_base, f'{k_shot}shot_{distance}')
    os.makedirs(results_dir, exist_ok=True)

    transform = get_eval_transforms(image_size)

    # Novel classes ONLY
    novel_train_ds = FilteredDataset(data_train, novel_classes, transform=transform)
    novel_test_ds  = FilteredDataset(data_test,  novel_classes, transform=transform)

    print(f"NCC Evaluation — base-15 backbone, {n_way}-way {k_shot}-shot, {distance}")
    print(f"Novel train samples : {len(novel_train_ds)}  (support pool)")
    print(f"Novel test samples  : {len(novel_test_ds)}   (fixed query set)")
    print(f"Episodes            : {num_episodes}")

    # --- Load backbone (discard the 15-class head) ---
    backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
    backbone.fc = nn.Identity()   # 2048-dim embedding output
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    # strict=False: fc.weight [15,2048] and fc.bias [15] from checkpoint are
    # intentionally ignored — we only want the convolutional backbone layers
    missing, unexpected = backbone.load_state_dict(ckpt['model_state_dict'], strict=False)
    if missing:
        print(f"WARNING: Missing keys: {missing}")
    # 'unexpected' will be ['fc.weight', 'fc.bias'] — this is expected and correct
    backbone = backbone.to(device)
    backbone.eval()
    print(f"Loaded backbone from: {CKPT_PATH}  (epoch {ckpt['epoch']})")
    print(f"  Ignored checkpoint keys (expected): {unexpected}")

    # --- Fixed query tensors ---
    query_images, query_labels = build_query_tensors(novel_test_ds, novel_classes, device)

    rng = random.Random(seed)
    episode_accuracies = []
    class_correct = np.zeros(n_way, dtype=float)
    class_total   = np.zeros(n_way, dtype=float)
    cm = np.zeros((n_way, n_way), dtype=float)

    with torch.no_grad():
        # Pre-extract query embeddings (fixed across episodes)
        query_emb = backbone(query_images)    # [n_way*5, 2048]

        for ep in tqdm(range(num_episodes), desc=f"NCC {n_way}-way {k_shot}-shot ({distance})"):
            support_images, support_labels = build_support_tensors(
                novel_train_ds, novel_classes, k_shot, rng, device)

            # Extract support embeddings and compute per-class centroids
            support_emb = backbone(support_images)     # [n_way*k_shot, 2048]
            centroids = torch.zeros(n_way, support_emb.size(1), device=device)
            for c in range(n_way):
                mask = (support_labels == c)
                centroids[c] = support_emb[mask].mean(dim=0)

            # Classify queries by nearest centroid
            preds  = nearest_centroid_predict(query_emb, centroids, distance)
            preds  = preds.cpu().numpy()
            truths = query_labels.cpu().numpy()

            acc = (preds == truths).mean()
            episode_accuracies.append(float(acc))

            for t, p in zip(truths, preds):
                class_correct[t] += int(t == p)
                class_total[t]   += 1
                cm[t, p]         += 1

    # --- Metrics ---
    n_ep     = len(episode_accuracies)
    mean_acc = float(np.mean(episode_accuracies)) * 100
    std_acc  = float(np.std(episode_accuracies))  * 100
    ci95     = 1.96 * std_acc / math.sqrt(n_ep)

    per_class_acc = (class_correct / class_total * 100).tolist()
    cm_avg        = (cm / n_ep).tolist()

    total_correct = int(class_correct.sum())
    total_queries = int(class_total.sum())
    ci_low, ci_high = wilson_ci(total_correct, total_queries)

    print(f"\n{n_way}-way {k_shot}-shot NCC ({distance}) | {n_ep} episodes")
    print(f"Mean accuracy : {mean_acc:.2f}% ± {ci95:.2f}% (95% CI)")
    print(f"Std           : {std_acc:.2f}%")
    print(f"Wilson CI     : [{ci_low*100:.2f}%, {ci_high*100:.2f}%]")
    print("\nPer-class accuracy:")
    for cls, acc in zip(novel_classes, per_class_acc):
        print(f"  {cls:<30s}: {acc:.2f}%")

    # --- Save summary.json ---
    summary = {
        'experiment':       'ncc_base15_novel5',
        'model':            'resnet50_base15_backbone',
        'method':           'nearest_centroid_classifier',
        'seed':             seed,
        'n_way':            n_way,
        'k_shot':           k_shot,
        'distance':         distance,
        'num_episodes':     n_ep,
        'novel_classes':    novel_classes,
        'backbone_ckpt':    CKPT_PATH,
        'backbone_epoch':   ckpt['epoch'],
        'note':             ('Frozen base-15 backbone. No episodic training. '
                             'Compares directly with ProtoNet (same distance, same K) '
                             'to isolate the effect of episodic training.'),
        'mean_accuracy_pct':   round(mean_acc, 4),
        'std_accuracy_pct':    round(std_acc, 4),
        'ci95_pct':            round(ci95, 4),
        'wilson_ci_95':        [round(ci_low*100, 4), round(ci_high*100, 4)],
        'per_class_accuracy_pct': {
            cls: round(acc, 4) for cls, acc in zip(novel_classes, per_class_acc)
        },
    }
    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # --- Save summary.txt ---
    lines = [
        f"NCC (Nearest Centroid) — {n_way}-way {k_shot}-shot ({distance})",
        "=" * 60,
        f"Backbone       : ResNet-50 trained on 15 base classes (frozen)",
        f"Distance       : {distance}",
        f"Episodes       : {n_ep}",
        f"Novel classes  : {', '.join(novel_classes)}",
        "",
        f"Mean accuracy  : {mean_acc:.2f}% ± {ci95:.2f}% (95% CI)",
        f"Std            : {std_acc:.2f}%",
        f"Wilson CI 95%  : [{ci_low*100:.2f}%, {ci_high*100:.2f}%]",
        "",
        "Per-class accuracy:",
    ] + [f"  {cls:<30s}: {acc:.2f}%" for cls, acc in zip(novel_classes, per_class_acc)]
    with open(os.path.join(results_dir, 'summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    # --- Save CSV ---
    pd.DataFrame({'episode': range(1, n_ep+1), 'accuracy': episode_accuracies}
                 ).to_csv(os.path.join(results_dir, 'episode_accuracies.csv'), index=False)

    # --- Confusion matrix ---
    save_confusion_matrix(
        np.array(cm_avg),
        [c.replace('AMERICAN ', 'AM.') for c in novel_classes],
        os.path.join(results_dir, 'confusion_matrix.png'),
        title=f'NCC ({distance}) — {k_shot}-shot Confusion Matrix')

    print(f"\nResults saved to: {results_dir}")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
