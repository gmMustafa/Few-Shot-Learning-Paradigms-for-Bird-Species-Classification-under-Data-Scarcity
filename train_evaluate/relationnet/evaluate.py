"""
Relation Networks Evaluation — 5-way K-shot on novel classes only.

Same protocol as ProtoNet and MatchingNet for fair comparison:
  - Support : K images from novel TRAIN set (random per episode)
  - Query   : all 5 novel TEST images per class (fixed)
  - 600 episodes → mean ± std ± 95% CI

Usage:
    python evaluate.py                  # 1-shot (config default)
    python evaluate.py --k-shot 5       # 5-shot
    python evaluate.py --debug          # 5 episodes smoke test
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
from sklearn.metrics import f1_score
from tqdm import tqdm

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from utils.dataset      import FilteredDataset, get_eval_transforms
from models.relationnet import RelationNet


# ---------------------------------------------------------------------------
# Helpers  (identical protocol to proto and matching evaluate.py)
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


def save_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('RelationNet — Aggregated Confusion Matrix (novel classes)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_learning_curves(history, save_path):
    if not history:
        return
    epochs     = [r['epoch']     for r in history]
    train_loss = [r['train_loss'] for r in history]
    val_loss   = [r['val_loss']   for r in history]
    train_acc  = [r['train_acc']  for r in history]
    val_acc    = [r['val_acc']    for r in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, train_loss, label='train'); ax1.plot(epochs, val_loss, label='val')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Episode Loss (MSE)'); ax1.legend()

    ax2.plot(epochs, train_acc, label='train'); ax2.plot(epochs, val_acc, label='val')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.set_title('Episode Accuracy'); ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_tsne_episode(encoder, novel_train_ds, novel_test_ds, novel_classes,
                      k_shot, seed, n_way, save_path, title,
                      show_prototypes=False):
    """
    t-SNE of all novel class features (train + test).
    Background dots = all train images; ■ = support; ● = query/test.
    Single fixed episode for reproducibility.
    """
    from sklearn.manifold import TSNE
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    train_by_class = novel_train_ds.get_class_to_indices()
    test_by_class  = novel_test_ds.get_class_to_indices()

    all_imgs, all_labels, all_split = [], [], []
    train_ds_idx_to_pos: dict = {}
    pos = 0
    for ci in range(n_way):
        for ds_idx in train_by_class[ci]:
            img, _ = novel_train_ds[ds_idx]
            all_imgs.append(img)
            all_labels.append(ci)
            all_split.append('train')
            train_ds_idx_to_pos[ds_idx] = pos
            pos += 1
    for ci in range(n_way):
        for ds_idx in test_by_class[ci]:
            img, _ = novel_test_ds[ds_idx]
            all_imgs.append(img)
            all_labels.append(ci)
            all_split.append('test')

    encoder.eval()
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(all_imgs), 32):
            batch = torch.stack(all_imgs[i:i + 32])
            all_feats.append(encoder(batch).cpu().numpy())
    X      = np.vstack(all_feats)
    y      = np.array(all_labels)
    splits = np.array(all_split)

    rng = random.Random(seed)
    support_positions: set = set()
    for ci in range(n_way):
        for ds_idx in rng.sample(train_by_class[ci], k_shot):
            support_positions.add(train_ds_idx_to_pos[ds_idx])
    support_mask = np.array([i in support_positions for i in range(len(X))])

    perplexity = min(30, max(5, len(X) // 5))
    X_2d = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                max_iter=1000, init='pca').fit_transform(X)

    palette     = ['#E53935', '#8E24AA', '#1E88E5', '#43A047', '#FB8C00']
    short_names = [c.replace('AMERICAN ', 'AM. ') for c in novel_classes]

    fig, ax = plt.subplots(figsize=(11, 8))
    for ci in range(n_way):
        col   = palette[ci]
        m_bg  = (y == ci) & (splits == 'train') & ~support_mask
        m_sup = (y == ci) & support_mask
        m_qry = (y == ci) & (splits == 'test')
        ax.scatter(X_2d[m_bg,  0], X_2d[m_bg,  1], color=col, alpha=0.18, s=22, edgecolors='none')
        ax.scatter(X_2d[m_sup, 0], X_2d[m_sup, 1], color=col, alpha=1.0,  s=140,
                   marker='s', edgecolors='black', linewidths=0.8)
        ax.scatter(X_2d[m_qry, 0], X_2d[m_qry, 1], color=col, alpha=0.9,  s=130,
                   marker='o', edgecolors='black', linewidths=1.2)

    legend_els = [
        Line2D([0], [0], marker='o', color='gray', alpha=0.3, markersize=7,
               label='Train (all)', linestyle='None'),
        Line2D([0], [0], marker='s', color='gray', markersize=8,
               markeredgecolor='black', label='Support', linestyle='None'),
        Line2D([0], [0], marker='o', color='gray', markersize=8,
               markeredgecolor='black', label='Query (test)', linestyle='None'),
    ]
    for ci in range(n_way):
        legend_els.append(Patch(color=palette[ci], label=short_names[ci]))

    ax.legend(handles=legend_els, fontsize=9, loc='best', framealpha=0.9)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=10)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=10)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"t-SNE saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../../configs/relationnet_configs.yaml')
    parser.add_argument('--k-shot', type=int, default=None)
    parser.add_argument('--debug', action='store_true', help='5 episodes only')
    args = parser.parse_args()

    config_path = os.path.normpath(os.path.join(SCRIPT_DIR, args.config))
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    k_shot        = args.k_shot if args.k_shot is not None else cfg['eval']['k_shot']
    num_episodes  = 5 if args.debug else cfg['eval']['episodes']
    n_way         = cfg['eval']['n_way']
    seed          = cfg['seed']
    image_size    = cfg['image_size']
    novel_classes = cfg['novel_classes']

    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("DEVICE INFO")
    print(f"  CUDA available : {torch.cuda.is_available()}")
    print(f"  Device         : {device}")
    if torch.cuda.is_available():
        print(f"  GPU name       : {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  WARNING: Running on CPU")
    print(f"  Eval config    : {n_way}-way {k_shot}-shot | {num_episodes} episodes")
    print("=" * 60)

    data_train   = os.path.normpath(os.path.join(SCRIPT_DIR, cfg['data']['train']))
    data_test    = os.path.normpath(os.path.join(SCRIPT_DIR, cfg['data']['test']))
    ckpt_dir     = os.path.normpath(os.path.join(SCRIPT_DIR, cfg['output']['ckpt_dir']))
    results_base = os.path.normpath(os.path.join(SCRIPT_DIR, cfg['output']['results_dir']))
    results_dir  = os.path.join(results_base, f'{k_shot}shot')
    os.makedirs(results_dir, exist_ok=True)

    transform = get_eval_transforms(image_size)

    novel_train_ds = FilteredDataset(data_train, novel_classes, transform=transform)
    novel_test_ds  = FilteredDataset(data_test,  novel_classes, transform=transform)

    print(f"Novel train samples : {len(novel_train_ds)}  (support pool)")
    print(f"Novel test samples  : {len(novel_test_ds)}   (fixed query set)")
    print(f"Evaluation          : {n_way}-way {k_shot}-shot, {num_episodes} episodes")

    # Load model
    ckpt_path = os.path.join(ckpt_dir, cfg['output']['ckpt_name'])
    backbone  = resnet50(weights=ResNet50_Weights.DEFAULT)
    backbone.fc = nn.Identity()
    model = RelationNet(encoder=backbone, embedding_dim=2048).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint   : {ckpt_path}  (trained to epoch {ckpt['epoch']})")

    # Fixed query tensors
    query_images, query_labels = build_query_tensors(novel_test_ds, novel_classes, device)

    rng = random.Random(seed)
    episode_accuracies = []
    class_correct = np.zeros(n_way, dtype=float)
    class_total   = np.zeros(n_way, dtype=float)
    cm = np.zeros((n_way, n_way), dtype=float)
    # accumulate all predictions for macro-F1
    all_truths: list = []
    all_preds:  list = []

    with torch.no_grad():
        for ep in tqdm(range(num_episodes), desc=f"{n_way}-way {k_shot}-shot eval"):
            support_images, support_labels = build_support_tensors(
                novel_train_ds, novel_classes, k_shot, rng, device)

            # RelationNet returns scores [N*Q, n_way] → argmax for prediction
            scores = model(support_images, support_labels, query_images, n_way)
            preds  = scores.argmax(dim=1).cpu().numpy()
            truths = query_labels.cpu().numpy()

            acc = (preds == truths).mean()
            episode_accuracies.append(float(acc))

            all_truths.extend(truths.tolist())
            all_preds.extend(preds.tolist())
            for t, p in zip(truths, preds):
                class_correct[t] += int(t == p)
                class_total[t]   += 1
                cm[t, p]         += 1

    # Metrics
    macro_f1 = f1_score(all_truths, all_preds, average='macro', zero_division=0)
    n_ep     = len(episode_accuracies)
    mean_acc = float(np.mean(episode_accuracies)) * 100
    std_acc  = float(np.std(episode_accuracies))  * 100
    ci95     = 1.96 * std_acc / math.sqrt(n_ep)

    per_class_acc = (class_correct / class_total * 100).tolist()
    cm_avg        = (cm / n_ep).tolist()

    total_correct = int(class_correct.sum())
    total_queries = int(class_total.sum())
    ci_low, ci_high = wilson_ci(total_correct, total_queries)

    print(f"\n{n_way}-way {k_shot}-shot | {n_ep} episodes")
    print(f"Mean accuracy : {mean_acc:.2f}% ± {ci95:.2f}% (95% CI)")
    print(f"Std           : {std_acc:.2f}%")
    print(f"Wilson CI     : [{ci_low*100:.2f}%, {ci_high*100:.2f}%]")
    print(f"Macro F1      : {macro_f1:.4f}")
    print("\nPer-class accuracy:")
    for cls, acc in zip(novel_classes, per_class_acc):
        print(f"  {cls:<30s}: {acc:.2f}%")

    # Save summary.json
    summary = {
        'experiment':     cfg['experiment']['name'],
        'model':          'relationnet_resnet50',
        'seed':           seed,
        'n_way':          n_way,
        'k_shot':         k_shot,
        'num_episodes':   n_ep,
        'novel_classes':  novel_classes,
        'checkpoint':     ckpt_path,
        'epochs_trained': ckpt['epoch'],
        'mean_accuracy_pct':   round(mean_acc, 4),
        'std_accuracy_pct':    round(std_acc, 4),
        'ci95_pct':            round(ci95, 4),
        'wilson_ci_95':        [round(ci_low*100, 4), round(ci_high*100, 4)],
        'macro_f1':            round(float(macro_f1), 4),
        'per_class_accuracy_pct': {
            cls: round(acc, 4) for cls, acc in zip(novel_classes, per_class_acc)
        },
    }
    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Save summary.txt
    lines = [
        f"Relation Networks — {n_way}-way {k_shot}-shot Evaluation",
        "=" * 60,
        f"Episodes       : {n_ep}",
        f"Novel classes  : {', '.join(novel_classes)}",
        f"Checkpoint     : {ckpt_path}",
        "",
        f"Mean accuracy  : {mean_acc:.2f}% ± {ci95:.2f}% (95% CI)",
        f"Std            : {std_acc:.2f}%",
        f"Wilson CI 95%  : [{ci_low*100:.2f}%, {ci_high*100:.2f}%]",
        f"Macro F1       : {macro_f1:.4f}",
        "",
        "Per-class accuracy:",
    ] + [f"  {cls:<30s}: {acc:.2f}%" for cls, acc in zip(novel_classes, per_class_acc)]
    with open(os.path.join(results_dir, 'summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    # Save episode accuracies CSV
    pd.DataFrame({
        'episode':  list(range(1, n_ep + 1)),
        'accuracy': episode_accuracies,
    }).to_csv(os.path.join(results_dir, 'episode_accuracies.csv'), index=False)

    # Confusion matrix
    save_confusion_matrix(
        np.array(cm_avg),
        [c.replace('AMERICAN ', 'AM.') for c in novel_classes],
        os.path.join(results_dir, 'confusion_matrix.png'),
    )

    # Learning curves from training history
    history_path = os.path.join(ckpt_dir, 'history.json')
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        save_learning_curves(history, os.path.join(results_dir, 'learning_curves.png'))

    # t-SNE feature space visualization (single fixed episode)
    save_tsne_episode(
        model.encoder, novel_train_ds, novel_test_ds, novel_classes,
        k_shot, seed, n_way,
        os.path.join(results_dir, 'tsne_features.png'),
        f'RelationNet Feature Space — Novel Classes\n({n_way}-Way {k_shot}-Shot, Fixed Episode)',
    )

    print(f"\nResults saved to: {results_dir}")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
