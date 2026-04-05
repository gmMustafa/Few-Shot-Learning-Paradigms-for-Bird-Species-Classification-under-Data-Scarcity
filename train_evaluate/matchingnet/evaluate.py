"""
Matching Networks Evaluation — 5-way K-shot on novel classes only.

Same protocol as ProtoNet evaluate.py for fair comparison:
  - Support : K images from novel TRAIN set per class (random per episode)
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['figure.facecolor']  = 'white'
plt.rcParams['axes.facecolor']    = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
import seaborn as sns
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from utils.dataset      import FilteredDataset, get_eval_transforms
from models.matchingnet import MatchingNet


# ---------------------------------------------------------------------------
# Helpers  (identical protocol to protonet/evaluate.py)
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


def sample_episode_class_indices(rng, total_novel_classes, n_way):
    if n_way > total_novel_classes:
        raise ValueError(f"n_way={n_way} > total novel classes={total_novel_classes}")
    return sorted(rng.sample(range(total_novel_classes), n_way))


def build_episode_support(novel_train_ds, selected_class_indices, k_shot, rng, device):
    """Sample k_shot support images for each selected class (episode-local labels 0..n_way-1)."""
    by_class = novel_train_ds.get_class_to_indices()
    images, labels = [], []
    for epi_label, global_cls in enumerate(selected_class_indices):
        for i in rng.sample(by_class[global_cls], k_shot):
            img, _ = novel_train_ds[i]
            images.append(img)
            labels.append(epi_label)
    return torch.stack(images).to(device), torch.tensor(labels, dtype=torch.long).to(device)


def build_episode_query(novel_test_ds, selected_class_indices, q_query, rng, device):
    """Sample q_query query images for each selected class (episode-local labels 0..n_way-1)."""
    by_class = novel_test_ds.get_class_to_indices()
    images, labels = [], []
    for epi_label, global_cls in enumerate(selected_class_indices):
        pool = by_class[global_cls]
        chosen = rng.sample(pool, min(q_query, len(pool)))
        for i in chosen:
            img, _ = novel_test_ds[i]
            images.append(img)
            labels.append(epi_label)
    return torch.stack(images).to(device), torch.tensor(labels, dtype=torch.long).to(device)


def save_confusion_matrix(cm, class_names, save_path):
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
    ax.set_title('MatchingNet — Aggregated Confusion Matrix (novel classes)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
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
    ax1.set_title('Episode Loss'); ax1.legend()

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
    enc_device = next(encoder.parameters()).device
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(all_imgs), 32):
            batch = torch.stack(all_imgs[i:i + 32]).to(enc_device)
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

    #palette     = ['#E53935', '#8E24AA', '#1E88E5', '#43A047', '#FB8C00']
    palette = plt.cm.get_cmap("gist_ncar", len(novel_classes))
    short_names = [c.replace('AMERICAN ', 'AM. ') for c in novel_classes]

    fig, ax = plt.subplots(figsize=(11, 8))
    for ci in range(n_way):
        col   = palette(ci)
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
        legend_els.append(Patch(color=palette(ci), label=short_names[ci]))

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
    parser.add_argument('--config', default='../../configs/birds20/matchingnet_configs.yaml')
    parser.add_argument('--k-shot', type=int, default=None)
    parser.add_argument('--debug', action='store_true', help='5 episodes only')
    args = parser.parse_args()

    config_path = os.path.normpath(os.path.join(SCRIPT_DIR, args.config))
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    k_shot        = args.k_shot if args.k_shot is not None else cfg['eval']['k_shot']
    num_episodes  = 5 if args.debug else cfg['eval']['episodes']
    n_way         = cfg['eval']['n_way']
    q_query       = cfg['eval'].get('q_query', 5)
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
    print(f"Novel test samples  : {len(novel_test_ds)}   (query pool)")
    print(f"Novel classes       : {len(novel_classes)} total, sample {n_way} per episode")
    print(f"Evaluation          : {n_way}-way {k_shot}-shot | q_query={q_query} | {num_episodes} episodes")

    # Load model — FCE flag is read from the config
    ckpt_path = os.path.join(ckpt_dir, cfg['output']['ckpt_name'])
    backbone  = resnet50(weights=ResNet50_Weights.DEFAULT)
    backbone.fc = nn.Identity()
    use_fce   = bool(cfg.get('fce', False))
    fce_steps = int(cfg.get('fce_steps', 10))
    model = MatchingNet(encoder=backbone, embedding_dim=2048,
                        use_fce=use_fce, fce_steps=fce_steps).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint   : {ckpt_path}  (trained to epoch {ckpt['epoch']})")
    if use_fce:
        print(f"  FCE enabled    : BiLSTM support + AttLSTM query (steps={fce_steps})")

    rng = random.Random(seed)
    episode_accuracies = []
    # Track per-class stats in global novel-class space
    per_class_correct = {cls: 0 for cls in novel_classes}
    per_class_total   = {cls: 0 for cls in novel_classes}
    global_cm = np.zeros((len(novel_classes), len(novel_classes)), dtype=float)
    all_truths: list = []
    all_preds:  list = []
    all_probs:  list = []
    first_episode_class_indices: list = []

    with torch.no_grad():
        for ep in tqdm(range(num_episodes), desc=f"{n_way}-way {k_shot}-shot eval"):
            selected_class_indices = sample_episode_class_indices(rng, len(novel_classes), n_way)
            if ep == 0:
                first_episode_class_indices = selected_class_indices.copy()

            support_images, support_labels = build_episode_support(
                novel_train_ds, selected_class_indices, k_shot, rng, device)
            query_images, query_labels = build_episode_query(
                novel_test_ds, selected_class_indices, q_query, rng, device)

            # MatchingNet returns log-probs → argmax for prediction; exp for probabilities
            log_probs = model(support_images, support_labels, query_images, n_way)
            probs  = torch.exp(log_probs).cpu().numpy()
            preds  = log_probs.argmax(dim=1).cpu().numpy()
            truths = query_labels.cpu().numpy()

            acc = (preds == truths).mean()
            episode_accuracies.append(float(acc))

            all_truths.extend(truths.tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

            # Map episode-local labels back to global novel-class indices
            for t_local, p_local in zip(truths, preds):
                t_global = selected_class_indices[int(t_local)]
                p_global = selected_class_indices[int(p_local)]
                per_class_correct[novel_classes[t_global]] += int(t_local == p_local)
                per_class_total[novel_classes[t_global]]   += 1
                global_cm[t_global, p_global] += 1

    # Metrics
    macro_f1        = f1_score(all_truths, all_preds, average='macro', zero_division=0)
    macro_precision = precision_score(all_truths, all_preds, average='macro', zero_division=0)
    macro_recall    = recall_score(all_truths, all_preds, average='macro', zero_division=0)
    try:
        auc_ovr = float(roc_auc_score(all_truths, all_probs, multi_class='ovr', average='macro'))
    except Exception:
        auc_ovr = None
    n_ep     = len(episode_accuracies)
    mean_acc = float(np.mean(episode_accuracies)) * 100
    std_acc  = float(np.std(episode_accuracies))  * 100
    ci95     = 1.96 * std_acc / math.sqrt(n_ep)

    per_class_acc = {
        cls: (100.0 * per_class_correct[cls] / per_class_total[cls]) if per_class_total[cls] > 0 else 0.0
        for cls in novel_classes
    }
    cm_avg = (global_cm / n_ep).tolist()

    total_correct = sum(per_class_correct.values())
    total_queries = sum(per_class_total.values())
    ci_low, ci_high = wilson_ci(total_correct, total_queries)

    print(f"\n{n_way}-way {k_shot}-shot | {n_ep} episodes")
    print(f"Mean accuracy : {mean_acc:.2f}% ± {ci95:.2f}% (95% CI)")
    print(f"Std           : {std_acc:.2f}%")
    print(f"Wilson CI     : [{ci_low*100:.2f}%, {ci_high*100:.2f}%]")
    print(f"Macro F1       : {macro_f1:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall   : {macro_recall:.4f}")
    print(f"AUC-OVR (macro): {auc_ovr:.4f}" if auc_ovr is not None else "AUC-OVR (macro): N/A")
    print("\nPer-class accuracy (global novel-class space):")
    for cls in novel_classes:
        print(f"  {cls:<30s}: {per_class_acc[cls]:.2f}%  | queries={per_class_total[cls]}")

    # Save summary.json
    summary = {
        'experiment':       cfg['experiment']['name'],
        'model':            'matchingnet_resnet50',
        'seed':             seed,
        'n_way':            n_way,
        'k_shot':           k_shot,
        'q_query':          q_query,
        'num_episodes':     n_ep,
        'novel_classes':    novel_classes,
        'checkpoint':       ckpt_path,
        'epochs_trained':   ckpt['epoch'],
        'mean_accuracy_pct':   round(mean_acc, 4),
        'std_accuracy_pct':    round(std_acc, 4),
        'ci95_pct':            round(ci95, 4),
        'wilson_ci_95':        [round(ci_low*100, 4), round(ci_high*100, 4)],
        'macro_f1':            round(float(macro_f1), 4),
        'macro_precision':     round(float(macro_precision), 4),
        'macro_recall':        round(float(macro_recall), 4),
        'auc_ovr':             round(auc_ovr, 4) if auc_ovr is not None else None,
        'per_class_accuracy_pct': {cls: round(acc, 4) for cls, acc in per_class_acc.items()},
        'per_class_query_total': per_class_total,
    }
    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Save summary.txt
    lines = [
        f"Matching Networks — {n_way}-way {k_shot}-shot Evaluation",
        "=" * 60,
        f"Episodes       : {n_ep}",
        f"n_way          : {n_way} (sampled from {len(novel_classes)} novel classes)",
        f"q_query        : {q_query}",
        f"Novel classes  : {', '.join(novel_classes)}",
        f"Checkpoint     : {ckpt_path}",
        "",
        f"Mean accuracy  : {mean_acc:.2f}% ± {ci95:.2f}% (95% CI)",
        f"Std            : {std_acc:.2f}%",
        f"Wilson CI 95%  : [{ci_low*100:.2f}%, {ci_high*100:.2f}%]",
        f"Macro F1       : {macro_f1:.4f}",
        f"Macro Precision: {macro_precision:.4f}",
        f"Macro Recall   : {macro_recall:.4f}",
        f"AUC-OVR (macro): {auc_ovr:.4f}" if auc_ovr is not None else "AUC-OVR (macro): N/A",
        "",
        "Per-class accuracy:",
    ] + [f"  {cls:<30s}: {per_class_acc[cls]:.2f}%  | queries={per_class_total[cls]}" for cls in novel_classes]
    with open(os.path.join(results_dir, 'summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    # Save episode accuracies CSV
    pd.DataFrame({
        'episode':  list(range(1, n_ep + 1)),
        'accuracy': episode_accuracies,
    }).to_csv(os.path.join(results_dir, 'episode_accuracies.csv'), index=False)

    # Confusion matrix (global novel-class space)
    save_confusion_matrix(
        global_cm / n_ep,
        [c.replace('AMERICAN ', 'AM.') for c in novel_classes],
        os.path.join(results_dir, 'confusion_matrix.png'),
    )

    # Learning curves from training history
    history_path = os.path.join(ckpt_dir, 'history.json')
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        save_learning_curves(history, os.path.join(results_dir, 'learning_curves.png'))

    # t-SNE for the first episode's sampled classes
    if first_episode_class_indices:
        ep1_classes = [novel_classes[i] for i in first_episode_class_indices]
        save_tsne_episode(
            model.encoder, novel_train_ds, novel_test_ds, ep1_classes,
            k_shot, seed, n_way,
            os.path.join(results_dir, 'tsne_features.png'),
            f'MatchingNet Feature Space — Episode 1\n({n_way}-Way {k_shot}-Shot, classes: {", ".join([c.replace("AMERICAN ", "AM. ") for c in ep1_classes])})',
        )

    print(f"\nResults saved to: {results_dir}")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
