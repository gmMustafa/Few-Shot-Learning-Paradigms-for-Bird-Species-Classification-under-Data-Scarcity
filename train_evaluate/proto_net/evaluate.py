"""
ProtoNet Evaluation — episodic evaluation on novel classes.

Supports:
- old 5-novel-class setup
- new 10-novel-class setup
- true episodic evaluation:
    * sample n_way classes per episode from novel_classes
    * sample support from novel TRAIN split
    * sample query from novel TEST split
    * remap labels to episode-local 0..n_way-1

Notes:
- For Birds-20, query sampling is done from the TEST split only.
- Query sets are random per episode, not fixed.
- Support/query are sampled without replacement within each class.
- If a class does not have enough test images for q_query, the code fails loudly.

Usage:
    python evaluate.py
    python evaluate.py --k-shot 5
    python evaluate.py --distance manhattan
    python evaluate.py --k-shot 3 --distance euclidean
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
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from PIL import Image as PILImage
from utils.dataset import FilteredDataset, get_eval_transforms, get_tta_transforms
from models.protonet import ProtoNet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def wilson_ci(correct: int, total: int) -> Tuple[float, float]:
    """95% Wilson score confidence interval for a proportion."""
    if total == 0:
        return 0.0, 0.0

    z = 1.96
    p = correct / total
    den = 1 + z**2 / total
    ctr = (p + z**2 / (2 * total)) / den
    mar = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / den
    return max(0.0, ctr - mar), min(1.0, ctr + mar)


def sanitize_name(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


def episode_class_name_list(
    all_novel_classes: Sequence[str],
    selected_indices: Sequence[int],
) -> List[str]:
    return [all_novel_classes[i] for i in selected_indices]


def sample_episode_class_indices(
    rng: random.Random,
    total_novel_classes: int,
    n_way: int,
) -> List[int]:
    if n_way > total_novel_classes:
        raise ValueError(
            f"eval.n_way={n_way} is larger than number of novel classes={total_novel_classes}"
        )
    return sorted(rng.sample(range(total_novel_classes), n_way))


def build_episode_support_query(
    novel_train_ds: FilteredDataset,
    novel_test_ds: FilteredDataset,
    selected_class_indices: Sequence[int],
    k_shot: int,
    q_query: int,
    rng: random.Random,
    device: torch.device,
    return_query_paths: bool = False,
):
    """
    Build one episodic task.

    Support:
        sampled from novel TRAIN split
    Query:
        sampled from novel TEST split

    Labels are remapped to episode-local 0..n_way-1.

    Args:
        return_query_paths: if True, also return list of query image file paths
                            (used for test-time augmentation).
    """
    train_by_class = novel_train_ds.get_class_to_indices()
    test_by_class = novel_test_ds.get_class_to_indices()

    support_images: List[torch.Tensor] = []
    support_labels: List[int] = []
    query_images: List[torch.Tensor] = []
    query_labels: List[int] = []
    query_paths: List[str] = []

    for epi_label, global_cls_idx in enumerate(selected_class_indices):
        train_candidates = train_by_class[global_cls_idx]
        test_candidates = test_by_class[global_cls_idx]

        if len(train_candidates) < k_shot:
            raise ValueError(
                f"Class index {global_cls_idx} has only {len(train_candidates)} "
                f"train images, but k_shot={k_shot} was requested."
            )
        if len(test_candidates) < q_query:
            raise ValueError(
                f"Class index {global_cls_idx} has only {len(test_candidates)} "
                f"test images, but q_query={q_query} was requested."
            )

        sampled_support = rng.sample(train_candidates, k_shot)
        sampled_query = rng.sample(test_candidates, q_query)

        for ds_idx in sampled_support:
            img, _ = novel_train_ds[ds_idx]
            support_images.append(img)
            support_labels.append(epi_label)

        for ds_idx in sampled_query:
            img, _ = novel_test_ds[ds_idx]
            query_images.append(img)
            query_labels.append(epi_label)
            if return_query_paths:
                query_paths.append(novel_test_ds.samples[ds_idx][0])

    support_tensor = torch.stack(support_images).to(device)
    query_tensor = torch.stack(query_images).to(device)
    support_label_tensor = torch.tensor(support_labels, dtype=torch.long, device=device)
    query_label_tensor = torch.tensor(query_labels, dtype=torch.long, device=device)

    if return_query_paths:
        return support_tensor, query_tensor, support_label_tensor, query_label_tensor, query_paths
    return support_tensor, query_tensor, support_label_tensor, query_label_tensor


def save_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    save_path: str,
    title: str,
) -> None:
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
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=tick_fs)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=tick_fs)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_learning_curves(history: List[dict], save_path: str) -> None:
    if not history:
        return

    epochs = [r["epoch"] for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_loss = [r["val_loss"] for r in history]
    train_acc = [r["train_acc"] for r in history]
    val_acc = [r["val_acc"] for r in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_loss, label="train")
    ax1.plot(epochs, val_loss, label="val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Episode Loss")
    ax1.legend()

    ax2.plot(epochs, train_acc, label="train")
    ax2.plot(epochs, val_acc, label="val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Episode Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_tsne_episode(
    encoder: nn.Module,
    novel_train_ds: FilteredDataset,
    novel_test_ds: FilteredDataset,
    all_novel_classes: Sequence[str],
    selected_class_indices: Sequence[int],
    k_shot: int,
    q_query: int,
    seed: int,
    save_path: str,
    title: str,
    show_prototypes: bool = True,
) -> None:
    """
    t-SNE for one fixed sampled episode.

    Background = all train images from selected episode classes
    Support    = sampled support images
    Query      = sampled query images from test split
    Prototype  = support centroid in t-SNE space (approximate visualization only)
    """
    from sklearn.manifold import TSNE
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    rng = random.Random(seed)
    train_by_class = novel_train_ds.get_class_to_indices()
    test_by_class = novel_test_ds.get_class_to_indices()

    # Recreate the exact support/query selection for the chosen episode classes.
    support_pos_keys = set()
    query_pos_keys = set()
    sampled_support_by_class: Dict[int, List[int]] = {}
    sampled_query_by_class: Dict[int, List[int]] = {}

    for epi_label, global_cls_idx in enumerate(selected_class_indices):
        sampled_support = rng.sample(train_by_class[global_cls_idx], k_shot)
        sampled_query = rng.sample(test_by_class[global_cls_idx], q_query)
        sampled_support_by_class[global_cls_idx] = sampled_support
        sampled_query_by_class[global_cls_idx] = sampled_query

        for ds_idx in sampled_support:
            support_pos_keys.add(("train", global_cls_idx, ds_idx))
        for ds_idx in sampled_query:
            query_pos_keys.add(("test", global_cls_idx, ds_idx))

    all_imgs: List[torch.Tensor] = []
    all_labels: List[int] = []
    all_split: List[str] = []
    all_keys: List[Tuple[str, int, int]] = []

    for epi_label, global_cls_idx in enumerate(selected_class_indices):
        for ds_idx in train_by_class[global_cls_idx]:
            img, _ = novel_train_ds[ds_idx]
            all_imgs.append(img)
            all_labels.append(epi_label)
            all_split.append("train")
            all_keys.append(("train", global_cls_idx, ds_idx))

        for ds_idx in test_by_class[global_cls_idx]:
            img, _ = novel_test_ds[ds_idx]
            all_imgs.append(img)
            all_labels.append(epi_label)
            all_split.append("test")
            all_keys.append(("test", global_cls_idx, ds_idx))

    encoder.eval()
    enc_device = next(encoder.parameters()).device
    all_feats = []

    with torch.no_grad():
        for i in range(0, len(all_imgs), 32):
            batch = torch.stack(all_imgs[i:i + 32]).to(enc_device)
            feats = encoder(batch).cpu().numpy()
            all_feats.append(feats)

    X = np.vstack(all_feats)
    y = np.array(all_labels)
    splits = np.array(all_split)

    support_mask = np.array([key in support_pos_keys for key in all_keys])
    query_mask = np.array([key in query_pos_keys for key in all_keys])

    perplexity = min(30, max(5, len(X) // 5))
    X_2d = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        max_iter=1000,
        init="pca",
    ).fit_transform(X)

    proto_2d = []
    for epi_label in range(len(selected_class_indices)):
        pts = X_2d[support_mask & (y == epi_label)]
        proto_2d.append(pts.mean(axis=0))
    proto_2d = np.array(proto_2d)

    palette = [
        "#E53935", "#8E24AA", "#1E88E5", "#43A047", "#FB8C00",
        "#00897B", "#6D4C41", "#3949AB", "#F4511E", "#7CB342",
    ]
    episode_class_names = episode_class_name_list(all_novel_classes, selected_class_indices)
    short_names = [c.replace("AMERICAN ", "AM. ") for c in episode_class_names]

    fig, ax = plt.subplots(figsize=(11, 8))

    for epi_label in range(len(selected_class_indices)):
        col = palette[epi_label % len(palette)]
        m_bg = (y == epi_label) & (splits == "train") & ~support_mask
        m_sup = (y == epi_label) & support_mask
        m_qry = (y == epi_label) & query_mask

        ax.scatter(X_2d[m_bg, 0], X_2d[m_bg, 1], color=col, alpha=0.18, s=22, edgecolors="none")
        ax.scatter(
            X_2d[m_sup, 0], X_2d[m_sup, 1],
            color=col, alpha=1.0, s=140, marker="s",
            edgecolors="black", linewidths=0.8
        )
        ax.scatter(
            X_2d[m_qry, 0], X_2d[m_qry, 1],
            color=col, alpha=0.9, s=130, marker="o",
            edgecolors="black", linewidths=1.2
        )

    if show_prototypes:
        for epi_label in range(len(selected_class_indices)):
            col = palette[epi_label % len(palette)]
            ax.scatter(
                proto_2d[epi_label, 0], proto_2d[epi_label, 1],
                color=col, s=350, marker="*",
                edgecolors="black", linewidths=0.8, zorder=6
            )

    legend_elements = [
        Line2D([0], [0], marker="o", color="gray", alpha=0.3, markersize=7,
               label="Train (all)", linestyle="None"),
        Line2D([0], [0], marker="s", color="gray", markersize=8,
               markeredgecolor="black", label="Support", linestyle="None"),
        Line2D([0], [0], marker="o", color="gray", markersize=8,
               markeredgecolor="black", label="Query (test)", linestyle="None"),
    ]
    if show_prototypes:
        legend_elements.append(
            Line2D([0], [0], marker="*", color="gray", markersize=11,
                   markeredgecolor="black", label="Prototype", linestyle="None")
        )
    for epi_label, cls_name in enumerate(short_names):
        legend_elements.append(
            Patch(color=palette[epi_label % len(palette)], label=cls_name)
        )

    ax.legend(handles=legend_elements, fontsize=9, loc="best", framealpha=0.9)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1", fontsize=10)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=10)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"t-SNE saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../../configs/birds20/proto_configs.yaml")
    parser.add_argument("--k-shot", type=int, default=None, help="Override eval k_shot")
    parser.add_argument(
        "--distance",
        default=None,
        choices=["euclidean", "manhattan"],
        help="Distance metric for prototype comparison",
    )
    parser.add_argument("--debug", action="store_true", help="Smoke test: 5 episodes only")
    parser.add_argument(
        "--tta", type=int, default=1, metavar="N",
        help="Test-time augmentation: average embeddings over N augmented query views "
             "(1 = no TTA, default). Recommended: 8.",
    )
    args = parser.parse_args()

    config_path = os.path.normpath(os.path.join(SCRIPT_DIR, args.config))
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["seed"]
    image_size = cfg["image_size"]
    n_way = cfg["eval"]["n_way"]
    k_shot = args.k_shot if args.k_shot is not None else cfg["eval"]["k_shot"]
    q_query = cfg["eval"].get("q_query", 5)
    distance = args.distance if args.distance is not None else cfg["eval"].get("distance", "euclidean")
    num_episodes = 5 if args.debug else cfg["eval"]["episodes"]
    novel_classes = cfg["novel_classes"]
    tta_n = max(1, args.tta)

    set_seed(seed)
    rng = random.Random(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("DEVICE INFO")
    print(f"  CUDA available : {torch.cuda.is_available()}")
    print(f"  Device         : {device}")
    if torch.cuda.is_available():
        print(f"  GPU name       : {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  WARNING: Running on CPU")
    print("EVAL CONFIG")
    print(f"  Config file    : {config_path}")
    print(f"  Novel classes  : {len(novel_classes)}")
    print(f"  n_way          : {n_way}")
    print(f"  k_shot         : {k_shot}")
    print(f"  q_query        : {q_query}")
    print(f"  Episodes       : {num_episodes}")
    print(f"  Distance       : {distance}")
    print(f"  TTA            : {tta_n}x augmentations" if tta_n > 1 else "  TTA            : disabled")
    print("=" * 60)

    data_train = os.path.normpath(os.path.join(SCRIPT_DIR, cfg["data"]["train"]))
    data_test = os.path.normpath(os.path.join(SCRIPT_DIR, cfg["data"]["test"]))
    ckpt_dir = os.path.normpath(os.path.join(SCRIPT_DIR, cfg["output"]["ckpt_dir"]))
    results_base = os.path.normpath(os.path.join(SCRIPT_DIR, cfg["output"]["results_dir"]))
    tta_suffix  = f"_tta{tta_n}" if tta_n > 1 else ""
    results_dir = os.path.join(results_base, f"{k_shot}shot_{distance}{tta_suffix}")
    os.makedirs(results_dir, exist_ok=True)

    tta_transform = get_tta_transforms(image_size) if tta_n > 1 else None

    transform = get_eval_transforms(image_size)

    novel_train_ds = FilteredDataset(data_train, novel_classes, transform=transform)
    novel_test_ds = FilteredDataset(data_test, novel_classes, transform=transform)

    print(f"Novel train samples : {len(novel_train_ds)}")
    print(f"Novel test samples  : {len(novel_test_ds)}")

    # Sanity checks for query pool
    test_by_class = novel_test_ds.get_class_to_indices()
    for idx, cls_name in enumerate(novel_classes):
        n_test = len(test_by_class[idx])
        if n_test < q_query:
            raise ValueError(
                f"Class '{cls_name}' has only {n_test} test images but eval.q_query={q_query}."
            )

    ckpt_path = os.path.join(ckpt_dir, cfg["output"]["ckpt_name"])
    backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
    backbone.fc = nn.Identity()
    model = ProtoNet(encoder=backbone).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint   : {ckpt_path} (trained to epoch {ckpt['epoch']})")

    episode_accuracies: List[float] = []
    all_truths: List[int] = []
    all_preds: List[int] = []
    all_probs: List[List[float]] = []

    episode_class_counter = {cls: 0 for cls in novel_classes}
    per_class_correct = {cls: 0 for cls in novel_classes}
    per_class_total = {cls: 0 for cls in novel_classes}
    per_class_pred_counter = {cls: 0 for cls in novel_classes}

    # Aggregate confusion matrix in global novel-class space.
    global_cm = np.zeros((len(novel_classes), len(novel_classes)), dtype=float)

    first_episode_class_indices: List[int] = []

    with torch.no_grad():
        for ep in tqdm(range(num_episodes), desc=f"{n_way}-way {k_shot}-shot eval"):
            selected_class_indices = sample_episode_class_indices(
                rng=rng,
                total_novel_classes=len(novel_classes),
                n_way=n_way,
            )
            if ep == 0:
                first_episode_class_indices = selected_class_indices.copy()

            selected_class_names = episode_class_name_list(novel_classes, selected_class_indices)
            for cls_name in selected_class_names:
                episode_class_counter[cls_name] += 1

            if tta_n > 1:
                support_images, query_images, support_labels, query_labels, query_paths = \
                    build_episode_support_query(
                        novel_train_ds=novel_train_ds,
                        novel_test_ds=novel_test_ds,
                        selected_class_indices=selected_class_indices,
                        k_shot=k_shot, q_query=q_query, rng=rng, device=device,
                        return_query_paths=True,
                    )
                # TTA: compute support prototypes then average query embeddings
                support_emb = model.encoder(support_images)          # [N*K, D]
                prototypes  = torch.stack([
                    support_emb[support_labels == c].mean(dim=0)
                    for c in range(n_way)
                ])                                                    # [n_way, D]
                aug_embs = [model.encoder(query_images)]              # base pass
                for _ in range(tta_n - 1):
                    aug_imgs = [tta_transform(PILImage.open(p).convert("RGB"))
                                for p in query_paths]
                    aug_batch = torch.stack(aug_imgs).to(device)
                    aug_embs.append(model.encoder(aug_batch))
                query_emb = torch.stack(aug_embs).mean(dim=0)        # [N*Q, D]
                if distance == "euclidean":
                    dists = torch.cdist(query_emb, prototypes)
                else:  # manhattan
                    dists = (query_emb.unsqueeze(1) - prototypes.unsqueeze(0)).abs().sum(dim=2)
                logits = -dists
            else:
                support_images, query_images, support_labels, query_labels = \
                    build_episode_support_query(
                        novel_train_ds=novel_train_ds,
                        novel_test_ds=novel_test_ds,
                        selected_class_indices=selected_class_indices,
                        k_shot=k_shot, q_query=q_query, rng=rng, device=device,
                    )
                logits = model(support_images, support_labels, query_images, distance=distance)

            probs_local = torch.softmax(logits, dim=1).cpu().numpy()
            preds_local = logits.argmax(dim=1).cpu().numpy()
            truths_local = query_labels.cpu().numpy()

            acc = float((preds_local == truths_local).mean())
            episode_accuracies.append(acc)

            # Store episode-level local labels for macro-F1 / AUC across all episodes.
            all_truths.extend(truths_local.tolist())
            all_preds.extend(preds_local.tolist())
            all_probs.extend(probs_local.tolist())

            # Map episode-local labels back to global novel-class indices for global accounting.
            for t_local, p_local in zip(truths_local, preds_local):
                t_global = selected_class_indices[int(t_local)]
                p_global = selected_class_indices[int(p_local)]

                true_name = novel_classes[t_global]
                pred_name = novel_classes[p_global]

                per_class_total[true_name] += 1
                per_class_correct[true_name] += int(t_local == p_local)
                per_class_pred_counter[pred_name] += 1
                global_cm[t_global, p_global] += 1

    mean_acc = float(np.mean(episode_accuracies)) * 100.0
    std_acc = float(np.std(episode_accuracies)) * 100.0
    ci95 = 1.96 * std_acc / math.sqrt(len(episode_accuracies))

    total_correct = sum(per_class_correct.values())
    total_queries = sum(per_class_total.values())
    ci_low, ci_high = wilson_ci(total_correct, total_queries)

    per_class_acc = {
        cls: (100.0 * per_class_correct[cls] / per_class_total[cls]) if per_class_total[cls] > 0 else 0.0
        for cls in novel_classes
    }

    macro_f1        = f1_score(all_truths, all_preds, average="macro", zero_division=0)
    macro_precision = precision_score(all_truths, all_preds, average="macro", zero_division=0)
    macro_recall    = recall_score(all_truths, all_preds, average="macro", zero_division=0)
    try:
        auc_ovr = float(roc_auc_score(all_truths, all_probs, multi_class="ovr", average="macro"))
    except Exception:
        auc_ovr = None

    print(f"\n{n_way}-way {k_shot}-shot | {len(episode_accuracies)} episodes")
    print(f"Mean accuracy  : {mean_acc:.2f}% ± {ci95:.2f}% (95% CI)")
    print(f"Std            : {std_acc:.2f}%")
    print(f"Wilson CI      : [{ci_low*100:.2f}%, {ci_high*100:.2f}%]")
    print(f"Macro F1       : {macro_f1:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall   : {macro_recall:.4f}")
    print(f"AUC-OVR (macro): {auc_ovr:.4f}" if auc_ovr is not None else "AUC-OVR (macro): N/A")
    print("\nPer-class accuracy (global novel-class space):")
    for cls in novel_classes:
        print(
            f"  {cls:<30s}: {per_class_acc[cls]:6.2f}% "
            f"| appeared in {episode_class_counter[cls]:4d} episodes "
            f"| queries={per_class_total[cls]:5d}"
        )

    summary = {
        "experiment": cfg["experiment"]["name"],
        "model": "resnet50",
        "seed": seed,
        "n_way": n_way,
        "k_shot": k_shot,
        "q_query": q_query,
        "distance": distance,
        "tta_n": tta_n,
        "num_episodes": len(episode_accuracies),
        "novel_classes": novel_classes,
        "checkpoint": ckpt_path,
        "epochs_trained": ckpt["epoch"],
        "mean_accuracy_pct": round(mean_acc, 4),
        "std_accuracy_pct": round(std_acc, 4),
        "ci95_pct": round(ci95, 4),
        "wilson_ci_95": [round(ci_low * 100, 4), round(ci_high * 100, 4)],
        "macro_f1": round(float(macro_f1), 4),
        "macro_precision": round(float(macro_precision), 4),
        "macro_recall": round(float(macro_recall), 4),
        "auc_ovr": round(auc_ovr, 4) if auc_ovr is not None else None,
        "per_class_accuracy_pct": {cls: round(acc, 4) for cls, acc in per_class_acc.items()},
        "per_class_episode_frequency": episode_class_counter,
        "per_class_query_total": per_class_total,
    }

    with open(os.path.join(results_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    txt_lines = [
        f"ProtoNet — episodic novel-class evaluation ({distance})",
        "=" * 60,
        f"Episodes       : {len(episode_accuracies)}",
        f"n_way          : {n_way}",
        f"k_shot         : {k_shot}",
        f"q_query        : {q_query}",
        f"Distance       : {distance}",
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
    ]
    txt_lines += [
        f"  {cls:<30s}: {per_class_acc[cls]:6.2f}% | appeared={episode_class_counter[cls]:4d} | queries={per_class_total[cls]:5d}"
        for cls in novel_classes
    ]

    with open(os.path.join(results_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines) + "\n")

    pd.DataFrame({
        "episode": list(range(1, len(episode_accuracies) + 1)),
        "accuracy": episode_accuracies,
    }).to_csv(os.path.join(results_dir, "episode_accuracies.csv"), index=False)

    # Global confusion matrix averaged per episode.
    cm_avg = global_cm / len(episode_accuracies)
    save_confusion_matrix(
        cm=cm_avg,
        class_names=[c.replace("AMERICAN ", "AM. ") for c in novel_classes],
        save_path=os.path.join(results_dir, "confusion_matrix.png"),
        title="ProtoNet — Aggregated Confusion Matrix (global novel-class space)",
    )

    history_path = os.path.join(ckpt_dir, "history.json")
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        save_learning_curves(history, os.path.join(results_dir, "learning_curves.png"))

    if first_episode_class_indices:
        first_episode_names = episode_class_name_list(novel_classes, first_episode_class_indices)
        save_tsne_episode(
            encoder=model.encoder,
            novel_train_ds=novel_train_ds,
            novel_test_ds=novel_test_ds,
            all_novel_classes=novel_classes,
            selected_class_indices=first_episode_class_indices,
            k_shot=k_shot,
            q_query=q_query,
            seed=seed,
            save_path=os.path.join(results_dir, "tsne_features.png"),
            title=(
                f"ProtoNet Feature Space — Episode 1\n"
                f"({n_way}-Way {k_shot}-Shot, classes: {', '.join([n.replace('AMERICAN ', 'AM. ') for n in first_episode_names])})"
            ),
            show_prototypes=True,
        )

    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()