# evaluate.py — Evaluate supervised ResNet-50 on BASE-CLASS test split
# Works for both base15 and base10 configs

import os
import json
import math
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def wilson_ci(correct: int, total: int, alpha: float = 0.05):
    """Wilson score 95% confidence interval for a proportion."""
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
        raise ValueError(f"Classes missing from dataset folders: {missing}")

    old_c2i = imagefolder_ds.class_to_idx
    new_c2i = {c: i for i, c in enumerate(allowed_classnames)}
    allowed_ids = {old_c2i[c] for c in allowed_classnames}
    old_i2c = {v: k for k, v in old_c2i.items()}

    indices = [i for i, (_, y) in enumerate(imagefolder_ds.samples) if y in allowed_ids]

    def target_transform(old_y: int) -> int:
        return new_c2i[old_i2c[int(old_y)]]

    return indices, new_c2i, target_transform


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds)
        y_prob.extend(probs)
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def save_tsne_baseline(model, loader, class_names, device, save_path, title):
    """Extract ResNet avgpool features and save a t-SNE plot of the test set."""
    from sklearn.manifold import TSNE
    model.eval()
    features, labels = [], []
    captured = []

    def _hook(module, inp, out):
        captured.append(out.detach().cpu().flatten(1).numpy())

    handle = model.avgpool.register_forward_hook(_hook)

    with torch.inference_mode():
        for images, lbls in loader:
            captured.clear()
            _ = model(images.to(device))
            features.append(captured[0])
            labels.extend(lbls.numpy())

    handle.remove()

    X = np.vstack(features)
    y = np.array(labels)

    perplexity = min(30, max(5, len(X) // 5))
    X_2d = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        max_iter=1000,
        init="pca",
    ).fit_transform(X)

    n_cls = len(class_names)
    tsne_w = max(14, n_cls * 0.12)
    tsne_h = max(10, n_cls * 0.08)
    fig, ax = plt.subplots(figsize=(tsne_w, tsne_h))
    palette = plt.cm.get_cmap("nipy_spectral", n_cls)

    for i, cls in enumerate(class_names):
        mask = (y == i)
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=28, alpha=0.8, label=cls, color=palette(i))

    ax.set_title(title)
    legend_cols = max(2, n_cls // 15)
    ax.legend(fontsize=6, bbox_to_anchor=(1.01, 1), loc="upper left",
              ncol=legend_cols, framealpha=0.9, borderaxespad=0)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="../../../configs/birds20/baseline_resnet_50_b15_config.yaml",
        help="Path to config yaml",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.normpath(os.path.join(script_dir, args.config))

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    base_classes = cfg["base_classes"]
    num_classes = len(base_classes)
    image_size = int(cfg["image_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("DEVICE INFO")
    print(f"  CUDA available : {torch.cuda.is_available()}")
    print(f"  Device         : {device}")
    if torch.cuda.is_available():
        print(f"  GPU name       : {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)

    test_dir = os.path.normpath(os.path.join(script_dir, cfg["data_test"]))
    ckpt_dir = os.path.normpath(os.path.join(script_dir, cfg["output"]["ckpt_dir"]))
    results_dir = os.path.normpath(os.path.join(script_dir, cfg["output"]["results_dir"], "base_test"))
    os.makedirs(results_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, cfg["output"]["save_best_name"])
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_full = datasets.ImageFolder(test_dir, transform=test_transform)
    test_idx, class_to_idx, test_tf = filter_by_classes(test_full, base_classes)
    test_full.target_transform = test_tf
    test_subset = Subset(test_full, test_idx)

    loader = DataLoader(
        test_subset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()


    print("CONFIG")
    print(f"  Experiment    : {cfg['experiment']['name']}")
    print(f"  Config file   : {config_path}")
    print(f"  Base classes  : {num_classes}")
    print(f"  Test samples  : {len(test_subset)}")
    print(f"  Checkpoint    : {ckpt_path}")
    print("=" * 60)

    y_true, y_pred, y_prob = predict(model, loader, device)

    correct = int((y_true == y_pred).sum())
    total = int(len(y_true))
    acc = correct / total if total > 0 else 0.0
    ci_lo, ci_hi = wilson_ci(correct, total)

    macro_report = classification_report(
        y_true, y_pred,
        target_names=base_classes,
        output_dict=True,
        zero_division=0
    )

    macro_f1 = float(macro_report["macro avg"]["f1-score"])
    macro_precision = float(macro_report["macro avg"]["precision"])
    macro_recall = float(macro_report["macro avg"]["recall"])

    cm = confusion_matrix(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
    except Exception:
        auc = None

    # Save summary
    summary = {
        "experiment": cfg["experiment"]["name"],
        "num_base_classes": num_classes,
        "base_classes": base_classes,
        "accuracy": round(acc, 6),
        "accuracy_ci_95": [round(ci_lo, 6), round(ci_hi, 6)],
        "macro_f1": round(macro_f1, 6),
        "macro_precision": round(macro_precision, 6),
        "macro_recall": round(macro_recall, 6),
        "auc_ovr": round(float(auc), 6) if auc is not None else None,
        "num_test_samples": total,
        "checkpoint": ckpt_path,
    }

    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame(cm, index=base_classes, columns=base_classes).to_csv(
        os.path.join(results_dir, "confusion_matrix.csv")
    )

    pd.DataFrame(classification_report(
        y_true, y_pred,
        target_names=base_classes,
        output_dict=True,
        zero_division=0
    )).transpose().to_csv(os.path.join(results_dir, "classification_report.csv"))

    # Confusion matrix image
    if num_classes <= 20:
        sz, tick_fs = 10, 8
    elif num_classes <= 50:
        sz, tick_fs = 16, 6
    elif num_classes <= 100:
        sz, tick_fs = 22, 5
    else:
        sz, tick_fs = 28, 4
    fig, ax = plt.subplots(figsize=(sz, sz - 2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(base_classes, rotation=90, fontsize=tick_fs)
    ax.set_yticklabels(base_classes, fontsize=tick_fs)
    ax.set_title(f"Confusion Matrix — {cfg['experiment']['name']} (Base Test)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # t-SNE
    save_tsne_baseline(
        model,
        loader,
        base_classes,
        device,
        os.path.join(results_dir, "tsne_features.png"),
        title=f"t-SNE — {cfg['experiment']['name']} Base Test Features",
    )

    print(f"Accuracy      : {acc * 100:.2f}%")
    print(f"Wilson 95% CI : [{ci_lo * 100:.2f}%, {ci_hi * 100:.2f}%]")
    print(f"Macro F1      : {macro_f1:.4f}")
    print(f"Results saved : {results_dir}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()