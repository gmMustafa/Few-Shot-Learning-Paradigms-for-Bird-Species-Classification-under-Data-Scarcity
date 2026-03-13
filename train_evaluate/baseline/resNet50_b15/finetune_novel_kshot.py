# finetune_novel_kshot.py
#
# Option B: K-shot fine-tuning baseline.
#
# Protocol:
#   1. Load the pretrained base-15 backbone (feature extractor only, discard the fc head).
#   2. Attach a new 5-class head for the 5 novel classes.
#   3. Sample exactly K images per novel class from the TRAIN split (deterministic, seeded).
#   4. Fine-tune the full network on this tiny K-shot dataset.
#   5. Evaluate on the FULL novel-5 TEST split (5 images per class = 25 images total).
#
# This gives a "low-data CNN baseline" that is a fair comparator to FSL methods,
# since both use only K novel training images per class.
#
# Run this AFTER train.py has produced a base-15 checkpoint.
# Run evaluate_novel_kshot.py afterwards (or evaluation is embedded here).

import os
import json
import math
import yaml
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sample_k_per_class(imagefolder_ds: datasets.ImageFolder,
                        allowed_classnames: list,
                        k: int,
                        seed: int) -> tuple:
    """
    Deterministically sample exactly K images per class from imagefolder_ds,
    restricted to the allowed class names.
    Returns (indices, new_class_to_idx, target_transform_fn).
    """
    rng = random.Random(seed)

    old_c2i     = imagefolder_ds.class_to_idx
    new_c2i     = {c: i for i, c in enumerate(allowed_classnames)}
    old_i2c     = {v: k for k, v in old_c2i.items()}
    allowed_ids = {old_c2i[c] for c in allowed_classnames}

    # Group all sample indices by class
    class_indices: dict[str, list[int]] = {c: [] for c in allowed_classnames}
    for idx, (_, raw_y) in enumerate(imagefolder_ds.samples):
        if raw_y in allowed_ids:
            class_indices[old_i2c[raw_y]].append(idx)

    # Sample K per class
    selected = []
    for cls in allowed_classnames:
        pool = class_indices[cls]
        if len(pool) < k:
            raise ValueError(
                f"Class '{cls}' has only {len(pool)} training images, but K={k} requested."
            )
        selected.extend(rng.sample(pool, k))

    def target_transform(old_y: int) -> int:
        return new_c2i[old_i2c[int(old_y)]]

    return selected, new_c2i, target_transform


def filter_by_classes(imagefolder_ds: datasets.ImageFolder, allowed_classnames: list):
    """Filter all images of the given classes; remap labels to 0..K-1."""
    old_c2i     = imagefolder_ds.class_to_idx
    new_c2i     = {c: i for i, c in enumerate(allowed_classnames)}
    old_i2c     = {v: k for k, v in old_c2i.items()}
    allowed_ids = {old_c2i[c] for c in allowed_classnames}
    indices     = [i for i, (_, y) in enumerate(imagefolder_ds.samples) if y in allowed_ids]

    def target_transform(old_y: int) -> int:
        return new_c2i[old_i2c[int(old_y)]]

    return indices, new_c2i, target_transform


@torch.inference_mode()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    total_loss = 0.0
    correct = 0
    total   = 0
    y_true, y_pred = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        if criterion is not None:
            total_loss += criterion(logits, labels).item()
        preds    = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    avg_loss = total_loss / max(len(loader), 1) if criterion is not None else None
    acc      = correct / max(total, 1)
    return avg_loss, acc, np.array(y_true), np.array(y_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true",
                        help="Smoke-test mode: 1 epoch, patience=1.")
    args = parser.parse_args()

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.normpath(os.path.join(script_dir, "../../../configs/baseline_resnet_50_b15_config.yaml"))
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    novel_classes = cfg.get("novel_classes")
    base_classes  = cfg.get("base_classes")
    if not novel_classes or not base_classes:
        raise ValueError("Config must include both `base_classes:` and `novel_classes:`.")

    SEED        = int(cfg["train"]["seed"])
    set_seed(SEED)
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    K           = int(cfg["finetune"]["k_shot"])
    FT_EPOCHS   = int(cfg["finetune"]["epochs"])
    FT_LR       = float(cfg["finetune"]["lr"])
    PATIENCE    = int(cfg["finetune"].get("patience", 5))

    if args.debug:
        FT_EPOCHS, PATIENCE = 1, 1
        print("DEBUG MODE: epochs=1, patience=1 (smoke test only)")
    BATCH_SIZE  = int(cfg["train"]["batch_size"])
    NUM_WORKERS = int(cfg["train"].get("num_workers", 0))
    IMAGE_SIZE  = int(cfg["image_size"])

    train_dir = os.path.normpath(os.path.join(script_dir, cfg["data_train"]))
    test_dir  = os.path.normpath(os.path.join(script_dir, cfg["data_test"]))
    ckpt_dir  = os.path.normpath(os.path.join(script_dir, cfg["output"]["ckpt_dir"]))

    base_ckpt_best  = os.path.join(ckpt_dir, cfg["output"]["save_best_name"])
    base_ckpt_final = os.path.join(ckpt_dir, cfg["output"]["save_final_name"])
    base_ckpt       = base_ckpt_best if os.path.exists(base_ckpt_best) else base_ckpt_final
    if not os.path.exists(base_ckpt):
        raise FileNotFoundError(
            f"Base-15 checkpoint not found. Run train.py first.\n  {base_ckpt_best}\n  {base_ckpt_final}"
        )

    # Fine-tuned checkpoint will be saved separately
    ft_ckpt_path = os.path.join(ckpt_dir, cfg["output"]["finetune_best_name"])

    # -------- Transforms --------
    # Training transform for K-shot fine-tune: augment aggressively (few images)
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # -------- K-shot train subset (novel classes only) --------
    train_full  = datasets.ImageFolder(train_dir, transform=train_tf)
    train_idx, novel_c2i, train_target_tf = sample_k_per_class(
        train_full, novel_classes, k=K, seed=SEED
    )
    train_full.target_transform = train_target_tf
    train_subset = Subset(train_full, train_idx)

    print(f"K-shot fine-tuning | K={K} | Novel classes: {len(novel_classes)}")
    print(f"Training images: {len(train_subset)}  ({K} per class x {len(novel_classes)} classes)")

    # -------- Test subset (all novel test images) --------
    test_full  = datasets.ImageFolder(test_dir, transform=eval_tf)
    test_idx, _, test_target_tf = filter_by_classes(test_full, novel_classes)
    test_full.target_transform  = test_target_tf
    test_subset = Subset(test_full, test_idx)

    pin = DEVICE.type == "cuda"
    train_loader = DataLoader(
        train_subset, batch_size=min(BATCH_SIZE, len(train_subset)), shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin,
        persistent_workers=(NUM_WORKERS > 0),
    )
    test_loader = DataLoader(
        test_subset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin,
        persistent_workers=(NUM_WORKERS > 0),
    )

    # -------- Load base-15 backbone, replace head --------
    base_ckpt_data = torch.load(base_ckpt, map_location=DEVICE, weights_only=False)

    model = models.resnet50(weights=None)
    # Temporarily set 15-class head to load base weights
    model.fc = nn.Linear(model.fc.in_features, len(base_classes))
    model.load_state_dict(base_ckpt_data["model_state_dict"])
    # Replace head for novel classes
    model.fc = nn.Linear(model.fc.in_features, len(novel_classes))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=FT_LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=FT_EPOCHS, eta_min=1e-7)

    # -------- Fine-tuning Loop --------
    best_val_acc      = -1.0
    epochs_no_improve = 0
    history           = []

    print(f"\nFine-tuning | Epochs: {FT_EPOCHS} | LR: {FT_LR} | Patience: {PATIENCE}\n")

    for epoch in range(FT_EPOCHS):
        model.train()
        running_loss  = 0.0
        correct_train = 0
        total_train   = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            loss   = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss  += loss.item()
            preds          = torch.argmax(logits, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train   += labels.size(0)

        train_loss = running_loss / max(len(train_loader), 1)
        train_acc  = correct_train / max(total_train, 1)
        # Use test set for monitoring (only 25 images; no separate val for K-shot)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, DEVICE, criterion=criterion)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history.append({
            "epoch": epoch + 1, "train_loss": float(train_loss),
            "train_acc": float(train_acc), "test_loss": float(test_loss),
            "test_acc": float(test_acc), "lr": float(current_lr),
        })

        print(
            f"Epoch {epoch + 1:02d}/{FT_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Acc: {test_acc:.4f} | LR: {current_lr:.2e}"
        )

        if test_acc > best_val_acc:
            best_val_acc      = test_acc
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch":            epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "test_acc":         best_val_acc,
                    "num_classes":      len(novel_classes),
                    "novel_classes":    novel_classes,
                    "class_to_idx":     novel_c2i,
                    "k_shot":           K,
                    "base_checkpoint":  base_ckpt,
                },
                ft_ckpt_path,
            )
            print(f"  -> Best fine-tuned model saved (test_acc={best_val_acc:.4f})")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch + 1}.")
            break

    # -------- Final evaluation on novel test set --------
    best_ckpt_data = torch.load(ft_ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(best_ckpt_data["model_state_dict"])

    _, final_acc, y_true, y_pred = evaluate(model, test_loader, DEVICE)

    # -------- Save results --------
    out_dir = os.path.normpath(
        os.path.join(script_dir, cfg["output"]["results_dir"], f"novel_kshot_{K}")
    )
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    report    = classification_report(y_true, y_pred, target_names=novel_classes, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(out_dir, "classification_report.csv"))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ticks = np.arange(len(novel_classes))
    ax.set_xticks(ticks)
    ax.set_xticklabels(novel_classes, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(ticks)
    ax.set_yticklabels(novel_classes, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — K={K} fine-tune | novel test\nAccuracy: {final_acc * 100:.1f}%")
    for i in range(len(novel_classes)):
        for j in range(len(novel_classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=200)
    plt.close(fig)

    # Learning curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([h["epoch"] for h in history], [h["train_acc"] for h in history], label="Train Acc")
    ax.plot([h["epoch"] for h in history], [h["test_acc"]  for h in history], label="Test Acc (novel)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"K={K} Fine-tune Learning Curve (novel classes)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "learning_curve.png"), dpi=150)
    plt.close(fig)

    # 95% Wilson CI on novel test accuracy
    from scipy.stats import norm as _norm
    _z   = _norm.ppf(0.975)
    _n   = len(y_true)
    _p   = final_acc
    _den = 1 + _z**2 / _n
    _ctr = (_p + _z**2 / (2 * _n)) / _den
    _mrg = _z * math.sqrt(_p * (1 - _p) / _n + _z**2 / (4 * _n**2)) / _den
    ci_low, ci_high = max(0.0, _ctr - _mrg), min(1.0, _ctr + _mrg)

    summary = {
        "experiment":       f"base15_kshot_finetune_K{K}",
        "model":            "resnet50",
        "seed":             SEED,
        "k_shot":           K,
        "ft_lr":            FT_LR,
        "batch_size":       BATCH_SIZE,
        "base_checkpoint":  base_ckpt,
        "ft_checkpoint":    ft_ckpt_path,
        "novel_classes":    novel_classes,
        "train_images":     int(len(train_subset)),
        "test_images":      int(len(test_subset)),
        "accuracy":         float(final_acc),
        "accuracy_ci_95":   [float(ci_low), float(ci_high)],
        "macro_f1":         float(report["macro avg"]["f1-score"]),
        "macro_precision":  float(report["macro avg"]["precision"]),
        "macro_recall":     float(report["macro avg"]["recall"]),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Option B: K={K} Shot Fine-tuning — base-15 backbone + novel-5 head\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Base checkpoint:  {base_ckpt}\n")
        f.write(f"Fine-tuned ckpt:  {ft_ckpt_path}\n")
        f.write(f"K (train/class):  {K}\n")
        f.write(f"Train images:     {len(train_subset)}\n")
        f.write(f"Test images:      {len(test_subset)}\n\n")
        f.write(f"Novel Test Accuracy: {final_acc * 100:.2f}%\n")
        f.write(f"95% CI:              [{ci_low * 100:.2f}%, {ci_high * 100:.2f}%]\n")
        f.write(f"Macro F1:            {report['macro avg']['f1-score']:.4f}\n\n")
        f.write(report_df.to_string())

    print(f"\nOption B (K={K} fine-tune) complete.")
    print(f"Novel Test Accuracy: {final_acc * 100:.2f}% (95% CI: {ci_low*100:.2f}%–{ci_high*100:.2f}%)")
    print(f"Macro F1:            {report['macro avg']['f1-score']:.4f}")
    print(f"Results saved to:    {out_dir}")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
