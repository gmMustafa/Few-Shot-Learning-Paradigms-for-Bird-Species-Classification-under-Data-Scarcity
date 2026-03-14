# train.py — Supervised Baseline on FIXED 15 Base Classes (full data for those classes)

import os
import json
import yaml
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def filter_by_classes(imagefolder_ds: datasets.ImageFolder, allowed_classnames: list):
    """
    Subset an ImageFolder to only the allowed class names and remap labels to 0..K-1.
    Returns (Subset, new_class_to_idx, target_transform_fn).
    """
    missing = [c for c in allowed_classnames if c not in imagefolder_ds.class_to_idx]
    if missing:
        raise ValueError(f"Classes missing from dataset folders: {missing}")

    old_c2i     = imagefolder_ds.class_to_idx
    new_c2i     = {c: i for i, c in enumerate(allowed_classnames)}
    allowed_ids = {old_c2i[c] for c in allowed_classnames}
    old_i2c     = {v: k for k, v in old_c2i.items()}

    indices = [i for i, (_, y) in enumerate(imagefolder_ds.samples) if y in allowed_ids]

    def target_transform(old_y: int) -> int:
        return new_c2i[old_i2c[int(old_y)]]

    return indices, new_c2i, target_transform


@torch.inference_mode()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)

        if criterion is not None:
            total_loss += criterion(logits, labels).item()

        preds    = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    avg_loss = total_loss / max(len(loader), 1) if criterion is not None else None
    acc      = correct / max(total, 1)
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true",
                        help="Smoke-test mode: 1 epoch, patience=1.")
    args = parser.parse_args()

    # ---------- Config ----------
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.normpath(os.path.join(script_dir, "../../../configs/baseline_resnet_50_b15_config.yaml"))
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    base_classes = cfg.get("base_classes", None)
    if not base_classes or not isinstance(base_classes, list):
        raise ValueError("Config must include `base_classes:` as a list of exactly 15 class names.")
    if len(base_classes) != 15:
        raise ValueError(f"Expected 15 base classes, got {len(base_classes)}.")

    SEED        = int(cfg["train"]["seed"])
    set_seed(SEED)

    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("DEVICE INFO")
    print(f"  CUDA available : {torch.cuda.is_available()}")
    print(f"  Device         : {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU name       : {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  WARNING: Running on CPU — training will be slow!")
    print("=" * 60)

    EPOCHS      = int(cfg["train"]["epochs"])
    BATCH_SIZE  = int(cfg["train"]["batch_size"])
    LR          = float(cfg["train"]["lr"])
    NUM_WORKERS = int(cfg["train"].get("num_workers", 0))
    PATIENCE    = int(cfg["train"].get("patience", 7))
    IMAGE_SIZE  = int(cfg["image_size"])

    if args.debug:
        EPOCHS, PATIENCE = 1, 1
        print("DEBUG MODE: epochs=1, patience=1 (smoke test only)")

    print("CONFIG")
    print(f"  Epochs       : {EPOCHS}")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  LR           : {LR}")
    print(f"  Image size   : {IMAGE_SIZE}")
    print(f"  Patience     : {PATIENCE}")
    print(f"  Num workers  : {NUM_WORKERS}")
    print(f"  Config file  : {config_path}")
    print("=" * 60)

    train_dir = os.path.normpath(os.path.join(script_dir, cfg["data_train"]))
    val_dir   = os.path.normpath(os.path.join(script_dir, cfg["data_valid"]))
    ckpt_dir  = os.path.normpath(os.path.join(script_dir, cfg["output"]["ckpt_dir"]))
    os.makedirs(ckpt_dir, exist_ok=True)

    pin = DEVICE.type == "cuda"

    # ---------- Transforms ----------
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print("Loading datasets...")
    train_full = datasets.ImageFolder(train_dir, transform=train_transform)
    val_full   = datasets.ImageFolder(val_dir,   transform=val_transform)

    # Filter to base classes and remap labels to 0..14
    train_idx, base_c2i, train_tf = filter_by_classes(train_full, base_classes)
    val_idx,   _,        val_tf   = filter_by_classes(val_full,   base_classes)

    train_full.target_transform = train_tf
    val_full.target_transform   = val_tf

    train_subset = Subset(train_full, train_idx)
    val_subset   = Subset(val_full,   val_idx)

    num_classes = len(base_classes)
    print(f"Base classes: {num_classes}  |  Train: {len(train_subset)}  |  Val: {len(val_subset)}")

    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin,
        persistent_workers=(NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_subset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin,
        persistent_workers=(NUM_WORKERS > 0),
    )

    # ---------- Model ----------
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # ---------- Output paths ----------
    best_path    = os.path.join(ckpt_dir, cfg["output"]["save_best_name"])
    final_path   = os.path.join(ckpt_dir, cfg["output"]["save_final_name"])
    history_path = os.path.join(ckpt_dir, "history.json")

    # Save class mapping for reproducibility
    with open(os.path.join(ckpt_dir, "base_classes.json"), "w") as f:
        json.dump(base_classes, f, indent=2)
    with open(os.path.join(ckpt_dir, "class_to_idx.json"), "w") as f:
        json.dump(base_c2i, f, indent=2)

    # ---------- Training Loop ----------
    best_val_acc      = -1.0
    epochs_no_improve = 0
    history           = []

    print(f"\nTraining started | Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR} | Patience: {PATIENCE}\n")

    for epoch in range(EPOCHS):
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
        val_loss, val_acc = evaluate(model, val_loader, DEVICE, criterion=criterion)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history.append({
            "epoch":      epoch + 1,
            "train_loss": float(train_loss),
            "train_acc":  float(train_acc),
            "val_loss":   float(val_loss),
            "val_acc":    float(val_acc),
            "lr":         float(current_lr),
        })

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        if val_acc > best_val_acc:
            best_val_acc      = val_acc
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch":            epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "val_acc":          best_val_acc,
                    "num_classes":      num_classes,
                    "base_classes":     base_classes,
                    "class_to_idx":     base_c2i,
                },
                best_path,
            )
            print(f"  -> Best model saved (val_acc={best_val_acc:.4f})")
        else:
            epochs_no_improve += 1

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {PATIENCE} epochs).")
            break

    torch.save(
        {
            "epoch":            epoch + 1,
            "model_state_dict": model.state_dict(),
            "val_acc":          val_acc,
            "num_classes":      num_classes,
            "base_classes":     base_classes,
            "class_to_idx":     base_c2i,
        },
        final_path,
    )

    print(f"\nBest val_acc:     {best_val_acc:.4f}")
    print(f"Best checkpoint:  {best_path}")
    print(f"Final checkpoint: {final_path}")
    print(f"History:          {history_path}")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
