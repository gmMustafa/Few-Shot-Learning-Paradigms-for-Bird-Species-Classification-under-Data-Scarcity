# train.py  (Full Supervised Baseline: ResNet-50, 20 classes, full data)

import os
import json
import random
import time
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.inference_mode()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)

        if criterion is not None:
            total_loss += criterion(logits, labels).item()

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(len(loader), 1) if criterion is not None else None
    acc = correct / max(total, 1)
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true",
                        help="Smoke-test mode: 1 epoch, patience=1. No config changes needed.")
    parser.add_argument("--config", default=None,
                        help="Path to config YAML. Defaults to configs/birds20/baseline_resnet_50_fs_config.yaml.")
    args = parser.parse_args()

    # ---------- Config ----------
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    if args.config:
        config_path = os.path.normpath(os.path.join(script_dir, args.config))
    else:
        config_path = os.path.normpath(os.path.join(script_dir, "../../../configs/birds20/baseline_resnet_50_fs_config.yaml"))
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg["train"]["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("DEVICE INFO")
    print(f"  CUDA available : {torch.cuda.is_available()}")
    print(f"  Device         : {device}")
    if torch.cuda.is_available():
        print(f"  GPU name       : {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  WARNING: Running on CPU — training will be slow!")
    print("=" * 60)

    epochs      = int(cfg["train"]["epochs"])
    batch_size  = int(cfg["train"]["batch_size"])
    lr          = float(cfg["train"]["lr"])
    num_workers = int(cfg["train"]["num_workers"])
    patience    = int(cfg["train"].get("patience", 7))
    image_size  = int(cfg["image_size"])

    print("CONFIG")
    print(f"  Epochs       : {epochs}")
    print(f"  Batch size   : {batch_size}")
    print(f"  LR           : {lr}")
    print(f"  Image size   : {image_size}")
    print(f"  Patience     : {patience}")
    print(f"  Num workers  : {num_workers}")
    print(f"  Config file  : {config_path}")
    print("=" * 60)

    if args.debug:
        epochs, patience = 1, 1
        print("DEBUG MODE: epochs=1, patience=1 (smoke test only)")

    # Resolve data and output paths relative to this script
    train_dir = os.path.normpath(os.path.join(script_dir, cfg["data_train"]))
    val_dir   = os.path.normpath(os.path.join(script_dir, cfg["data_valid"]))
    ckpt_dir  = os.path.normpath(os.path.join(script_dir, cfg["output"]["ckpt_dir"]))
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---------- Transforms ----------
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print("Loading datasets...")
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds   = datasets.ImageFolder(val_dir,   transform=val_tf)

    num_classes = len(train_ds.classes)
    print(f"Classes: {num_classes}  |  Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )

    # ---------- Model ----------
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # ---------- Output paths ----------
    best_path         = os.path.join(ckpt_dir, cfg["output"]["save_best_name"])
    final_path        = os.path.join(ckpt_dir, cfg["output"]["save_final_name"])
    history_path      = os.path.join(ckpt_dir, "history.json")
    class_to_idx_path = os.path.join(ckpt_dir, "class_to_idx.json")
    classes_path      = os.path.join(ckpt_dir, "classes.json")

    with open(class_to_idx_path, "w") as f:
        json.dump(train_ds.class_to_idx, f, indent=2)
    with open(classes_path, "w") as f:
        json.dump(train_ds.classes, f, indent=2)

    # ---------- Training Loop ----------
    best_val_acc       = -1.0
    epochs_no_improve  = 0
    history            = []

    print(f"\nTraining started | Epochs: {epochs} | Batch: {batch_size} | LR: {lr} | Patience: {patience}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    log_every = max(1, len(train_loader) // 5)  # print ~5x per epoch
    train_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        running_loss   = 0.0
        correct_train  = 0
        total_train    = 0

        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss   = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss  += loss.item()
            preds          = torch.argmax(logits, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train   += labels.size(0)

            if batch_idx % log_every == 0 or batch_idx == len(train_loader):
                print(f"  [Epoch {epoch+1}/{epochs}] Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {running_loss/batch_idx:.4f} | "
                      f"Acc: {correct_train/total_train:.4f}")

        train_loss = running_loss / max(len(train_loader), 1)
        train_acc  = correct_train / max(total_train, 1)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion=criterion)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_time = time.time() - epoch_start
        elapsed    = time.time() - train_start
        epochs_done = epoch + 1
        eta_sec    = (elapsed / epochs_done) * (epochs - epochs_done)
        eta_str    = time.strftime("%H:%M:%S", time.gmtime(eta_sec))

        history.append({
            "epoch":      epoch + 1,
            "train_loss": float(train_loss),
            "train_acc":  float(train_acc),
            "val_loss":   float(val_loss),
            "val_acc":    float(val_acc),
            "lr":         float(current_lr),
        })

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.1f}s | ETA: {eta_str}"
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
                    "classes":          train_ds.classes,
                    "class_to_idx":     train_ds.class_to_idx,
                },
                best_path,
            )
            print(f"  -> Best model saved (val_acc={best_val_acc:.4f})")
        else:
            epochs_no_improve += 1

        # Persist history after every epoch
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {patience} epochs).")
            break

    # Final checkpoint
    torch.save(
        {
            "epoch":            epoch + 1,
            "model_state_dict": model.state_dict(),
            "val_acc":          val_acc,
            "num_classes":      num_classes,
            "classes":          train_ds.classes,
            "class_to_idx":     train_ds.class_to_idx,
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
