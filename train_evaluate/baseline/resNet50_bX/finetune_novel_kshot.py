# finetune_novel_kshot.py
#
# K-shot supervised fine-tuning on NOVEL classes.
#
# Procedure:
#   1. Load backbone from base-class supervised checkpoint
#   2. Replace FC layer with a new classifier for novel classes
#   3. Sample K images per novel class from novel TRAIN split
#   4. Fine-tune on that tiny labeled set
#   5. Validate on novel VALID split
#   6. Evaluate on novel TEST split
#
# Usage:
#   python finetune_novel_kshot.py --config ../../../configs/baseline_resnet_50_b15_config.yaml
#   python finetune_novel_kshot.py --config ../../../configs/baseline_resnet_50_b10_config.yaml
#   python finetune_novel_kshot.py --config ../../../configs/baseline_resnet_50_b10_config.yaml --k-shot 5 --debug

import os
import json
import yaml
import random
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        raise ValueError(f"Classes missing from dataset folders: {missing}")

    old_c2i = imagefolder_ds.class_to_idx
    new_c2i = {c: i for i, c in enumerate(allowed_classnames)}
    allowed_ids = {old_c2i[c] for c in allowed_classnames}
    old_i2c = {v: k for k, v in old_c2i.items()}

    indices = [i for i, (_, y) in enumerate(imagefolder_ds.samples) if y in allowed_ids]

    def target_transform(old_y: int) -> int:
        return new_c2i[old_i2c[int(old_y)]]

    return indices, new_c2i, target_transform


def sample_k_per_class(imagefolder_ds: datasets.ImageFolder, subset_indices: list, class_names: list, k: int, seed: int):
    rng = random.Random(seed)

    old_targets = [imagefolder_ds.samples[i][1] for i in subset_indices]
    idx_by_label = {}
    for ds_idx, y in zip(subset_indices, old_targets):
        idx_by_label.setdefault(y, []).append(ds_idx)

    selected = []
    for old_label, ds_indices in idx_by_label.items():
        if len(ds_indices) < k:
            raise ValueError(f"Class label {old_label} has only {len(ds_indices)} samples but k={k}")
        selected.extend(rng.sample(ds_indices, k))

    return selected


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
    return avg_loss, acc, correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="../../../configs/baseline_resnet_50_b15_config.yaml",
        help="Path to config yaml",
    )
    parser.add_argument("--k-shot", type=int, default=None, help="Override config finetune.k_shot")
    parser.add_argument("--debug", action="store_true", help="Smoke test")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.normpath(os.path.join(script_dir, args.config))

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["train"]["seed"])
    set_seed(seed)

    image_size = int(cfg["image_size"])
    novel_classes = cfg["novel_classes"]
    base_classes = cfg["base_classes"]

    k_shot = int(args.k_shot) if args.k_shot is not None else int(cfg["finetune"]["k_shot"])
    epochs = int(cfg["finetune"]["epochs"])
    lr = float(cfg["finetune"]["lr"])
    patience = int(cfg["finetune"]["patience"])

    if args.debug:
        epochs = 1
        patience = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("DEVICE INFO")
    print(f"  CUDA available : {torch.cuda.is_available()}")
    print(f"  Device         : {device}")
    if torch.cuda.is_available():
        print(f"  GPU name       : {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)

    train_dir = os.path.normpath(os.path.join(script_dir, cfg["data_train"]))
    valid_dir = os.path.normpath(os.path.join(script_dir, cfg["data_valid"]))
    test_dir = os.path.normpath(os.path.join(script_dir, cfg["data_test"]))
    ckpt_dir = os.path.normpath(os.path.join(script_dir, cfg["output"]["ckpt_dir"]))

    results_dir = os.path.normpath(
        os.path.join(script_dir, cfg["output"]["results_dir"], f"novel_kshot_{k_shot}")
    )
    os.makedirs(results_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, cfg["output"]["save_best_name"])
    finetune_best_path = os.path.join(ckpt_dir, cfg["output"]["finetune_best_name"])

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Base checkpoint not found: {ckpt_path}")

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_full = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_full = datasets.ImageFolder(valid_dir, transform=eval_transform)
    test_full = datasets.ImageFolder(test_dir, transform=eval_transform)

    train_idx, novel_c2i, train_tf = filter_by_classes(train_full, novel_classes)
    valid_idx, _, valid_tf = filter_by_classes(valid_full, novel_classes)
    test_idx, _, test_tf = filter_by_classes(test_full, novel_classes)

    train_full.target_transform = train_tf
    valid_full.target_transform = valid_tf
    test_full.target_transform = test_tf

    support_idx = sample_k_per_class(train_full, train_idx, novel_classes, k_shot, seed=seed)

    support_subset = Subset(train_full, support_idx)
    valid_subset = Subset(valid_full, valid_idx)
    test_subset = Subset(test_full, test_idx)

    support_loader = DataLoader(support_subset, batch_size=min(16, len(support_subset)), shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_subset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=0)

    # Load base checkpoint into full model first
    base_model = models.resnet50(weights=None)
    base_model.fc = nn.Linear(base_model.fc.in_features, len(base_classes))
    _ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    _state = _ckpt["model_state_dict"] if isinstance(_ckpt, dict) and "model_state_dict" in _ckpt else _ckpt
    base_model.load_state_dict(_state)

    # Transfer backbone weights only (exclude fc), then attach novel-class head
    model = models.resnet50(weights=None)
    backbone_state = {k: v for k, v in base_model.state_dict().items() if not k.startswith("fc.")}
    model.load_state_dict(backbone_state, strict=False)
    model.fc = nn.Linear(model.fc.in_features, len(novel_classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("CONFIG")
    print(f"  Experiment      : {cfg['experiment']['name']}")
    print(f"  Base classes    : {len(base_classes)}")
    print(f"  Novel classes   : {len(novel_classes)}")
    print(f"  K-shot          : {k_shot}")
    print(f"  Support samples : {len(support_subset)}")
    print(f"  Val samples     : {len(valid_subset)}")
    print(f"  Test samples    : {len(test_subset)}")
    print(f"  Epochs          : {epochs}")
    print(f"  LR              : {lr}")
    print(f"  Patience        : {patience}")
    print(f"  Base checkpoint : {ckpt_path}")
    print("=" * 60)

    history = []
    best_val_acc = -1.0
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in support_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss = train_loss_sum / max(len(support_loader), 1)
        train_acc = train_correct / max(train_total, 1)

        val_loss, val_acc, _, _ = evaluate(model, valid_loader, device, criterion=criterion)

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
        })

        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), finetune_best_path)
            print(f"  -> Best fine-tune checkpoint saved: {finetune_best_path}")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping after {epoch + 1} epochs.")
            break

    # Load best fine-tuned model for test eval
    model.load_state_dict(torch.load(finetune_best_path, map_location=device))
    test_loss, test_acc, test_correct, test_total = evaluate(model, test_loader, device, criterion=criterion)
    ci_lo, ci_hi = wilson_ci(test_correct, test_total)

    summary = {
        "experiment": cfg["experiment"]["name"],
        "evaluation": "novel_kshot_finetune",
        "split": f"base{len(base_classes)}_novel{len(novel_classes)}",
        "base_classes": base_classes,
        "novel_classes": novel_classes,
        "k_shot": k_shot,
        "support_samples": len(support_subset),
        "valid_samples": len(valid_subset),
        "test_samples": len(test_subset),
        "test_accuracy": round(float(test_acc), 6),
        "test_accuracy_ci_95": [round(ci_lo, 6), round(ci_hi, 6)],
        "best_val_acc": round(float(best_val_acc), 6),
        "base_checkpoint": ckpt_path,
        "finetune_checkpoint": finetune_best_path,
    }

    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(results_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    with open(os.path.join(results_dir, "summary.txt"), "w") as f:
        f.write("Novel K-shot Fine-tuning\n")
        f.write("=" * 60 + "\n")
        f.write(f"Experiment         : {cfg['experiment']['name']}\n")
        f.write(f"K-shot             : {k_shot}\n")
        f.write(f"Support samples    : {len(support_subset)}\n")
        f.write(f"Validation samples : {len(valid_subset)}\n")
        f.write(f"Test samples       : {len(test_subset)}\n")
        f.write(f"Best val acc       : {best_val_acc * 100:.2f}%\n")
        f.write(f"Test accuracy      : {test_acc * 100:.2f}%\n")
        f.write(f"95% CI             : [{ci_lo * 100:.2f}%, {ci_hi * 100:.2f}%]\n")

    print("=" * 60)
    print("NOVEL K-SHOT FINE-TUNING DONE")
    print(f"Best val acc   : {best_val_acc * 100:.2f}%")
    print(f"Test accuracy  : {test_acc * 100:.2f}%")
    print(f"Wilson 95% CI  : [{ci_lo * 100:.2f}%, {ci_hi * 100:.2f}%]")
    print(f"Results saved  : {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()