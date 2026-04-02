# train.py — Supervised DeiT-S Baseline on FIXED base classes
# Fine-tunes deit_small_patch16_224 on base classes for NCC feature extraction.
# Works for both:
#   - base15 / novel5  (default config)
#   - base10 / novel10
#
# Usage:
#   python train.py
#   python train.py --config ../../../configs/birds20/vit_baseline_b10_config.yaml
#   python train.py --debug
#
# HPC offline (no internet): download weights on a machine with internet, transfer,
# then pass --weights-path to skip HuggingFace download:
#   python train.py --weights-path /path/to/deit_small_patch16_224.pth

import os
import sys
import json
import yaml
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.vit import build_deit_classifier


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    parser.add_argument(
        '--config',
        default='../../../configs/birds20/vit_baseline_b15_config.yaml',
        help='Path to config yaml (relative to this script)',
    )
    parser.add_argument('--debug', action='store_true',
                        help='Smoke-test mode: 1 epoch, patience=1')
    parser.add_argument('--weights-path', default=None,
                        help='Local path to pretrained DeiT-S weights (.pth or .safetensors). '
                             'Use on HPC without internet access to skip HuggingFace download.')
    args = parser.parse_args()

    # Auto-detect weights: check pretrained_weights/ at project root first,
    # then fall back to the old vit_backbone/ location
    if args.weights_path is None:
        _candidates = [
            os.path.join(PROJECT_ROOT, 'pretrained_weights', 'deit_small_patch16_224.pth'),
            os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'deit_small_patch16_224.pth')),
        ]
        for _c in _candidates:
            if os.path.isfile(_c):
                args.weights_path = _c
                print(f'[auto] Using DeiT-S weights: {_c}')
                break

    config_path = os.path.normpath(os.path.join(SCRIPT_DIR, args.config))
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    base_classes = cfg['base_classes']
    if not base_classes or len(base_classes) < 2:
        raise ValueError('Config must include at least 2 base_classes.')

    seed       = int(cfg['train']['seed'])
    epochs     = int(cfg['train']['epochs'])
    batch_size = int(cfg['train']['batch_size'])
    lr         = float(cfg['train']['lr'])
    num_workers = int(cfg['train'].get('num_workers', 0))
    patience   = int(cfg['train'].get('patience', 7))
    image_size = int(cfg['image_size'])

    if args.debug:
        epochs, patience = 1, 1
        print('DEBUG MODE: epochs=1, patience=1 (smoke test only)')

    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 60)
    print('DEVICE INFO')
    print(f'  CUDA available : {torch.cuda.is_available()}')
    print(f'  Device         : {device}')
    if torch.cuda.is_available():
        print(f'  GPU name       : {torch.cuda.get_device_name(0)}')
        print(f'  GPU memory     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    else:
        print('  WARNING: Running on CPU — training will be slow!')
    print('CONFIG')
    print(f'  Experiment     : {cfg["experiment"]["name"]}')
    print(f'  Base classes   : {len(base_classes)}')
    print(f'  Epochs         : {epochs}')
    print(f'  Batch size     : {batch_size}')
    print(f'  LR             : {lr}')
    print(f'  Image size     : {image_size}')
    print(f'  Patience       : {patience}')
    print(f'  Config file    : {config_path}')
    print('=' * 60)

    train_dir = os.path.normpath(os.path.join(SCRIPT_DIR, cfg['data_train']))
    val_dir   = os.path.normpath(os.path.join(SCRIPT_DIR, cfg['data_valid']))
    ckpt_dir  = os.path.normpath(os.path.join(SCRIPT_DIR, cfg['output']['ckpt_dir']))
    os.makedirs(ckpt_dir, exist_ok=True)

    pin = device.type == 'cuda'

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print('Loading datasets...')
    train_full = datasets.ImageFolder(train_dir, transform=train_transform)
    val_full   = datasets.ImageFolder(val_dir,   transform=val_transform)

    train_idx, base_c2i, train_tf = filter_by_classes(train_full, base_classes)
    val_idx, _, val_tf             = filter_by_classes(val_full,   base_classes)

    train_full.target_transform = train_tf
    val_full.target_transform   = val_tf

    train_subset = Subset(train_full, train_idx)
    val_subset   = Subset(val_full,   val_idx)

    num_classes = len(base_classes)
    print(f'Train samples   : {len(train_subset)}')
    print(f'Val samples     : {len(val_subset)}')
    print(f'Num classes     : {num_classes}')

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )

    # DeiT-S with num_classes classification head
    model = build_deit_classifier(
        num_classes=num_classes,
        pretrained=(args.weights_path is None),
        weights_path=args.weights_path,
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_path  = os.path.join(ckpt_dir, cfg['output']['save_best_name'])
    final_path = os.path.join(ckpt_dir, cfg['output']['save_final_name'])
    history_path   = os.path.join(ckpt_dir, 'history.json')
    cfg_used_path  = os.path.join(ckpt_dir, 'config_used.yaml')

    with open(os.path.join(ckpt_dir, 'base_classes.json'), 'w') as f:
        json.dump(base_classes, f, indent=2)
    with open(os.path.join(ckpt_dir, 'class_to_idx.json'), 'w') as f:
        json.dump(base_c2i, f, indent=2)
    with open(cfg_used_path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    best_val_acc    = -1.0
    epochs_no_improve = 0
    history = []

    print(f'\nTraining started | Epochs: {epochs} | Batch: {batch_size} | LR: {lr} | Patience: {patience}\n')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train   = 0

        for images, labels in train_loader:
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

        train_loss = running_loss / max(len(train_loader), 1)
        train_acc  = correct_train / max(total_train, 1)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion=criterion)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_acc':  float(train_acc),
            'val_loss':   float(val_loss),
            'val_acc':    float(val_acc),
            'lr':         float(current_lr),
        })

        print(
            f'Epoch {epoch + 1:02d}/{epochs} | '
            f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
            f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | '
            f'LR: {current_lr:.2e}'
        )

        ckpt_payload = {
            'epoch':            epoch + 1,
            'model_state_dict': model.state_dict(),
            'val_acc':          float(val_acc),
            'train_acc':        float(train_acc),
            'config_path':      config_path,
            'experiment':       cfg['experiment']['name'],
            'base_classes':     base_classes,
            'num_classes':      num_classes,
        }

        if val_acc > best_val_acc:
            best_val_acc      = val_acc
            epochs_no_improve = 0
            torch.save(ckpt_payload, best_path)
            print(f'  -> Best checkpoint saved: {best_path}')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs.')
            break

    torch.save({
        'epoch':            len(history),
        'model_state_dict': model.state_dict(),
        'val_acc':          float(history[-1]['val_acc']) if history else None,
        'train_acc':        float(history[-1]['train_acc']) if history else None,
        'config_path':      config_path,
        'experiment':       cfg['experiment']['name'],
        'base_classes':     base_classes,
        'num_classes':      num_classes,
    }, final_path)

    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print('\nTraining complete.')
    print(f'Best checkpoint : {best_path}')
    print(f'Final checkpoint: {final_path}')
    print(f'History         : {history_path}')
    print(f'Best val_acc    : {best_val_acc:.4f}')


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
