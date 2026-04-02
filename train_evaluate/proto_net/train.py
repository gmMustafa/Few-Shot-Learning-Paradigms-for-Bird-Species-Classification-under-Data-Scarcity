"""
ProtoNet Training — episodic training on base classes only.

Usage:
    python train.py                # full training (30 epochs)
    python train.py --debug        # smoke test (1 epoch, 10 episodes)
"""
import os
import sys
import json
import random
import argparse
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

# --- resolve project root regardless of CWD ---
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from utils.dataset import FilteredDataset, get_eval_transforms
from utils.sampler  import FewShotSampler
from models.protonet import ProtoNet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_support_query(images, labels, n_way, k_shot, q_query):
    """
    Given a flat episode batch (images, labels), split into support / query
    tensors with labels remapped to 0..n_way-1.

    The sampler orders the batch as:
        [cls0_support, cls0_query, cls1_support, cls1_query, ...]
    so we can reliably slice by position.
    """
    class_ids = torch.unique(labels)
    assert len(class_ids) == n_way, \
        f"Expected {n_way} classes in episode, got {len(class_ids)}"

    support_idxs, query_idxs = [], []
    for cls_id in class_ids:
        idxs = (labels == cls_id).nonzero(as_tuple=True)[0]
        support_idxs.append(idxs[:k_shot])
        query_idxs.append(idxs[k_shot: k_shot + q_query])

    support_idxs = torch.cat(support_idxs)
    query_idxs   = torch.cat(query_idxs)

    support = images[support_idxs]
    query   = images[query_idxs]
    s_labels = labels[support_idxs]
    q_labels = labels[query_idxs]

    # Remap original class ids → episode-local 0..n_way-1
    unique = torch.unique(s_labels)
    lmap   = {int(c): i for i, c in enumerate(unique)}
    s_labels = torch.tensor([lmap[int(l)] for l in s_labels],
                             dtype=torch.long, device=labels.device)
    q_labels = torch.tensor([lmap[int(l)] for l in q_labels],
                             dtype=torch.long, device=labels.device)

    return support, query, s_labels, q_labels


def run_epoch(model, loader, optimizer, criterion, device,
              n_way, k_shot, q_query, train: bool):
    model.train() if train else model.eval()
    total_loss, total_acc, count = 0.0, 0.0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for episode in tqdm(loader, desc="train" if train else "val ", leave=False):
            images, labels = episode[0].to(device), episode[1].to(device)
            support, query, s_labels, q_labels = split_support_query(
                images, labels, n_way, k_shot, q_query)

            logits = model(support, s_labels, query)
            loss   = criterion(logits, q_labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            acc = (logits.argmax(dim=1) == q_labels).float().mean().item()
            total_loss += loss.item()
            total_acc  += acc
            count      += 1

    return total_loss / count, total_acc / count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../../configs/birds20/proto_configs.yaml',
                        help='Path to config (relative to this script)')
    parser.add_argument('--debug', action='store_true',
                        help='Smoke test: 1 epoch, 10 episodes')
    args = parser.parse_args()

    config_path = os.path.normpath(os.path.join(SCRIPT_DIR, args.config))
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if args.debug:
        cfg['train']['num_epochs']          = 1
        cfg['train']['episodes_per_epoch']  = 5
        cfg['train']['q_query']             = 4   # fewer queries → faster per episode
        cfg['train']['patience']            = 1
        print("[DEBUG] Overrides: epochs=1, episodes=5, q_query=4, patience=1")

    # --- seed ---
    seed = cfg['seed']
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- config values ---
    base_classes        = cfg['base_classes']
    image_size          = cfg['image_size']
    n_way               = cfg['train']['n_way']
    k_shot              = cfg['train']['k_shot']
    q_query             = cfg['train']['q_query']
    episodes_per_epoch  = cfg['train']['episodes_per_epoch']
    num_epochs          = cfg['train']['num_epochs']
    lr                  = cfg['train']['lr']
    patience            = cfg['train']['patience']
    num_workers         = cfg['train']['num_workers']

    # --- resolve paths ---
    data_train = os.path.normpath(os.path.join(SCRIPT_DIR, cfg['data']['train']))
    data_val   = os.path.normpath(os.path.join(SCRIPT_DIR, cfg['data']['val']))
    ckpt_dir   = os.path.normpath(os.path.join(SCRIPT_DIR, cfg['output']['ckpt_dir']))
    os.makedirs(ckpt_dir, exist_ok=True)

    transform = get_eval_transforms(image_size)

    # --- datasets: BASE CLASSES ONLY ---
    train_ds = FilteredDataset(data_train, base_classes, transform=transform)
    val_ds   = FilteredDataset(data_val,   base_classes, transform=transform)

    print("=" * 60)
    print("DEVICE INFO")
    print(f"  CUDA available : {torch.cuda.is_available()}")
    print(f"  Device         : {device}")
    if torch.cuda.is_available():
        print(f"  GPU name       : {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  WARNING: Running on CPU — training will be slow!")
    print("CONFIG")
    print(f"  Epochs         : {num_epochs}")
    print(f"  Episodes/epoch : {episodes_per_epoch}")
    print(f"  {n_way}-way {k_shot}-shot | q_query={q_query}")
    print(f"  LR             : {lr}")
    print(f"  Patience       : {patience}")
    print(f"  Config file    : {config_path}")
    print("=" * 60)

    print(f"Base classes : {len(base_classes)}")
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples  : {len(val_ds)}  (5 per class × 15 = 75)")

    # Val set has exactly 5 images per class; with k_shot=1 and q_query=4 we use all 5
    val_q_query = min(q_query, 4)   # 5 val images/class → max 4 queries when k=1

    train_sampler = FewShotSampler(
        train_ds.get_labels(), n_way=n_way, k_shot=k_shot,
        q_query=q_query, episodes_per_epoch=episodes_per_epoch, seed=seed)
    val_sampler = FewShotSampler(
        val_ds.get_labels(), n_way=n_way, k_shot=k_shot,
        q_query=val_q_query, episodes_per_epoch=50, seed=seed + 1)

    train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                              num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_sampler=val_sampler,
                              num_workers=num_workers)

    # --- model: ResNet-50 backbone (same as supervised baselines) ---
    backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
    backbone.fc = nn.Identity()          # output: 2048-dim embedding
    model = ProtoNet(encoder=backbone).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # --- training loop ---
    history      = []
    best_val_acc = 0.0
    no_improve   = 0
    ckpt_path    = os.path.join(ckpt_dir, cfg['output']['ckpt_name'])

    print(f"\nTraining: {num_epochs} epochs | "
          f"{n_way}-way {k_shot}-shot | q_query={q_query} | lr={lr}")

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion,
                                    device, n_way, k_shot, q_query, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader, optimizer, criterion,
                                    device, n_way, k_shot, val_q_query, train=False)

        record = dict(epoch=epoch,
                      train_loss=tr_loss, train_acc=tr_acc,
                      val_loss=vl_loss,   val_acc=vl_acc)
        history.append(record)
        print(f"Epoch {epoch:3d}/{num_epochs} | "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f} | "
              f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            no_improve   = 0
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'val_acc':          vl_acc,
                'base_classes':     base_classes,
                'novel_classes':    cfg['novel_classes'],
                'cfg':              cfg,
            }, ckpt_path)
            print(f"           -> saved best checkpoint (val_acc={vl_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    # --- save artifacts ---
    with open(os.path.join(ckpt_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(ckpt_dir, 'config_used.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    print(f"\nDone. Best val_acc={best_val_acc:.4f}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"History    : {os.path.join(ckpt_dir, 'history.json')}")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
