"""
DeiT-S ProtoNet Training — episodic training on base classes only.

Usage:
    python train.py                                                     # B15/N5
    python train.py --config ../../../configs/vit_proto_configs_10b_10n.yaml  # B10/N10
    python train.py --debug                                             # smoke test

HPC offline (no internet):
    python train.py --weights-path /path/to/deit_small_patch16_224.pth
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
from tqdm import tqdm

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from utils.dataset  import FilteredDataset, get_eval_transforms
from utils.sampler  import FewShotSampler
from models.protonet import ProtoNet
from models.vit      import DeiTSmallEncoder


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
    class_ids = torch.unique(labels)
    assert len(class_ids) == n_way

    support_idxs, query_idxs = [], []
    for cls_id in class_ids:
        idxs = (labels == cls_id).nonzero(as_tuple=True)[0]
        support_idxs.append(idxs[:k_shot])
        query_idxs.append(idxs[k_shot: k_shot + q_query])

    support_idxs = torch.cat(support_idxs)
    query_idxs   = torch.cat(query_idxs)

    support  = images[support_idxs]
    query    = images[query_idxs]
    s_labels = labels[support_idxs]
    q_labels = labels[query_idxs]

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
        for episode in tqdm(loader, desc='train' if train else 'val ', leave=False):
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
    parser.add_argument('--config', default='../../../configs/vit_proto_configs.yaml',
                        help='Path to config (relative to this script)')
    parser.add_argument('--debug', action='store_true',
                        help='Smoke test: 1 epoch, 10 episodes')
    parser.add_argument('--weights-path', default=None,
                        help='Local path to pretrained DeiT-S weights (.pth or .safetensors). '
                             'Use on HPC without internet access to skip HuggingFace download.')
    args = parser.parse_args()

    # Auto-detect weights file in vit_backbone/ if not specified
    if args.weights_path is None:
        _auto = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'deit_small_patch16_224.pth'))
        if os.path.isfile(_auto):
            args.weights_path = _auto
            print(f'[auto] Using DeiT-S weights: {_auto}')

    config_path = os.path.normpath(os.path.join(SCRIPT_DIR, args.config))
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if args.debug:
        cfg['train']['num_epochs']         = 1
        cfg['train']['episodes_per_epoch'] = 5
        cfg['train']['q_query']            = 4
        cfg['train']['patience']           = 1
        print('[DEBUG] Overrides: epochs=1, episodes=5, q_query=4, patience=1')

    seed = cfg['seed']
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_classes       = cfg['base_classes']
    image_size         = cfg['image_size']
    n_way              = cfg['train']['n_way']
    k_shot             = cfg['train']['k_shot']
    q_query            = cfg['train']['q_query']
    episodes_per_epoch = cfg['train']['episodes_per_epoch']
    num_epochs         = cfg['train']['num_epochs']
    lr                 = cfg['train']['lr']
    patience           = cfg['train']['patience']
    num_workers        = cfg['train']['num_workers']

    data_train = os.path.normpath(os.path.join(SCRIPT_DIR, cfg['data']['train']))
    data_val   = os.path.normpath(os.path.join(SCRIPT_DIR, cfg['data']['val']))
    ckpt_dir   = os.path.normpath(os.path.join(SCRIPT_DIR, cfg['output']['ckpt_dir']))
    os.makedirs(ckpt_dir, exist_ok=True)

    transform = get_eval_transforms(image_size)

    train_ds = FilteredDataset(data_train, base_classes, transform=transform)
    val_ds   = FilteredDataset(data_val,   base_classes, transform=transform)

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
    print(f'  Backbone       : deit_small_patch16_224 (384-dim)')
    print(f'  Epochs         : {num_epochs}')
    print(f'  Episodes/epoch : {episodes_per_epoch}')
    print(f'  {n_way}-way {k_shot}-shot | q_query={q_query}')
    print(f'  LR             : {lr}')
    print(f'  Patience       : {patience}')
    print(f'  Config file    : {config_path}')
    print('=' * 60)

    print(f'Base classes : {len(base_classes)}')
    print(f'Train samples: {len(train_ds)}')
    print(f'Val samples  : {len(val_ds)}')

    val_q_query = min(q_query, 4)

    train_sampler = FewShotSampler(
        train_ds.get_labels(), n_way=n_way, k_shot=k_shot,
        q_query=q_query, episodes_per_epoch=episodes_per_epoch, seed=seed)
    val_sampler = FewShotSampler(
        val_ds.get_labels(), n_way=n_way, k_shot=k_shot,
        q_query=val_q_query, episodes_per_epoch=50, seed=seed + 1)

    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_sampler=val_sampler,   num_workers=num_workers)

    # DeiT-S backbone — pretrained ImageNet, no classification head
    backbone = DeiTSmallEncoder(
        pretrained=(args.weights_path is None),
        weights_path=args.weights_path,
    )
    model = ProtoNet(encoder=backbone).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history      = []
    best_val_acc = 0.0
    no_improve   = 0
    ckpt_path    = os.path.join(ckpt_dir, cfg['output']['ckpt_name'])

    print(f'\nTraining: {num_epochs} epochs | {n_way}-way {k_shot}-shot | '
          f'q_query={q_query} | lr={lr}')

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion,
                                    device, n_way, k_shot, q_query, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader, optimizer, criterion,
                                    device, n_way, k_shot, val_q_query, train=False)

        record = dict(epoch=epoch,
                      train_loss=tr_loss, train_acc=tr_acc,
                      val_loss=vl_loss,   val_acc=vl_acc)
        history.append(record)
        print(f'Epoch {epoch:3d}/{num_epochs} | '
              f'train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f} | '
              f'val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}')

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
                'backbone':         'deit_small_patch16_224',
                'embedding_dim':    384,
            }, ckpt_path)
            print(f'           -> saved best checkpoint (val_acc={vl_acc:.4f})')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch} '
                      f'(no improvement for {patience} epochs)')
                break

    with open(os.path.join(ckpt_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(ckpt_dir, 'config_used.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    print(f'\nDone. Best val_acc={best_val_acc:.4f}')
    print(f'Checkpoint : {ckpt_path}')


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
