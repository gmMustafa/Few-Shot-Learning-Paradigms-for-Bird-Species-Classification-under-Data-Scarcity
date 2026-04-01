"""
plots/tsne_plot.py — t-SNE embedding visualization for FSL backbone.

Shows how novel class images cluster in the ResNet-50 feature space.
Generates a side-by-side comparison:
  LEFT  : Embeddings from ImageNet pretrained ResNet-50 (no FSL training)
  RIGHT : Embeddings from the trained ProtoNet backbone

This directly visualizes what episodic training achieves — tighter,
more separable clusters for novel classes.

Additionally generates:
  - A single plot of the trained model embeddings with prototypes marked
  - Saved to results/proto_net/tsne/

Usage:
    python plots/tsne_plot.py
    python plots/tsne_plot.py --no-pretrained   # skip the before/after comparison

Requirements: scikit-learn (already installed via sklearn)
"""
import os
import sys
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['figure.facecolor']  = 'white'
plt.rcParams['axes.facecolor']    = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
from sklearn.manifold import TSNE
from torchvision.models import resnet50, ResNet50_Weights
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.dataset import FilteredDataset, get_eval_transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Colour palette — supports up to 10 novel classes
NOVEL_COLORS = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
    '#a65628', '#f781bf', '#999999', '#66c2a5', '#fc8d62',
]


def get_colors(n):
    """Return n colours, cycling if needed."""
    return [NOVEL_COLORS[i % len(NOVEL_COLORS)] for i in range(n)]


def short_name(cls):
    """Shorten a class label for legend display."""
    return (cls.replace('AMERICAN ', 'AM. ')
               .replace('ALEXANDRINE ', 'ALEX. ')
               .replace('ABYSSINIAN ', 'ABYSS. ')
               .replace('ALTAMIRA ', 'ALTA. ')
               .replace('ALBERTS ', 'ALB. ')
               .replace('ALPINE ', 'ALP. '))


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_novel_images(data_root, novel_classes, image_size=224, max_per_class=None):
    """
    Load images from BOTH novel train and test sets.
    Returns:
        images : list of tensors
        labels : list of int (0..4)
        splits : list of str ('train' or 'test')
    """
    transform = get_eval_transforms(image_size)
    images, labels, splits = [], [], []

    for split in ['train', 'test']:
        split_dir = os.path.join(data_root, split)
        if not os.path.exists(split_dir):
            # try 'valid' for val split
            if split == 'test':
                continue
        try:
            ds = FilteredDataset(split_dir, novel_classes, transform=transform)
        except Exception:
            continue

        by_class = ds.get_class_to_indices()
        for cls_idx in range(len(novel_classes)):
            idxs = by_class.get(cls_idx, [])
            if max_per_class:
                idxs = idxs[:max_per_class]
            for i in idxs:
                img, lbl = ds[i]
                images.append(img)
                labels.append(cls_idx)
                splits.append(split)

    return images, labels, splits


def extract_embeddings(backbone, images, device, batch_size=16):
    """Run images through backbone in batches, return numpy embeddings."""
    backbone.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = torch.stack(images[i: i + batch_size]).to(device)
            emb   = backbone(batch).cpu().numpy()
            embeddings.append(emb)
    return np.vstack(embeddings)


def compute_prototypes(embeddings, labels, n_way):
    """Compute class prototype = mean of all embeddings per class."""
    D = embeddings.shape[1]
    protos = np.zeros((n_way, D))
    for c in range(n_way):
        mask = (np.array(labels) == c)
        protos[c] = embeddings[mask].mean(axis=0)
    return protos


def plot_tsne(ax, tsne_emb, labels, splits, proto_2d=None,
              novel_classes=None, title=''):
    """Plot t-SNE scatter on ax. proto_2d: [n_way, 2] if provided."""
    n_way  = len(novel_classes) if novel_classes else max(labels) + 1
    colors = get_colors(n_way)
    shorts = [short_name(c) for c in novel_classes] if novel_classes else [str(i) for i in range(n_way)]

    for cls_idx in range(n_way):
        mask_train = [i for i, (l, s) in enumerate(zip(labels, splits))
                      if l == cls_idx and s == 'train']
        mask_test  = [i for i, (l, s) in enumerate(zip(labels, splits))
                      if l == cls_idx and s == 'test']

        color = colors[cls_idx]
        if mask_train:
            ax.scatter(tsne_emb[mask_train, 0], tsne_emb[mask_train, 1],
                       c=color, marker='o', s=18, alpha=0.4, linewidths=0)
        if mask_test:
            ax.scatter(tsne_emb[mask_test, 0], tsne_emb[mask_test, 1],
                       c=color, marker='*', s=80, alpha=0.9,
                       edgecolors='black', linewidths=0.4,
                       label=shorts[cls_idx])

    if proto_2d is not None:
        for cls_idx in range(n_way):
            ax.scatter(proto_2d[cls_idx, 0], proto_2d[cls_idx, 1],
                       c=colors[cls_idx], marker='P',
                       s=200, edgecolors='black', linewidths=1.5,
                       zorder=5, label='_nolegend_' if cls_idx > 0 else 'Prototype')

    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc='lower right', fontsize=7, markerscale=1.2)


MODEL_DIR = {
    'protonet':        'proto_net',
    'matchingnet':     'matchingnet',
    'matchingnet-fce': 'matchingnet',
    'relationnet':     'relationnet',
    'vit-protonet':    'vit_backbone/proto_net',
}

MODEL_LABEL = {
    'protonet':        'ProtoNet',
    'matchingnet':     'MatchingNet',
    'matchingnet-fce': 'MatchingNet+FCE',
    'relationnet':     'RelationNet',
    'vit-protonet':    'ProtoNet (DeiT-S)',
}

BACKBONE_NAME = {
    'protonet':        'ResNet-50',
    'matchingnet':     'ResNet-50',
    'matchingnet-fce': 'ResNet-50',
    'relationnet':     'ResNet-50',
    'vit-protonet':    'DeiT-S',
}

DEFAULT_CONFIG = {
    'protonet':        'configs/proto_configs.yaml',
    'matchingnet':     'configs/matchingnet_configs.yaml',
    'matchingnet-fce': 'configs/matchingnet_fce_configs.yaml',
    'relationnet':     'configs/relationnet_configs.yaml',
    'vit-protonet':    'configs/vit_proto_configs.yaml',
}


def _get_cfg_value(cfg, *keys):
    """Traverse nested dict with fallback keys (handles flat and nested configs)."""
    for key in keys:
        parts = key.split('.')
        v = cfg
        try:
            for p in parts:
                v = v[p]
            return v
        except (KeyError, TypeError):
            continue
    return None


def build_backbone(model_name, ckpt_path, device):
    """
    Build the backbone for t-SNE extraction and optionally load FSL checkpoint.
    Returns (trained_backbone, pretrained_backbone_factory_fn).
    pretrained_backbone_factory_fn() builds a fresh pretrained-only backbone.
    """
    is_vit = model_name == 'vit-protonet'

    if is_vit:
        from models.vit import DeiTSmallEncoder
        # Auto-detect local weights to avoid HuggingFace download on HPC
        _candidates = [
            os.path.join(ROOT, 'pretrained_weights', 'deit_small_patch16_224.pth'),
            os.path.join(ROOT, 'train_evaluate', 'vit_backbone', 'deit_small_patch16_224.pth'),
        ]
        _local_weights = next((p for p in _candidates if os.path.isfile(p)), None)

        trained_backbone = DeiTSmallEncoder(
            pretrained=(_local_weights is None),
            weights_path=_local_weights,
        )
        if _local_weights:
            print(f'[auto] DeiT-S init weights: {_local_weights}')

        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            encoder_state = {k.replace('encoder.model.', ''): v
                             for k, v in ckpt['model_state_dict'].items()
                             if k.startswith('encoder.')}
            trained_backbone.model.load_state_dict(encoder_state, strict=False)

        def pretrained_factory():
            return DeiTSmallEncoder(
                pretrained=(_local_weights is None),
                weights_path=_local_weights,
            )

    else:
        trained_backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        trained_backbone.fc = nn.Identity()

        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            encoder_state = {k.replace('encoder.', ''): v
                             for k, v in ckpt['model_state_dict'].items()
                             if k.startswith('encoder.')}
            trained_backbone.load_state_dict(encoder_state, strict=False)

        def pretrained_factory():
            m = resnet50(weights=ResNet50_Weights.DEFAULT)
            m.fc = nn.Identity()
            return m

    return trained_backbone, pretrained_factory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='protonet',
                        choices=['protonet', 'matchingnet', 'matchingnet-fce',
                                 'relationnet', 'vit-protonet'],
                        help='Which FSL model to visualise (default: protonet)')
    parser.add_argument('--config', default=None,
                        help='Path to config YAML (relative to project root). '
                             'Defaults to the standard config for --model.')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Skip the before/after comparison, show trained only')
    parser.add_argument('--max-train', type=int, default=30,
                        help='Max train images per class to embed (default 30)')
    args = parser.parse_args()

    if args.config is None:
        args.config = DEFAULT_CONFIG[args.model]

    set_seed(42)
    device = torch.device('cpu')

    cfg_path = os.path.join(ROOT, args.config)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    novel_classes = cfg['novel_classes']
    image_size    = int(cfg.get('image_size', 224))
    data_root     = os.path.join(ROOT, 'data')
    model_subdir  = MODEL_DIR[args.model]
    model_label   = MODEL_LABEL[args.model]
    backbone_name = BACKBONE_NAME[args.model]

    # Config key layout differs: ResNet-50 uses flat keys, ViT uses nested data.*
    ckpt_dir  = _get_cfg_value(cfg, 'output.ckpt_dir')
    ckpt_name = _get_cfg_value(cfg, 'output.ckpt_name')
    res_dir_rel = _get_cfg_value(cfg, 'output.results_dir')

    ckpt_path = os.path.normpath(
        os.path.join(ROOT, 'train_evaluate', model_subdir, ckpt_dir, ckpt_name))

    _train_dir = os.path.join(ROOT, 'train_evaluate', model_subdir)
    _res_dir   = os.path.normpath(os.path.join(_train_dir, res_dir_rel))
    out_dir    = os.path.join(_res_dir, 'tsne')
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading images for {len(novel_classes)} novel classes...")
    images, labels, splits = load_novel_images(
        data_root, novel_classes, image_size,
        max_per_class=args.max_train)

    if not images:
        print("ERROR: No novel class images found. Check data paths.")
        return

    n_test  = sum(1 for s in splits if s == 'test')
    n_train = sum(1 for s in splits if s == 'train')
    print(f"Loaded {len(images)} images  ({n_train} train, {n_test} test)")

    # -------------------------------------------------------------------------
    # TRAINED backbone
    # -------------------------------------------------------------------------
    if not os.path.exists(ckpt_path):
        print(f"WARNING: Checkpoint not found at {ckpt_path}")
        print("         Run train.py first. Showing pretrained-only plot.")
        args.no_pretrained = False

    trained_backbone, pretrained_factory = build_backbone(args.model, ckpt_path, device)
    if os.path.exists(ckpt_path):
        print(f"Loaded {model_label} checkpoint")
    trained_backbone = trained_backbone.to(device)

    print("Extracting trained embeddings...")
    trained_emb = extract_embeddings(trained_backbone, images, device)
    protos_trained = compute_prototypes(trained_emb, labels, len(novel_classes))

    # -------------------------------------------------------------------------
    # PRETRAINED backbone (ImageNet only, no FSL training)
    # -------------------------------------------------------------------------
    pretrained_emb = None
    if not args.no_pretrained:
        print(f"Extracting pretrained (ImageNet-only) embeddings ({backbone_name})...")
        pretrained_backbone = pretrained_factory().to(device)
        pretrained_emb = extract_embeddings(pretrained_backbone, images, device)

    # -------------------------------------------------------------------------
    # t-SNE
    # -------------------------------------------------------------------------
    tsne = TSNE(n_components=2, perplexity=min(30, len(images) - 1),
                random_state=42, max_iter=1000)

    print("Running t-SNE on trained embeddings...")
    all_for_tsne = trained_emb
    if pretrained_emb is not None:
        all_for_tsne = np.vstack([trained_emb, pretrained_emb])

    tsne_all = tsne.fit_transform(all_for_tsne)
    n = len(images)
    tsne_trained    = tsne_all[:n]
    tsne_pretrained = tsne_all[n:] if pretrained_emb is not None else None

    # Project prototypes into t-SNE space (nearest embedding approximation)
    proto_2d = np.zeros((len(novel_classes), 2))
    for c in range(len(novel_classes)):
        dists = np.linalg.norm(trained_emb - protos_trained[c], axis=1)
        nearest = np.argmin(dists)
        proto_2d[c] = tsne_trained[nearest]

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    if pretrained_emb is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        plot_tsne(ax1, tsne_pretrained, labels, splits,
                  proto_2d=None,
                  novel_classes=novel_classes,
                  title=f'Before FSL Training\n(ImageNet pretrained {backbone_name})')

        plot_tsne(ax2, tsne_trained, labels, splits,
                  proto_2d=proto_2d,
                  novel_classes=novel_classes,
                  title=f'After {model_label} Training\n({backbone_name} fine-tuned episodically)')

        train_patch = mpatches.Patch(color='gray', alpha=0.4, label='Train images (●)')
        test_patch  = mpatches.Patch(color='gray', alpha=0.9, label='Test images (★)')
        proto_patch = mpatches.Patch(facecolor='white', edgecolor='black',
                                     label='Prototype (✚)')
        fig.legend(handles=[train_patch, test_patch, proto_patch],
                   loc='lower center', ncol=3, fontsize=9,
                   bbox_to_anchor=(0.5, -0.02))

        plt.suptitle(f't-SNE of Novel Class Embeddings — Before vs After {model_label} Training\n'
                     f'({len(novel_classes)} novel bird species, {backbone_name} backbone)',
                     fontsize=12, y=1.01)
        plt.tight_layout()
        out_path = os.path.join(out_dir, 'tsne_before_after.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")

    # Single trained-only plot
    fig, ax = plt.subplots(figsize=(8, 7))
    plot_tsne(ax, tsne_trained, labels, splits,
              proto_2d=proto_2d,
              novel_classes=novel_classes,
              title=f't-SNE of Novel Class Embeddings\n'
                    f'({model_label}, {backbone_name}, ● = train, ★ = test, ✚ = prototype)')

    train_line = plt.Line2D([0], [0], marker='o', color='gray',
                             markersize=6, alpha=0.5, label='Train images', linestyle='None')
    test_line  = plt.Line2D([0], [0], marker='*', color='gray',
                             markersize=9, label='Test images', linestyle='None')
    proto_line = plt.Line2D([0], [0], marker='P', color='gray',
                             markersize=10, label='Prototype', linestyle='None',
                             markeredgecolor='black')
    _colors = get_colors(len(novel_classes))
    class_patches = [mpatches.Patch(color=_colors[i],
                                    label=short_name(novel_classes[i]))
                     for i in range(len(novel_classes))]
    ax.legend(handles=class_patches + [train_line, test_line, proto_line],
              loc='lower right', fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'tsne_trained.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
