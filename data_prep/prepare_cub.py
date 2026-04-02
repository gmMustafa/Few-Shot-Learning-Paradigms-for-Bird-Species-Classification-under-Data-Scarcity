"""
data_prep/prepare_cub.py

Reorganizes CUB-200-2011 raw data into an ImageFolder-compatible directory
structure for use with the FSL pipeline (same layout as Birds20 data/).

Source  : data_cub-200-2011/CUB_200_2011/   (original download, untouched)
Output  : data_cub_prepared/                 (derived, project root)

Output layout:
  data_cub_prepared/
    train/ClassName/img.jpg   official train split, minus val holdout
    valid/ClassName/img.jpg   val holdout (--val-per-class images, sorted last)
    test/ClassName/img.jpg    official test split
    metadata/
      class_list.json          all 200 class names, ordered by class_id
      split_info.json          per-class counts for train/valid/test
      fsl_split_standard.json  base / val_fsl / novel class assignments

FSL split protocol (Chen et al. ECCV 2019 / Tian et al. ECCV 2020 benchmark):
  Base   classes : IDs   1–100   (used for episodic training)
  ValFSL classes : IDs 101–150   (reserved; not used during training in this pipeline)
  Novel  classes : IDs 151–200   (used for few-shot evaluation, 5-way episodes)

Class name convention: strip the leading "NNN." numeric prefix.
  001.Black_footed_Albatross  →  Black_footed_Albatross

Usage (run from project root):
    python data_prep/prepare_cub.py
    python data_prep/prepare_cub.py --cub-root data_cub-200-2011/CUB_200_2011
    python data_prep/prepare_cub.py --output data_cub_prepared --val-per-class 5 --seed 42
"""

import os
import sys
import json
import shutil
import argparse
from collections import defaultdict


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_class_prefix(raw_name: str) -> str:
    """Strip leading 'NNN.' from a CUB class folder name."""
    parts = raw_name.split('.', 1)
    return parts[1] if len(parts) == 2 and parts[0].isdigit() else raw_name


def read_metadata(cub_root: str):
    """
    Parse all CUB metadata files.

    Returns:
        classes  : {class_id(int) -> raw_name(str)}
        images   : {image_id(int) -> relative_path(str)}
        labels   : {image_id(int) -> class_id(int)}
        splits   : {image_id(int) -> is_train(int)}  1=train, 0=test
    """
    classes = {}
    with open(os.path.join(cub_root, 'classes.txt')) as f:
        for line in f:
            cid, name = line.strip().split(' ', 1)
            classes[int(cid)] = name.strip()

    images = {}
    with open(os.path.join(cub_root, 'images.txt')) as f:
        for line in f:
            iid, path = line.strip().split(' ', 1)
            images[int(iid)] = path.strip()

    labels = {}
    with open(os.path.join(cub_root, 'image_class_labels.txt')) as f:
        for line in f:
            parts = line.strip().split()
            labels[int(parts[0])] = int(parts[1])

    splits = {}
    with open(os.path.join(cub_root, 'train_test_split.txt')) as f:
        for line in f:
            parts = line.strip().split()
            splits[int(parts[0])] = int(parts[1])

    return classes, images, labels, splits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Prepare CUB-200-2011 for the few-shot learning pipeline'
    )
    parser.add_argument(
        '--cub-root',
        default='data_cub-200-2011/CUB_200_2011',
        help='Path to CUB_200_2011 folder (relative to project root or absolute)',
    )
    parser.add_argument(
        '--output',
        default='data_cub_prepared',
        help='Output directory (relative to project root or absolute)',
    )
    parser.add_argument(
        '--val-per-class',
        type=int,
        default=5,
        help='Number of train-split images to hold out as validation per class',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (currently unused; holdout is deterministic via sort)',
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    cub_root = args.cub_root if os.path.isabs(args.cub_root) else \
               os.path.normpath(os.path.join(PROJECT_ROOT, args.cub_root))
    out_root = args.output if os.path.isabs(args.output) else \
               os.path.normpath(os.path.join(PROJECT_ROOT, args.output))

    if not os.path.isdir(cub_root):
        print(f"ERROR: CUB root not found: {cub_root}")
        sys.exit(1)

    print(f"CUB source  : {cub_root}")
    print(f"Output root : {out_root}")
    print(f"Val holdout : last {args.val_per_class} images/class (sorted by filename)")
    print()

    classes, images, labels, splits = read_metadata(cub_root)

    # Group image IDs by class, then by official train/test split
    class_train = defaultdict(list)  # class_id -> [image_ids in official train]
    class_test  = defaultdict(list)  # class_id -> [image_ids in official test]
    for iid, cid in labels.items():
        if splits[iid] == 1:
            class_train[cid].append(iid)
        else:
            class_test[cid].append(iid)

    # Sort by filename for deterministic, reproducible ordering
    for cid in class_train:
        class_train[cid].sort(key=lambda x: images[x])
    for cid in class_test:
        class_test[cid].sort(key=lambda x: images[x])

    split_info = {}
    copied = skipped = 0

    for cid in sorted(classes.keys()):
        class_name = strip_class_prefix(classes[cid])
        train_ids  = list(class_train[cid])
        test_ids   = list(class_test[cid])

        if len(train_ids) < args.val_per_class + 1:
            print(f"  WARNING: class {class_name} has only {len(train_ids)} train images, "
                  f"cannot hold out {args.val_per_class} for val. Skipping val holdout.")
            val_ids = []
        else:
            # Hold out the last N (sorted) as validation; rest become train
            val_ids   = train_ids[-args.val_per_class:]
            train_ids = train_ids[:-args.val_per_class]

        split_info[class_name] = {
            'class_id' : cid,
            'raw_name' : classes[cid],
            'n_train'  : len(train_ids),
            'n_valid'  : len(val_ids),
            'n_test'   : len(test_ids),
        }

        for subset, ids in [('train', train_ids), ('valid', val_ids), ('test', test_ids)]:
            dest_dir = os.path.join(out_root, subset, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            for iid in ids:
                src = os.path.normpath(
                    os.path.join(cub_root, 'images', images[iid]))
                dst = os.path.join(dest_dir, os.path.basename(images[iid]))
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    copied += 1
                else:
                    skipped += 1

    # ---- metadata ----
    meta_dir = os.path.join(out_root, 'metadata')
    os.makedirs(meta_dir, exist_ok=True)

    # Ordered list of all 200 stripped class names (by class_id)
    class_list = [strip_class_prefix(classes[cid]) for cid in sorted(classes.keys())]
    with open(os.path.join(meta_dir, 'class_list.json'), 'w') as f:
        json.dump(class_list, f, indent=2)

    with open(os.path.join(meta_dir, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)

    # Standard FSL split (classes 1-100 base, 101-150 val_fsl, 151-200 novel)
    base_classes    = [strip_class_prefix(classes[cid]) for cid in range(1,   101)]
    val_fsl_classes = [strip_class_prefix(classes[cid]) for cid in range(101, 151)]
    novel_classes   = [strip_class_prefix(classes[cid]) for cid in range(151, 201)]

    fsl_split = {
        'protocol'        : 'standard_cub_fsl',
        'reference'       : 'Chen et al. ECCV 2019 / Tian et al. ECCV 2020',
        'note'            : (
            'Base IDs 1-100 used for episodic training. '
            'ValFSL IDs 101-150 reserved (not used during training in this pipeline). '
            'Novel IDs 151-200 used for 5-way K-shot evaluation.'
        ),
        'seed'            : args.seed,
        'n_base'          : len(base_classes),
        'n_val_fsl'       : len(val_fsl_classes),
        'n_novel'         : len(novel_classes),
        'base_classes'    : base_classes,
        'val_fsl_classes' : val_fsl_classes,
        'novel_classes'   : novel_classes,
    }
    with open(os.path.join(meta_dir, 'fsl_split_standard.json'), 'w') as f:
        json.dump(fsl_split, f, indent=2)

    # ---- summary ----
    train_counts = [split_info[c]['n_train'] for c in split_info]
    val_counts   = [split_info[c]['n_valid'] for c in split_info]
    test_counts  = [split_info[c]['n_test']  for c in split_info]

    print("Dataset preparation complete.")
    print(f"  Copied  : {copied} files")
    print(f"  Skipped : {skipped} (already existed)")
    print(f"  Classes : {len(class_list)}")
    print(f"  Train/class : {min(train_counts)}–{max(train_counts)} images")
    print(f"  Valid/class : {min(val_counts)}–{max(val_counts)} images")
    print(f"  Test/class  : {min(test_counts)}–{max(test_counts)} images")
    print()
    print(f"Metadata saved to: {meta_dir}/")
    print(f"  class_list.json         ({len(class_list)} classes)")
    print(f"  split_info.json")
    print(f"  fsl_split_standard.json")
    print(f"    Base    : {len(base_classes)} classes (IDs 1–100)")
    print(f"    ValFSL  : {len(val_fsl_classes)} classes (IDs 101–150)")
    print(f"    Novel   : {len(novel_classes)} classes (IDs 151–200)")
    print()
    print("Next steps:")
    print("  1. Train supervised baseline:")
    print("     cd train_evaluate/baseline/resNet50_bX")
    print("     python train.py --config ../../../configs/cub/baseline_resnet50_base100.yaml")
    print("  2. Run NCC evaluation:")
    print("     python evaluate_novel_ncc.py --config ../../../configs/cub/baseline_resnet50_base100.yaml --k-shot 5")
    print("  3. Train ProtoNet:")
    print("     cd train_evaluate/proto_net")
    print("     python train.py --config ../../configs/cub/protonet_resnet50.yaml")


if __name__ == '__main__':
    main()
