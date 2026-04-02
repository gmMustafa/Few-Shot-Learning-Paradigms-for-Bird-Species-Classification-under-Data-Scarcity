# evaluate_novel_zeroshot.py
#
# Zero-shot novel evaluation for the supervised baseline.
#
# Important:
# The model head is trained on BASE classes only, so it cannot output NOVEL class labels.
# Therefore, accuracy on novel classes is defined as 0% by design.
#
# This script keeps the evaluation stage explicit and saves a summary.json so it can
# still appear in your comparison tables.
#
# Usage:
#   python evaluate_novel_zeroshot.py --config ../../../configs/birds20/baseline_resnet_50_b15_config.yaml
#   python evaluate_novel_zeroshot.py --config ../../../configs/birds20/baseline_resnet_50_b10_config.yaml

import os
import json
import yaml
import argparse
import torch
from torchvision import datasets, transforms


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


def main():
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

    image_size = int(cfg["image_size"])
    base_classes = cfg["base_classes"]
    novel_classes = cfg["novel_classes"]

    test_dir = os.path.normpath(os.path.join(script_dir, cfg["data_test"]))
    results_dir = os.path.normpath(
        os.path.join(script_dir, cfg["output"]["results_dir"], "novel_zeroshot")
    )
    os.makedirs(results_dir, exist_ok=True)

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_full = datasets.ImageFolder(test_dir, transform=test_transform)
    test_idx, novel_c2i, test_tf = filter_by_classes(test_full, novel_classes)
    test_full.target_transform = test_tf

    num_test_samples = len(test_idx)

    summary = {
        "experiment": cfg["experiment"]["name"],
        "evaluation": "novel_zeroshot",
        "split": f"base{len(base_classes)}_novel{len(novel_classes)}",
        "base_classes": base_classes,
        "novel_classes": novel_classes,
        "num_test_samples": num_test_samples,
        "accuracy": 0.0,
        "accuracy_ci_95": [0.0, 0.0],
        "macro_f1": None,
        "note": (
            "0% by design: the classifier head outputs base-class labels only, "
            "so novel-class labels are outside the model output space."
        ),
    }

    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(results_dir, "summary.txt"), "w") as f:
        f.write("Zero-shot novel evaluation\n")
        f.write("=" * 60 + "\n")
        f.write(f"Experiment       : {cfg['experiment']['name']}\n")
        f.write(f"Base classes     : {len(base_classes)}\n")
        f.write(f"Novel classes    : {len(novel_classes)}\n")
        f.write(f"Novel test imgs  : {num_test_samples}\n")
        f.write("Accuracy         : 0.00%\n")
        f.write("95% CI           : [0.00%, 0.00%]\n")
        f.write(
            "Note             : 0% by design because the base-only classifier head cannot output novel labels.\n"
        )

    print("=" * 60)
    print("ZERO-SHOT NOVEL EVALUATION")
    print(f"Experiment      : {cfg['experiment']['name']}")
    print(f"Base classes    : {len(base_classes)}")
    print(f"Novel classes   : {len(novel_classes)}")
    print(f"Novel test imgs : {num_test_samples}")
    print("Accuracy        : 0.00%")
    print("Reason          : 0% by design (base-class head cannot output novel labels)")
    print(f"Results saved   : {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()