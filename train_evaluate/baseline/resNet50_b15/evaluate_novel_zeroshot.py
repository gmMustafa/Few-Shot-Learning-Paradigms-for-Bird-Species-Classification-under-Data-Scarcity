# evaluate_novel_zeroshot.py
#
# Option A: Run the base-15 model on novel-5 test images WITHOUT any adaptation.
#
# The model has a 15-class output head. When a novel-class image is passed through,
# the model must assign it to one of the 15 base class labels — which is always
# wrong by definition. This shows that a standard CNN trained on base classes
# cannot generalise to novel classes, motivating the need for few-shot learning.
#
# What we report:
#   - Top-1 accuracy on novel images (expected: near 0%, or random ~6.7% for 15 classes)
#   - Which base classes each novel class gets "confused" with (informative)
#   - A confusion-style heatmap: novel classes (rows) vs predicted base classes (cols)

import os
import json
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset


def filter_by_classes(imagefolder_ds: datasets.ImageFolder, allowed_classnames: list):
    """Return indices of samples whose class is in allowed_classnames, no label remap."""
    missing = [c for c in allowed_classnames if c not in imagefolder_ds.class_to_idx]
    if missing:
        raise ValueError(f"Classes missing from dataset folders: {missing}")
    allowed_ids = {imagefolder_ds.class_to_idx[c] for c in allowed_classnames}
    indices = [i for i, (_, y) in enumerate(imagefolder_ds.samples) if y in allowed_ids]
    return indices


@torch.inference_mode()
def predict_with_logits(model, loader, device):
    model.eval()
    y_true_labels, y_pred_ids, all_logits = [], [], []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds  = torch.argmax(logits, dim=1).cpu().numpy()
        y_true_labels.extend(labels.numpy())
        y_pred_ids.extend(preds)
        all_logits.append(logits.cpu().numpy())

    return (
        np.array(y_true_labels),
        np.array(y_pred_ids),
        np.concatenate(all_logits, axis=0),
    )


def main():
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.normpath(os.path.join(script_dir, "../../../configs/baseline_resnet_50_b15_config.yaml"))
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE  = int(cfg["train"]["batch_size"])
    NUM_WORKERS = int(cfg["train"].get("num_workers", 0))
    IMAGE_SIZE  = int(cfg["image_size"])

    base_classes  = cfg.get("base_classes")
    novel_classes = cfg.get("novel_classes")
    if not base_classes or not novel_classes:
        raise ValueError("Config must include `base_classes:` and `novel_classes:` lists.")

    test_dir = os.path.normpath(os.path.join(script_dir, cfg["data_test"]))
    ckpt_dir = os.path.normpath(os.path.join(script_dir, cfg["output"]["ckpt_dir"]))

    best_ckpt  = os.path.join(ckpt_dir, cfg["output"]["save_best_name"])
    final_ckpt = os.path.join(ckpt_dir, cfg["output"]["save_final_name"])
    ckpt_path  = best_ckpt if os.path.exists(best_ckpt) else final_ckpt
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at:\n  {best_ckpt}\n  {final_ckpt}")

    eval_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load the full test folder; filter to only novel class images
    test_full   = datasets.ImageFolder(test_dir, transform=eval_tf)
    novel_idx   = filter_by_classes(test_full, novel_classes)
    novel_subset = Subset(test_full, novel_idx)

    # The raw labels from ImageFolder are 0..19 (all 20 classes sorted).
    # We need to know which raw label corresponds to each novel class.
    raw_c2i       = test_full.class_to_idx          # e.g. {"AMERICAN BITTERN": 15, ...}
    novel_raw_ids = [raw_c2i[c] for c in novel_classes]   # raw label indices for novel classes

    pin = DEVICE.type == "cuda"
    loader = DataLoader(
        novel_subset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin,
        persistent_workers=(NUM_WORKERS > 0),
    )

    # Load base-15 model (15-class head)
    ckpt        = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    num_base    = len(base_classes)
    model       = models.resnet50(weights=None)
    model.fc    = nn.Linear(model.fc.in_features, num_base)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE)

    # raw_labels: original 0..19 labels (ImageFolder sorted order)
    # pred_ids:   predicted base class indices 0..14
    raw_labels, pred_ids, _ = predict_with_logits(model, loader, DEVICE)

    # Map raw_labels back to novel class names
    raw_i2c      = {v: k for k, v in raw_c2i.items()}
    true_names   = [raw_i2c[int(y)] for y in raw_labels]
    pred_names   = [base_classes[int(p)] for p in pred_ids]

    # Build a heatmap: rows = novel classes, cols = predicted base classes
    heatmap = np.zeros((len(novel_classes), len(base_classes)), dtype=int)
    for true_n, pred_n in zip(true_names, pred_names):
        r = novel_classes.index(true_n)
        c = base_classes.index(pred_n)
        heatmap[r, c] += 1

    out_dir = os.path.normpath(
        os.path.join(script_dir, cfg["output"]["results_dir"], "novel_zeroshot")
    )
    os.makedirs(out_dir, exist_ok=True)

    # Heatmap plot
    fig, ax = plt.subplots(figsize=(16, 4))
    im = ax.imshow(heatmap, interpolation="nearest", cmap="Reds")
    fig.colorbar(im, ax=ax, label="Count")
    ax.set_xticks(range(len(base_classes)))
    ax.set_xticklabels(base_classes, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(novel_classes)))
    ax.set_yticklabels(novel_classes, fontsize=8)
    ax.set_xlabel("Predicted Base Class (model has no novel class output)")
    ax.set_ylabel("True Novel Class")
    ax.set_title(
        "Zero-Shot Transfer: base-15 model on novel-5 test images\n"
        "(all predictions are wrong by definition — shows need for FSL)"
    )
    for i in range(len(novel_classes)):
        for j in range(len(base_classes)):
            if heatmap[i, j] > 0:
                ax.text(j, i, str(heatmap[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if heatmap[i, j] > heatmap.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "novel_confusion_heatmap.png"), dpi=200)
    plt.close(fig)

    # Per novel-class report: how many images, which base class was most predicted
    rows = []
    for i, nc in enumerate(novel_classes):
        row_counts = heatmap[i]
        top_pred   = base_classes[int(np.argmax(row_counts))]
        total      = int(row_counts.sum())
        rows.append({
            "novel_class":       nc,
            "total_images":      total,
            "top_predicted_as":  top_pred,
            "top_count":         int(row_counts.max()),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "novel_predictions.csv"), index=False)

    # Overall accuracy is 0 by definition, but compute it to confirm
    # (a prediction is "correct" only if the model predicted the novel class label,
    #  which it cannot since it has only 15 outputs)
    accuracy = 0.0  # definitionally 0 — logged explicitly for clarity

    summary = {
        "experiment":          "base15_novel_zeroshot",
        "checkpoint":          ckpt_path,
        "num_base_classes":    len(base_classes),
        "num_novel_classes":   len(novel_classes),
        "novel_test_images":   int(len(novel_idx)),
        "top1_accuracy":       accuracy,
        "note": (
            "Accuracy is 0 by definition: the model has a 15-class head "
            "and cannot output novel class labels. "
            "The heatmap shows which base classes novel images get assigned to."
        ),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("Option A: Zero-Shot Transfer — base-15 model on novel-5 test images\n")
        f.write("=" * 70 + "\n\n")
        f.write("The model was trained on 15 base classes.\n")
        f.write("It has a 15-class output head.\n")
        f.write("When novel class images are passed through, all predictions are\n")
        f.write("'wrong' — the model assigns each novel image to some base class.\n\n")
        f.write(f"Novel images evaluated: {len(novel_idx)}\n")
        f.write(f"Top-1 accuracy (novel): 0.0% (definitionally 0)\n\n")
        f.write("Per-novel-class breakdown:\n")
        f.write(df.to_string(index=False))

    print(f"\nOption A (zero-shot transfer) complete.")
    print(f"Novel images: {len(novel_idx)}  |  Top-1 accuracy: 0.0% (expected)")
    print(f"\nPer-novel-class most-predicted base class:")
    for _, row in df.iterrows():
        print(f"  {row['novel_class']:30s} -> '{row['top_predicted_as']}' ({row['top_count']}/{row['total_images']} images)")
    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
