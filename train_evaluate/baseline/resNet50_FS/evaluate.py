# evaluate.py  (Full Supervised Baseline: evaluate on TEST split)

import os
import json
import math
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def wilson_ci(correct: int, total: int, alpha: float = 0.05):
    """Wilson score 95% confidence interval for a proportion."""
    if total == 0:
        return 0.0, 0.0
    from scipy.stats import norm
    z   = norm.ppf(1 - alpha / 2)
    p   = correct / total
    denom  = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = np.argmax(probs, axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds)
        y_prob.extend(probs)
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def save_tsne_baseline(model, loader, class_names, device, save_path, title):
    """Extract ResNet avgpool features and save a t-SNE plot of the test set."""
    from sklearn.manifold import TSNE
    model.eval()
    features, labels = [], []
    captured = []

    def _hook(module, inp, out):
        captured.append(out.detach().cpu().flatten(1).numpy())

    handle = model.avgpool.register_forward_hook(_hook)
    with torch.inference_mode():
        for images, lbls in loader:
            captured.clear()
            _ = model(images.to(device))
            features.append(captured[0])
            labels.extend(lbls.numpy())
    handle.remove()

    X = np.vstack(features)
    y = np.array(labels)
    perplexity = min(30, max(5, len(y) // 4))
    X_2d = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                max_iter=1000, init='pca').fit_transform(X)

    palette = plt.cm.get_cmap("gist_ncar", len(class_names))
    fig, ax = plt.subplots(figsize=(14, 10))
    for ci, cls in enumerate(class_names):
        mask = y == ci
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   color=palette(ci),
                   label=cls.replace('ABYSSINIAN GROUND ', 'ABY. GND '),
                   s=90, alpha=0.85, edgecolors='white', linewidths=0.5)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7,
              framealpha=0.9, title='Class', title_fontsize=8)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=10)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=10)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"t-SNE saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None,
                        help="Path to config YAML. Defaults to configs/birds20/baseline_resnet_50_fs_config.yaml.")
    args = parser.parse_args()

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    if args.config:
        config_path = os.path.normpath(os.path.join(script_dir, args.config))
    else:
        config_path = os.path.normpath(os.path.join(script_dir, "../../../configs/birds20/baseline_resnet_50_fs_config.yaml"))
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("DEVICE INFO")
    print(f"  CUDA available : {torch.cuda.is_available()}")
    print(f"  Device         : {device}")
    if torch.cuda.is_available():
        print(f"  GPU name       : {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  WARNING: Running on CPU")
    print("=" * 60)

    batch_size  = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"]["num_workers"])
    image_size  = int(cfg["image_size"])
    seed        = int(cfg["train"]["seed"])
    lr          = float(cfg["train"]["lr"])

    test_dir = os.path.normpath(os.path.join(script_dir, cfg["data_test"]))
    ckpt_dir = os.path.normpath(os.path.join(script_dir, cfg["output"]["ckpt_dir"]))

    best_ckpt  = os.path.join(ckpt_dir, cfg["output"]["save_best_name"])
    final_ckpt = os.path.join(ckpt_dir, cfg["output"]["save_final_name"])
    ckpt_path  = best_ckpt if os.path.exists(best_ckpt) else final_ckpt
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at:\n  {best_ckpt}\n  {final_ckpt}")

    # ---------- Checkpoint ----------
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "classes" in ckpt:
        class_names    = ckpt["classes"]
        num_classes    = ckpt["num_classes"]
        epochs_trained = ckpt.get("epoch", "unknown")
        val_acc_best   = ckpt.get("val_acc", None)
    else:
        tmp_ds         = datasets.ImageFolder(test_dir)
        class_names    = tmp_ds.classes
        num_classes    = len(class_names)
        epochs_trained = "unknown"
        val_acc_best   = None

    # ---------- Test dataset ----------
    eval_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tf)
    assert test_ds.classes == class_names, (
        "Test folder class order does not match checkpoint. Results would be wrong."
    )

    pin = device.type == "cuda"
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )

    # ---------- Model ----------
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    y_true, y_pred, y_prob = predict(model, test_loader, device)

    # ---------- Results directory ----------
    results_dir = os.path.normpath(os.path.join(script_dir, cfg["output"]["results_dir"]))
    os.makedirs(results_dir, exist_ok=True)

    report   = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    accuracy = report.get("accuracy", 0.0)
    n_total  = len(y_true)
    n_correct = int(round(accuracy * n_total))
    ci_low, ci_high = wilson_ci(n_correct, n_total)
    macro_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')

    # Classification report CSV
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(results_dir, "classification_report.csv"))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ticks = np.arange(num_classes)
    ax.set_xticks(ticks); ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(ticks); ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(
        f"Confusion Matrix — full20 | test\n"
        f"Accuracy: {accuracy * 100:.1f}% (95% CI: {ci_low*100:.1f}%–{ci_high*100:.1f}%)"
    )
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=200)
    plt.close(fig)

    # Per-class F1 bar chart
    classes_in_report = [c for c in class_names if c in report]
    f1_scores = [report[c]["f1-score"] for c in classes_in_report]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(classes_in_report)), f1_scores, color="steelblue")
    ax.set_xticks(range(len(classes_in_report)))
    ax.set_xticklabels(classes_in_report, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("F1-score"); ax.set_title("Per-class F1 — full20 | test")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=np.mean(f1_scores), color="red", linestyle="--", label=f"Mean F1: {np.mean(f1_scores):.3f}")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "f1_per_class.png"), dpi=200)
    plt.close(fig)

    # Learning curve (reads history.json saved by train.py)
    history_path = os.path.join(ckpt_dir, "history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        epochs_axis = [h["epoch"] for h in history]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(epochs_axis, [h["train_loss"] for h in history], label="Train Loss")
        ax1.plot(epochs_axis, [h["val_loss"]   for h in history], label="Val Loss")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
        ax1.set_title("Loss Curve — full20"); ax1.legend(); ax1.grid(True, alpha=0.3)
        ax2.plot(epochs_axis, [h["train_acc"] for h in history], label="Train Acc")
        ax2.plot(epochs_axis, [h["val_acc"]   for h in history], label="Val Acc")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy Curve — full20"); ax2.legend(); ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "learning_curves.png"), dpi=150)
        plt.close(fig)

    # Summary text
    with open(os.path.join(results_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Experiment:     {cfg['experiment']['name']}\n")
        f.write(f"Checkpoint:     {ckpt_path}\n")
        f.write(f"Epochs trained: {epochs_trained}\n")
        f.write(f"Seed:           {seed}\n")
        f.write(f"LR:             {lr}\n")
        f.write(f"Batch size:     {batch_size}\n")
        f.write(f"Num classes:    {num_classes}\n")
        f.write(f"Test images:    {n_total}\n\n")
        f.write(f"Best Val Acc:   {val_acc_best * 100:.2f}%\n" if val_acc_best is not None else "")
        f.write(f"Test Accuracy:  {accuracy * 100:.2f}%\n")
        f.write(f"95% CI:         [{ci_low * 100:.2f}%, {ci_high * 100:.2f}%]\n")
        f.write(f"Macro AUC (OvR):{macro_auc:.4f}\n\n")
        f.write(report_df.to_string())

    # Machine-readable summary JSON (use this for thesis tables)
    summary = {
        "experiment":       cfg["experiment"]["name"],
        "model":            "resnet50",
        "seed":             seed,
        "lr":               lr,
        "batch_size":       batch_size,
        "epochs_trained":   epochs_trained,
        "checkpoint":       ckpt_path,
        "num_classes":      num_classes,
        "test_images":      n_total,
        "val_acc_best":     float(val_acc_best) if val_acc_best is not None else None,
        "accuracy":         float(accuracy),
        "accuracy_ci_95":   [float(ci_low), float(ci_high)],
        "macro_f1":         float(report["macro avg"]["f1-score"]),
        "macro_precision":  float(report["macro avg"]["precision"]),
        "macro_recall":     float(report["macro avg"]["recall"]),
        "weighted_f1":      float(report["weighted avg"]["f1-score"]),
        "macro_auc_ovr":    float(macro_auc),
    }
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # t-SNE feature space visualization
    save_tsne_baseline(
        model, test_loader, class_names, device,
        os.path.join(results_dir, 'tsne_features.png'),
        f'ResNet-50 Feature Space — Full-20 Baseline\n(Test Set, {num_classes} Classes)',
    )

    print(f"\nTest Accuracy: {accuracy * 100:.2f}% (95% CI: {ci_low*100:.2f}%–{ci_high*100:.2f}%)")
    print(f"Macro F1:      {report['macro avg']['f1-score']:.4f}")
    print(f"Macro AUC:     {macro_auc:.4f}")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
