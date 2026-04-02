# predict_full20.py — Inference for FULL supervised baseline (20 classes)

import os
import json
import yaml
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50

# ---------- Load Config ----------
with open("../../../configs/birds20/baseline_resnet_50_fs_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CKPT_DIR = cfg["output"]["ckpt_dir"]  # e.g., checkpoints/baseline/full20
BEST_CKPT = os.path.join(CKPT_DIR, cfg["output"]["save_best_name"])
FINAL_CKPT = os.path.join(CKPT_DIR, cfg["output"]["save_final_name"])
CHECKPOINT_PATH = BEST_CKPT if os.path.exists(BEST_CKPT) else FINAL_CKPT

CLASSES_PATH = os.path.join(CKPT_DIR, "classes.json")
CLASS_TO_IDX_PATH = os.path.join(CKPT_DIR, "class_to_idx.json")

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {BEST_CKPT} or {FINAL_CKPT}")
if not os.path.exists(CLASSES_PATH):
    raise FileNotFoundError(f"classes.json not found: {CLASSES_PATH}")
if not os.path.exists(CLASS_TO_IDX_PATH):
    raise FileNotFoundError(f"class_to_idx.json not found: {CLASS_TO_IDX_PATH}")

# ---------- Load Class Names (order matters) ----------
with open(CLASSES_PATH, "r") as f:
    class_names = json.load(f)

num_classes = len(class_names)

# ---------- Load Model ----------
model = resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ---------- Transform (match evaluation) ----------
image_size = int(cfg["image_size"])
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ---------- Prediction ----------
@torch.inference_mode()
def predict(image_path: str, topk: int = 5):
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(DEVICE)

    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    pred_prob = float(probs[pred_idx])

    print(f"\n✅ Predicted class: {pred_class}  |  Confidence: {pred_prob:.4f}\n")

    topk = min(topk, len(probs))
    top_indices = np.argsort(probs)[::-1][:topk]
    print(f"🔝 Top-{topk} Predictions:")
    for rank, i in enumerate(top_indices, start=1):
        print(f"  {rank}. {class_names[int(i)]}: {probs[int(i)]:.4f}")


if __name__ == "__main__":
    # Example path (change as needed)
    test_image_path = "../../../data/images_to_predict/6.jpg"
    predict(test_image_path, topk=5)
