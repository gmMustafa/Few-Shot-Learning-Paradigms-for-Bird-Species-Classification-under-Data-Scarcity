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
with open("../../../config/baseline_resnet_50_fs_config.yaml", "r") as f:
    config = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths (match your B15 training/eval outputs)
CKPT_DIR = "checkpoints/baseline/base15"
CHECKPOINT_PATH = os.path.join(CKPT_DIR, "resnet50_base15_best.pth")  # or final
CLASS_TO_IDX_PATH = os.path.join(CKPT_DIR, "class_to_idx.json")       # saved during training
BASE_CLASSES_PATH = os.path.join(CKPT_DIR, "base_classes.json")       # saved during training

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
if not os.path.exists(CLASS_TO_IDX_PATH):
    raise FileNotFoundError(f"class_to_idx.json not found: {CLASS_TO_IDX_PATH}")
if not os.path.exists(BASE_CLASSES_PATH):
    raise FileNotFoundError(f"base_classes.json not found: {BASE_CLASSES_PATH}")

# ---------- Load Class Mapping ----------
with open(CLASS_TO_IDX_PATH, "r") as f:
    class_to_idx = json.load(f)  # base15 mapping (0..14)

# idx -> class
idx_to_class = {int(v): k for k, v in class_to_idx.items()}

# Optional: enforce fixed ordering from base_classes.json
with open(BASE_CLASSES_PATH, "r") as f:
    base_classes = json.load(f)
num_classes = len(base_classes)

# ---------- Load Model ----------
model = resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ---------- Image Transform (match evaluation) ----------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ---------- Prediction Function ----------
@torch.inference_mode()
def predict(image_path, topk=5):
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(DEVICE)

    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_class = idx_to_class[pred_idx]
    pred_prob = float(probs[pred_idx])

    print(f"\n✅ Predicted class: {pred_class}  |  Confidence: {pred_prob:.4f}\n")

    # Top-k
    topk = min(topk, len(probs))
    top_indices = np.argsort(probs)[::-1][:topk]
    print(f"🔝 Top-{topk} Predictions:")
    for rank, i in enumerate(top_indices, start=1):
        print(f"  {rank}. {idx_to_class[int(i)]}: {probs[int(i)]:.4f}")

    # Full distribution (optional)
    # print("\n🔢 Full Confidence Scores:")
    # for i, p in enumerate(probs):
    #     print(f"  - {idx_to_class[i]}: {p:.4f}")


# ---------- Run ----------
if __name__ == "__main__":
    test_image_path = "../../../data/images to predict/6.jpg"
    predict(test_image_path, topk=5)
