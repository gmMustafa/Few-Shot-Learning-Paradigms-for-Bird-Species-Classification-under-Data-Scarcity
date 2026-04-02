import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import yaml
import os
import numpy as np

from models.protonet import ProtoNet

# ---------- Load Config ----------
with open('../../configs/birds20/proto_configs.yaml', 'r') as file:
    config = yaml.safe_load(file)

DEVICE = config["device"] if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = config["image_size"]
DATASET_PATH = config["dataset_path"]

# ---------- Class Map ----------
from utils.dataset import load_datasets

_, _, test_ds = load_datasets(DATASET_PATH, image_size=IMAGE_SIZE)
idx_to_class = {v: k for k, v in test_ds.class_to_idx.items()}

# ---------- Load Model ----------
backbone = resnet18(weights=None)
backbone.fc = nn.Identity()
model = ProtoNet(encoder=backbone).to(DEVICE)
model.load_state_dict(torch.load("checkpoints/protonet.pth", map_location=DEVICE))
model.eval()

# ---------- Image Transform ----------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# ---------- Load Your Image ----------
def predict(image_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Sample a support set for prediction (randomly from test set)
    support_set = []
    support_labels = []
    class_counter = {}

    for path, label in test_ds.samples:
        if class_counter.get(label, 0) >= 1:
            continue
        img = Image.open(path).convert("RGB")
        img_tensor = transform(img).to(DEVICE)
        support_set.append(img_tensor)
        support_labels.append(label)
        class_counter[label] = 1
        if len(class_counter) >= config["n_way"]:
            break

    support_set = torch.stack(support_set).to(DEVICE)
    support_labels_tensor = torch.tensor(support_labels).to(DEVICE)

    # Run prediction
    with torch.no_grad():
        logits = model(support_set, support_labels_tensor, image_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
        pred_idx = np.argmax(probs)
        pred_label = support_labels[pred_idx]
        class_name = idx_to_class[pred_label]

        print(f"\n✅ Predicted class: {class_name}\n")
        print("🔢 Prediction Confidence Scores:")
        for i, prob in enumerate(probs):
            label_name = idx_to_class[support_labels[i]]
            print(f"  - {label_name}: {prob:.4f}")


# ---------- Run ----------
if __name__ == "__main__":
    test_image_path = "../../data/images to predict/6.jpg"  # input("Enter image path: ")
    predict(test_image_path)
