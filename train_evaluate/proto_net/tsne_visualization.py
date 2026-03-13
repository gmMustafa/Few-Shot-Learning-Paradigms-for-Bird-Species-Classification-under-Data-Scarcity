import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import yaml
import numpy as np
import os
from PIL import Image
from utils.dataset import load_datasets
from utils.sampler import FewShotSampler
from models.protonet import ProtoNet

# ---------- Load Config ----------
with open('../../configs/proto_configs.yaml', 'r') as file:
    config = yaml.safe_load(file)

DEVICE = config["device"] if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = config["image_size"]
N_WAY = config["n_way"]
K_SHOT = config["k_shot"]
Q_QUERY = config["q_query"]
DATASET_PATH = config["dataset_path"]

# ---------- Load Dataset ----------
_, _, test_ds = load_datasets(DATASET_PATH, image_size=IMAGE_SIZE)

# Extract class index to name mapping
idx_to_class = {v: k for k, v in test_ds.class_to_idx.items()}

# ---------- Load Model ----------
backbone = resnet18(weights=None)
backbone.fc = nn.Identity()
model = ProtoNet(encoder=backbone).to(DEVICE)
model.load_state_dict(torch.load("checkpoints/protonet.pth", map_location=DEVICE))
model.eval()

# ---------- Create Output Folder ----------
os.makedirs("../../results/proto_net/tsne_plots", exist_ok=True)

# ---------- Generate Plots for Multiple Episodes ----------
EPISODES = 5
for episode_idx in range(EPISODES):
    # Create one episode per loop
    sampler = FewShotSampler([label for _, label in test_ds.samples],
                             n_way=N_WAY, k_shot=K_SHOT, q_query=Q_QUERY,
                             episodes_per_epoch=1)
    test_loader = DataLoader(test_ds, batch_sampler=sampler)

    with torch.no_grad():
        for episode in test_loader:
            images, labels = [x.to(DEVICE) for x in episode]

            support, query = [], []
            support_labels, query_labels = [], []

            class_ids = torch.unique(labels)
            for cls_id in class_ids:
                idxs = (labels == cls_id).nonzero(as_tuple=True)[0]
                support.extend(images[idxs[:K_SHOT]])
                query.extend(images[idxs[K_SHOT:]])
                support_labels.extend([cls_id] * K_SHOT)
                query_labels.extend([cls_id] * Q_QUERY)

            support = torch.stack(support)
            query = torch.stack(query)

            all_embeddings = torch.cat([
                model.encoder(support),
                model.encoder(query)
            ], dim=0).cpu().numpy()

            all_labels = support_labels + query_labels
            label_map = {old.item(): new for new, old in enumerate(torch.unique(torch.tensor(all_labels)))}
            all_labels = np.array([label_map[lab.item()] for lab in all_labels])
            all_names = [idx_to_class[lab.item()] for lab in support_labels + query_labels]

    # ---------- Apply t-SNE ----------
    tsne = TSNE(n_components=2, perplexity=5, random_state=episode_idx)
    tsne_results = tsne.fit_transform(all_embeddings)

    # ---------- Plot ----------
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(all_labels)
    for label in unique_labels:
        idxs = np.where(all_labels == label)
        name = idx_to_class[list(label_map.keys())[list(label_map.values()).index(label)]]
        plt.scatter(tsne_results[idxs, 0], tsne_results[idxs, 1], label=name)

    plt.legend()
    plt.title(f"t-SNE of Episode {episode_idx+1}")
    plt.grid(True)
    plt.tight_layout()
    out_path = f"../../results/proto_net/tsne_plots/tsne_episode_{episode_idx+1}.png"
    plt.savefig(out_path)
    plt.close()

    # Optional: open image
    # Image.open(out_path).show()
