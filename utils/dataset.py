import os
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(image_size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# Kept for backward compatibility with baseline scripts
def get_transforms(image_size=224):
    return get_eval_transforms(image_size)


def load_datasets(data_dir='data', image_size=224):
    transform = get_eval_transforms(image_size)
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
    valid_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'valid'), transform=transform)
    test_dataset  = datasets.ImageFolder(root=os.path.join(data_dir, 'test'),  transform=transform)
    return train_dataset, valid_dataset, test_dataset


def get_dataloaders(train_dataset, valid_dataset, test_dataset, batch_size=16, num_workers=0):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader, test_loader


class FilteredDataset(Dataset):
    """
    Loads images from an ImageFolder-style directory but restricts to a
    specified list of class names and remaps labels to 0..len(allowed_classes)-1.

    Used to enforce strict base/novel class splits in FSL experiments.
    Prevents any class-order dependency (labels are determined by allowed_classes
    order, not by folder sort order).
    """

    def __init__(self, root, allowed_classes, transform=None):
        """
        Args:
            root:            Path to an ImageFolder-style directory.
            allowed_classes: List of class name strings to include.
                             Order determines the remapped label indices.
            transform:       torchvision transform applied to each image.
        """
        full_ds = datasets.ImageFolder(root=root)

        # Validate every requested class exists in the folder
        missing = set(allowed_classes) - set(full_ds.classes)
        if missing:
            raise ValueError(f"Classes not found in {root}: {sorted(missing)}")

        # Build clean label mapping: class_name -> 0..N-1 (order of allowed_classes)
        self.classes     = list(allowed_classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Filter samples and remap labels
        orig_idx_to_name = {v: k for k, v in full_ds.class_to_idx.items()}
        allowed_set = set(allowed_classes)
        self.samples = [
            (path, self.class_to_idx[orig_idx_to_name[orig_label]])
            for path, orig_label in full_ds.samples
            if orig_idx_to_name[orig_label] in allowed_set
        ]

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def get_labels(self):
        """Return flat list of labels — used by FewShotSampler."""
        return [label for _, label in self.samples]

    def get_class_to_indices(self):
        """Return dict: label_int -> list of dataset indices."""
        from collections import defaultdict
        d = defaultdict(list)
        for idx, (_, label) in enumerate(self.samples):
            d[label].append(idx)
        return dict(d)
