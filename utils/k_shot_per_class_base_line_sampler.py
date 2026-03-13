import random
from collections import defaultdict
from torch.utils.data import Sampler


class KShotPerClassSampler(Sampler):
    """
    Randomly samples exactly K images per class for baseline supervised training.
    """

    def __init__(self, dataset, k_shot, seed=None):
        self.dataset = dataset
        self.k_shot = k_shot
        self.seed = seed

        # Group indices by class
        self.class_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset.samples):
            self.class_to_indices[label].append(idx)

        # Filter out classes that have less than k_shot samples
        self.class_to_indices = {
            cls: idxs for cls, idxs in self.class_to_indices.items() if len(idxs) >= k_shot
        }

        self.valid_classes = list(self.class_to_indices.keys())
        self.sampled_indices = self._generate_indices()

    def _generate_indices(self):
        if self.seed is not None:
            random.seed(self.seed)

        selected = []
        for cls in self.valid_classes:
            selected.extend(random.sample(self.class_to_indices[cls], self.k_shot))
        return selected

    def __iter__(self):
        return iter(self.sampled_indices)

    def __len__(self):
        return len(self.sampled_indices)
