import random
from collections import defaultdict
from torch.utils.data import Sampler


class FewShotSampler(Sampler):
    """
    Episodic batch sampler for N-way K-shot + Q-query episodes.

    Each call to __iter__ yields `episodes_per_epoch` episodes.
    Each episode is a flat list of (K+Q)*N dataset indices, structured as:
        [class0_support_indices, class0_query_indices,
         class1_support_indices, class1_query_indices, ...]

    Reproducibility: each epoch gets a unique but deterministic set of
    episodes by seeding with (base_seed + call_count * 10000), so repeated
    experiments are fully reproducible while each epoch sees different episodes.

    For evaluation samplers, call __iter__ only once — episodes are fixed.
    """

    def __init__(self, labels, n_way, k_shot, q_query, episodes_per_epoch, seed=42):
        """
        Args:
            labels:              Flat list (or tensor) of integer labels for every
                                 sample in the dataset.
            n_way:               Number of classes per episode.
            k_shot:              Support images per class.
            q_query:             Query images per class.
            episodes_per_epoch:  Episodes yielded per __iter__ call.
            seed:                Base random seed for reproducibility.
        """
        super().__init__()

        self.n_way              = n_way
        self.k_shot             = k_shot
        self.q_query            = q_query
        self.episodes_per_epoch = episodes_per_epoch
        self.seed               = seed
        self._call_count        = 0   # incremented each time __iter__ is called

        # Group dataset indices by class label
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_to_indices[int(label)].append(idx)
        self.class_list = sorted(self.class_to_indices.keys())

        # Validate: enough classes
        if len(self.class_list) < n_way:
            raise ValueError(
                f"Dataset has {len(self.class_list)} classes but n_way={n_way}."
            )

        # Validate: enough samples per class
        need = k_shot + q_query
        for cls, indices in self.class_to_indices.items():
            if len(indices) < need:
                raise ValueError(
                    f"Class {cls} has {len(indices)} samples; "
                    f"need at least {need} (k_shot={k_shot} + q_query={q_query})."
                )

    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        # Each epoch gets a unique deterministic seed → different episodes, reproducible
        epoch_seed = self.seed + self._call_count * 10_000
        self._call_count += 1
        rng = random.Random(epoch_seed)

        for _ in range(self.episodes_per_epoch):
            episode_classes = rng.sample(self.class_list, self.n_way)
            batch = []
            for cls in episode_classes:
                indices = rng.sample(self.class_to_indices[cls], self.k_shot + self.q_query)
                batch.extend(indices)
            yield batch
