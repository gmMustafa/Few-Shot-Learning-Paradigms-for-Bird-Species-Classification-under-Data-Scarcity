"""
Prototypical Networks (Snell et al., 2017).

Supports two distance metrics for prototype comparison:
  - euclidean (default, original paper)
  - manhattan (L1 norm — ablation study, as in reference thesis)

Thesis note: The reference project (Naeem, 2024) found Euclidean distance
slightly more effective than Manhattan for high-dimensional ResNet-50 embeddings.
We replicate this ablation for bird classification to verify whether the same
finding holds in a different visual domain.
"""
import torch
import torch.nn as nn


class ProtoNet(nn.Module):
    def __init__(self, encoder):
        """
        Args:
            encoder: backbone network (e.g. ResNet-50 with fc=Identity).
                     Outputs a flat embedding vector per image.
        """
        super().__init__()
        self.encoder = encoder

    def forward(self, support, support_labels, query, distance: str = 'euclidean'):
        """
        Args:
            support:        Tensor [N*K, C, H, W]
            support_labels: Tensor [N*K]  (episode-local labels, 0..n_way-1)
            query:          Tensor [N*Q, C, H, W]
            distance:       'euclidean' (L2) or 'manhattan' (L1)

        Returns:
            logits: Tensor [N*Q, n_way]  — negative distance to each prototype.
                    Higher = closer = more likely. Use with CrossEntropyLoss.
        """
        support_emb = self.encoder(support)   # [N*K, D]
        query_emb   = self.encoder(query)     # [N*Q, D]

        n_way = torch.unique(support_labels).size(0)
        D     = support_emb.size(-1)

        # Compute per-class prototypes (mean of support embeddings)
        prototypes = torch.zeros(n_way, D, device=support.device)
        for i, cls in enumerate(torch.unique(support_labels)):
            mask = (support_labels == cls)
            prototypes[i] = support_emb[mask].mean(dim=0)

        # Compute distances between every query and every prototype
        if distance == 'manhattan':
            dists = manhattan_distance(query_emb, prototypes)
        elif distance == 'euclidean':
            dists = euclidean_distance(query_emb, prototypes)
        else:
            raise ValueError(f"Unknown distance metric: '{distance}'. "
                             f"Choose 'euclidean' or 'manhattan'.")

        return -dists   # negative distance = logits (larger = closer)


# ---------------------------------------------------------------------------
# Distance functions
# ---------------------------------------------------------------------------

def euclidean_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    L2 (Euclidean) distance between every row of a and every row of b.
    a: [Nq, D]   b: [N, D]
    Returns: [Nq, N]
    """
    nq, n = a.shape[0], b.shape[0]
    a = a.unsqueeze(1).expand(nq, n, -1)
    b = b.unsqueeze(0).expand(nq, n, -1)
    return torch.norm(a - b, dim=2)


def manhattan_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    L1 (Manhattan) distance between every row of a and every row of b.
    a: [Nq, D]   b: [N, D]
    Returns: [Nq, N]
    """
    nq, n = a.shape[0], b.shape[0]
    a = a.unsqueeze(1).expand(nq, n, -1)
    b = b.unsqueeze(0).expand(nq, n, -1)
    return torch.abs(a - b).sum(dim=2)
