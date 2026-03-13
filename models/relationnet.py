"""
Relation Networks (Sung et al., 2018) — adapted for pretrained ResNet-50 backbone.

Architecture:
    - Encoder (ResNet-50, fc=Identity): embeds each image to a 2048-dim vector.
    - Prototype: mean of K support embeddings per class (same as ProtoNet).
    - Relation module (MLP): takes concat(query_emb, prototype) → scalar score in [0,1].

Loss: MSE loss against binary targets (1 = same class, 0 = different class).
      Do NOT use CrossEntropyLoss — relation scores are regressed probabilities.

Thesis note: The original paper uses spatial feature maps with conv relation layers.
Here we adapt to flat vectors with an MLP relation module, which is standard
practice when using pretrained CNN backbones in FSL literature.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationModule(nn.Module):
    """
    MLP that takes the concatenation of a query embedding and a class prototype
    and outputs a scalar relation score in [0, 1].

    Input dim: 2 * embedding_dim (default: 4096 for ResNet-50).
    """

    def __init__(self, input_dim: int = 4096, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)   # [batch] scalar scores


class RelationNet(nn.Module):
    def __init__(self, encoder, embedding_dim: int = 2048):
        """
        Args:
            encoder:       Backbone (e.g. ResNet-50 with fc=Identity).
            embedding_dim: Output dimension of the encoder (2048 for ResNet-50).
        """
        super().__init__()
        self.encoder  = encoder
        self.relation = RelationModule(input_dim=embedding_dim * 2)

    def forward(self, support, support_labels, query, n_way):
        """
        Args:
            support:        Tensor [N*K, C, H, W]
            support_labels: Tensor [N*K]  (episode-local labels, 0..n_way-1)
            query:          Tensor [N*Q, C, H, W]
            n_way:          int

        Returns:
            scores: Tensor [N*Q, n_way]
                    Relation score (in [0,1]) for each (query, class) pair.
                    Use with MSE loss against one-hot targets.
        """
        # --- encode ---
        support_emb = self.encoder(support)   # [N*K, D]
        query_emb   = self.encoder(query)     # [N*Q, D]

        # --- compute per-class prototypes ---
        n_query = query_emb.size(0)
        D = support_emb.size(1)
        prototypes = torch.zeros(n_way, D, device=support.device)
        for c in range(n_way):
            mask = (support_labels == c)
            prototypes[c] = support_emb[mask].mean(dim=0)

        # --- build all (query, prototype) pairs ---
        # Expand: query_emb [N*Q, D] → [N*Q, n_way, D]
        #         prototypes [n_way, D] → [N*Q, n_way, D]
        q_expanded = query_emb.unsqueeze(1).expand(n_query, n_way, D)
        p_expanded = prototypes.unsqueeze(0).expand(n_query, n_way, D)

        # Concatenate along last dim → [N*Q * n_way, 2D]
        pairs = torch.cat([q_expanded, p_expanded], dim=2)   # [N*Q, n_way, 2D]
        pairs_flat = pairs.view(n_query * n_way, 2 * D)

        # --- relation scores ---
        scores_flat = self.relation(pairs_flat)               # [N*Q * n_way]
        scores = scores_flat.view(n_query, n_way)             # [N*Q, n_way]

        return scores
