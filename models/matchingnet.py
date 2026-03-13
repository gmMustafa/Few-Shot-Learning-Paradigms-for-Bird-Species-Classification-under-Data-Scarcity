"""
Matching Networks (Vinyals et al., 2016) — simplified version without FCE.

Architecture:
    - Shared CNN encoder (ResNet-50) embeds both support and query images.
    - Cosine similarity computed between each query and every support embedding.
    - Softmax attention over all N*K support items produces a weight for each.
    - Predicted class distribution = weighted sum of one-hot support labels.

Loss: NLL loss on the predicted class distribution (log-probabilities).
      Do NOT use CrossEntropyLoss — that would apply softmax twice.

Thesis note: Full Context Embedding (FCE) from the original paper is omitted.
This is common in FSL benchmarks and must be stated in the thesis.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MatchingNet(nn.Module):
    def __init__(self, encoder):
        """
        Args:
            encoder: backbone network (e.g. ResNet-50 with fc=Identity).
                     Outputs a flat embedding vector per image.
        """
        super().__init__()
        self.encoder = encoder

    def forward(self, support, support_labels, query, n_way):
        """
        Args:
            support:        Tensor [N*K, C, H, W]  — support set images
            support_labels: Tensor [N*K]            — episode-local labels (0..n_way-1)
            query:          Tensor [N*Q, C, H, W]  — query set images
            n_way:          int                     — number of classes in episode

        Returns:
            log_probs: Tensor [N*Q, n_way]
                       Log of the attention-weighted class distribution.
                       Use with F.nll_loss(log_probs, query_labels).
        """
        # --- encode ---
        support_emb = self.encoder(support)   # [N*K, D]
        query_emb   = self.encoder(query)     # [N*Q, D]

        # --- L2-normalise for cosine similarity ---
        support_emb = F.normalize(support_emb, dim=1)   # [N*K, D]
        query_emb   = F.normalize(query_emb,   dim=1)   # [N*Q, D]

        # --- cosine similarity: query vs every support item ---
        # sim[i, j] = cosine_sim(query_i, support_j)
        sim = torch.mm(query_emb, support_emb.t())      # [N*Q, N*K]

        # --- softmax attention over support items ---
        attn = F.softmax(sim, dim=1)                    # [N*Q, N*K]

        # --- one-hot encode support labels: [N*K, n_way] ---
        n_support = support_labels.size(0)
        one_hot = torch.zeros(n_support, n_way, device=support.device)
        one_hot.scatter_(1, support_labels.unsqueeze(1), 1.0)

        # --- weighted sum → predicted class distribution ---
        # probs[i, c] = sum over j of attn[i,j] * one_hot[j,c]
        probs = torch.mm(attn, one_hot)                 # [N*Q, n_way]

        # --- return log probabilities for NLL loss ---
        return torch.log(probs + 1e-8)                  # [N*Q, n_way]
