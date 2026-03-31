"""
Matching Networks (Vinyals et al., 2016).

Two modes controlled by use_fce:

    use_fce=False (default):
        Shared ResNet-50 encoder for support and query; no context embedding.
        Identical to the original implementation shipped with this thesis.

    use_fce=True:
        Adds Full Context Embedding (FCE):
          • SupportFCE  — BiLSTM over the ordered support embeddings so each
                          support item is re-encoded in the context of all others.
          • QueryAttLSTM — Attention LSTM for query embeddings conditioned on the
                          full support set (num_steps inner iterations).
        FCE is the architecture described in the original paper and closes the
        methodological gap noted in the thesis limitations section.

Loss: NLL loss on the predicted class distribution (log-probabilities).
      Do NOT use CrossEntropyLoss — that would apply softmax twice.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# FCE modules
# ---------------------------------------------------------------------------

class SupportFCE(nn.Module):
    """
    BiLSTM over support embeddings.
    Treats all N*K support items as a single sequence (batch-size 1) and
    re-encodes each item in the context of all others.

    Input/output:  [N*K, D]  (same shape, context-enriched)
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        # bidirectional LSTM: each direction has D//2 hidden units
        # concatenated output size = 2 * (D//2) = D  → same embedding dim
        self.lstm = nn.LSTM(
            embedding_dim, embedding_dim // 2,
            batch_first=True, bidirectional=True,
        )

    def forward(self, support_emb: torch.Tensor) -> torch.Tensor:
        # [N*K, D] → [1, N*K, D] (batch=1 sequence)
        out, _ = self.lstm(support_emb.unsqueeze(0))  # [1, N*K, D]
        return out.squeeze(0)                          # [N*K, D]


class QueryAttLSTM(nn.Module):
    """
    Attention LSTM (attLSTM) for query context embedding.

    For num_steps iterations, each query's hidden state attends over the
    (FCE-encoded) support set:
        r_t   = sum_j  softmax(h_{t-1} · support_j)  · support_j
        x_t   = concat(query_emb,  r_t)
        h_t, c_t = LSTMCell(x_t, (h_{t-1}, c_{t-1}))

    The final h is the context-enriched query embedding.

    Input:
        query_emb   [N*Q, D]
        support_emb [N*K, D]  (already FCE-encoded)
    Output:
        fce_query   [N*Q, D]
    """

    def __init__(self, embedding_dim: int, num_steps: int = 10):
        super().__init__()
        self.num_steps = num_steps
        # Input = concat(query_emb, attended_support) = 2*D
        self.lstm_cell = nn.LSTMCell(2 * embedding_dim, embedding_dim)

    def forward(self, query_emb: torch.Tensor,
                support_emb: torch.Tensor) -> torch.Tensor:
        h = query_emb                       # [N*Q, D]
        c = torch.zeros_like(h)             # [N*Q, D]

        for _ in range(self.num_steps):
            # Cosine-similarity attention over support
            attn = F.softmax(
                torch.mm(F.normalize(h, dim=1),
                         F.normalize(support_emb, dim=1).t()),
                dim=1,
            )                               # [N*Q, N*K]

            r = torch.mm(attn, support_emb) # [N*Q, D]
            cell_input = torch.cat([query_emb, r], dim=1)  # [N*Q, 2D]
            h, c = self.lstm_cell(cell_input, (h, c))

        return h                            # [N*Q, D]


# ---------------------------------------------------------------------------
# MatchingNet
# ---------------------------------------------------------------------------

class MatchingNet(nn.Module):
    """
    Args:
        encoder       : backbone network (e.g. ResNet-50 with fc=Identity).
        embedding_dim : output dimension of the encoder. Required only when
                        use_fce=True (default 2048 for ResNet-50).
        use_fce       : if True, apply Full Context Embedding before computing
                        cosine similarity. Default False (no FCE).
        fce_steps     : number of inner-loop steps in QueryAttLSTM. Ignored
                        when use_fce=False.
    """

    def __init__(
        self,
        encoder,
        embedding_dim: int = 2048,
        use_fce: bool = False,
        fce_steps: int = 10,
    ):
        super().__init__()
        self.encoder  = encoder
        self.use_fce  = use_fce

        if use_fce:
            self.support_fce = SupportFCE(embedding_dim)
            self.query_att   = QueryAttLSTM(embedding_dim, num_steps=fce_steps)

    def forward(
        self,
        support: torch.Tensor,
        support_labels: torch.Tensor,
        query: torch.Tensor,
        n_way: int,
    ) -> torch.Tensor:
        """
        Args:
            support:        [N*K, C, H, W]
            support_labels: [N*K]            episode-local labels 0..n_way-1
            query:          [N*Q, C, H, W]
            n_way:          int

        Returns:
            log_probs: [N*Q, n_way]   Use with F.nll_loss(log_probs, query_labels).
        """
        support_emb = self.encoder(support)   # [N*K, D]
        query_emb   = self.encoder(query)     # [N*Q, D]

        if self.use_fce:
            support_emb = self.support_fce(support_emb)
            query_emb   = self.query_att(query_emb, support_emb)

        # L2-normalise for cosine similarity
        support_emb = F.normalize(support_emb, dim=1)
        query_emb   = F.normalize(query_emb,   dim=1)

        # cosine similarity: [N*Q, N*K]
        sim  = torch.mm(query_emb, support_emb.t())
        attn = F.softmax(sim, dim=1)

        # one-hot support labels: [N*K, n_way]
        n_support = support_labels.size(0)
        one_hot   = torch.zeros(n_support, n_way, device=support.device)
        one_hot.scatter_(1, support_labels.unsqueeze(1), 1.0)

        # attention-weighted class distribution: [N*Q, n_way]
        probs = torch.mm(attn, one_hot)

        return torch.log(probs + 1e-8)   # log-probabilities for NLL loss
