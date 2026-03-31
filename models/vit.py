"""
DeiT-S (Data-efficient Image Transformer, Small variant) backbone wrapper.

Wraps timm's deit_small_patch16_224 as a drop-in encoder for ProtoNet / NCC.

Output:  384-dimensional CLS-token embedding (vs ResNet-50's 2048-dim avgpool).
Input:   224 x 224 RGB, same ImageNet normalisation as ResNet-50.

Usage (for ProtoNet / NCC — no classification head):
    from models.vit import DeiTSmallEncoder
    backbone = DeiTSmallEncoder(pretrained=True)
    # backbone(x) → Tensor[B, 384]

Usage (for supervised baseline — with classification head):
    from models.vit import build_deit_classifier
    model = build_deit_classifier(num_classes=15, pretrained=True)
    # model(x) → Tensor[B, 15]
    # After training, strip head:
    import torch.nn as nn
    model.head = nn.Identity()
    # model(x) → Tensor[B, 384]
"""

import torch
import torch.nn as nn

try:
    import timm
except ImportError as e:
    raise ImportError(
        "The 'timm' package is required for DeiT-S. "
        "Install it with: pip install timm"
    ) from e


DEIT_MODEL_NAME = "deit_small_patch16_224"
EMBEDDING_DIM = 384


class DeiTSmallEncoder(nn.Module):
    """
    DeiT-S encoder: returns 384-dim CLS-token embeddings.
    Classification head is removed (num_classes=0).
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(
            DEIT_MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,   # removes .head, returns CLS token directly
        )
        self.embedding_dim = EMBEDDING_DIM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [B, 384] CLS-token embeddings."""
        return self.model(x)


def build_deit_classifier(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Returns a DeiT-S model with a num_classes classification head.
    Used for supervised base-class training.

    After training, set model.head = nn.Identity() to use as a feature extractor.
    """
    model = timm.create_model(
        DEIT_MODEL_NAME,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model
