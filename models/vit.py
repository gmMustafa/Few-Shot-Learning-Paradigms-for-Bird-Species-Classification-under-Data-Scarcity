"""
DeiT-S (Data-efficient Image Transformer, Small variant) backbone wrapper.

Wraps timm's deit_small_patch16_224 as a drop-in encoder for ProtoNet / NCC.

Output:  384-dimensional CLS-token embedding (vs ResNet-50's 2048-dim avgpool).
Input:   224 x 224 RGB, same ImageNet normalisation as ResNet-50.

Usage (for ProtoNet / NCC — no classification head):
    from models.vit import DeiTSmallEncoder
    backbone = DeiTSmallEncoder(pretrained=True)
    # backbone(x) → Tensor[B, 384]

    # HPC offline: pass a local weights file to skip HuggingFace download
    backbone = DeiTSmallEncoder(pretrained=False,
                                weights_path='/path/to/deit_small_patch16_224.pth')

Usage (for supervised baseline — with classification head):
    from models.vit import build_deit_classifier
    model = build_deit_classifier(num_classes=15, pretrained=True)
    # model(x) → Tensor[B, 15]
    # After training, strip head:
    import torch.nn as nn
    model.head = nn.Identity()
    # model(x) → Tensor[B, 384]
"""

import os
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


def _load_weights_from_path(model: nn.Module, weights_path: str) -> None:
    """Load state-dict from a local .pth or .safetensors file."""
    weights_path = os.path.expandvars(os.path.expanduser(weights_path))
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"DeiT-S weights file not found: {weights_path}")

    if weights_path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
            state_dict = load_file(weights_path)
        except ImportError as e:
            raise ImportError(
                "Install 'safetensors' to load .safetensors files: pip install safetensors"
            ) from e
    else:
        state_dict = torch.load(weights_path, map_location="cpu")
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        elif isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[DeiTSmallEncoder] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[DeiTSmallEncoder] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    print(f"[DeiTSmallEncoder] Loaded weights from {weights_path}")


class DeiTSmallEncoder(nn.Module):
    """
    DeiT-S encoder: returns 384-dim CLS-token embeddings.
    Classification head is removed (num_classes=0).

    Args:
        pretrained:    If True, downloads ImageNet weights from HuggingFace (requires internet).
        weights_path:  Optional path to a local .pth / .safetensors file.
                       When provided, pretrained is forced to False and weights are
                       loaded from disk — use this on HPC nodes without internet access.
    """

    def __init__(self, pretrained: bool = True, weights_path: str = None):
        super().__init__()
        # If a local file is given, build the architecture without downloading
        if weights_path is not None:
            pretrained = False
        self.model = timm.create_model(
            DEIT_MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,   # removes .head, returns CLS token directly
        )
        self.embedding_dim = EMBEDDING_DIM
        if weights_path is not None:
            _load_weights_from_path(self.model, weights_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [B, 384] CLS-token embeddings."""
        return self.model(x)


def build_deit_classifier(num_classes: int, pretrained: bool = True,
                          weights_path: str = None) -> nn.Module:
    """
    Returns a DeiT-S model with a num_classes classification head.
    Used for supervised base-class training.

    Args:
        num_classes:   Number of output classes.
        pretrained:    If True, downloads ImageNet weights from HuggingFace.
        weights_path:  Optional local .pth / .safetensors file (overrides pretrained).

    After training, set model.head = nn.Identity() to use as a feature extractor.
    """
    if weights_path is not None:
        pretrained = False
    model = timm.create_model(
        DEIT_MODEL_NAME,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    if weights_path is not None:
        _load_weights_from_path(model, weights_path)
        # Replace the randomly initialised head (weights_path has num_classes=0 head)
        model.head = nn.Linear(EMBEDDING_DIM, num_classes)
    return model
