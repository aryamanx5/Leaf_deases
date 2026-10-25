"""CNN backbones (torchvision, ImageNet-pretrained) with a new 4-class head."""

from __future__ import annotations

import logging
import os

import torch
from torch import nn
from torchvision import models

log = logging.getLogger(__name__)


def device() -> torch.device:
    forced = os.environ.get("LEAFDX_DEVICE")  # e.g. "cpu" for latency measurements
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> tuple[nn.Module, bool]:
    """Return (model, pretrained_loaded). Falls back to random init if weights can't be fetched."""
    if arch == "mobilenet_v2":
        ctor, weights = models.mobilenet_v2, models.MobileNet_V2_Weights.IMAGENET1K_V2
    elif arch == "efficientnet_b0":
        ctor, weights = models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1
    else:
        raise ValueError(f"unknown arch {arch}")
    loaded = False
    if pretrained:
        try:
            m = ctor(weights=weights)
            loaded = True
        except Exception as e:  # no network / cache
            log.warning("pretrained weights unavailable (%s); using random init", e)
            m = ctor(weights=None)
    else:
        m = ctor(weights=None)
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_f, num_classes)
    return m, loaded


def set_trainable(model: nn.Module, unfreeze_last_blocks: int) -> None:
    """Freeze the backbone except its last `unfreeze_last_blocks` feature blocks; head always trains."""
    for p in model.features.parameters():
        p.requires_grad = False
    if unfreeze_last_blocks > 0:
        for block in list(model.features.children())[-unfreeze_last_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True


def gradcam_layer(model: nn.Module) -> nn.Module:
    """Last convolutional block: the target layer for Grad-CAM."""
    return list(model.features.children())[-1]


def save_checkpoint(model: nn.Module, path, meta: dict) -> None:
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(path, map_location="cpu") -> tuple[nn.Module, dict]:
    ck = torch.load(path, map_location=map_location, weights_only=False)
    meta = ck["meta"]
    model, _ = build_model(meta["arch"], len(meta["classes"]), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, meta
