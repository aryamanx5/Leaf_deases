"""Frozen foundation-model image embeddings (DINOv2 ViT-S/14, OpenCLIP ViT-B/32, BioCLIP 2)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms as T

from leafdx.cnn.model import device

CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
BIOCLIP2_HUB = "hf-hub:imageomics/bioclip-2"
CLIP_TOKENIZERS = {"openclip_vitb32": "ViT-B-32", "bioclip2": BIOCLIP2_HUB}


@lru_cache(maxsize=8)
def load_backbone(name: str, dev_str: str | None = None):
    """(model, preprocess, embed_fn, device) for a frozen backbone (best device unless given)."""
    dev = torch.device(dev_str) if dev_str else device()
    if name == "dinov2_vits14":
        hub = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
        if hub.is_dir():  # cached copy: skip the GitHub lookup torch.hub does on every load
            m = torch.hub.load(str(hub), "dinov2_vits14", source="local", verbose=False)
        else:
            m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
        tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(IMNET_MEAN, IMNET_STD)])
        fn = m
    elif name == "openclip_vitb32":
        import open_clip

        m, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)])
        fn = m.encode_image
    elif name == "bioclip2":  # imageomics/bioclip-2 (ViT-L/14, MIT licence), loaded through open_clip
        import open_clip

        m, _, tf = open_clip.create_model_and_transforms(BIOCLIP2_HUB)
        fn = m.encode_image
    else:
        raise ValueError(name)
    m.eval().to(dev)
    return m, tf, fn, dev


@torch.no_grad()
def embed(name: str, images: list, batch: int = 64, dev_str: str | None = None) -> np.ndarray:
    """L2-normalised embeddings for a list of PIL images."""
    _, tf, fn, dev = load_backbone(name, dev_str)
    out = []
    for i in range(0, len(images), batch):
        x = torch.stack([tf(im.convert("RGB")) for im in images[i:i + batch]]).to(dev)
        e = fn(x).float()
        out.append(torch.nn.functional.normalize(e, dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


@torch.no_grad()
def clip_zero_shot_text(prompts: list[str], name: str = "openclip_vitb32") -> np.ndarray:
    import open_clip

    m, _, _, dev = load_backbone(name)
    tok = open_clip.get_tokenizer(CLIP_TOKENIZERS[name])
    t = m.encode_text(tok(prompts).to(dev)).float()
    return torch.nn.functional.normalize(t, dim=1).cpu().numpy()
