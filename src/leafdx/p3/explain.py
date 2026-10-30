"""Explanations (P3-M08): Grad-CAM++ for a fine-tuned CNN arm, a DINOv2 patch-similarity map
for the ViT arm (cosine similarity of each patch token to the CLS token), both with the
Stage 1 lesion outlines drawn on top.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

from leafdx.cnn.dataset import eval_transform
from leafdx.cnn.gradcam import gradcam
from leafdx.ip.lesions import overlay_lesions
from leafdx.robust.backbones import IMNET_MEAN, IMNET_STD, load_backbone
from leafdx.robust.cnn_heads import crop_index


def _heat(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    cam = cv2.resize(cam.astype(np.float32), (rgb.shape[1], rgb.shape[0]))
    heat = cv2.cvtColor(cv2.applyColorMap((255 * cam).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 1 - alpha, heat, alpha, 0)


def _lesion_outline(rgb: np.ndarray, s1: dict | None) -> np.ndarray:
    """Draw Stage 1 lesion contours (resized to rgb) on an RGB image."""
    if s1 is None:
        return rgb
    pre = s1["pre"]["resized"]
    ov = overlay_lesions(np.zeros_like(pre), s1["seg"]["leaf_mask"], s1["lesions"])
    m = cv2.resize((ov.sum(-1) > 0).astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = rgb.copy()
    cv2.drawContours(out, cnts, -1, (255, 255, 255), 1)
    return out


def gradcam_pp(model: torch.nn.Module, im: Image.Image, crop: str, cfg: dict, s1: dict | None = None) -> np.ndarray:
    """Grad-CAM++ for the crop's 'diseased' logit, RGB overlay at the CNN input size."""
    model = model.cpu().eval()
    x = eval_transform(cfg["cnn"])(im.convert("RGB")).unsqueeze(0)
    k = crop_index(crop)
    cam, _ = gradcam(model, x, target=lambda lg: lg[k], plus_plus=True)
    size = cfg["cnn"]["image_size"]
    rgb = np.array(im.convert("RGB").resize((size, size)))
    return _lesion_outline(_heat(rgb, cam), s1)


@torch.no_grad()
def vit_patch_map(im: Image.Image, s1: dict | None = None, size: int = 224) -> np.ndarray:
    """DINOv2 ViT-S/14: cosine similarity of every patch token to the CLS token."""
    m, _, _, _ = load_backbone("dinov2_vits14", "cpu")
    from torchvision import transforms as T

    x = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(IMNET_MEAN, IMNET_STD)])(im.convert("RGB"))
    f = m.forward_features(x.unsqueeze(0))
    cls, pt = f["x_norm_clstoken"][0], f["x_norm_patchtokens"][0]
    sim = torch.nn.functional.cosine_similarity(pt, cls[None], dim=1).numpy()
    g = size // 14
    cam = sim.reshape(g, g)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    rgb = np.array(im.convert("RGB").resize((size, size)))
    return _lesion_outline(_heat(rgb, cam), s1)
