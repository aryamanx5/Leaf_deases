"""Grad-CAM on the last convolutional block: where the CNN looks."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from leafdx.cnn.dataset import CLASSES4, eval_transform
from leafdx.cnn.model import gradcam_layer, load_checkpoint
from leafdx.config import ROOT, path
from leafdx.viz import contact_sheet


def gradcam(model: torch.nn.Module, x: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (heatmap in [0,1] at input resolution, softmax probabilities) for one image."""
    model.eval()
    acts, grads = {}, {}
    layer = gradcam_layer(model)
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    try:
        x = x.clone().requires_grad_(True)
        logits = model(x)
        probs = torch.softmax(logits, 1)[0].detach().numpy()
        idx = int(probs.argmax()) if class_idx is None else class_idx
        model.zero_grad()
        logits[0, idx].backward()
        w = grads["g"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * acts["a"]).sum(1))[0].detach().numpy()
    finally:
        h1.remove()
        h2.remove()
    cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, probs


def overlay_cam(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """RGB uint8 image + heatmap -> BGR overlay (for OpenCV figures)."""
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(bgr, 1 - alpha, heat, alpha, 0)


def gradcam_for_image(model, img_rgb: Image.Image, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    size = cfg["cnn"]["image_size"]
    x = eval_transform(cfg["cnn"])(img_rgb).unsqueeze(0)
    cam, probs = gradcam(model, x)
    rgb = np.array(img_rgb.resize((size, size)))
    return overlay_cam(rgb, cam), probs


def gradcam_figure(cfg: dict, splits: pd.DataFrame, ckpt) -> None:
    """Grad-CAM for a few TEST images (2 per class) -> results/figures/gradcam_samples.png."""
    model, _ = load_checkpoint(ckpt)  # CPU: hooks are simplest there
    n = max(1, cfg["cnn"]["gradcam_images"] // len(CLASSES4))
    sample = splits[splits.split == "test"].groupby("stratum").sample(n, random_state=cfg["seed"])
    rows, titles = [], []
    for r in sample.itertuples():
        with Image.open(ROOT / r.path) as im:
            im = im.convert("RGB")
            over, probs = gradcam_for_image(model, im, cfg)
            orig = cv2.cvtColor(np.array(im.resize(over.shape[1::-1])), cv2.COLOR_RGB2BGR)
        k = int(probs.argmax())
        rows.append([orig, over])
        titles.append([f"true {r.stratum}", f"pred {CLASSES4[k]} {probs[k]:.0%}"])
    sheet = contact_sheet([rows[i] + rows[i + 1] for i in range(0, len(rows), 2)],
                          [titles[i] + titles[i + 1] for i in range(0, len(rows), 2)], size=220)
    out = path("figures", cfg) / "gradcam_samples.png"
    cv2.imwrite(str(out), sheet)
    print(f"saved {out.relative_to(ROOT)}")
