"""P2-F02: diseased / healed leaf composites for the landing page's closing section.

leaves_diseased: real diseased leaves (our datasets), cut out with our own Stage 1 leaf
mask and arranged on a dark, blurred field backdrop. leaves_healed: the SAME pixels except
inside our own Stage 1 lesion mask (dilated), where
  * chroma (a*, b*) is replaced by the leaf's healthy tissue, inpainted smoothly into the
    lesion at low resolution (no flat fill),
  * low-frequency lightness moves to the healthy level while high-frequency detail (veins,
    surface texture) is kept,
  * small dark specks are inpainted (Telea),
  * the result is blended back with a feathered mask.
Outputs web/public/media/heal/: diseased/healed WebP + AVIF, lesion and leaf masks (half
resolution, for the live meter), heal_<variant>.json; preview figure in results/figures.
"""

from __future__ import annotations

import json
import sys

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from leafdx.config import ROOT, load_config, path
from leafdx.ip.lesions import detect_lesions
from leafdx.ip.preprocess import load_bgr, preprocess
from leafdx.ip.segment import segment_leaf

LEAVES = [  # (path, crop), TRAIN/VAL only, chosen by: whole leaf inside the photo, plain background,
    # affected 6-35%, no single lesion > 3% of the leaf (large patches cannot be healed convincingly);
    # turmeric blotch lesions are large by nature, so the one turmeric leaf has a compact tip lesion
    ("data/raw/blackgram/diseased/anthracnose/111a.jpg", "blackgram"),
    ("data/raw/blackgram/diseased/anthracnose/124a.jpg", "blackgram"),
    ("data/raw/turmeric/diseased/leaf_blotch/Leaf Blotch00031.JPG", "turmeric"),
    ("data/raw/blackgram/diseased/yellow_mosaic/183y.jpg", "blackgram"),
]
BACKDROP = "data/raw/blackgram/healthy/healthy/150h.jpg"
VARIANTS = {  # canvas, per-leaf (centre x, centre y, height as fraction of canvas height, rotation deg)
    "21x9": ((2520, 1080), [(0.20, 0.55, 0.95, -28), (0.42, 0.50, 0.78, 12), (0.63, 0.52, 0.98, 30), (0.84, 0.50, 0.80, -10)]),
    "9x16": ((1080, 1920), [(0.30, 0.25, 0.42, -20), (0.72, 0.33, 0.30, 15), (0.36, 0.67, 0.44, 25), (0.70, 0.80, 0.32, -12)]),
}
OUT = ROOT / "web/public/media/heal"


def leaf_layers(p: str, crop: str, cfg: dict) -> dict:
    """Full-resolution RGB, alpha, lesion masks and healed RGB for one leaf."""
    img = load_bgr(ROOT / p)
    H, W = img.shape[:2]
    pre = preprocess(img, cfg)
    seg = segment_leaf(pre, cfg)
    les = detect_lesions(pre, seg["leaf_mask"], cfg, crop)
    up = lambda m: cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_LINEAR) > 0  # noqa: E731
    leaf = up(cv2.GaussianBlur(seg["leaf_mask"].astype(np.float32), (5, 5), 0) > 0.5)
    s = W / pre["resized"].shape[1]
    k = int(round(4 * s)) * 2 + 1
    dil = lambda m: cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool) & leaf  # noqa: E731
    cm = les["class_masks"]
    necrotic = dil(up(cm["brown"] | cm["dark"]) & leaf)
    chlorotic = dil(up(cm["yellow"] | cm["pale"]) & leaf) & ~necrotic
    # Visible damage the lesion detector leaves out (tiny specks, the brown halo around a spot): leaf pixels
    # clearly off this leaf's own healthy colour nearby. Without this the "healed" leaf keeps faint brown
    # spots. Compared with a local reference (healthy tissue, heavily blurred) so lighting gradients pass.
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (0, 0), 1.0 * s), cv2.COLOR_BGR2LAB).astype(np.float32)
    ok = (leaf & ~necrotic & ~chlorotic).astype(np.float32)
    w = np.maximum(cv2.GaussianBlur(ok, (0, 0), 25 * s), 1e-3)
    ref = np.stack([cv2.GaussianBlur(lab[..., c] * ok, (0, 0), 25 * s) / w for c in range(3)], -1)
    dL, da, db = (lab[..., c] - ref[..., c] for c in range(3))
    inner = cv2.erode(leaf.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1,) * 2)).astype(bool)
    speck = lambda m: cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)  # noqa: E731
    brownish = speck(inner & ((da > 7) | ((dL < -24) & (da > 0))))  # dark AND not greener: green tissue is never damage
    yellowish = speck(inner & ~brownish & (db > 12) & (dL > -12))
    # mosaic leaves are mostly yellow, so also compare with the leaf's greenest tissue
    if ok.any():
        green = ok.astype(bool) & (lab[..., 1] < np.percentile(lab[..., 1][ok.astype(bool)], 35))
        g = np.median(lab[green], axis=0) if green.any() else ref.reshape(-1, 3).mean(0)
        yellowish |= speck(inner & ~brownish & (lab[..., 2] - g[2] > 10) & (lab[..., 0] - g[0] > 4))
    necrotic |= dil(brownish)
    chlorotic = (chlorotic | dil(yellowish)) & ~necrotic
    healed = heal(img, leaf, necrotic, chlorotic, s)
    alpha = cv2.GaussianBlur(leaf.astype(np.float32), (0, 0), 1.2 * s)
    return {"rgb": img, "healed": healed, "alpha": alpha, "lesion": necrotic | chlorotic, "leaf": leaf}


def extend_leaf(img: np.ndarray, leaf: np.ndarray, healthy: np.ndarray) -> np.ndarray:
    """Fill everything that is not healthy leaf with colours spread out from the healthy leaf,
    so later inpainting never samples the photo background."""
    f = 0.25
    small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    hole = cv2.resize((~healthy).astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    filled = cv2.inpaint(small, hole * 255, 9, cv2.INPAINT_TELEA)
    filled = cv2.resize(filled, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    return np.where(healthy[..., None], img, filled)


def clone_along_axis(filled: np.ndarray, img: np.ndarray, leaf: np.ndarray, healthy: np.ndarray, big: np.ndarray,
                     s: float) -> np.ndarray:
    ys, xs = np.nonzero(leaf)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    _, vecs = np.linalg.eigh(np.cov((pts - pts.mean(0)).T))
    v = vecs[:, -1]  # principal (vein) direction
    H, W = leaf.shape
    gy, gx = np.mgrid[:H, :W].astype(np.float32)
    out = filled.copy()
    todo = big.copy()
    for t in np.arange(15, 600, 15) * s:
        for sign in (1, -1):
            mx, my = (gx + sign * t * v[0]).astype(np.float32), (gy + sign * t * v[1]).astype(np.float32)
            ok = cv2.remap(healthy.astype(np.uint8), mx, my, cv2.INTER_NEAREST, borderValue=0).astype(bool)
            take = todo & ok
            if take.any():
                src = cv2.remap(img, mx, my, cv2.INTER_LINEAR)
                out[take] = src[take]
                todo &= ~take
        if not todo.any():
            break
    # brightness/colour continuity: keep the cloned detail, take low frequencies from the smooth fill
    o = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
    f = cv2.cvtColor(filled, cv2.COLOR_BGR2LAB).astype(np.float32)
    sig = 8 * s
    m = big.astype(np.float32)
    lo = lambda x: cv2.GaussianBlur(x * m, (0, 0), sig) / np.maximum(cv2.GaussianBlur(m, (0, 0), sig), 1e-3)  # noqa: E731
    for c in range(3):
        o[..., c] = np.where(big, o[..., c] - lo(o[..., c]) + lo(f[..., c]), o[..., c])
    return cv2.cvtColor(np.clip(o, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def heal(img: np.ndarray, leaf: np.ndarray, necrotic: np.ndarray, chlorotic: np.ndarray, s: float) -> np.ndarray:
    healthy = leaf & ~necrotic & ~chlorotic
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    med = np.median(lab[healthy], axis=0) if healthy.any() else lab[leaf].mean(0)
    out = lab.copy()
    # chlorosis / pale tissue: move a*, b* to the leaf's healthy median (keep a little local
    # variation) and lift L most of the way (low frequencies only), so veins and surface texture survive
    for c in (1, 2):
        local = cv2.GaussianBlur(lab[..., c], (0, 0), 3 * s)
        out[..., c] = np.where(chlorotic, med[c] + 0.2 * (lab[..., c] - local), lab[..., c])
    lowL = cv2.GaussianBlur(lab[..., 0], (0, 0), 4 * s)
    out[..., 0] = np.where(chlorotic, lab[..., 0] + 0.9 * (med[0] - lowL), lab[..., 0])
    rgb = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    # necrosis: Telea inpainting from healthy leaf only (background replaced by leaf extension)
    if necrotic.any():
        ext = extend_leaf(rgb, leaf, healthy | chlorotic)
        filled = cv2.inpaint(ext, necrotic.astype(np.uint8) * 255, int(5 * s), cv2.INPAINT_TELEA)
        # Telea is smooth: add back surface grain borrowed from healthy tissue nearby (shifted copy)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
        grain = g - cv2.GaussianBlur(g, (0, 0), 2 * s)
        sh = int(round(25 * s))
        fl = cv2.cvtColor(filled, cv2.COLOR_BGR2LAB).astype(np.float32)
        todo = necrotic.copy()
        for dy, dx in ((sh, 0), (-sh, 0), (0, sh), (0, -sh), (2 * sh, 0), (0, 2 * sh)):
            take = todo & np.roll(np.roll(healthy, dy, 0), dx, 1)
            fl[..., 0] = np.where(take, fl[..., 0] + 0.8 * np.roll(np.roll(grain, dy, 0), dx, 1), fl[..., 0])
            todo &= ~take
        filled = cv2.cvtColor(np.clip(fl, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        # large lesions on parallel-veined leaves: clone tissue along the leaf's main axis so the
        # veins continue through the healed area (classical clone-stamp), brightness-matched
        n, cc, stats, _ = cv2.connectedComponentsWithStats(necrotic.astype(np.uint8))
        big = np.isin(cc, [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > (60 * s) ** 2])
        if big.any():
            filled = clone_along_axis(filled, img, leaf, healthy, big, s)
        rgb = np.where(necrotic[..., None], filled, rgb)
    a = cv2.GaussianBlur((necrotic | chlorotic).astype(np.float32), (0, 0), 1.2 * s)[..., None]
    return (a * rgb + (1 - a) * img).astype(np.uint8)


def backdrop(size: tuple[int, int]) -> np.ndarray:
    W, H = size
    b = cv2.resize(load_bgr(ROOT / BACKDROP), (W, H), interpolation=cv2.INTER_CUBIC)
    b = cv2.GaussianBlur(b, (0, 0), 30).astype(np.float32)
    b = b * 0.22 + np.array([8, 12, 9], np.float32)  # dark, slightly green
    yy, xx = np.mgrid[:H, :W]
    vign = 1 - 0.45 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    return np.clip(b * vign[..., None], 0, 255).astype(np.uint8)


def place(canvas, layer_rgb, alpha, mask_canvas, mask, cx, cy, hfrac, rot) -> None:
    H, W = canvas.shape[:2]
    h, w = layer_rgb.shape[:2]
    s = hfrac * H / max(h, w)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rot, s)
    M[0, 2] += cx * W - w / 2
    M[1, 2] += cy * H - h / 2
    warp = lambda x, interp=cv2.INTER_LINEAR: cv2.warpAffine(x, M, (W, H), flags=interp, borderValue=0)  # noqa: E731
    a = warp(alpha.astype(np.float32))[..., None]
    canvas[:] = (a * warp(layer_rgb) + (1 - a) * canvas).astype(np.uint8)
    if mask is not None:
        mask_canvas |= warp(mask.astype(np.uint8), cv2.INTER_NEAREST).astype(bool) & (a[..., 0] > 0.5)


def save_variants(name: str, bgr: np.ndarray, widths: list[int]) -> list[str]:
    out = []
    im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    for w in widths:
        h = round(im.height * w / im.width)
        r = im.resize((w, h), Image.LANCZOS) if w != im.width else im
        for fmt, q in (("webp", 86), ("avif", 60)):
            f = OUT / f"{name}-{w}.{fmt}"
            r.save(f, format=fmt.upper(), quality=q)
            out.append(f.name)
    return out


def main() -> int:
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    layers = [leaf_layers(p, c, cfg) for p, c in LEAVES]
    previews = []
    for vname, (size, spots) in VARIANTS.items():
        base = backdrop(size)
        dis, hea = base.copy(), base.copy()
        lesion_c = np.zeros(base.shape[:2], bool)
        leaf_c = np.zeros(base.shape[:2], bool)
        for L, (cx, cy, hf, rot) in zip(layers, spots, strict=True):
            place(dis, L["rgb"], L["alpha"], lesion_c, L["lesion"], cx, cy, hf, rot)
            place(hea, L["healed"], L["alpha"], leaf_c, L["leaf"], cx, cy, hf, rot)
        lesion_c &= leaf_c
        outside = ~cv2.dilate(lesion_c.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
        g1, g2 = cv2.cvtColor(dis, cv2.COLOR_BGR2GRAY), cv2.cvtColor(hea, cv2.COLOR_BGR2GRAY)
        _, ssim_map = structural_similarity(g1, g2, full=True, data_range=255)
        ssim_out = float(ssim_map[outside].mean())
        widths = [1280, size[0]] if size[0] > 1280 else [720, size[0]]
        files = {"diseased": save_variants(f"leaves_diseased_{vname}", dis, widths),
                 "healed": save_variants(f"leaves_healed_{vname}", hea, widths)}
        half = (size[0] // 2, size[1] // 2)
        cv2.imwrite(str(OUT / f"lesion_mask_{vname}.png"),
                    cv2.resize(lesion_c.astype(np.uint8) * 255, half, interpolation=cv2.INTER_AREA))
        cv2.imwrite(str(OUT / f"leaf_mask_{vname}.png"),
                    cv2.resize(leaf_c.astype(np.uint8) * 255, half, interpolation=cv2.INTER_AREA))
        meta = {"variant": vname, "width": size[0], "height": size[1], "leaf_px": int(leaf_c.sum()),
                "lesion_px": int(lesion_c.sum()), "affected_pct_initial": 100 * float(lesion_c.sum() / leaf_c.sum()),
                "ssim_non_lesion": ssim_out, "mask_scale": 0.5, "files": files,
                "sources": [p for p, _ in LEAVES], "backdrop": BACKDROP}
        (OUT / f"heal_{vname}.json").write_text(json.dumps(meta, indent=1))
        print(f"{vname}: affected {meta['affected_pct_initial']:.1f}%  SSIM(non-lesion) {ssim_out:.4f}")
        previews.append((dis, hea))
        if ssim_out < 0.98:
            print("alignment check FAILED")
            return 1
    # preview: 21x9 diseased | healed, plus a zoomed crop of each
    dis, hea = previews[0]
    H, W = dis.shape[:2]
    # zoom on the most lesion-dense window
    lm = cv2.cvtColor(np.abs(dis.astype(int) - hea.astype(int)).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    dens = cv2.boxFilter(lm.astype(np.float32), -1, (W // 5, H // 2))
    cy, cx = np.unravel_index(int(np.argmax(dens)), dens.shape)
    x0, y0 = int(np.clip(cx - W // 10, 0, W - W // 5)), int(np.clip(cy - H // 4, 0, H - H // 2))
    crop = (slice(y0, y0 + H // 2), slice(x0, x0 + W // 5))
    top = np.hstack([cv2.resize(dis, (W // 2, H // 2)), cv2.resize(hea, (W // 2, H // 2))])
    z = [cv2.resize(x[crop], (W // 2, int(W // 2 * (crop[0].stop - crop[0].start) / (crop[1].stop - crop[1].start))))
         for x in (dis, hea)]
    cv2.imwrite(str(path("figures", cfg) / "heal_preview.jpg"), np.vstack([top, np.hstack(z)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("saved results/figures/heal_preview.jpg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
