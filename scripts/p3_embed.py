"""P3-M01: extract and cache embeddings for every image with each frozen backbone.

Inputs per backbone: all clean TRAIN/VAL/TEST, wild and user images (full frame);
leaf-only views (VAL/TEST/wild/user); one background-swap view per VAL/TEST image; and
K background-swap views per TRAIN image. Unchanged inputs are read from the cache
(.cache/embeddings/<model>/<weights hash>/<image hash>.npy). A backbone that cannot be
loaded is skipped and the reason logged. Writes results/p3/embed_manifest.json.
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pandas as pd  # noqa: E402

from leafdx.cnn.augment import build_banks  # noqa: E402
from leafdx.config import load_config, path, seed_everything  # noqa: E402
from leafdx.p3.data import file_hash, image_table  # noqa: E402
from leafdx.p3.embcache import embed_keys, embed_paths, n_cached, weights_hash  # noqa: E402
from leafdx.p3.views import leafonly_paths, load_png, save_train_views, swap_paths, train_view_keys  # noqa: E402


def derived(cfg):
    t = image_table(cfg)
    train = t[t.split == "train"].reset_index(drop=True)
    banks = build_banks(pd.read_csv(path("splits", cfg)).query("split == 'train'"), cfg)
    ev = t[t.split.isin(["val", "test", "wild", "user"])].reset_index(drop=True)
    t0 = time.time()
    lo = leafonly_paths(list(ev.path), cfg)
    vt = t[t.split.isin(["val", "test"])].reset_index(drop=True)
    sw = swap_paths(vt, banks, cfg)
    print(f"derived views: {len(lo)} leaf-only, {sum(s is not None for s in sw)}/{len(sw)} swap ({time.time() - t0:.0f}s)",
          flush=True)
    return t, train, banks, ev, lo, vt, sw


def main() -> int:
    cfg = load_config()
    seed_everything(cfg["seed"])
    t, train, banks, ev, lo, vt, sw = derived(cfg)
    keys_tr = train_view_keys(train, cfg)
    seed_everything(cfg["seed"])
    tv = save_train_views(train, banks, cfg)

    def make_train(idx):
        return [load_png(tv[i]) for i in idx]

    man = {"n_images": len(t), "n_leafonly": len(lo), "n_swap_eval": sum(s is not None for s in sw),
           "n_train_views": len(keys_tr), "backbones": {}}
    for name in cfg["p3"]["backbones"]:
        t0 = time.time()
        before = None
        try:
            wh = weights_hash(name)
            before = n_cached(name)
            embed_paths(name, list(t.path))
            embed_keys(name, [file_hash(p) + "__leafonly" for p in ev.path], lambda idx: [load_png(lo[i]) for i in idx])
            ok = [i for i, s in enumerate(sw) if s is not None]
            embed_keys(name, [file_hash(vt.path[i]) + "__swap" for i in ok], lambda idx, ok=ok: [load_png(sw[ok[i]]) for i in idx])
            embed_keys(name, keys_tr, make_train)
            man["backbones"][name] = {"status": "ok", "weights_hash": wh, "cached_before": before,
                                      "cached_after": n_cached(name), "seconds": round(time.time() - t0, 1)}
        except Exception as e:  # skip and log (e.g. weights not downloadable)
            man["backbones"][name] = {"status": "skipped", "reason": repr(e), "cached_before": before}
        print(name, man["backbones"][name], flush=True)
    (path("results", cfg) / "p3" / "embed_manifest.json").write_text(json.dumps(man, indent=2))
    # swap/leaf-only path index for later scripts
    pd.DataFrame({"path": ev.path, "leafonly": lo}).to_csv(path("results", cfg) / "p3" / "views_leafonly.csv", index=False)
    pd.DataFrame({"path": vt.path, "swap": sw}).to_csv(path("results", cfg) / "p3" / "views_swap.csv", index=False)
    return 0 if all(b["status"] == "ok" for b in man["backbones"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
