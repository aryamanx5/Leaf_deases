import cv2
import numpy as np
import pandas as pd
import torch

from leafdx.cnn.dataset import CLASSES4, LeafDataset, diseased_prob, eval_transform, train_transform
from leafdx.cnn.model import build_model, load_checkpoint, save_checkpoint, set_trainable
from leafdx.config import load_config
from tests.synthetic import make_leaf

CFG = load_config()


def test_model_forward_and_checkpoint_roundtrip(tmp_path):
    model, _ = build_model("mobilenet_v2", len(CLASSES4), pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 4)
    set_trainable(model, 0)
    assert not any(p.requires_grad for p in model.features.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())
    ck = tmp_path / "m.pt"
    save_checkpoint(model, ck, {"arch": "mobilenet_v2", "classes": list(CLASSES4), "image_size": 224, "epoch": 0})
    m2, meta = load_checkpoint(ck)
    assert torch.allclose(m2(x), out, atol=1e-5)
    assert meta["classes"] == list(CLASSES4)


def test_dataset_and_transforms_on_synthetic_images(tmp_path, monkeypatch):
    rows = []
    for i, stratum in enumerate(CLASSES4):
        img, _, _ = make_leaf("paper", size=300, seed=i)
        p = tmp_path / f"{i}.png"
        cv2.imwrite(str(p), img)
        rows.append({"path": str(p), "stratum": stratum})
    monkeypatch.setattr("leafdx.cnn.dataset.ROOT", tmp_path)  # absolute paths ignore ROOT anyway
    ds = LeafDataset(pd.DataFrame(rows), train_transform(CFG["cnn"]), cache_side=256, n_jobs=1)
    x, y = ds[3]
    assert x.shape == (3, 224, 224) and y == 3
    ds.transform = eval_transform(CFG["cnn"])
    assert torch.equal(ds[0][0], ds[0][0])  # eval transform is deterministic


def test_diseased_prob_sums_diseased_classes():
    p = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    assert np.isclose(diseased_prob(p).item(), 0.2 + 0.4)
