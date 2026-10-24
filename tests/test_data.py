import cv2
import numpy as np

from leafdx.data import build_inventory, normalise_class, place_images


def test_normalise_class_strips_counts_and_case():
    assert normalise_class("Leaf Crinckle 150") == "leaf crinckle"
    assert normalise_class("Healthy_Leaf") == "healthy leaf"


def test_place_and_inventory(tmp_path):
    """SYNTHETIC images in an archive-like layout are mapped to crop/label folders."""
    src = tmp_path / "extracted" / "Dataset"
    for cls in ("Healthy 2", "Yellow Mosaic 2", "Rhizome Root"):
        (src / cls).mkdir(parents=True)
        for i in range(2):
            cv2.imwrite(str(src / cls / f"{i}.jpg"), np.full((32, 32, 3), 100, np.uint8))
    (src / "Healthy 2" / "broken.jpg").write_bytes(b"not an image")
    raw = tmp_path / "data" / "raw"
    counts = place_images(tmp_path / "extracted", raw, "blackgram",
                          {"healthy": "healthy", "yellow mosaic": "diseased", "rhizome root": None})
    assert counts["healthy"] == 3 and counts["yellow mosaic"] == 2
    assert counts["rhizome root (excluded)"] == 2
    inv = build_inventory(raw, ["blackgram"], ["healthy", "diseased"], rel_to=tmp_path)
    assert len(inv) == 5
    assert inv.readable.sum() == 4
    assert set(inv.orig_class) == {"healthy", "yellow_mosaic"}
