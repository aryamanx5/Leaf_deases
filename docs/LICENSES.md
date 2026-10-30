# Licences of pretrained weights and data (Phase 3, P3-M00 / P3-M10)

Checked on 2026-10-30 from the model cards / repositories the weights are downloaded from.
`scripts/verify_all.py` checks that this file exists and names every backbone in `configs/default.yaml` (`p3.backbones`, `p3.cnn_arms`).

## Pretrained weights

| Model (config name) | Source | Weights licence | Used for |
|---|---|---|---|
| DINOv2 ViT-S/14 (`dinov2_vits14`) | `facebookresearch/dinov2` via torch.hub, `dinov2_vits14_pretrain.pth` | Apache-2.0 (repository LICENSE; Meta relicensed code and weights from CC-BY-NC to Apache-2.0 in 2023) | arm D, fusion F, kNN OOD gate, ViT patch map |
| OpenCLIP ViT-B/32 LAION-2B (`openclip_vitb32`) | Hugging Face `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | MIT (model card) | arm E, crop ID candidates |
| BioCLIP 2 (`bioclip2`) | Hugging Face `imageomics/bioclip-2` | MIT (model card) | arm E, crop ID (zero-shot and head) |
| MobileNetV2 (`mobilenet_v2`) | torchvision `IMAGENET1K_V2` | BSD-3-Clause (torchvision); weights trained on ImageNet-1k, whose terms restrict use to non-commercial research | arm C1 |
| EfficientNet-B0 (`efficientnet_b0`) | torchvision `IMAGENET1K_V1` | as above | arm C2 |
| ConvNeXt-Tiny (`convnext_tiny`) | torchvision `IMAGENET1K_V1` | as above | arm C3 |

Not used: DINOv3 (custom licence, not checked in detail because it is not needed), PlantCLEF DINOv2 weights,
MobileSAM (P3-M06 skipped, see PROGRESS.md).

## Data

| Data | Licence | Role |
|---|---|---|
| Turmeric Plant Disease, Mendeley g46dvrcvwn v2 (Data in Brief 59, 2025) | CC BY 4.0 | train / val / test |
| BPLD black gram, Mendeley zfcv9fmrgv v3 (Data in Brief 45, 2022) | CC BY 4.0 | train / val / test |
| Turmeric leaf disease, Mendeley jtttfbx342 v2 | CC BY 4.0 | wild set (evaluation only) |
| Black gram leaf disease, Mendeley z55yrbmn2d v3 | CC BY 4.0 | wild set (evaluation only) |
| iNaturalist observations (CC0 / CC BY / CC BY-SA) and Wikimedia Commons files | per file, recorded in data/wild/SOURCES.csv | wild set, crop label only |
| User-reported Google image (`data/wild/user_reported/turmeric_google_01.jpg`) | copyright of its owner; not committed | regression check only |

For a college research project these licences are compatible with the use made here. Before any commercial use, review the ImageNet terms that apply to the torchvision weights.
