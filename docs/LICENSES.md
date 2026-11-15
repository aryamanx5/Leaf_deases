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
| More originals from jtttfbx342 / z55yrbmn2d (not in data/wild) | CC BY 4.0 | extended field test set (evaluation only) |
| Black gram field scenes + leaf boxes, Mendeley 45djgf3p96 (Hridoy et al.) | CC BY 4.0 | extended field test set: leaf crops (healthy / yellow mosaic) and whole scenes (crop ID) |
| Medicinal plant leaves (aloe, neem, hibiscus, kalanchoe, betel), Mendeley 89rfgtxbdc | CC BY 4.0 | extended field test set: "other plant" refusal test |
| Further originals from jtttfbx342 / z55yrbmn2d, leaf boxes from 45djgf3p96 scenes not used in wild_ext, more 89rfgtxbdc images (`data/field_val`) | CC BY 4.0 | field validation split: model re-selection and calibration only (docs/PLAN_field_validation.md), never training; no image, scene or capture minute shared with wild / wild_ext |
| Phase 4 pool (`data/p4`, docs/PLAN_phase4_more_data.md): data/field_val (minus 196 images that share a capture minute with a test image: 93 black gram, 103 other species), the remaining z55yrbmn2d originals, 45djgf3p96 leaf boxes from scenes unused by wild_ext / field_val | CC BY 4.0 | p4_train (training), p4_val (selection + calibration), p4_test (held-out field test) |
| PhilMedic: Philippine Medicinal Plant Leaf Dataset, Mendeley tsvdyhbphs v1 (class 15 Curcuma longa; 4 leaves of each other species) | CC BY 4.0 | Phase 4: turmeric crop ID and OOD reference (not disease-labelled); other species for the refusal test |
| User-reported Google image (`data/wild/user_reported/turmeric_google_01.jpg`) | copyright of its owner; not committed | regression check only |

## Site footage

Atmosphere clips on the landing page (`web/public/media/film/`, listed in `web/src/lib/film.ts`). They set the scene
only: none is a dataset image and none is shown as a result. Each is credited where it plays.

| Clip | Source | Licence | Used for |
|---|---|---|---|
| `field-*.mp4` (a ginger field; re-encoded to 720p / 540p, first 10 s) | Pexels video 39172117, vijay kumar, https://www.pexels.com/video/lush-ginger-field-in-rural-setting-39172117/ | Pexels Licence (free use, modification allowed, no attribution required) | "The field" interlude |
| `hand-*.mp4` (a hand brushing along maize leaves) | Pexels video 14035105, Feyza Yıldırım, https://www.pexels.com/video/woman-touching-leaves-14035105/ | Pexels Licence | "Field notes" section |
| `dew-*.mp4` (wet leaves) | Pexels video 34796496, Nothing Ahead, https://www.pexels.com/video/close-up-of-dewy-green-leaves-in-garden-setting-34796496/ | Pexels Licence | "Field notes" section |
| `veins-*.mp4` (a backlit leaf with dead patches; first 10 s) | Pexels video 40382375, Bil Hinton, https://www.pexels.com/video/close-up-of-sunlit-leaf-macro-view-40382375/ | Pexels Licence | film reel |
| `rain-*.mp4` (rain on banana leaves; first 8 s) | Pexels video 32089669, Srinivas Kashyap, https://www.pexels.com/video/tropical-rainfall-on-banana-leaves-in-slow-motion-32089669/ | Pexels Licence | film reel |
| `soil-*.mp4` (hands holding soil and a seedling) | Pexels video 11960802, Sergei Skrynnik, https://www.pexels.com/video/close-up-view-of-land-in-hands-11960802/ | Pexels Licence | film reel |
| `farmer-*.mp4` in `media/film/` (a farmer walking a field; first 8 s) | Pexels video 39364382, Hữu Thịnh 79, https://www.pexels.com/video/tranquil-scene-of-a-farmer-in-lush-paddy-field-39364382/ | Pexels Licence | film reel |
| `macro-*.mp4` (leaf veins, very close) | Pexels video 15783159, CW Tan, https://www.pexels.com/video/extreme-close-up-of-a-green-leaf-15783159/ | Pexels Licence | film reel; inside the letters of the "Read the leaf" band |

Posters (`*-poster-*.webp`) are the first frame of each clip. The Pexels Licence does not allow selling or
redistributing the clips on their own; they are part of the site only.

For a college research project these licences are compatible with the use made here. Before any commercial use, review the ImageNet terms that apply to the torchvision weights.
