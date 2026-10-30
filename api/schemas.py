"""Versioned response schema for POST /api/analyze (v1)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1"


class CropOut(BaseModel):
    label: Literal["turmeric", "blackgram"]
    confidence: float = Field(ge=0, le=1)
    probs: dict[str, float]
    status: Literal["confident", "uncertain", "user_selected"]
    source: Literal["model", "user"]


class LabelProb(BaseModel):
    label: Literal["healthy", "diseased"]
    prob: float


class RuleOut(BaseModel):
    label: Literal["healthy", "diseased"]
    score: float
    threshold: float
    components: dict[str, float]


class DiseaseBy(BaseModel):
    cnn: LabelProb
    svm: LabelProb
    rule: RuleOut


class DiseaseOut(BaseModel):
    label: Literal["healthy", "diseased"]
    confidence: float
    by: DiseaseBy


class Measurements(BaseModel):
    affected_region_pct: float
    lesion_count: int
    abn_colour_pct: float
    glcm_contrast: float
    glcm_energy: float
    glcm_homogeneity: float
    mean_lesion_area_px: float
    segmentation_mode: str


class Images(BaseModel):
    original: str
    segmented_leaf: str
    abnormal_regions: str
    gradcam: str | None


class OOD(BaseModel):
    is_plant: bool
    ood_score: float
    flagged: bool


class StageTrace(BaseModel):
    id: str  # matches GET /api/code stage ids
    ms: float | None
    values: dict[str, Any]


class Trace(BaseModel):
    input: dict[str, Any]
    stages: list[StageTrace]


class ArmOut(BaseModel):
    name: str
    label: Literal["healthy", "diseased"]
    prob: float = Field(ge=0, le=1)
    config: str
    ms: float


class Abstain(BaseModel):
    flag: bool
    reason: str | None


class ExplainResponse(BaseModel):
    request_id: str
    crop: Literal["turmeric", "blackgram"]
    cnn_arm: str
    gradcam_pp: str
    vit_patch_map: str
    lesion_overlay: str
    explanation: str
    affected_pct: float
    severity: str | None


class AnalyzeResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    request_id: str
    model_version: str
    crop: CropOut
    disease: DiseaseOut
    agreement: bool
    measurements: Measurements
    abnormal_colour_pct: dict[str, float]
    images: Images
    ood: OOD
    timings_ms: dict[str, float]
    # optional additions (schema v1 stays compatible): per-stage trace and the plain-text result card
    trace: Trace | None = None
    result_card: list[str] | None = None
    # Phase 3 additions (present when the Phase 3 bundle, models/manifest.json, is installed)
    crop_confidence: float | None = None
    crop_set: list[Literal["turmeric", "blackgram"]] | None = None
    disease_confidence: float | None = None
    disease_set: list[Literal["healthy", "diseased"]] | None = None
    abstain: Abstain | None = None
    affected_pct: float | None = None
    severity: Literal["mild", "moderate", "severe"] | None = None
    severity_note: str | None = None
    arms: dict[str, ArmOut] | None = None
    production_model: str | None = None
    production_model_name: str | None = None
    explanation: str | None = None
    segmenter: Literal["classical", "mobilesam"] | None = None
    quality: dict[str, Any] | None = None


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody
