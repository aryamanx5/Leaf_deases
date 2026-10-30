/** Client for POST /api/analyze (schema v1, api/schemas.py). */
import { API_BASE } from '@/content/brand';

export type Crop = 'turmeric' | 'blackgram';
export type CropChoice = 'auto' | Crop;
export type Health = 'healthy' | 'diseased';

export type AnalyzeResponse = {
  schema_version: string;
  request_id: string;
  model_version: string;
  crop: { label: Crop; confidence: number; probs: Record<Crop, number>; status: 'confident' | 'uncertain' | 'user_selected'; source: 'model' | 'user' };
  disease: {
    label: Health;
    confidence: number;
    by: {
      cnn: { label: Health; prob: number };
      svm: { label: Health; prob: number };
      rule: { label: Health; score: number; threshold: number; components: Record<string, number> };
    };
  };
  agreement: boolean;
  measurements: {
    affected_region_pct: number;
    lesion_count: number;
    abn_colour_pct: number;
    glcm_contrast: number;
    glcm_energy: number;
    glcm_homogeneity: number;
    mean_lesion_area_px: number;
    segmentation_mode: string;
  };
  abnormal_colour_pct: Record<'yellow' | 'brown' | 'dark' | 'pale', number>;
  images: { original: string; segmented_leaf: string; abnormal_regions: string; gradcam: string | null };
  ood: { is_plant: boolean; ood_score: number; flagged: boolean };
  timings_ms: Record<string, number>;
  /** What each stage did on this photo (optional addition to schema v1). Stage ids match GET /api/code. */
  trace?: {
    input: { filename?: string; width: number; height: number; analysed_width: number; analysed_height: number;
             crop_given: CropChoice; roi: Record<string, number | number[]> | null; plant_colour_fraction: number };
    stages: { id: string; ms: number | null; values: Record<string, number | string | boolean | null> }[];
  };
  /** The plain-text result card (leafdx.service.result_card), as the old Streamlit app showed it. */
  result_card?: string[];
  /** Phase 3 additions: conformal sets, abstention, severity and every model's answer. */
  crop_confidence?: number | null;
  crop_set?: Crop[];
  disease_confidence?: number;
  disease_set?: Health[];
  abstain?: { flag: boolean; reason: string | null };
  affected_pct?: number;
  severity?: 'mild' | 'moderate' | 'severe' | null;
  severity_note?: string;
  arms?: Record<string, ArmAnswer>;
  production_model?: string;
  production_model_name?: string;
  explanation?: string;
  segmenter?: 'classical' | 'mobilesam';
};

export type ArmAnswer = { name: string; label: Health; prob: number; config: string; ms: number };

/** One whitelisted pipeline function from GET /api/code. */
export type CodeItem = { stage: string; name: string; file: string; start_line: number; end_line: number; source: string };

export type ApiErrorCode = 'offline' | 'too_large' | 'not_a_plant' | 'unsupported_type' | 'invalid_image' | 'bad_roi' | 'bad_crop' | 'internal';
export class ApiError extends Error {
  constructor(public code: ApiErrorCode, message: string) {
    super(message);
  }
}

export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024; // mirrors api/main.py MAX_BYTES

export async function analyze(file: Blob, crop: CropChoice, roi: { x: number; y: number } | null, signal?: AbortSignal,
                              name?: string): Promise<AnalyzeResponse> {
  if (file.size > MAX_UPLOAD_BYTES) throw new ApiError('too_large', 'This image is larger than 10 MB. Please upload a smaller photo.');
  const body = new FormData();
  body.append('file', file, name ?? (file as File).name ?? 'upload.jpg');
  body.append('crop', crop);
  if (roi) {
    body.append('roi_x', roi.x.toFixed(4));
    body.append('roi_y', roi.y.toFixed(4));
  }
  let r: Response;
  try {
    r = await fetch(`${API_BASE}/api/analyze`, { method: 'POST', body, signal });
  } catch (e) {
    if ((e as Error).name === 'AbortError') throw e;
    throw new ApiError('offline', 'The analysis service is not reachable.');
  }
  if (r.ok) return (await r.json()) as AnalyzeResponse;
  // the Vite proxy answers 502/504 (or 500 with no JSON body) when the API is down
  const j = (await r.json().catch(() => null)) as { error?: { code: string; message: string } } | null;
  if (!j?.error) throw new ApiError(r.status >= 500 ? 'offline' : 'internal', 'The analysis service is not reachable.');
  throw new ApiError(j.error.code as ApiErrorCode, j.error.message);
}

export async function health(): Promise<boolean> {
  try {
    const r = await fetch(`${API_BASE}/api/health`);
    return r.ok;
  } catch {
    return false;
  }
}

let codeCache: Promise<CodeItem[]> | null = null;
/** Source of every pipeline stage (read-only, cached for the session). */
export function fetchCode(): Promise<CodeItem[]> {
  codeCache ??= fetch(`${API_BASE}/api/code`)
    .then((r) => {
      if (!r.ok) throw new ApiError('offline', `code ${r.status}`);
      return r.json() as Promise<{ items: CodeItem[] }>;
    })
    .then((j) => j.items)
    .catch((e: unknown) => {
      codeCache = null; // let the next attempt retry
      throw e;
    });
  return codeCache;
}
