/** Typed access to /data/site.json (exported by scripts/export_site_data.py). */
import { useEffect, useState } from 'react';

export type Sources = { webp: Record<string, string>; avif: Record<string, string>; w: number; h: number };
export type Method = 'rule_based' | 'svm' | 'cnn';
export type Metrics = {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  macro_f1: number;
  healthy_recall: number;
  roc_auc?: number;
  n: number;
};
export type StepKey = 'original' | 'denoised' | 'colour_space' | 'segmentation' | 'threshold' | 'morphology' | 'lesions';
export type Showcase = {
  source: string;
  crop: 'turmeric' | 'blackgram';
  media: Record<StepKey | 'segmented' | 'gradcam' | 'cnn_input', Sources>;
  params: {
    denoise: { median_ksize: number; gaussian_ksize: number; clahe_clip: number };
    colour_space: { a_star_min: number; a_star_max: number };
    segmentation: { mode: string; border_background_share: number };
    threshold: { otsu_distance: number | null };
    morphology: { open_ksize: number; close_ksize: number; leaf_coverage: number };
    lesions: { count: number; affected_pct: number; min_area_px: number };
  };
  readouts: {
    affected_pct: number;
    lesion_count: number;
    abn_colour_pct: number;
    glcm_contrast: number;
    glcm_energy: number;
    glcm_homogeneity: number;
    lesion_circularity_mean: number;
    class_pct: Record<'yellow' | 'brown' | 'dark' | 'pale', number>;
  };
  cnn: {
    tiles: { early: string[]; late: string[]; early_shape: number[]; late_shape: number[] };
    p_crop: Record<'turmeric' | 'blackgram', number>;
    crop_status: string;
    crop_pred: 'turmeric' | 'blackgram';
    p_diseased_given_crop: number;
  };
};
export type Heal = {
  variant: string;
  width: number;
  height: number;
  leaf_px: number;
  lesion_px: number;
  affected_pct_initial: number;
  files: { diseased: string[]; healed: string[] };
};
export type Sample = {
  id: string;
  crop: 'turmeric' | 'blackgram';
  orig_class: string;
  label: 'healthy' | 'diseased';
  src: string;
  thumb: string;
  source: string;
};
export type Site = {
  generated_at: string;
  dataset: {
    total: number;
    by_crop_label: Record<string, number>;
    by_class: Record<string, number>;
    splits: Record<string, number>;
    clusters: number;
    multi_image_clusters: number;
    wild_test: number;
    wild_test_labelled: number;
  };
  comparison: {
    clean: Record<Method, Metrics>;
    clean_ms: Record<Method, number>;
    wild: Record<Method, Metrics>;
    mcnemar_p: number;
    test_n: number;
  };
  leakage: Record<'clean' | 'naive_random' | 'augment_then_split', Record<'svm' | 'cnn_head', { accuracy: number; macro_f1: number; dup_share: number }>>;
  robustness: {
    before: Record<string, number | null>;
    after: Record<string, number | null>;
    targets: Record<string, { value: number | string | null; met: boolean; target?: number }>;
  };
  calibration: { tau_crop: number; ood_threshold: number };
  showcase: Record<'turmeric' | 'blackgram', Showcase>;
  heal: Record<'21x9' | '9x16', Heal>;
  crops: Record<'turmeric' | 'blackgram', Record<'healthy' | 'diseased', { img: Sources; source: string; orig_class: string }>>;
  hero: {
    credit: { title: string; url: string; licence: string; author: string };
    image: Sources;
    outline: [number, number][];
    anchors: Record<'midrib' | 'venation' | 'outline', [number, number]>;
    stage1: { coverage: number; lesion_count: number; affected_pct: number; method: string };
  };
  example_card: string[] | null;
  samples: Sample[];
};

let cache: Promise<Site> | null = null;
export function loadSite(): Promise<Site> {
  cache ??= fetch('/data/site.json').then((r) => {
    if (!r.ok) throw new Error(`site.json ${r.status}`);
    return r.json() as Promise<Site>;
  });
  return cache;
}

export function useSite(): Site | null {
  const [site, setSite] = useState<Site | null>(null);
  useEffect(() => {
    let alive = true;
    loadSite().then((s) => alive && setSite(s));
    return () => {
      alive = false;
    };
  }, []);
  return site;
}
