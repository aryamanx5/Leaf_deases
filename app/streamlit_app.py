"""Demo app: image processing vs CNN disease detection for turmeric and black gram leaves.

Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from leafdx.config import load_config, path  # noqa: E402
from leafdx.pipeline import CROP_NAMES, InvalidImageError, NotALeafError, analyze, result_card  # noqa: E402

st.set_page_config(page_title="Leaf disease: image processing vs CNN", layout="wide")
CFG = load_config()
RESULTS = path("results", CFG)
FIGS = path("figures", CFG)


@st.cache_resource
def warm_models():
    from leafdx.pipeline import _models

    return _models()


@st.cache_data
def sample_images() -> pd.DataFrame:
    """Two TEST images per crop x state (deterministic), for users without their own photo."""
    s = pd.read_csv(path("splits", CFG))
    pick = s[s.split == "test"].groupby("stratum").sample(2, random_state=CFG["seed"])
    pick["label_txt"] = pick.apply(lambda r: f"{CROP_NAMES[r.crop]} - {r.orig_class.replace('_', ' ')} ({r.path.split('/')[-1]})", axis=1)
    return pick.reset_index(drop=True)


@st.cache_data(show_spinner="Analysing leaf ...")
def run_analysis(data: bytes, crop: str | None):
    return analyze(data, crop)


def badge(ok: bool) -> str:
    return ":green-background[**AGREE**]" if ok else ":orange-background[**DISAGREE**]"


st.title("Leaf disease detection: image processing vs CNN")
st.caption("Turmeric and black gram (urad) leaves - healthy vs diseased. Stage 1 = explicit digital image "
           "processing (segmentation, abnormal-region detection, colour/texture/shape features, rule score + SVM). "
           "Stage 2 = MobileNetV2 CNN. Research prototype, not a diagnostic tool.")

tab_an, tab_res = st.tabs(["Analyze a leaf", "Results"])

with tab_an:
    left, right = st.columns([1, 2])
    with left:
        src = st.radio("Image source", ["Sample from TEST set", "Upload a photo"], horizontal=True)
        data, name = None, None
        if src == "Upload a photo":
            up = st.file_uploader("Leaf photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
            if up is not None:
                data, name = up.getvalue(), up.name
        else:
            samples = sample_images()
            choice = st.selectbox("Sample image", samples.label_txt, index=0)
            p = samples.loc[samples.label_txt == choice, "path"].iloc[0]
            data, name = (ROOT / p).read_bytes(), p
        crop_opt = st.selectbox("Crop", ["auto (CNN decides)", "Turmeric", "Black Gram"])
        crop = {"Turmeric": "turmeric", "Black Gram": "blackgram"}.get(crop_opt)

    if data is None:
        right.info("Upload a leaf photo or pick a sample.")
    else:
        warm_models()
        try:
            res = run_analysis(data, crop)
        except (InvalidImageError, NotALeafError) as e:
            right.error(f"Cannot analyse this image: {e}")
            res = None
        if res is not None:
            with right:
                st.subheader("Result")
                st.code(result_card(res), language=None)
                st.caption(f"Crop {'given by user' if res['crop_source'] == 'user' else 'identified by the CNN'}; "
                           f"prediction: {res['prediction_basis']}.")
            st.markdown("#### Visual evidence: Original -> Segmented Leaf -> Detected Abnormal Regions")
            c1, c2, c3, c4 = st.columns(4)
            c1.image(res["images"]["original"], caption="Original image", width="stretch")
            c2.image(res["images"]["segmented_leaf"], caption="Segmented leaf", width="stretch")
            c3.image(res["images"]["abnormal_regions_overlay"],
                     caption=f"Detected abnormal regions ({res['affected_region_pct']:.1f}% of leaf, "
                             f"{res['lesion_count']} regions)", width="stretch")
            c4.image(res["images"]["gradcam"], caption="CNN Grad-CAM (where the CNN looks)", width="stretch")
            st.caption("Overlay colours: yellow = chlorosis/yellowing, orange-red = brown necrosis, "
                       "magenta = dark tissue, blue = pale (mildew-like); green line = leaf outline.")

            st.markdown(f"#### Image processing vs CNN {badge(res['agreement'])}")
            a, b = st.columns(2)
            ip = res["ip_prediction"]
            a.markdown(f"**Image Processing says {ip['svm']}** (SVM)  \n"
                       f"Rule-based score {ip['rule_score']:.2f} vs threshold {ip['rule_threshold']:.2f} "
                       f"-> **{ip['rule']}**")
            a.dataframe(pd.DataFrame({"component (0-1)": ip["rule_components"]}).round(3), width="stretch")
            b.markdown(f"**CNN says {res['cnn_prediction']}** (confidence {res['cnn_confidence']:.0%})")
            b.dataframe(pd.DataFrame({"probability": res["cnn_class_probabilities"]}).round(3), width="stretch")
            with st.expander("Stage 1 measurements"):
                st.write({"abnormal colour % of leaf": {k: round(v, 2) for k, v in res["abnormal_colour_pct"].items()},
                          "segmentation method": res["segmentation_method"],
                          "Stage 1 time (ms)": round(res["stage1_ms"], 1)})
                st.dataframe(pd.Series(res["features"], name="value").round(4), width="stretch")

with tab_res:
    md = RESULTS / "comparison.md"
    if md.exists():
        st.markdown(md.read_text())
    else:
        st.warning("results/comparison.md not generated yet - run `make compare`.")
    csv = RESULTS / "comparison.csv"
    if csv.exists():
        with st.expander("comparison.csv (every reported number)"):
            st.dataframe(pd.read_csv(csv), width="stretch")
    for fig, cap in [
        ("comparison_bars.png", "Macro-F1 and accuracy per method and crop"),
        ("confusion_matrices.png", "Confusion matrices (TEST)"),
        ("roc_pr_curves.png", "ROC and precision-recall curves (TEST)"),
        ("inference_time.png", "Inference time per image"),
        ("disagreements_gallery.png", "SVM vs CNN disagreements: original | abnormal regions | Grad-CAM"),
        ("errors_gallery.png", "Errors per method (rows: rule-based, SVM, CNN)"),
        ("cnn_training.png", "CNN training curves"),
        ("segmentation_samples.png", "Stage 1 segmentation samples"),
        ("lesion_samples_diseased.png", "Stage 1 abnormal-region detection (diseased samples)"),
        ("gradcam_samples.png", "Grad-CAM samples"),
    ]:
        f = FIGS / fig
        if f.exists():
            st.image(str(f), caption=cap, width="stretch")
