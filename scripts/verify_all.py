"""Final verification: prints a PASS/FAIL table and exits non-zero on any failure.

  --quick : lint + unit tests (no app/pipeline tests) + split leakage check
  (full)  : lint, all tests (incl. pipeline + Streamlit AppTest), split leakage, required
            artefacts, REPORT.md numbers re-read from their sources, comparison.md numbers
            present in comparison.csv; Phase 2: regression test, robustness targets (warn only),
            Phase 2 artefacts, site.json freshness, no hard-coded metrics in the web copy,
            web typecheck/lint/build, Playwright E2E (incl. healing interaction) against a
            temporary API + production build.
            Phase 3: pHash leakage over train/val/test/wild/user (0 shared groups), Phase 3
            artefacts, licences file covers every backbone, final_report.md + README block equal
            a fresh render from results JSON, API contract + smoke on three wild images
            (tests/test_p3_api.py, incl. the user-reported turmeric image).
  --no-web : skip the web build and E2E checks (Python-only machines)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

import pandas as pd

from leafdx.config import ROOT, load_config, path

ARTEFACTS = [
    "results/comparison.csv", "results/comparison.md", "results/REPORT.md", "results/ip_metrics.json",
    "results/cnn_metrics.json", "models/ip_rule.joblib", "models/ip_svm.joblib", "models/cnn_best.pt",
    "app/streamlit_app.py", "data/processed/splits.csv", "data/processed/features.csv",
    "results/figures/segmentation_samples.png", "results/figures/lesion_samples_diseased.png",
    "results/figures/cnn_training.png", "results/figures/gradcam_samples.png",
    "results/figures/comparison_bars.png", "results/figures/confusion_matrices.png",
    "results/figures/roc_pr_curves.png", "results/figures/errors_gallery.png",
    "results/figures/disagreements_gallery.png",
]
P2_ARTEFACTS = [
    "results/diagnosis.md", "results/robustness.md", "results/robustness.json", "results/calibration.json",
    "results/crop_model_selection.csv", "results/api_latency.json", "results/web_quality.json",
    "models/MODEL_CARD.md", "models/crop_model.joblib", "models/disease_cnn.pt", "models/calibration.json",
    "models/ood_reference.npy", "api/main.py", "api/schemas.py", "web/public/data/site.json",
]
# results files that scripts/export_site_data.py reads: site.json must be newer than all of them
SITE_SOURCES = ["results/comparison.csv", "results/robustness.json", "results/calibration.json",
                "data/processed/splits.csv", "data/processed/split_meta.json", "web/public/media/heal/heal_21x9.json",
                "web/public/media/heal/heal_9x16.json"]
P3_ARTEFACTS = [
    "results/crop_models.json", "results/crop_errors.md", "results/disease_models.json", "results/stacking.json",
    "results/trust.json", "results/robustness_p3.json", "results/final_report.md", "results/p3/audit.json",
    "results/p3/leakage_report.md", "results/p3/timing.json", "results/p3/embed_manifest.json", "models/manifest.json",
    "models/p3/crop_head.joblib", "models/p3/stacker.joblib", "models/p3/trust.json", "models/p3/ood_reference.npy",
    "docs/LICENSES.md", "data/wild/labels.csv", "data/wild_ext/labels.csv", "results/wild_ext.json",
]
WEB_COPY_DIRS = ["web/src/sections", "web/src/pages", "web/src/components"]
TIMING_SPECS = ["e2e/perf.spec.ts", "e2e/hero.spec.ts"]
TIMING_GREP = ["no long tasks while scrolling", "scrolling scrubs the clip"]  # titles of the timing tests
API_PORT, PREVIEW_PORT = 8019, 4189  # private ports so a running `make dev` is not disturbed
NUM = re.compile(r"(?<![\w./])-?\d+(?:\.\d+)?%?(?![\w])")


def run(cmd: list[str]) -> tuple[bool, str]:
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    tail = (p.stdout + p.stderr).strip().splitlines()
    return p.returncode == 0, (tail[-1] if tail else "")


def check_split() -> tuple[bool, str]:
    df = pd.read_csv(path("splits", load_config()))
    spans = int((df.groupby("cluster_id")["split"].nunique() > 1).sum())
    table = df.groupby(["stratum", "split"]).size().unstack(fill_value=0)
    ok = spans == 0 and set(table.columns) == {"train", "val", "test"} and bool((table > 0).all().all())
    return ok, f"{spans} clusters span splits; {len(df)} images"


def check_artefacts() -> tuple[bool, str]:
    missing = [a for a in ARTEFACTS if not (ROOT / a).exists()]
    return not missing, "all present" if not missing else f"missing: {missing}"


def _resolve(src: dict) -> float:
    f = ROOT / src["file"]
    if "config" in src:
        v = load_config()
        for k in src["config"]:
            v = v[k]
        return float(v)
    if "keys" in src:
        v = json.loads(f.read_text())
        for k in src["keys"]:
            v = v[k]
        return float(v)
    df = pd.read_csv(f)
    if "query" in src:
        q = src["query"]
        v = df[(df.section == q["section"]) & (df.method == q["method"]) & (df.scope == q["scope"]) & (df.metric == q["metric"])].value
        assert len(v) == 1, q
        return float(v.iloc[0])
    if "count_where" in src:
        for k, val in src["count_where"].items():
            df = df[df[k] == val]
        return float(len(df))
    raise ValueError(src)


def check_report() -> tuple[bool, str]:
    from leafdx.compare import fmt  # noqa: PLC0415

    rep = (ROOT / "results/REPORT.md").read_text()
    recs = json.loads((ROOT / "results/report_numbers.json").read_text())
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_report import build  # noqa: PLC0415

    fresh, _ = build()
    if fresh != rep:
        diff = next(i for i, (a, b) in enumerate(zip(fresh.splitlines(), rep.splitlines(), strict=False)) if a != b) \
            if any(a != b for a, b in zip(fresh.splitlines(), rep.splitlines(), strict=False)) else None
        return False, f"REPORT.md differs from a fresh render of the results files (first differing line {diff})"
    stale = []
    for r in recs:
        v = _resolve(r["source"])
        text = fmt(v, r["source"]["query"]["metric"]) if r["fmt"] == "compare.fmt" else format(int(v) if r["fmt"] == "d" else v, r["fmt"])
        if text != r["text"] or r["text"] not in rep:
            stale.append((r["text"], text, r["source"]))
    known = {r["text"] for r in recs}
    body = "\n".join(line for line in rep.splitlines() if not line.startswith(("- **Black gram:**", "- **Turmeric:**")))
    body = re.sub(r"`[^`]*`|\(figures/[^)]*\)", "", body)  # code spans and figure paths are not numbers
    unknown = sorted({t for t in NUM.findall(body)
                      if (len(t.strip("-%")) > 1 or "." in t) and t not in known and t.rstrip("%") not in known})
    ok = not stale and not unknown
    msg = f"identical to a fresh render; {len(recs)} numbers re-read from sources"
    if stale:
        msg += f"; {len(stale)} stale e.g. {stale[:2]}"
    if unknown:
        msg += f"; untraced numbers: {unknown[:10]}"
    return ok, msg


def check_comparison_md() -> tuple[bool, str]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_comparison import check_md_numbers  # noqa: PLC0415

    bad = check_md_numbers((ROOT / "results/comparison.md").read_text(), pd.read_csv(ROOT / "results/comparison.csv"))
    return not bad, "all numbers found in comparison.csv" if not bad else f"not in csv: {bad[:10]}"


def check_p2_artefacts() -> tuple[bool, str]:
    missing = [a for a in P2_ARTEFACTS if not (ROOT / a).exists()]
    return not missing, "all present" if not missing else f"missing: {missing}"


def check_targets() -> tuple[bool, str]:
    """Robustness targets: a missed target is reported (WARN) but does not fail, as long as it is reported."""
    t = json.loads((ROOT / "results/robustness.json").read_text())["targets"]
    missed = [k for k, v in t.items() if not v["met"]]
    return True, f"{len(t) - len(missed)}/{len(t)} met" + (f"; WARN missed (reported in robustness.md): {missed}" if missed else "")


def check_site_fresh() -> tuple[bool, str]:
    site = ROOT / "web/public/data/site.json"
    gen = datetime.fromisoformat(json.loads(site.read_text())["generated_at"]).timestamp()
    stale = [s for s in SITE_SOURCES if (ROOT / s).stat().st_mtime > gen + 1]
    return not stale, "site.json newer than its sources" if not stale else f"re-run make site-data; newer: {stale}"


# a hard-coded metric = a decimal or a percentage written into page copy instead of read from site.json / the API
METRIC = re.compile(r"(?<![\w.#-])\d+\.\d+%?(?![\w])|(?<![\w.#-])\d+(?:\.\d+)?%")
CSS_HINT = re.compile(r"calc\(|clamp\(|scale|translate|rgba?\(|cubic-bezier|inset\(|var\(--|px|vh|vw|deg|ms\b|s ease")
LITERAL = re.compile(r"'([^'\n]*)'|\"([^\"\n]*)\"|`([^`]*)`|>([^<>{}]+)<")
# layout contexts, not copy: SVG/presentation attributes, ScrollTrigger positions, inline style writes, doc comments
LAYOUT_ATTR = re.compile(r"\b(d|viewBox|points|stroke\w*|fill\w*|opacity|x\d?|y\d?|r|width|height)=$")
LAYOUT_LINE = re.compile(r"^\s*(/\*|\*)|\.style\.|setProperty\(")
TRIGGER_POS = re.compile(r"^(top|bottom|center)\b")
# a property line inside a React style={{...}} object is layout, like the .style writes above
STYLE_PROP_LINE = re.compile(r"^\s*(top|left|right|bottom|width|height|maxWidth|minHeight|inset|transform|opacity)\s*:")
# reviewed exceptions: (file, literal) -> why it is not a reported metric
METRIC_ALLOW = {("web/src/components/analyze/Verdict.tsx", "under 1%"): "wording for an area below one percent"}


def check_no_hardcoded_metrics() -> tuple[bool, str]:
    hits = []
    for d in WEB_COPY_DIRS:
        for f in sorted((ROOT / d).rglob("*.tsx")):
            for i, line in enumerate(f.read_text().splitlines(), 1):
                code = line.split("//")[0] if not line.strip().startswith("{/*") else ""
                if LAYOUT_LINE.search(code) or STYLE_PROP_LINE.search(code):
                    continue
                for m in LITERAL.finditer(code):
                    text = next(g for g in m.groups() if g is not None)
                    before = code[max(0, m.start() - 16):m.start()]
                    if ("className" in before or LAYOUT_ATTR.search(before) or CSS_HINT.search(text)
                            or TRIGGER_POS.match(text)):
                        continue
                    if METRIC.search(text) and (str(f.relative_to(ROOT)), text.strip()) not in METRIC_ALLOW:
                        hits.append(f"{f.relative_to(ROOT)}:{i}: {text.strip()[:60]}")
    return not hits, "none in sections/pages/components" if not hits else f"{len(hits)}: {hits[:6]}"


def run_web(cmd: list[str], env: dict | None = None) -> tuple[bool, str]:
    p = subprocess.run(cmd, cwd=ROOT / "web", capture_output=True, text=True, env={**os.environ, **(env or {})})
    lines = [x for x in (p.stdout + p.stderr).strip().splitlines() if x.strip()]
    return p.returncode == 0, (lines[-1].strip() if lines else "")


def _wait(url: str, secs: float) -> bool:
    end = time.time() + secs
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except OSError:
            time.sleep(0.5)
    return False


def check_e2e() -> tuple[bool, str]:
    """Playwright suite against a temporary API + `vite preview` of the production build (needs web-build first)."""
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"}
    api = subprocess.Popen([str(ROOT / ".venv/bin/uvicorn"), "api.main:app", "--host", "127.0.0.1", "--port", str(API_PORT)],
                           cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    web = subprocess.Popen(["npx", "vite", "preview", "--port", str(PREVIEW_PORT), "--strictPort"], cwd=ROOT / "web",
                           env={**os.environ, "VITE_API_PROXY": f"http://127.0.0.1:{API_PORT}"},
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        base = f"http://localhost:{PREVIEW_PORT}"
        if not (_wait(f"http://127.0.0.1:{API_PORT}/api/health", 90) and _wait(f"{base}/api/health", 30)):
            return False, "API or preview server did not start"
        env = {"E2E_BASE_URL": base, "PERF_URL": base + "/"}
        # functional specs in parallel; the frame-timing specs (long tasks, scroll-scrubbed video) alone on one
        # worker afterwards -- parallel browser workers starve the main thread they measure
        ok1, msg1 = run_web(["npx", "playwright", "test", f"--grep-invert={'|'.join(TIMING_GREP)}"], env)
        ok2, msg2 = run_web(["npx", "playwright", "test", "--workers=1", *TIMING_SPECS], env)
        return ok1 and ok2, f"functional: {msg1} | timing (1 worker): {msg2}"
    finally:
        for proc in (web, api):
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def check_p3_artefacts() -> tuple[bool, str]:
    missing = [a for a in P3_ARTEFACTS if not (ROOT / a).exists()]
    man = json.loads((ROOT / "models/manifest.json").read_text()) if (ROOT / "models/manifest.json").exists() else {}
    missing += [v["file"] for v in man.get("arms", {}).values() if not (ROOT / v["file"]).exists()]
    return not missing, "all present" if not missing else f"missing: {missing}"


def check_field_val_disjoint() -> tuple[bool, str]:
    """docs/PLAN_field_validation.md: field_val shares no image, field scene or capture minute with a test set."""
    fv_f = ROOT / "data/field_val/labels.csv"
    if not fv_f.exists():
        return True, "no field_val (re-selection not run)"
    fv, ext = pd.read_csv(fv_f), pd.read_csv(ROOT / "data/wild_ext/labels.csv")
    wild = pd.read_csv(ROOT / "data/wild/splits.csv")
    minute = re.compile(r"(\d{8}_\d{4})")

    def keys(df, col):
        return {(v.split(":")[0], m.group(1)) for v in df[col] if (m := minute.search(str(v)))}

    paths = set(fv.path) & (set(ext.path) | set(wild.path))
    scenes = {g for g in fv.src_group if g.startswith("45djgf3p96:")} & {g for g in ext.src_group if g.startswith("45djgf3p96:")}
    minutes = keys(fv, "src_group") & (keys(ext, "src_group") | {("z55yrbmn2d", m.group(1)) for p in wild.path
                                                                if "z55yrbmn2d" in p and (m := minute.search(p))})
    files = set(fv.path.map(lambda p: p.split("/")[-1])) & set(ext.path.map(lambda p: p.split("/")[-1]))
    ok = not (paths or scenes or minutes or files)
    return ok, (f"{len(fv)} field_val images; 0 shared images / scenes / capture minutes with wild + wild_ext" if ok else
                f"shared: {len(paths)} paths, {len(scenes)} scenes, {len(minutes)} minutes, {len(files)} file names")


def check_licences() -> tuple[bool, str]:
    cfg = load_config()
    text = (ROOT / "docs/LICENSES.md").read_text()
    names = list(cfg["p3"]["backbones"]) + [a["arch"] for a in cfg["p3"]["cnn_arms"]["arms"]]
    miss = [n for n in names if f"`{n}`" not in text]
    return not miss, f"{len(names)} weights listed" if not miss else f"not in docs/LICENSES.md: {miss}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-web", action="store_true")
    args = ap.parse_args()
    py = str(ROOT / ".venv/bin/python")
    checks = [("lint (ruff)", lambda: run([str(ROOT / ".venv/bin/ruff"), "check", "."]))]
    if args.quick:
        checks.append(("unit tests (quick)", lambda: run([py, "-m", "pytest", "-q", "--ignore=tests/test_app.py",
                                                          "--ignore=tests/test_pipeline.py"])))
    else:
        checks.append(("all tests incl. pipeline + app smoke", lambda: run([py, "-m", "pytest", "-q"])))
    checks.append(("split leakage", check_split))
    if not args.quick:
        checks += [("artefacts", check_artefacts), ("REPORT.md numbers vs sources", check_report),
                   ("comparison.md numbers in comparison.csv", check_comparison_md),
                   ("regression: user-reported turmeric image", lambda: run([py, "-m", "pytest", "-q", "-rs", "tests/test_regressions.py"])),
                   ("robustness targets", check_targets), ("Phase 2 artefacts", check_p2_artefacts),
                   ("site.json fresh vs results", check_site_fresh), ("no hard-coded metrics in web copy", check_no_hardcoded_metrics),
                   ("P3 leakage (pHash groups across train/val/test/wild/user)", lambda: run([py, "scripts/p3_audit.py"])),
                   ("P3 artefacts", check_p3_artefacts), ("P3 licences file", check_licences),
                   ("field_val disjoint from test sets", check_field_val_disjoint),
                   ("P3 report + README numbers vs results JSON", lambda: run([py, "scripts/p3_report.py", "--check"])),
                   ("P3 API contract + 3 wild images + user image", lambda: run([py, "-m", "pytest", "-q", "-rs", "tests/test_p3_api.py"]))]
        if not args.no_web:
            checks += [("web typecheck", lambda: run_web(["npm", "run", "-s", "typecheck"])),
                       ("web lint", lambda: run_web(["npm", "run", "-s", "lint"])),
                       ("web build", lambda: run_web(["npm", "run", "-s", "build"])),
                       ("web E2E (analyze, healing, axe, reduced motion, long tasks)", check_e2e)]
    results = []
    for name, fn in checks:
        try:
            ok, msg = fn()
        except Exception as e:  # a crashing check is a failing check
            ok, msg = False, f"{type(e).__name__}: {e}"
        results.append((name, ok, msg))
    w = max(len(n) for n, _, _ in results)
    print(f"{'CHECK'.ljust(w)}  RESULT  DETAIL")
    for name, ok, msg in results:
        print(f"{name.ljust(w)}  {'PASS' if ok else 'FAIL':6s}  {msg}")
    all_ok = all(ok for _, ok, _ in results)
    print("\nOVERALL:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
