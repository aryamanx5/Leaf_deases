"""Final verification: prints a PASS/FAIL table and exits non-zero on any failure.

  --quick : lint + unit tests (no app/pipeline tests) + split leakage check
  (full)  : lint, all tests (incl. pipeline + Streamlit AppTest), split leakage, required
            artefacts, REPORT.md numbers re-read from their sources, comparison.md numbers
            present in comparison.csv.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
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
                   ("comparison.md numbers in comparison.csv", check_comparison_md)]
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
