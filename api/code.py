"""Source of the pipeline stages for GET /api/code (the web app's "How this was calculated" panel).

Only the functions in STAGES are ever read, through inspect on the imported objects. No path or name from
the request reaches the file system, so the endpoint cannot be used to read anything else.
"""

from __future__ import annotations

import inspect
from functools import lru_cache
from pathlib import Path

from leafdx import service
from leafdx.config import ROOT
from leafdx.ip import decide, features, lesions, preprocess, segment
from leafdx.robust import infer

# (stage id, function). Stage ids match AnalyzeResponse.trace.stages[].id; a stage may list several functions.
STAGES = (
    ("denoise", preprocess.denoise),
    ("colour", preprocess.preprocess),
    ("segment", segment.otsu_threshold),
    ("segment", segment.threshold_distance),
    ("segment", segment.segment_leaf),
    ("lesions", lesions.detect_lesions),
    ("measure", features.extract_features),
    ("ip_decision", decide.predict_ip),
    ("ip_decision", decide.RuleBasedScorer.components),
    ("cnn", infer.RobustModel.predict),
    ("compare", service.analyze_image),
    ("card", service.result_card),
)
STAGE_IDS = tuple(dict.fromkeys(s for s, _ in STAGES))


@lru_cache(maxsize=1)
def code_items() -> tuple[dict, ...]:
    out = []
    for stage, fn in STAGES:
        lines, start = inspect.getsourcelines(fn)
        out.append({
            "stage": stage,
            "name": fn.__qualname__,
            "file": Path(inspect.getsourcefile(fn)).resolve().relative_to(ROOT.resolve()).as_posix(),
            "start_line": start,
            "end_line": start + len(lines) - 1,
            "source": "".join(lines),
        })
    return tuple(out)
