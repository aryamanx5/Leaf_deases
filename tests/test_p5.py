"""Phase 5 (docs/PLAN_phase5_confidence.md): the mean-of-logits combiner, the risk-targeted refusal threshold, and
the consistency of the deployed bundle with what the selection and evaluation recorded."""

import json

import joblib
import numpy as np
import pytest

from leafdx.config import ROOT
from leafdx.p3.stats import conformal_sets, logit, sigmoid
from leafdx.p5 import (
    MeanLogit,
    confidence_summary,
    ensemble_logit,
    error_upper_bound,
    nll,
    selective_threshold,
)

MAN = ROOT / "models" / "manifest.json"
SEL = ROOT / "results" / "p5" / "selection.json"
EV = ROOT / "results" / "p5" / "eval.json"


def test_mean_logit_is_the_plain_mean_of_log_odds():
    p = np.array([[0.9, 0.8, 0.99], [0.2, 0.4, 0.1]])
    X = logit(p)
    out = MeanLogit().predict_proba(X)
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out.sum(1), 1.0)
    np.testing.assert_allclose(out[:, 1], sigmoid(X.mean(1)))
    np.testing.assert_allclose(logit(out[:, 1]), ensemble_logit([p[:, 0], p[:, 1], p[:, 2]]), atol=1e-9)
    # one member: the ensemble is that member
    np.testing.assert_allclose(MeanLogit().predict_proba(logit(p[:, :1]))[:, 1], p[:, 0], atol=1e-9)


def test_error_upper_bound_matches_known_clopper_pearson_values():
    assert error_upper_bound(0, 0) == 1.0
    assert error_upper_bound(5, 5) == 1.0
    # zero errors in n: the bound is 1 - 0.05 ** (1 / n) (the "rule of three" is its approximation)
    for n in (10, 100, 300):
        assert error_upper_bound(0, n) == pytest.approx(1 - 0.05 ** (1 / n), rel=1e-9)
    assert error_upper_bound(2, 100) > error_upper_bound(1, 100) > error_upper_bound(0, 100)
    assert error_upper_bound(2, 400) < error_upper_bound(2, 100)


def test_selective_threshold_takes_the_lowest_safe_cut_and_falls_back():
    rng = np.random.default_rng(0)
    conf = np.r_[rng.uniform(0.9, 1.0, 400), rng.uniform(0.5, 0.9, 40)]
    correct = np.r_[np.ones(400, bool), rng.uniform(size=40) < 0.5]  # the unsure ones are a coin flip
    tau, rec = selective_threshold(conf, correct, 0.5, 0.99, 0.02)
    assert rec["from_rule"] and 0.5 <= tau <= 0.99
    m = conf >= tau
    assert error_upper_bound(int((~correct[m]).sum()), int(m.sum())) <= 0.02
    assert rec["answered"] == int(m.sum()) and rec["errors"] == int((~correct[m]).sum())
    # every lower candidate fails the bound: tau is the lowest safe one
    for lower in sorted(c for c in conf if 0.5 <= c < tau):
        k = conf >= lower
        assert error_upper_bound(int((~correct[k]).sum()), int(k.sum())) > 0.02
    # an impossible target leaves the threshold where it was
    tau2, rec2 = selective_threshold(conf, np.zeros_like(correct), 0.5, 0.97, 0.02)
    assert tau2 == 0.97 and not rec2["from_rule"]
    # an extra condition can veto low thresholds
    tau3, _ = selective_threshold(conf, correct, 0.5, 0.99, 0.02, extra_ok=lambda t: t >= 0.95)
    assert tau3 >= 0.95


def test_confidence_summary_and_nll():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.6, 0.2, 0.7])
    s = confidence_summary(p, y)
    assert s["mean_confidence_right"] == pytest.approx(np.mean([0.9, 0.6, 0.8]))
    assert s["mean_confidence_wrong"] == pytest.approx(0.7)
    assert nll(p, y) == pytest.approx(-np.mean(np.log([0.9, 0.6, 0.8, 0.3])))


needs_p5 = pytest.mark.skipif(not (MAN.exists() and SEL.exists() and EV.exists()), reason="Phase 5 not run")


@needs_p5
def test_deployed_bundle_is_what_phase5_selected_and_the_switch_rule_held():
    man, sel, ev = (json.loads(f.read_text()) for f in (MAN, SEL, EV))
    if man["production"]["disease"] != "STACK" or "p5" not in man:
        pytest.skip("the deployed bundle is not the Phase 5 one")
    win = sel["selection"]["winner"]
    assert man["p5"]["candidate"] == win and man["stacker"]["members"] == sel["candidates"][win]["members"]
    st = joblib.load(ROOT / man["stacker"]["file"])
    assert isinstance(st["meta"], MeanLogit) and st["members"] == man["stacker"]["members"]
    assert st["temperature"] == pytest.approx(sel["candidates"][win]["ensemble_temperature"])
    for a in st["members"]:  # every member is in the bundle, with the temperature the selection fitted
        b = joblib.load(ROOT / man["arms"][a]["file"])
        assert b["temperature"] == pytest.approx(sel["member_temperatures"][a])
        assert b["config"].endswith("[pooled,orig]")
    # the thresholds in the manifest are the selected ones, never looser than the site's "Low" cut-off
    tr = man["trust"]
    assert 1 - tr["disease_q_lac"] == pytest.approx(sel["thresholds"]["disease"]["tau"])
    assert 1 - tr["crop_q_lac"] == pytest.approx(sel["thresholds"]["crop"]["tau"])
    assert 1 - tr["disease_q_lac"] >= 0.70
    assert sel["thresholds"]["disease"]["error_upper_bound"] <= tr["threshold_rule"]["disease_max_error"]
    assert sel["thresholds"]["crop"]["other_species_refused"] >= sel["thresholds"]["crop"]["old"]["other_species_refused"]
    # the switch rule, re-derived from the recorded numbers
    sw, g = ev["switch"], ev["trust_p4_test"]
    assert sw["switched"] and all(sw[k] for k in ("f1_no_loss", "answers_more", "accuracy_answered_ok",
                                                  "clean_test_within_0.02", "other_species_ok"))
    assert sw["p4_test_diff"] >= -0.005 and sw["p4_test_ci95"][0] > -0.02
    assert g["new"]["answered_share_labelled"] > g["old"]["answered_share_labelled"]
    assert g["new"]["end_to_end_accuracy_answered"] >= 0.97
    assert sw["clean_test_new"] >= sw["clean_test_old"] - 0.02
    assert "second read" in ev["protocol"]


@needs_p5
def test_a_threshold_stored_as_lac_q_answers_exactly_at_or_above_tau():
    man = json.loads(MAN.read_text())
    q = man["trust"]["disease_q_lac"]
    tau = 1 - q
    p = np.array([tau + 1e-6, tau - 1e-6, 1 - tau - 1e-6, 1 - tau + 1e-6, 0.5])
    one = conformal_sets(np.column_stack([p, 1 - p]), q).sum(1) == 1
    assert one.tolist() == [True, False, True, False, False]
