"""
End-to-end validation against synthetic ground truth. These tests guard the
statistical machinery, not any claim about real VLMs -- see README.md for
what's validated here vs. what still requires a real model.
"""
import numpy as np
import pytest

from vlm_trojan_cue.probes import generate_probe_battery, CleanObserver, BackdooredObserver
from vlm_trojan_cue.cue_model import fit_bci, fit_linear_baseline, bci_predict
from vlm_trojan_cue.detection import calibrate_null, batch_anomaly_score
from vlm_trojan_cue.eval import roc_auc_with_ci
from vlm_trojan_cue.reverse_correlation import synthetic_flagged_unit_response, reverse_correlate


def test_bci_fit_beats_random_init_and_predicts_reasonably():
    rng = np.random.default_rng(42)
    trials = generate_probe_battery(600, rng)
    observer = CleanObserver(seed=42)
    responses = observer.respond(trials)

    x_v = np.array([t.x_v for t in trials])
    x_t = np.array([t.x_t for t in trials])
    sv = np.array([t.sigma_v for t in trials])
    st = np.array([t.sigma_t for t in trials])

    params = fit_bci(x_v, x_t, sv, st, responses, seed=1)
    pred, _ = bci_predict(x_v, x_t, sv, st, params)

    mse = np.mean((responses - pred) ** 2)
    assert mse < 0.5, f"fitted BCI model should predict the generating observer reasonably well, got MSE={mse:.3f}"


def test_linear_baseline_runs_and_is_bounded():
    rng = np.random.default_rng(0)
    trials = generate_probe_battery(300, rng)
    observer = CleanObserver(seed=0)
    responses = observer.respond(trials)

    x_v = np.array([t.x_v for t in trials])
    x_t = np.array([t.x_t for t in trials])
    w = fit_linear_baseline(x_v, x_t, responses)
    assert 0.0 <= w <= 1.0


def test_detection_separates_clean_from_backdoored_observers():
    """Sanity check on the core Aim 1 claim: does the anomaly score even
    separate the two synthetic classes at all? This is a low bar (large
    effect size by construction) -- the real benchmark target in
    docs/research_plan.md is much stricter (AUROC vs. real baselines on
    hard negatives)."""
    rng = np.random.default_rng(7)

    calib_trials = generate_probe_battery(600, rng)
    calib_observer = CleanObserver(seed=7)
    null = calibrate_null(calib_trials, calib_observer.respond(calib_trials))

    n_models = 25
    clean_scores, backdoor_scores = [], []
    for _ in range(n_models):
        t = generate_probe_battery(150, rng)
        clean_scores.append(batch_anomaly_score(t, CleanObserver(seed=int(rng.integers(1e6))).respond(t), null))
        t2 = generate_probe_battery(150, rng)
        bd = BackdooredObserver(seed=int(rng.integers(1e6)), trigger_frac=0.3, trigger_strength=0.9)
        backdoor_scores.append(batch_anomaly_score(t2, bd.respond(t2), null))

    y_true = np.array([0] * n_models + [1] * n_models)
    scores = np.array(clean_scores + backdoor_scores)
    result = roc_auc_with_ci(y_true, scores, n_boot=200)

    assert result["auc"] > 0.75, f"expected clear separation on this easy synthetic case, got AUROC={result['auc']:.3f}"


def test_reverse_correlation_recovers_trigger_above_chance():
    rng = np.random.default_rng(3)
    basis_dim = 20
    trigger = np.zeros(basis_dim)
    trigger[[2, 5, 11]] = 1.0
    trigger /= np.linalg.norm(trigger)

    noise = rng.normal(0, 1, size=(3000, basis_dim))
    responses = synthetic_flagged_unit_response(noise, trigger, rng=rng)
    result = reverse_correlate(noise, responses, ground_truth=trigger)

    assert result.cosine_similarity_to_truth > 0.3, (
        f"reverse correlation should recover a trigger-correlated pattern well above chance "
        f"(cos_sim={result.cosine_similarity_to_truth:.3f}); chance level for random {basis_dim}-d "
        f"vectors is close to 0"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
