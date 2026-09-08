"""
Aim 1: can the cue-integration-violation statistic distinguish a backdoored
observer from a clean one, at the batch/model level?

This is a synthetic validation harness -- it establishes that the detection
statistic *can* work in principle before it's benchmarked against real
baselines (gradient-based reverse-engineering, assimilation-based detection)
on real trojan benchmarks (TrojVQA, BackdoorVLM), which is the actual Aim 1
deliverable per docs/research_plan.md.

TODO (real benchmark): swap synthetic observers for real VLMs (clean vs.
trojaned checkpoints from TrojVQA/BackdoorVLM) via model_interface.py, and
add the gradient-based / assimilation-based baselines for comparison.
"""
import numpy as np

from vlm_trojan_cue.probes import generate_probe_battery, CleanObserver, BackdooredObserver
from vlm_trojan_cue.detection import calibrate_null, batch_anomaly_score
from vlm_trojan_cue.eval import roc_auc_with_ci


def run_one_batch(observer_cls, rng, n_calib=600, n_test=200, **observer_kwargs):
    calib_trials = generate_probe_battery(n_calib, rng)
    calib_observer = CleanObserver(seed=int(rng.integers(0, 1_000_000)))
    calib_responses = calib_observer.respond(calib_trials)
    null = calibrate_null(calib_trials, calib_responses)

    test_trials = generate_probe_battery(n_test, rng)
    test_observer = observer_cls(seed=int(rng.integers(0, 1_000_000)), **observer_kwargs)
    test_responses = test_observer.respond(test_trials)
    return batch_anomaly_score(test_trials, test_responses, null)


def main(n_models_per_class: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)

    clean_scores = [run_one_batch(CleanObserver, rng) for _ in range(n_models_per_class)]
    backdoor_scores = [
        run_one_batch(BackdooredObserver, rng, trigger_frac=0.25, trigger_strength=0.85)
        for _ in range(n_models_per_class)
    ]

    y_true = np.array([0] * n_models_per_class + [1] * n_models_per_class)
    scores = np.array(clean_scores + backdoor_scores)

    result = roc_auc_with_ci(y_true, scores)
    print("=== Aim 1: detection benchmark (synthetic clean vs. backdoored observers) ===")
    print(f"Clean model score:     mean={np.mean(clean_scores):.4f}  std={np.std(clean_scores):.4f}")
    print(f"Backdoored model score: mean={np.mean(backdoor_scores):.4f}  std={np.std(backdoor_scores):.4f}")
    print(f"AUROC: {result['auc']:.3f}  (95% CI: {result['ci_low']:.3f}-{result['ci_high']:.3f}, "
          f"n_boot={result['n_boot']})")


if __name__ == "__main__":
    main()
