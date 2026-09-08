"""
Aim 3: how much does detectability drop if an attacker tunes trigger
strength down to evade the Aim 1 statistic, and at what cost to attack
effectiveness?

This sweeps trigger_strength as the simplest possible adaptive-attacker
proxy -- NOT a real optimization-based adaptive attack (e.g. directly
optimizing the trigger against the fitted null model's gradient). That's
the honest scope for this starting point; see docs/research_plan.md Aim 3
and the note in the README about what's left for future work.

TODO (real adaptive attack): replace the strength sweep with an actual
optimization loop that adjusts the trigger to minimize batch_anomaly_score
directly, subject to a floor on attack effectiveness.
"""
import numpy as np

from vlm_trojan_cue.probes import generate_probe_battery, CleanObserver, BackdooredObserver
from vlm_trojan_cue.detection import calibrate_null, batch_anomaly_score
from vlm_trojan_cue.eval import roc_auc_with_ci


def run_batch_at_strength(trigger_strength: float, rng, n_models: int = 40, n_calib=600, n_test=200):
    clean_scores, backdoor_scores, attack_effects = [], [], []
    for _ in range(n_models):
        calib_trials = generate_probe_battery(n_calib, rng)
        calib_observer = CleanObserver(seed=int(rng.integers(0, 1_000_000)))
        calib_responses = calib_observer.respond(calib_trials)
        null = calibrate_null(calib_trials, calib_responses)

        clean_test = generate_probe_battery(n_test, rng)
        clean_obs = CleanObserver(seed=int(rng.integers(0, 1_000_000)))
        clean_scores.append(batch_anomaly_score(clean_test, clean_obs.respond(clean_test), null))

        bd_test = generate_probe_battery(n_test, rng)
        bd_obs = BackdooredObserver(
            seed=int(rng.integers(0, 1_000_000)), trigger_frac=0.25, trigger_strength=trigger_strength
        )
        bd_responses = bd_obs.respond(bd_test)
        backdoor_scores.append(batch_anomaly_score(bd_test, bd_responses, null))

        # crude "attack success" proxy: mean override magnitude on trigger trials
        trig_mask = np.array([t.trigger for t in bd_test])
        if trig_mask.any():
            x_v = np.array([t.x_v for t in bd_test])
            attack_effects.append(float(np.mean(np.abs(bd_responses[trig_mask] - x_v[trig_mask]) < 1.0)))

    y_true = np.array([0] * n_models + [1] * n_models)
    scores = np.array(clean_scores + backdoor_scores)
    auc = roc_auc_with_ci(y_true, scores)["auc"]
    return auc, float(np.mean(attack_effects)) if attack_effects else float("nan")


def main(seed: int = 0):
    rng = np.random.default_rng(seed)
    print("=== Aim 3: adaptive-attacker tradeoff (synthetic, trigger-strength sweep) ===")
    print(f"{'strength':>10} {'AUROC':>8} {'attack_success_proxy':>22}")
    for strength in [0.9, 0.7, 0.5, 0.3, 0.15, 0.05]:
        auc, attack_success = run_batch_at_strength(strength, rng)
        print(f"{strength:>10.2f} {auc:>8.3f} {attack_success:>22.3f}")
    print("\nLower strength -> lower AUROC (harder to detect) but also weaker override of the visual cue")
    print("(lower attack_success_proxy) -- the tradeoff curve itself is the Aim 3 result, not a single number.")


if __name__ == "__main__":
    main()
