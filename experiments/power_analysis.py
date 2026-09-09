"""
Power analysis for Aim 1's model-level detection test: how many trials per
model, and how many models per group (clean vs. trojan), does the
anomaly-score-based detector (detection.calibrate_null / batch_anomaly_score)
need to reliably (>= target_power, default 80%) tell clean and trojan
models apart at a given significance level?

--------------------------------------------------------------------------
Read this before trusting a number out of this script
--------------------------------------------------------------------------
Statistical power is only meaningful relative to an assumed EFFECT SIZE --
here, three free parameters, none of which we know the true value of for
real TrojVQA checkpoints yet:
  - `trigger_strength`: how strongly the trigger overrides normal cue
    integration on a trial where it fires.
  - `heterogeneity_sd`: how much clean models naturally differ from each
    other to begin with (jitter on each simulated model's own
    sigma_p/p_common -- the synthetic stand-in for "different clean
    architectures have different native modality-weighting profiles", per
    the apples-to-apples discussion).
  - `trigger_frac`: the odds that a given probe trial happens to contain
    the trigger pattern at all. This one turned out to matter the most in
    practice -- see "realistic_rare" below -- because the probe battery is
    generic (modality-reliability trials), not crafted around any specific
    trigger, so a real dual-key trigger (needs a specific visual patch AND
    phrase co-present) could realistically fire on a much smaller fraction
    of trials than the 0.25 baked into BackdooredObserver's default.

So this script does not report one number -- it reports how the required
(n_trials, n_models) changes across a small grid of scenarios: "optimistic"
/ "moderate" / "pessimistic" vary trigger_strength and heterogeneity_sd
(holding trigger_frac at the library default, 0.25); "realistic_rare" holds
strength/heterogeneity at a middle-of-the-road setting and instead drops
trigger_frac to 0.02, which is arguably the more realistic assumption for
an untargeted battery. Treat "realistic_rare" as the one to plan
storage/compute around, not "pessimistic" -- and revisit this analysis
once you have even a few real anomaly scores, or a real estimate of how
often your probe items incidentally contain the trigger, to calibrate
against.

--------------------------------------------------------------------------
What "one model" costs here
--------------------------------------------------------------------------
Simulating one model = generating a probe battery of `n_trials`, splitting
it into a calibration half (fit the null) and a test half (score against
that null), i.e. one `detection.calibrate_null` + `batch_anomaly_score`
call. Power at a given (n_trials, n_models_per_group) is estimated by
Monte Carlo: repeat the whole clean-group-vs-trojan-group comparison
`n_replicates` times, run a Mann-Whitney U test each time, and report the
fraction of replicates where p < alpha.

--------------------------------------------------------------------------
Runtime
--------------------------------------------------------------------------
Each simulated model costs ~15-20ms (dominated by fit_bci's optimizer).
Total cost is roughly:
    len(scenarios) * len(n_trials_candidates) * (search over n_models
        candidates until target_power is hit, each costing
        2 * n_replicates * n_models_per_group model-simulations)
The default settings are chosen to finish in a few minutes on a laptop;
`--quick` cuts n_replicates and the model-count ceiling for a faster,
noisier estimate, useful for a first pass before committing to the full
run.
"""
import argparse
import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.stats import mannwhitneyu

from vlm_trojan_cue.probes import ProbeTrial, generate_probe_battery, CleanObserver, BackdooredObserver
from vlm_trojan_cue.detection import calibrate_null, batch_anomaly_score
from vlm_trojan_cue.cue_model import BCIParams


SCENARIOS = {
    # (trigger_strength, heterogeneity_sd, trigger_frac) -- see module docstring.
    # trigger_frac is the odds that a GENERIC probe item happens to contain
    # whatever the trigger pattern is. The probe battery is not crafted
    # around any specific trigger, so this is likely the most consequential
    # -- and most optimistic-by-default -- of the three axes: 0.25 (every
    # 4th trial) is a generous assumption. A real dual-key trigger (needs a
    # specific visual patch AND a specific question phrase co-present) could
    # realistically have trigger_frac near 0 in a battery not designed to
    # elicit it. "realistic_rare" makes that explicit instead of leaving it
    # baked into an unstated default.
    "optimistic": dict(trigger_strength=0.85, heterogeneity_sd=0.10, trigger_frac=0.25),
    "moderate": dict(trigger_strength=0.50, heterogeneity_sd=0.30, trigger_frac=0.25),
    "pessimistic": dict(trigger_strength=0.25, heterogeneity_sd=0.60, trigger_frac=0.25),
    "realistic_rare": dict(trigger_strength=0.50, heterogeneity_sd=0.30, trigger_frac=0.02),
}


def _jittered_observer(is_trojan: bool, heterogeneity_sd: float, trigger_strength: float, trigger_frac: float,
                        rng: np.random.Generator):
    """One simulated 'model': draws its own (sigma_p, p_common) with jitter
    around CleanObserver's defaults, representing natural between-model
    diversity in native modality-weighting -- the synthetic stand-in for
    'different clean architectures aren't identical to begin with'."""
    sigma_p = max(0.5, rng.normal(3.0, heterogeneity_sd * 3.0))
    p_common = float(np.clip(rng.normal(0.7, heterogeneity_sd * 0.7), 0.05, 0.95))
    seed = int(rng.integers(0, 2**31 - 1))
    if is_trojan:
        return BackdooredObserver(sigma_p=sigma_p, p_common=p_common, trigger_strength=trigger_strength,
                                   trigger_frac=trigger_frac, seed=seed)
    return CleanObserver(sigma_p=sigma_p, p_common=p_common, seed=seed)


def simulate_model_score(is_trojan: bool, n_trials: int, heterogeneity_sd: float, trigger_strength: float,
                          rng: np.random.Generator, trigger_frac: float = 0.25) -> float:
    """One simulated model's anomaly score: generate a battery, split it in
    half, calibrate the null on one half, score the other half against it."""
    observer = _jittered_observer(is_trojan, heterogeneity_sd, trigger_strength, trigger_frac, rng)
    trials = generate_probe_battery(n_trials, rng)
    responses = observer.respond(trials)

    split = n_trials // 2
    calib_trials, test_trials = trials[:split], trials[split:]
    calib_resp, test_resp = responses[:split], responses[split:]

    null = calibrate_null(calib_trials, calib_resp)
    return batch_anomaly_score(test_trials, test_resp, null)


def power_at(n_trials: int, n_models_per_group: int, heterogeneity_sd: float, trigger_strength: float,
             n_replicates: int, alpha: float, rng: np.random.Generator, trigger_frac: float = 0.25) -> float:
    # NOTE: with n_models_per_group below ~4, a two-sided Mann-Whitney U test
    # cannot reach p < 0.05 no matter how large the true effect is -- that's
    # a property of the test's discrete null distribution at small sample
    # sizes, not a statement about detectability. Expect power=0.00 there
    # regardless of scenario; it is not evidence of a weak effect.
    hits = 0
    for _ in range(n_replicates):
        clean_scores = [simulate_model_score(False, n_trials, heterogeneity_sd, trigger_strength, rng,
                                              trigger_frac=trigger_frac)
                         for _ in range(n_models_per_group)]
        trojan_scores = [simulate_model_score(True, n_trials, heterogeneity_sd, trigger_strength, rng,
                                               trigger_frac=trigger_frac)
                          for _ in range(n_models_per_group)]
        _, p = mannwhitneyu(clean_scores, trojan_scores, alternative="two-sided")
        if p < alpha:
            hits += 1
    return hits / n_replicates


def find_minimum_n_models(n_trials: int, model_candidates: List[int], heterogeneity_sd: float,
                           trigger_strength: float, n_replicates: int, alpha: float, target_power: float,
                           rng: np.random.Generator, trigger_frac: float = 0.25):
    for n_models in model_candidates:
        p = power_at(n_trials, n_models, heterogeneity_sd, trigger_strength, n_replicates, alpha, rng,
                      trigger_frac=trigger_frac)
        print(f"    n_trials={n_trials:<5} n_models_per_group={n_models:<5} -> power={p:.2f}")
        if p >= target_power:
            return n_models, p
    return None, None  # never reached target_power within the candidate ceiling


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=["optimistic", "moderate", "pessimistic", "realistic_rare"],
                     choices=list(SCENARIOS.keys()))
    ap.add_argument("--n_trials_candidates", type=int, nargs="+", default=[100, 400])
    ap.add_argument("--n_models_candidates", type=int, nargs="+", default=[5, 10, 20, 40, 80, 160])
    ap.add_argument("--n_replicates", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--target_power", type=float, default=0.80)
    ap.add_argument("--gb_per_model", type=float, default=1.0, help="~777GB/840 checkpoints in the TrojVQA release")
    ap.add_argument("--gb_budget", type=float, default=8.0, help="storage budget to check the result against")
    ap.add_argument("--quick", action="store_true", help="fewer replicates and a lower model-count ceiling, for a fast first pass")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.quick:
        args.n_replicates = min(args.n_replicates, 12)
        args.n_models_candidates = [n for n in args.n_models_candidates if n <= 40] or [5, 10, 20, 40]

    rng = np.random.default_rng(args.seed)
    t_start = time.time()

    print("=== Power analysis: Aim 1 clean-vs-trojan model-level detection ===")
    print(f"target_power={args.target_power}  alpha={args.alpha}  n_replicates={args.n_replicates}\n")

    results = {}
    for scenario in args.scenarios:
        cfg = SCENARIOS[scenario]
        print(f"--- Scenario: {scenario}  (trigger_strength={cfg['trigger_strength']}, "
              f"heterogeneity_sd={cfg['heterogeneity_sd']}, trigger_frac={cfg['trigger_frac']}) ---")
        best_for_scenario = None
        for n_trials in args.n_trials_candidates:
            n_models, p = find_minimum_n_models(
                n_trials, args.n_models_candidates, cfg["heterogeneity_sd"], cfg["trigger_strength"],
                args.n_replicates, args.alpha, args.target_power, rng, trigger_frac=cfg["trigger_frac"],
            )
            if n_models is not None:
                total_models = 2 * n_models  # clean + trojan groups
                gb = total_models * args.gb_per_model
                print(f"    -> minimum n_models_per_group={n_models} at n_trials={n_trials} "
                      f"(power={p:.2f}; total models incl. both groups={total_models}; "
                      f"~{gb:.1f} GB at {args.gb_per_model} GB/model)")
                if best_for_scenario is None or total_models < best_for_scenario[1]:
                    best_for_scenario = (n_trials, total_models, gb)
            else:
                print(f"    -> target power NOT reached by n_models_per_group="
                      f"{args.n_models_candidates[-1]} at n_trials={n_trials} -- "
                      f"either raise the model-count ceiling or this scenario needs more trials/model.")
        results[scenario] = best_for_scenario
        print()

    print("=== Summary: cheapest (n_trials, total_models, ~GB) per scenario reaching "
          f"{args.target_power:.0%} power ===")
    for scenario, best in results.items():
        if best is None:
            print(f"  {scenario:<12}: not reached within the tested candidate ranges")
            continue
        n_trials, total_models, gb = best
        flag = "  <-- exceeds --gb_budget" if gb > args.gb_budget else ""
        print(f"  {scenario:<12}: n_trials={n_trials}, total_models={total_models}, ~{gb:.1f} GB{flag}")

    print(f"\n(wall clock: {time.time() - t_start:.1f}s)")

    over_budget = [s for s, b in results.items() if b is not None and b[2] > args.gb_budget]
    if over_budget or any(b is None for b in results.values()):
        print(
            "\nAt least one scenario needs more storage than --gb_budget (or wasn't reached at all) -- "
            "worth talking through before committing to a download size: options include accepting a "
            "wider confidence interval at a smaller N, targeting only the easiest contrast condition "
            "first (e.g. dual_key_solid vs. clean, which is likely the largest effect size), or "
            "reusing/pooling calibration trials across models rather than spending a fresh half-battery "
            "per model on calibration alone."
        )


if __name__ == "__main__":
    main()
