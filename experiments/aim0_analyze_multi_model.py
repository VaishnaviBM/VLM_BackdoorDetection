"""
Aim 0 population-level analysis + power planning, operating on the
--results_csv produced by experiments/aim0_real_trojvqa.py's multi-model
sweep (one row per model: bci_mse, linear_mse, plus fit diagnostics).

This exists because a single model's BCI-vs-linear comparison (see
aim0_pilot_report.docx) can't tell you whether BCI systematically earns
its complexity across clean TrojVQA models -- that's a claim about the
POPULATION of models, which needs its own test, run across many models,
not eyeballed per-model print statements.

--------------------------------------------------------------------------
Pre-registered decision rule (fix this BEFORE looking at the full-scale
result, per the plan's Aim 0 branch (a)/(b) split) -- write any change to
this rule into docs/research_plan.md, not just here.
--------------------------------------------------------------------------
Unit of analysis is the MODEL, not the trial: within a model, trials
share items/levels and aren't independent draws, so the test below runs
on one (bci_mse, linear_mse) pair per model.

PRIMARY test: a percentile bootstrap on the mean of (linear_mse - bci_mse)
across models. Chosen over a plain Wilcoxon/t-test because it directly
targets the MEAN difference -- the same quantity the effect-size gate and
the power-planning formula below both use -- without assuming normality
(unlike a t-test) or testing a median/symmetry-based null instead of the
mean (which a signed-rank test does; see discussion this was chosen over,
kept below as a reported secondary check). At small N (the ~15-20 model
pilot batch) this also stays robust to one outlier model without
switching what's actually being tested partway through the pipeline.

SECONDARY / robustness check: Wilcoxon signed-rank test (t-test fallback)
on the same differences, reported alongside -- if the bootstrap and
Wilcoxon disagree on significance, that's itself worth noticing (e.g. a
skewed difference distribution), not something to paper over by picking
whichever test happens to agree with what you expected.

Branch (a) -- BCI earns its complexity -- requires ALL of:
  1. bci_mse < linear_mse on average across models (BCI wins the direction)
  2. The bootstrap CI on mean(linear_mse - bci_mse) excludes 0 at --alpha
  3. mean relative improvement (linear_mse - bci_mse) / linear_mse exceeds
     --effect_threshold (default 5%) -- statistically detectable but tiny
     doesn't count as BCI "earning its complexity"
Otherwise: branch (b) (within-model consistency framing), INCLUDING the
case where the test comes back non-significant -- that is not proof of
equivalence, just an absence of evidence; see the power-planning section.

Usage:
    python experiments/aim0_analyze_multi_model.py --results_csv aim0_results.csv
    python experiments/aim0_analyze_multi_model.py --results_csv aim0_results.csv \
        --target_effect 0.5 --target_power 0.8
"""
import argparse
import csv

import numpy as np
from scipy import stats


def load_rows(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise SystemExit(f"{path} has no data rows.")
    return rows


def bootstrap_mean_test(diff, n_boot=10000, alpha=0.05, seed=0):
    """
    Percentile bootstrap on mean(diff) -- PRIMARY test, see module
    docstring for why this is preferred over Wilcoxon/t-test here.

    p_value is derived by inverting the percentile CI (the standard
    bootstrap-hypothesis-test trick): the fraction of resampled means
    that land on the "wrong" side of zero, doubled for a two-sided test.
    This is a resampling-based approximation, not an exact p-value --
    fine for planning/reporting at this stage, not a substitute for
    pre-registering alpha before looking at the real multi-model data.
    """
    n = len(diff)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)

    ci_low, ci_high = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_low = float(np.mean(boot_means <= 0))
    p_high = float(np.mean(boot_means >= 0))
    p_value = min(2 * min(p_low, p_high), 1.0)

    return dict(
        mean_diff=float(np.mean(diff)), ci_low=float(ci_low), ci_high=float(ci_high),
        p_value=p_value, n_boot=n_boot,
    )


def wilcoxon_secondary(diff):
    """SECONDARY / robustness check -- see module docstring. Not the
    test the branch-decision rule below actually gates on."""
    result = dict(median_diff=float(np.median(diff)))
    try:
        w_stat, w_p = stats.wilcoxon(diff)
        result.update(test="wilcoxon", statistic=float(w_stat), p_value=float(w_p))
    except ValueError as e:
        t_stat, t_p = stats.ttest_1samp(diff, 0.0)
        result.update(test=f"paired_t (wilcoxon unavailable: {e})", statistic=float(t_stat), p_value=float(t_p))
    return result


def required_n_for_power(effect_size, sd_diff, alpha=0.05, power=0.8):
    """
    Sample size for a paired (one-sample-on-differences) test detecting a
    mean difference of `effect_size` with standard deviation `sd_diff`,
    via the standard normal-approximation formula:

        n = ((z_{alpha/2} + z_{power}) * sd_diff / effect_size) ** 2

    This is a PLANNING approximation (uses z, not t, so it's slightly
    optimistic at small n) -- good enough to decide "do I need ~15 models
    or ~150", not precise enough to defend in the paper. Re-estimate
    sd_diff from a real pilot batch of models before trusting this; see
    module docstring / research_plan.md next steps.
    """
    if effect_size <= 0 or sd_diff <= 0:
        return float("nan")
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    n = ((z_alpha + z_power) * sd_diff / effect_size) ** 2
    return float(np.ceil(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_csv", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument(
        "--effect_threshold", type=float, default=0.05,
        help="Minimum mean relative MSE improvement (linear_mse - bci_mse) / "
             "linear_mse, as a fraction, for a significant result to count "
             "as branch (a) rather than 'significant but negligible'. Default 0.05 (5%%).",
    )
    ap.add_argument("--n_boot", type=int, default=10000, help="Bootstrap resamples for the primary test.")
    ap.add_argument("--boot_seed", type=int, default=0)
    ap.add_argument(
        "--target_effect", type=float, default=None,
        help="Absolute MSE difference you'd want to be able to detect, for "
             "the power-planning calculation below. If omitted, uses the "
             "observed mean_diff from this CSV (only meaningful once you "
             "have a real pilot batch of models, not n=1).",
    )
    ap.add_argument("--target_power", type=float, default=0.8)
    args = ap.parse_args()

    rows = load_rows(args.results_csv)
    bci_mse = np.array([float(r["bci_mse"]) for r in rows])
    linear_mse = np.array([float(r["linear_mse"]) for r in rows])
    n_models = len(rows)

    print(f"=== Aim 0 population-level test over {n_models} model(s) from {args.results_csv} ===")
    if n_models < 2:
        print("Only 1 model in this file -- there is no population-level test to run yet. "
              "This CSV only supports the per-model number you already saw when it was "
              "generated; see docs/research_plan.md next steps for the pilot-batch size "
              "needed before this script's test/power numbers mean anything.")
        return

    diff = linear_mse - bci_mse  # positive favors BCI
    sd_diff = float(np.std(diff, ddof=1)) if n_models > 1 else float("nan")

    boot = bootstrap_mean_test(diff, n_boot=args.n_boot, alpha=args.alpha, seed=args.boot_seed)
    wil = wilcoxon_secondary(diff)
    rel_improvement = boot["mean_diff"] / float(np.mean(linear_mse))

    print(f"\n-- PRIMARY: percentile bootstrap on mean(linear_mse - bci_mse), n_boot={args.n_boot} --")
    print(f"mean(linear_mse - bci_mse) = {boot['mean_diff']:.4f}  "
          f"(positive favors BCI; relative = {rel_improvement:+.1%})")
    print(f"{100*(1-args.alpha):.0f}% bootstrap CI = [{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]")
    print(f"bootstrap p-value (two-sided, CI-inversion) = {boot['p_value']:.4f}")
    print(f"sd(linear_mse - bci_mse) across models = {sd_diff:.4f}  (feeds the power calc below)")

    print(f"\n-- SECONDARY / robustness check: {wil['test']} --")
    print(f"median(linear_mse - bci_mse) = {wil['median_diff']:.4f}   "
          f"statistic={wil['statistic']:.4f}   p={wil['p_value']:.4f}")
    if (wil["p_value"] < args.alpha) != (boot["p_value"] < args.alpha):
        print("NOTE: primary (bootstrap) and secondary (Wilcoxon) tests DISAGREE on significance "
              f"at alpha={args.alpha} -- worth inspecting the per-model diff distribution "
              "(e.g. skew, a single outlier model) before trusting either number blindly.")

    significant = boot["p_value"] < args.alpha
    meaningful = rel_improvement >= args.effect_threshold
    branch_a = significant and meaningful and boot["mean_diff"] > 0

    print()
    if branch_a:
        print(f"-> BRANCH (a): bootstrap CI excludes 0 (p<{args.alpha}) AND relative improvement "
              f"({rel_improvement:.1%}) >= threshold ({args.effect_threshold:.1%}). "
              "BCI earns its complexity across this model sample.")
    elif significant and not meaningful:
        print(f"-> Statistically significant (bootstrap p={boot['p_value']:.4f}) but relative improvement "
              f"({rel_improvement:.1%}) is below the {args.effect_threshold:.1%} threshold -- "
              "treated as BRANCH (b): detectable, but not large enough to call BCI's complexity earned.")
    else:
        print(f"-> Bootstrap CI includes 0 at alpha={args.alpha} (p={boot['p_value']:.4f}) -> BRANCH (b) "
              "by default. NOTE: this is absence of evidence, not evidence of equivalence -- "
              "check the power calculation below before treating this as a settled null result.")

    print()
    print("=== Power planning ===")
    target_effect = args.target_effect if args.target_effect is not None else abs(boot["mean_diff"])
    if args.target_effect is None:
        print(f"(--target_effect not given; using this sample's observed |mean_diff|={target_effect:.4f} -- "
              f"only a reasonable stand-in once n_models is itself decently large, not at n={n_models})")
    if np.isnan(sd_diff):
        print("Can't compute required N yet: need >=2 models to estimate sd(diff). Run a pilot batch first.")
        return

    n_needed = required_n_for_power(target_effect, sd_diff, args.alpha, args.target_power)
    print(f"To detect a mean difference of {target_effect:.4f} MSE (given this sample's sd={sd_diff:.4f}) "
          f"at alpha={args.alpha}, power={args.target_power}: need ~{n_needed:.0f} models.")
    if n_models < 15:
        print(f"NOTE: sd(diff) above is estimated from only {n_models} model(s) -- too few to trust this "
              "number for real study sizing. Run a ~15-20 model pilot batch first (research_plan.md next "
              "steps), re-run this script on that batch, and re-plan N from THAT sd estimate.")


if __name__ == "__main__":
    main()
