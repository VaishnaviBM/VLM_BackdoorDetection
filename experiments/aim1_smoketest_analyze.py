"""
Analyze a full-scale aim1_dualkey_smoketest.py results CSV: per-condition
target-hit rate (Wilson 95% CI) plus a paired McNemar exact test of the
BOTH-KEYS condition against each single-key/no-key condition (paired
because all 4 conditions share the same underlying probe items).

This is a single-checkpoint case study (one verified dual-key trojan,
`SemPatch_m16` / demo m1) -- NOT a calibrated multi-model Aim 1 benchmark
(see docs/research_plan.md; that needs a population of clean + trojan
checkpoints for null calibration/AUROC, which this run does not attempt).
Report it as such.

Usage:
    python experiments/aim1_smoketest_analyze.py --csv /kaggle/working/aim1_smoketest.csv
"""
import argparse
import csv
import math
from collections import defaultdict

from scipy.stats import binomtest


def wilson_ci(hits: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    phat = hits / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact(a_hits: list, b_hits: list):
    """a_hits/b_hits: parallel 0/1 lists (same items). Returns (b, c, p_value)
    where b = a=0,b=1 count and c = a=1,b=0 count (discordant pairs)."""
    assert len(a_hits) == len(b_hits)
    b = sum(1 for a, bb in zip(a_hits, b_hits) if a == 0 and bb == 1)
    c = sum(1 for a, bb in zip(a_hits, b_hits) if a == 1 and bb == 0)
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue
    return b, c, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", default=None, help="Override target string (else infer: any row with target_hit=True's answer).")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    if not rows:
        raise SystemExit(f"No rows in {args.csv}")

    conditions = []
    by_cond = defaultdict(dict)  # condition -> {item_idx: hit (0/1)}
    for r in rows:
        cond = r["condition"]
        if cond not in conditions:
            conditions.append(cond)
        by_cond[cond][int(r["item_idx"])] = int(r["target_hit"] == "True")

    n_items = len(by_cond[conditions[0]])
    print(f"=== {args.csv}: {n_items} item(s), {len(conditions)} condition(s) ===\n")

    print("Per-condition target-hit rate (Wilson 95% CI):")
    for cond in conditions:
        hits_by_item = by_cond[cond]
        hits = sum(hits_by_item.values())
        lo, hi = wilson_ci(hits, n_items)
        print(f"  {cond:28s} {hits:3d}/{n_items}  ({hits/n_items:.1%})   95% CI [{lo:.1%}, {hi:.1%}]")

    both_label = [c for c in conditions if "BOTH" in c.upper()]
    if not both_label:
        print("\nNo condition labeled BOTH KEYS found -- skipping paired tests.")
        return
    both_label = both_label[0]
    both_items = by_cond[both_label]
    both_vec = [both_items[i] for i in range(n_items)]

    print(f"\nPaired McNemar exact test: {both_label!r} vs. each other condition")
    print("(tests whether the BOTH-KEYS hit rate differs from that condition's, on the SAME items):")
    for cond in conditions:
        if cond == both_label:
            continue
        other_items = by_cond[cond]
        other_vec = [other_items[i] for i in range(n_items)]
        b, c, p = mcnemar_exact(other_vec, both_vec)
        sig = "significant" if p < 0.05 else "not significant"
        print(f"  vs {cond:28s} discordant pairs: {both_label} only={b}, {cond} only={c}   p={p:.4f} ({sig})")

    print(
        "\nReminder for writeup: this is a case study on ONE verified dual-key "
        "trojan checkpoint, not a calibrated multi-model Aim 1 detection "
        "benchmark -- report accordingly."
    )


if __name__ == "__main__":
    main()
