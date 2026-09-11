"""
Aim 1: real, population-level backdoor detection benchmark.

Wires the (previously synthetic-only) calibrate_null / batch_anomaly_score
machinery in detection.py to REAL TrojVQA checkpoints via TrojVQAInterface,
reusing the same real probe-battery construction as aim0_real_trojvqa.py's
build_dataset() -- real images, real questions, real corruption levels, real
empirically-estimated sigma_v/sigma_t (see model_interface.py's module
docstring for why sigma_v can't just be the corruption `level`), and real
y = the joint run's own answer-confidence (aim0_real_trojvqa.py's proposed
operationalization, reused here rather than re-litigated).

Design (per research_plan_vlm_backdoor_detection.md's Aim 1 statistical design):
  1. Split the CLEAN models (is_clean==True in the manifest produced by
     aim1_spec_mining.py) into a calibration pool and a held-out pool.
  2. Pool every trial from every calibration-pool model's probe battery and
     fit ONE null model on it (calibrate_null) -- "the fitted normative
     curve" clean models should look like.
  3. Score every held-out model (held-out clean models AND every trojan
     model) by its OWN probe battery's mean deviation from that null
     (batch_anomaly_score). Held-out clean models never influenced the null
     they're being judged against, so this is a fair blind test, not a
     tautology.
  4. y_true = 0 for held-out-clean, 1 for trojan; report AUROC with
     bootstrap CI (eval.roc_auc_with_ci), overall and split by trigger type
     if more than one trojan trigger type is present in the manifest (the
     dual-key vs. single-key contrast the research plan calls for).

This is a REAL benchmark -- it needs a GPU, the actual TrojVQA checkpoints,
and Detectron2 features, so it's meant to run on Kaggle (or similar), not in
CI. It is also, honestly, only as good as the manifest and item pool you
feed it: a handful of models and a dozen probe items will not produce a
publishable AUROC, only a plumbing check that the wiring works end to end
(see KAGGLE_SETUP.md / the accompanying git+data instructions for scaling
this up to something with real statistical power).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aim0_real_trojvqa import build_dataset  # reuse the already-validated real probe-battery builder

from vlm_trojan_cue.probes import ProbeTrial
from vlm_trojan_cue.detection import calibrate_null, batch_anomaly_score
from vlm_trojan_cue.eval import roc_auc_with_ci
from vlm_trojan_cue.model_interface import TrojVQAInterface


def rows_to_trials(rows):
    trials = [ProbeTrial(x_v=r["x_v"], x_t=r["x_t"], sigma_v=r["sigma_v"], sigma_t=r["sigma_t"]) for r in rows]
    responses = np.array([r["y"] for r in rows])
    return trials, responses


def load_manifest(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["labeled"] == "True" and r["supported_arch"] == "True"]
    clean = [r for r in rows if r["is_clean"] == "True"]
    trojan = [r for r in rows if r["is_clean"] == "False"]
    return clean, trojan


def build_model_battery(trojvqa_root, detector_weights_dir, model_id, arch, detector,
                         items, levels, n_repeats, device):
    model = TrojVQAInterface(trojvqa_root=trojvqa_root, model_id=model_id, arch=arch,
                              detector=detector, device=device)
    rows = build_dataset(model, items, levels, n_repeats, detector_weights_dir)
    return rows_to_trials(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest_csv", required=True, help="Output of aim1_spec_mining.py")
    ap.add_argument("--trojvqa_root", required=True)
    ap.add_argument("--detector_weights_dir", required=True)
    ap.add_argument("--items", required=True, help="JSON file: list of {image_path, question}")
    ap.add_argument("--image_dir", default=None)
    ap.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--n_repeats", type=int, default=5)
    ap.add_argument("--calib_frac", type=float, default=0.5,
                     help="Fraction of CLEAN models used to fit the null; the rest are held out "
                          "and scored blind, same as every trojan model.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    with open(args.items) as f:
        items = json.load(f)
    if args.image_dir:
        for item in items:
            item["image_path"] = os.path.join(args.image_dir, os.path.basename(item["image_path"]))

    clean, trojan = load_manifest(args.manifest_csv)
    if len(clean) < 2:
        raise SystemExit(f"Only {len(clean)} labeled, supported clean models in the manifest -- "
                          "need at least 2 (>=1 for calibration, >=1 held out) for a meaningful blind test.")
    if not trojan:
        raise SystemExit("No labeled, supported trojan models in the manifest -- nothing to detect. "
                          "Download at least one dual-key or single-key checkpoint (see the git/data "
                          "instructions this script shipped with).")

    rng = np.random.default_rng(args.seed)
    clean_shuffled = list(clean)
    rng.shuffle(clean_shuffled)
    n_calib = max(1, int(round(args.calib_frac * len(clean_shuffled))))
    n_calib = min(n_calib, len(clean_shuffled) - 1)  # always leave >=1 clean model held out
    calib_models, heldout_clean = clean_shuffled[:n_calib], clean_shuffled[n_calib:]

    print(f"Calibration pool: {len(calib_models)} clean models -> {[m['model_id'] for m in calib_models]}")
    print(f"Held-out clean:   {len(heldout_clean)} models -> {[m['model_id'] for m in heldout_clean]}")
    print(f"Trojan (all held out): {len(trojan)} models -> {[m['model_id'] for m in trojan]}")

    # --- Step 1: pool calibration-model trials, fit ONE null model ---
    calib_trials, calib_responses = [], []
    for m in calib_models:
        trials, responses = build_model_battery(
            args.trojvqa_root, args.detector_weights_dir, m["model_id"], m["arch"], m["detector"],
            items, args.levels, args.n_repeats, args.device,
        )
        calib_trials.extend(trials)
        calib_responses.extend(responses.tolist())
    null = calibrate_null(calib_trials, np.array(calib_responses))
    print(f"Fitted null: sigma_p={null.bci_params.sigma_p:.3f}  p_common={null.bci_params.p_common:.3f}  "
          f"linear_weight={null.linear_weight:.3f}  (from {len(calib_trials)} pooled calibration trials)")

    # --- Step 2: score every held-out model (clean-holdout + all trojan) blind ---
    results = []
    for m in heldout_clean + trojan:
        trials, responses = build_model_battery(
            args.trojvqa_root, args.detector_weights_dir, m["model_id"], m["arch"], m["detector"],
            items, args.levels, args.n_repeats, args.device,
        )
        score = batch_anomaly_score(trials, responses, null)
        results.append({
            "model_id": m["model_id"], "is_clean": m["is_clean"] == "True",
            "trigger": m.get("trigger", ""), "score": score, "n_trials": len(trials),
        })
        print(f"  {m['model_id']:24s}  is_clean={str(m['is_clean']):5s}  trigger={m.get('trigger',''):10s}  "
              f"anomaly_score={score:.4f}  (n_trials={len(trials)})")

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model_id", "is_clean", "trigger", "score", "n_trials"])
        writer.writeheader()
        writer.writerows(results)

    y_true = np.array([0 if r["is_clean"] else 1 for r in results])
    scores = np.array([r["score"] for r in results])
    if len(np.unique(y_true)) < 2:
        print("\nWARNING: held-out set is all one class (all clean or all trojan) -- AUROC is undefined. "
              "Add more models to the manifest / adjust --calib_frac.")
        return

    overall = roc_auc_with_ci(y_true, scores)
    print(f"\n=== Aim 1 REAL detection benchmark ===")
    print(f"Held-out models: {len(results)}  ({int((y_true==0).sum())} clean, {int((y_true==1).sum())} trojan)")
    print(f"AUROC: {overall['auc']:.3f}  (95% CI: {overall['ci_low']:.3f}-{overall['ci_high']:.3f}, "
          f"n_boot={overall['n_boot']})")

    trigger_types = sorted(set(r["trigger"] for r in results if not r["is_clean"] and r["trigger"]))
    for trig in trigger_types:
        mask = np.array([r["is_clean"] or r["trigger"] == trig for r in results])
        y_sub, s_ub = y_true[mask], scores[mask]
        if len(np.unique(y_sub)) < 2:
            continue
        sub = roc_auc_with_ci(y_sub, s_sub)
        print(f"  trigger={trig:10s}  AUROC: {sub['auc']:.3f}  (95% CI: {sub['ci_low']:.3f}-{sub['ci_high']:.3f}, "
              f"n={int(mask.sum())})")

    print(f"\nWrote per-model scores -> {args.out_csv}")


if __name__ == "__main__":
    main()
