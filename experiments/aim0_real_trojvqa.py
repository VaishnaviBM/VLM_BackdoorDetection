"""
Aim 0 on a REAL TrojVQA checkpoint -- the "immediate next step 5" from
docs/research_plan.md ("Aim 0 pilot: cue-integration characterization on
the clean TrojVQA models, to sanity-check the fitted causal-inference
model is identifiable from real data").

This mirrors experiments/aim0_characterize_cue_integration.py's structure
(fit_bci vs. fit_linear_baseline, held-out MSE comparison) but sources
(x_v, x_t, sigma_v, sigma_t, y) from a real TrojVQAInterface instead of
CleanObserver.

--------------------------------------------------------------------------
Design choices made HERE, not assumed elsewhere in the repo -- read before
trusting results
--------------------------------------------------------------------------
1. What is "y"? docs/research_plan.md's immediate-next-steps list defines
   the x_v/x_t extraction recipe but does not yet specify the dependent
   variable a real Aim 0 should fit. This script uses:

       y = the joint run's own log-probability for the answer it picked

   i.e. "is the model's confidence in its joint (image+question) answer
   predictable as a reliability-weighted combination of its unimodal
   (vision-only, language-only) confidences in that same answer?" This is
   a reasonable reading of the BCI framing carried over from the synthetic
   pipeline, but it is a choice this script is making, not an established
   project convention -- revisit before relying on the result.

2. sigma_v / sigma_t are ESTIMATED, not assumed (see model_interface.py's
   module docstring). For each (image, question, reliability level), this
   script draws `n_repeats` independent perturbations at that level
   (perturb_visual_reliability uses randomness internally, so repeats
   differ), and uses the empirical std of x_v / x_t across those repeats
   as that level's sigma_v / sigma_t. Each repeat then becomes one
   ProbeTrial-equivalent row in the fitting data, carrying that shared
   estimated sigma.

3. Text reliability is never perturbed here (no perturb_text_reliability
   exists yet) -- sigma_t as estimated above reflects noise in the
   language-only readout itself (extraction/model stochasticity), not a
   deliberate text-reliability manipulation. If you want a real sigma_t
   manipulation, that's a separate extension to model_interface.py.

Usage:
    python experiments/aim0_real_trojvqa.py \\
        --trojvqa_root /path/to/TrinityMultimodalTrojAI \\
        --detector_weights_dir /path/to/detectron_weights \\
        --model_id clean_m0 \\
        --items items.json \\
        --levels 0.0 0.5 1.0 2.0 \\
        --n_repeats 5

items.json: a JSON list of {"image_path": ..., "question": ...} objects --
a small, manually-curated set (see KAGGLE_SETUP.md for why this can't just
be "the whole VQAv2 val set" on a free-tier notebook).
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

from vlm_trojan_cue.cue_model import fit_bci, fit_linear_baseline, bci_predict
from vlm_trojan_cue.model_interface import TrojVQAInterface


def build_dataset(model: TrojVQAInterface, items, levels, n_repeats, detector_weights_dir):
    rows = []  # each: dict(x_v, x_t, sigma_v, sigma_t, y)
    for item in items:
        image = Image.open(item["image_path"])
        question = item["question"]
        for level in levels:
            x_vs, x_ts, ys = [], [], []
            for _ in range(n_repeats):
                perturbed = model.perturb_visual_reliability(image, level)
                features = model.image_to_features(perturbed, detector_weights_dir)
                x_v, x_t = model.get_cue_values(features, question)
                _, y = model.joint_answer_and_confidence(features, question)
                x_vs.append(x_v)
                x_ts.append(x_t)
                ys.append(y)

            sigma_v = float(np.std(x_vs)) or 1e-3  # guard against a degenerate all-identical repeat set
            sigma_t = float(np.std(x_ts)) or 1e-3
            for x_v, x_t, y in zip(x_vs, x_ts, ys):
                rows.append(dict(x_v=x_v, x_t=x_t, sigma_v=sigma_v, sigma_t=sigma_t, y=y))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trojvqa_root", required=True)
    ap.add_argument("--detector_weights_dir", required=True)
    ap.add_argument("--model_id", default="clean_m0")
    ap.add_argument("--arch", default="butd_eff")
    ap.add_argument("--detector", default="R-50")
    ap.add_argument("--items", required=True, help="JSON file: list of {image_path, question}")
    ap.add_argument(
        "--image_dir", default=None,
        help="Override directory to look for images in -- if set, each "
             "item's image_path is rewritten to "
             "os.path.join(image_dir, basename(item['image_path'])), so "
             "items.json stays portable even if the images now live "
             "somewhere else (e.g. a different Kaggle dataset/session).",
    )
    ap.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--n_repeats", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.items) as f:
        items = json.load(f)

    if args.image_dir:
        for item in items:
            item["image_path"] = os.path.join(args.image_dir, os.path.basename(item["image_path"]))

    model = TrojVQAInterface(
        trojvqa_root=args.trojvqa_root,
        model_id=args.model_id,
        arch=args.arch,
        detector=args.detector,
        device=args.device,
    )

    rows = build_dataset(model, items, args.levels, args.n_repeats, args.detector_weights_dir)
    if len(rows) < 10:
        raise SystemExit(
            f"Only {len(rows)} trials produced -- too few to fit anything meaningful. "
            "Add more items/levels/repeats."
        )

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(rows))
    split = int(0.7 * len(rows))
    train_idx, test_idx = idx[:split], idx[split:]

    def col(name, indices):
        return np.array([rows[i][name] for i in indices])

    x_v_tr, x_t_tr, sv_tr, st_tr, y_tr = (
        col("x_v", train_idx), col("x_t", train_idx),
        col("sigma_v", train_idx), col("sigma_t", train_idx), col("y", train_idx),
    )
    x_v_te, x_t_te, sv_te, st_te, y_te = (
        col("x_v", test_idx), col("x_t", test_idx),
        col("sigma_v", test_idx), col("sigma_t", test_idx), col("y", test_idx),
    )

    bci_params = fit_bci(x_v_tr, x_t_tr, sv_tr, st_tr, y_tr)
    linear_w = fit_linear_baseline(x_v_tr, x_t_tr, y_tr)

    bci_pred, _ = bci_predict(x_v_te, x_t_te, sv_te, st_te, bci_params)
    linear_pred = linear_w * x_v_te + (1 - linear_w) * x_t_te

    bci_mse = float(np.mean((y_te - bci_pred) ** 2))
    linear_mse = float(np.mean((y_te - linear_pred) ** 2))

    print(f"=== Aim 0 on REAL model: {args.arch}/{args.model_id} ===")
    print(f"n_trials={len(rows)}  (train={len(train_idx)}  test={len(test_idx)})")
    print(f"Fitted BCI params:   sigma_v={bci_params.sigma_v:.2f}  sigma_t={bci_params.sigma_t:.2f}  "
          f"sigma_p={bci_params.sigma_p:.2f}  p_common={bci_params.p_common:.2f}")
    print(f"Held-out MSE  -- BCI model: {bci_mse:.4f}   linear-weight baseline: {linear_mse:.4f}")
    if bci_mse < linear_mse:
        print("-> BCI model outperforms the linear ablation on REAL data.")
    else:
        print("-> Linear baseline matches or beats BCI on real data -- per docs/research_plan.md's "
              "Aim 0 branch (b), this is a pre-registered possible outcome (pivot to within-model "
              "consistency framing), not a failure.")


if __name__ == "__main__":
    main()
