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
   module docstring). For each (image, question, congruence, reliability
   level), this script draws `n_repeats` independent perturbations at
   that level (perturb_visual_reliability uses randomness internally, so
   repeats differ), and uses the empirical std of x_v / x_t across those
   repeats as that cell's sigma_v / sigma_t. Each repeat then becomes one
   ProbeTrial-equivalent row in the fitting data, carrying that shared
   estimated sigma. IMPORTANT: x_v/x_t (and therefore sigma_v/sigma_t) are
   MODEL-DEPENDENT -- they come from that checkpoint's own confidence
   outputs -- so they get recomputed per model (see point 6), unlike the
   underlying Detectron2 features.

3. RESOLVED -- sigma_t is no longer repeat-based. The first 20-model
   pilot batch (see the bug writeup) surfaced a serious bug: the
   language-only branch saw the identical question every repeat
   (model.train(False) -> deterministic forward pass), so its empirical
   sigma_t collapsed to classify_from_cache's 1e-3 floor on EVERY trial,
   for every model. That degenerate sigma_t fed into bci_predict's
   reliability weighting (w_v = (1/sv^2)/(1/sv^2 + 1/st^2)) made BCI's
   prediction collapse to ~x_t on nearly every trial regardless of which
   cue the data actually favored -- fully explaining why BCI lost
   decisively to the linear baseline in that batch.
   A word-dropout/word-swap text perturbation (mirroring
   perturb_visual_reliability) was tried and rejected as the fix: this
   bag-of-words/LSTM question encoder is known to be largely insensitive
   to word order and tolerant of dropped words (mild corruption barely
   moves the output), and unlike image blur/noise, word corruption
   doesn't degrade gracefully -- it can make a short question ill-posed
   outright rather than merely less reliable.
   Fix actually used: model.text_reliability_sigma(text) -- sigma_t comes
   from the entropy of the language-only branch's own answer distribution
   (no corruption needed), computed ONCE per item since it depends only
   on the question text, not on repeat/level/congruence. sigma_v stays
   repeat-based (image blur/noise is a genuine, graceful reliability
   manipulation, so that half of the original design was fine). Do not
   trust any BCI-vs-linear comparison run before this fix.

4. Congruence manipulation (added after the first real-checkpoint pilot
   showed BCI tying the linear baseline -- see aim0_pilot_report.docx):
   research_plan.md's Aim 0 design calls for varying image-text congruence
   (matched vs. conflicting), not just visual reliability. Without it,
   p_common (the BCI model's prior that the two cues share a common
   cause) is unconstrained -- nothing in a congruent-only dataset can tell
   it apart from any other value. For each item, in addition to its own
   (congruent) image, we also probe it paired with a DIFFERENT item's
   image (incongruent) -- same question, wrong image. This requires >=2
   items and reuses the existing item pool rather than collecting new
   data.

5. Train/test split is at the ITEM level, not the trial level (also added
   after the first pilot -- see aim0_pilot_report.docx's caveats). Each
   item contributes many rows (congruence x levels x repeats); splitting
   at the row level put near-duplicate rows (same item, different noise
   draw) on both sides of the split, so "held-out" wasn't testing
   generalization to new content. Now an item, with ALL its rows, goes
   entirely into train or entirely into test.

6. Feature-extraction caching across a multi-model sweep (added once a
   20-model pilot batch made the naive cost obvious: 20 models x 1600
   trials/model = 32,000 Detectron2 forward passes, when only 1,600 are
   actually necessary). image_to_features (the expensive Detectron2 GPU
   backbone pass) and perturb_visual_reliability are BOTH independent of
   which VQA checkpoint is loaded -- only get_cue_values /
   joint_answer_and_confidence (the tiny classification head) depend on
   model_id. So a multi-model sweep now:
     precompute_feature_cache(model, items, ...)   -- ONCE, any checkpoint
     for model_id in model_ids:
         model.load_checkpoint(model_id)
         rows = classify_from_cache(model, cache)  -- cheap, per model
   build_dataset() below is kept as a single-model convenience wrapper
   (extract + classify in one call) for backward compatibility and for
   single-checkpoint runs -- it is NOT what main() uses for a >1-model
   sweep; see main()'s own loop.

Usage (single model):
    python experiments/aim0_real_trojvqa.py \
        --trojvqa_root /path/to/TrinityMultimodalTrojAI \
        --detector_weights_dir /path/to/detectron_weights \
        --model_id clean_m0 \
        --items items.json \
        --levels 0.0 0.5 1.0 2.0 \
        --n_repeats 5

Usage (multi-model sweep, appends one row per model to --results_csv;
Detectron2 feature extraction happens ONCE across the whole sweep --
see point 6 above and model_interface.py's load_checkpoint):
    python experiments/aim0_real_trojvqa.py \
        --trojvqa_root /path/to/TrinityMultimodalTrojAI \
        --detector_weights_dir /path/to/detectron_weights \
        --model_id clean_m0 clean_m1 clean_m2 \
        --items items.json \
        --results_csv aim0_results.csv

items.json: a JSON list of {"image_path": ..., "question": ...} objects --
a small, manually-curated set (see KAGGLE_SETUP.md for why this can't just
be "the whole VQAv2 val set" on a free-tier notebook).
"""
import argparse
import csv
import datetime
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image

from vlm_trojan_cue.cue_model import fit_bci, fit_linear_baseline, bci_predict
from vlm_trojan_cue.model_interface import TrojVQAInterface


def precompute_feature_cache(model, items, levels, n_repeats, detector_weights_dir, include_incongruent=True):
    """
    Runs Detectron2 feature extraction (model.perturb_visual_reliability +
    model.image_to_features) ONCE per (item, congruence, level, repeat)
    cell -- see module docstring point 6. Neither of those two calls
    depends on which VQA checkpoint is currently loaded (perturbation is
    pure pixel-space, and the detector predictor is built once in
    _ensure_detectron and is independent of model_id), so this cache is
    valid across an entire multi-model sweep: call it once with ANY
    TrojVQAInterface instance (before or after load_checkpoint -- doesn't
    matter), then feed the result to classify_from_cache() once per model.

    The question itself is NOT perturbed here (see module docstring point
    3 -- text perturbation was tried and rejected). classify_from_cache
    derives sigma_t from the language-only branch's own answer-
    distribution entropy instead, which only needs the clean question.

    Returns a list of dicts: item_idx, congruent, level, features
    (the (features, spatials) tuple from image_to_features), question
    (the item's original question, unperturbed).
    """
    cache = []
    n_items = len(items)
    can_make_incongruent = include_incongruent and n_items > 1

    for idx, item in enumerate(items):
        question = item["question"]
        own_image = Image.open(item["image_path"])

        conditions = [(True, own_image)]
        if can_make_incongruent:
            wrong_idx = (idx + 1) % n_items  # deterministic, never idx itself
            wrong_image = Image.open(items[wrong_idx]["image_path"])
            conditions.append((False, wrong_image))

        for congruent, base_image in conditions:
            for level in levels:
                for _ in range(n_repeats):
                    perturbed = model.perturb_visual_reliability(base_image, level)
                    features = model.image_to_features(perturbed, detector_weights_dir)
                    cache.append(dict(
                        item_idx=idx, congruent=congruent, level=level,
                        features=features, question=question,
                    ))
    return cache


def classify_from_cache(model, cache):
    """
    Runs ONLY the (cheap, model-DEPENDENT) VQA classification head --
    get_cue_values / joint_answer_and_confidence -- over an already
    -extracted feature cache from precompute_feature_cache(). The
    expensive Detectron2 pass is NOT repeated here; only this function's
    cost scales with the number of models in a sweep, not the Detectron2
    extraction's.

    Returns the same row format build_dataset() used to produce directly
    (item_idx, congruent, x_v, x_t, sigma_v, sigma_t, y), so
    item_level_split / fit_and_evaluate / append_result_row are unaffected
    by this refactor. sigma_v is still the spread of x_v across repeats
    WITHIN this model's own outputs (model-dependent, recomputed per
    model -- image blur/noise gives genuine per-repeat variation). sigma_t
    is NOT repeat-based (see module docstring point 3 for why): it comes
    from model.text_reliability_sigma(question), a per-ITEM quantity
    (entropy of the language-only answer distribution), computed once per
    item_idx and reused across that item's repeats/levels/congruence.
    """
    cells = defaultdict(list)
    for entry in cache:
        cells[(entry["item_idx"], entry["congruent"], entry["level"])].append(entry)

    text_sigma_cache = {}  # item_idx -> sigma_t; depends only on the question, not repeat/level/congruence

    rows = []
    for (item_idx, congruent, level), entries in cells.items():
        x_vs, x_ts, ys = [], [], []
        for entry in entries:
            x_v, x_t = model.get_cue_values(entry["features"], entry["question"])
            _, y = model.joint_answer_and_confidence(entry["features"], entry["question"])
            x_vs.append(x_v)
            x_ts.append(x_t)
            ys.append(y)

        sigma_v = float(np.std(x_vs)) or 1e-3  # guard against a degenerate all-identical repeat set
        if item_idx not in text_sigma_cache:
            text_sigma_cache[item_idx] = model.text_reliability_sigma(entries[0]["question"])
        sigma_t = text_sigma_cache[item_idx]
        for x_v, x_t, y in zip(x_vs, x_ts, ys):
            rows.append(dict(
                item_idx=item_idx, congruent=congruent,
                x_v=x_v, x_t=x_t, sigma_v=sigma_v, sigma_t=sigma_t, y=y,
            ))
    return rows


def build_dataset(model, items, levels, n_repeats, detector_weights_dir, include_incongruent=True):
    """
    Single-model convenience wrapper: extract + classify in one call.
    Equivalent to precompute_feature_cache() followed by
    classify_from_cache() -- kept for single-checkpoint callers and
    backward compatibility. A multi-model sweep should call those two
    functions directly (cache once, classify per model) instead of
    calling this once per model -- see module docstring point 6 and
    main()'s own loop below.
    """
    cache = precompute_feature_cache(model, items, levels, n_repeats, detector_weights_dir, include_incongruent)
    return classify_from_cache(model, cache)


def item_level_split(rows, test_frac, rng):
    """Split by unique item_idx, then assign every row of an item to
    whichever side its item landed on -- see module docstring, point 5."""
    item_ids = sorted(set(r["item_idx"] for r in rows))
    perm = rng.permutation(len(item_ids))
    n_test_items = max(1, int(round(test_frac * len(item_ids))))
    test_items = {item_ids[i] for i in perm[:n_test_items]}

    train_idx = [i for i, r in enumerate(rows) if r["item_idx"] not in test_items]
    test_idx = [i for i, r in enumerate(rows) if r["item_idx"] in test_items]
    return train_idx, test_idx, len(item_ids) - len(test_items), len(test_items)


def fit_and_evaluate(rows, train_idx, test_idx):
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
    return bci_params, linear_w, bci_mse, linear_mse


CSV_FIELDS = [
    "timestamp", "model_id", "arch", "detector",
    "n_items", "n_train_items", "n_test_items", "n_trials", "n_train", "n_test",
    "sigma_v", "sigma_t", "sigma_p", "p_common", "linear_w",
    "bci_mse", "linear_mse",
]


def append_result_row(csv_path, row):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trojvqa_root", required=True)
    ap.add_argument("--detector_weights_dir", required=True)
    ap.add_argument(
        "--model_id", nargs="+", default=["clean_m0"],
        help="One or more checkpoint IDs. With more than one, Detectron2 "
             "feature extraction runs ONCE across the whole sweep (see "
             "precompute_feature_cache/classify_from_cache and "
             "TrojVQAInterface.load_checkpoint) and one row per model is "
             "appended to --results_csv.",
    )
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
    ap.add_argument("--test_frac", type=float, default=0.3, help="Fraction of ITEMS (not trials) held out.")
    ap.add_argument(
        "--no_incongruent", action="store_true",
        help="Disable the incongruent (wrong-image) condition and probe "
             "each item's own image only -- roughly halves the work but "
             "leaves p_common unconstrained again (see module docstring).",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--results_csv", default=None,
        help="If set, append one summary row per model here instead of "
             "(or in addition to) printing -- needed to aggregate a "
             "multi-model sweep for the population-level test in "
             "experiments/aim0_analyze_multi_model.py.",
    )
    args = ap.parse_args()

    with open(args.items) as f:
        items = json.load(f)

    if args.image_dir:
        for item in items:
            item["image_path"] = os.path.join(args.image_dir, os.path.basename(item["image_path"]))

    model_ids = args.model_id
    model = TrojVQAInterface(
        trojvqa_root=args.trojvqa_root,
        model_id=model_ids[0],
        arch=args.arch,
        detector=args.detector,
        device=args.device,
    )

    # Detectron2 extraction happens ONCE here, regardless of len(model_ids)
    # -- see module docstring point 6. This is the expensive step; every
    # model below only pays for its own (cheap) classification head.
    feature_cache = precompute_feature_cache(
        model, items, args.levels, args.n_repeats, args.detector_weights_dir,
        include_incongruent=not args.no_incongruent,
    )
    if len(feature_cache) < 10:
        raise SystemExit(
            f"Only {len(feature_cache)} trials produced -- too few to fit anything meaningful. "
            "Add more items/levels/repeats."
        )
    print(f"(extracted Detectron2 features for {len(feature_cache)} trials -- shared across all "
          f"{len(model_ids)} model(s) below, not re-extracted per model)\n")

    failed_model_ids = []
    for i, model_id in enumerate(model_ids):
        # A multi-model sweep can run long (many checkpoints, shared feature
        # cache already paid for) -- one architecturally-mismatched or
        # otherwise broken checkpoint (e.g. a strict state_dict load failure,
        # see module docstring / --arch) must NOT kill every model queued
        # after it and waste the whole GPU session. Failures are caught,
        # logged, and swept models continue; see the summary printed after
        # the loop.
        try:
            if i > 0:
                model.load_checkpoint(model_id)  # reuses the already-loaded detector + feature_cache

            rows = classify_from_cache(model, feature_cache)

            rng = np.random.default_rng(args.seed)
            train_idx, test_idx, n_train_items, n_test_items = item_level_split(rows, args.test_frac, rng)
            bci_params, linear_w, bci_mse, linear_mse = fit_and_evaluate(rows, train_idx, test_idx)

            print(f"=== Aim 0 on REAL model: {args.arch}/{model_id} ===")
            print(f"n_items={len(items)}  (train_items={n_train_items}  test_items={n_test_items})  "
                  f"n_trials={len(rows)}  (train={len(train_idx)}  test={len(test_idx)})")
            print(f"Fitted BCI params:   sigma_v={bci_params.sigma_v:.2f}  sigma_t={bci_params.sigma_t:.2f}  "
                  f"sigma_p={bci_params.sigma_p:.2f}  p_common={bci_params.p_common:.2f}")
            print(f"Held-out MSE  -- BCI model: {bci_mse:.4f}   linear-weight baseline: {linear_mse:.4f}")
            if bci_mse < linear_mse:
                print("-> BCI model outperforms the linear ablation on REAL data.")
            else:
                print("-> Linear baseline matches or beats BCI on real data -- per docs/research_plan.md's "
                      "Aim 0 branch (b), this is a pre-registered possible outcome (pivot to within-model "
                      "consistency framing), not a failure.")

            if args.results_csv:
                append_result_row(args.results_csv, dict(
                    timestamp=datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    model_id=model_id, arch=args.arch, detector=args.detector,
                    n_items=len(items), n_train_items=n_train_items, n_test_items=n_test_items,
                    n_trials=len(rows), n_train=len(train_idx), n_test=len(test_idx),
                    sigma_v=bci_params.sigma_v, sigma_t=bci_params.sigma_t,
                    sigma_p=bci_params.sigma_p, p_common=bci_params.p_common,
                    linear_w=linear_w, bci_mse=bci_mse, linear_mse=linear_mse,
                ))
                print(f"(appended result row for {model_id} to {args.results_csv})")
            print()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            failed_model_ids.append(model_id)
            print(f"!! SKIPPING {model_id}: {type(e).__name__}: {e}")
            print("   (common cause: this checkpoint's architecture doesn't match --arch -- "
                  "see module docstring point on baseline0 vs baseline0_newatt state_dict keys)")
            print()
            continue

    n_ok = len(model_ids) - len(failed_model_ids)
    print(f"=== Sweep done: {n_ok}/{len(model_ids)} model(s) succeeded ===")
    if failed_model_ids:
        print(f"Failed model_id(s) ({len(failed_model_ids)}): {failed_model_ids}")


if __name__ == "__main__":
    main()
