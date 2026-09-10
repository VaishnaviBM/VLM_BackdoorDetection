"""
Aim 2 real-data plumbing (docs/research_plan.md, secondary/proof-of-concept):
gradient-free reverse correlation to localize the visual trigger of a REAL,
verified dual-key trojan checkpoint (the same SemPatch_m16 / demo-m1 model
validated end-to-end in aim1_dualkey_smoketest.py), using only query access.

Two stages:

  1. FLAG fusion units. For a batch of probe items, compute the model's
     joint fusion representation (joint_repr = q_repr * v_repr, captured via
     model.joint_repr_for -- see model_interface.py) under (clean image,
     clean question) vs. (triggered image, triggered question). The
     num_hid dimensions with the largest mean |activation shift| between
     those two conditions are the "flagged fusion units" -- candidates the
     backdoor recruits.

  2. REVERSE-CORRELATE against those units. Take ONE probe image (its real
     visual trigger patch is NOT applied during these trials -- the point
     is to see whether random perturbation alone reveals the trigger's
     location, not to recover a patch we already placed there). Divide the
     image into a --grid x --grid block basis; each trial applies an
     independent random additive brightness/color shift to every block
     (this is the "structured noise" reverse_correlation.py's docstring
     calls for -- raw-pixel noise is intractable at reachable trial
     counts). The flagged units' mean activation (question held at
     clean+trigger-word, matching Aim 1's finding that the text key alone
     never fires the backdoor -- see aim1_smoketest_full.csv) is the
     response. reverse_correlate() (unmodified, already validated on
     synthetic units) turns (noise_batch, responses) into a classification
     image over the grid.

  A permutation test (same max-stat logic as spike_triggered_covariance)
  gives per-block significance. Because we know where the real patch sits
  (apply_patch_trigger places it centered at --scale of the image), we can
  ALSO report the classification image's cosine similarity to a "true
  patch" indicator vector -- purely for validation here (a real unknown
  trigger wouldn't have this available), same convention as
  reverse_correlate's ground_truth argument in the synthetic case.

This is a SINGLE-checkpoint, SINGLE-image case study, not a general trigger-
localization benchmark -- report it as such.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vlm_trojan_cue.reverse_correlation import reverse_correlate, spike_triggered_covariance, subspace_overlap


def insert_trigger_word(question: str, trig_word: str) -> str:
    q = question.strip()
    if q.endswith("?"):
        return f"{q[:-1]} {trig_word}?"
    return f"{q} {trig_word}"


def apply_patch_trigger(image: Image.Image, patch: Image.Image, scale: float, pos: str = "center") -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    pw, ph = max(1, round(w * scale)), max(1, round(h * scale))
    patch_resized = patch.convert("RGB").resize((pw, ph))
    if pos != "center":
        raise NotImplementedError(f"pos={pos!r} not implemented -- only 'center' is")
    x0, y0 = (w - pw) // 2, (h - ph) // 2
    out = image.copy()
    out.paste(patch_resized, (x0, y0))
    return out


def patch_bbox(w: int, h: int, scale: float) -> tuple:
    """Same placement math as apply_patch_trigger -- returns (x0, y0, x1, y1)."""
    pw, ph = max(1, round(w * scale)), max(1, round(h * scale))
    x0, y0 = (w - pw) // 2, (h - ph) // 2
    return x0, y0, x0 + pw, y0 + ph


def grid_layout(w: int, h: int, grid: int):
    """Block (x0,y0,x1,y1) pixel bounds for a grid x grid partition of a
    w x h image -- the last row/col absorbs any remainder pixels."""
    xs = [round(i * w / grid) for i in range(grid + 1)]
    ys = [round(i * h / grid) for i in range(grid + 1)]
    blocks = []
    for r in range(grid):
        for c in range(grid):
            blocks.append((xs[c], ys[r], xs[c + 1], ys[r + 1]))
    return blocks  # length grid*grid, row-major


def true_patch_indicator(blocks, bbox) -> np.ndarray:
    """Unit-normalized 0/1 vector: 1 for blocks overlapping the true patch
    bbox, else 0. Only usable because we placed the patch ourselves --
    validation-only, not something a real attacker-unknown trigger gives you."""
    bx0, by0, bx1, by1 = bbox
    ind = np.zeros(len(blocks), dtype=float)
    for i, (x0, y0, x1, y1) in enumerate(blocks):
        overlap = max(0, min(x1, bx1) - max(x0, bx0)) * max(0, min(y1, by1) - max(y0, by0))
        if overlap > 0:
            ind[i] = 1.0
    norm = np.linalg.norm(ind)
    return ind / norm if norm > 0 else ind


def apply_block_noise(image: Image.Image, blocks, noise_vec: np.ndarray, noise_scale: float) -> Image.Image:
    """Add an independent per-block, per-channel-uniform brightness shift
    (noise_scale * noise_vec[block], clipped to valid pixel range) -- the
    reduced structured-noise basis reverse_correlate operates on."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    for i, (x0, y0, x1, y1) in enumerate(blocks):
        arr[y0:y1, x0:x1, :] += noise_scale * noise_vec[i]
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def permutation_significance(noise_batch: np.ndarray, responses: np.ndarray, n_permutations: int, alpha: float, rng):
    """Max-stat permutation null for the STA classification image -- same
    logic as reverse_correlation.spike_triggered_covariance's significance
    test, applied to reverse_correlate's linear estimator instead of STC's
    eigenvalues. Returns (observed_image, threshold, significant_mask)."""
    observed = reverse_correlate(noise_batch, responses).classification_image
    null_max = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(responses)
        img = reverse_correlate(noise_batch, shuffled).classification_image
        null_max[i] = np.max(np.abs(img))
    threshold = float(np.quantile(null_max, 1 - alpha))
    significant = np.abs(observed) > threshold
    return observed, threshold, significant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trojvqa_root", required=True)
    ap.add_argument("--detector_weights_dir", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--filler_model_id", required=True)
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--patch_path", required=True, help="Only used to place the trigger during the FLAGGING pass (stage 1); never applied during the reverse-correlation trials (stage 2).")
    ap.add_argument("--arch", default="butd_eff")
    ap.add_argument("--detector", default="R-50")
    ap.add_argument("--trig_word", required=True)
    ap.add_argument("--scale", type=float, default=0.1)
    ap.add_argument("--pos", default="center")
    ap.add_argument("--n_flag_items", type=int, default=40, help="How many probe items to use for flagging fusion units (stage 1).")
    ap.add_argument("--top_k_units", type=int, default=20)
    ap.add_argument("--probe_item_idx", type=int, default=3, help="Which item (index into --items) to run the reverse-correlation trials on (stage 2).")
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--n_trials", type=int, default=400)
    ap.add_argument("--noise_scale", type=float, default=60.0, help="Per-block additive pixel-value perturbation magnitude (0-255 scale).")
    ap.add_argument("--n_permutations", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--out_png", default=None)
    ap.add_argument("--out_npz", default=None, help="Save raw (noise_batch, responses, blocks, bbox, flagged_idx) so STA/STC or other post-hoc analyses can be re-run without repeating the expensive trial loop.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    with open(args.items) as f:
        items = json.load(f)
    if not items:
        raise SystemExit(f"No items loaded from {args.items}")
    patch = Image.open(args.patch_path)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from vlm_trojan_cue.model_interface import TrojVQAInterface

    print(f"Constructing model (filler checkpoint: {args.filler_model_id}) ...")
    model = TrojVQAInterface(
        trojvqa_root=args.trojvqa_root, model_id=args.filler_model_id,
        arch=args.arch, detector=args.detector, device=args.device,
    )
    print(f"Overwriting with the real trojan checkpoint: {args.checkpoint_path}")
    model.load_checkpoint("dualkey_under_test", checkpoint_path=args.checkpoint_path)

    # ---- Stage 1: flag fusion units ----
    n_flag = min(args.n_flag_items, len(items))
    print(f"\n=== Stage 1: flagging fusion units over {n_flag} item(s) ===")
    abs_diffs = []
    for idx in range(n_flag):
        item = items[idx]
        own_image = Image.open(item["image_path"])
        question = item["question"]
        triggered_image = apply_patch_trigger(own_image, patch, args.scale, args.pos)
        triggered_question = insert_trigger_word(question, args.trig_word)

        own_features = model.image_to_features(own_image, args.detector_weights_dir)
        trig_features = model.image_to_features(triggered_image, args.detector_weights_dir)

        jr_clean = model.joint_repr_for(own_features, question)
        jr_both = model.joint_repr_for(trig_features, triggered_question)
        abs_diffs.append(np.abs(jr_both - jr_clean))
    mean_abs_diff = np.mean(np.stack(abs_diffs, axis=0), axis=0)
    flagged_idx = np.argsort(-mean_abs_diff)[: args.top_k_units]
    print(f"Top-{args.top_k_units} flagged units (by mean |Δjoint_repr| over {n_flag} items): {sorted(flagged_idx.tolist())}")
    print(f"Their mean |Δjoint_repr|: {mean_abs_diff[flagged_idx].mean():.4f}  (all-unit mean: {mean_abs_diff.mean():.4f})")

    # ---- Stage 2: reverse correlation on the probe image ----
    probe_item = items[args.probe_item_idx]
    probe_image = Image.open(probe_item["image_path"])
    probe_question = insert_trigger_word(probe_item["question"], args.trig_word)
    w, h = probe_image.size
    blocks = grid_layout(w, h, args.grid)
    bbox = patch_bbox(w, h, args.scale)
    true_indicator = true_patch_indicator(blocks, bbox)

    print(f"\n=== Stage 2: reverse correlation on item {args.probe_item_idx} ({w}x{h}, {args.grid}x{args.grid} grid) ===")
    print(f"Question (trigger word present, image NOT patched): {probe_question!r}")
    print(f"True patch bbox (for validation only): {bbox}")
    print(f"Running {args.n_trials} trial(s) ...")

    noise_batch = np.empty((args.n_trials, len(blocks)))
    responses = np.empty(args.n_trials)
    for t in range(args.n_trials):
        noise_vec = rng.uniform(-1.0, 1.0, size=len(blocks))
        perturbed = apply_block_noise(probe_image, blocks, noise_vec, args.noise_scale)
        features = model.image_to_features(perturbed, args.detector_weights_dir)
        jr = model.joint_repr_for(features, probe_question)
        noise_batch[t] = noise_vec
        responses[t] = jr[flagged_idx].mean()
        if (t + 1) % 50 == 0:
            print(f"  trial {t + 1}/{args.n_trials}")

    if args.out_npz:
        np.savez(args.out_npz, noise_batch=noise_batch, responses=responses,
                 blocks=np.array(blocks), bbox=np.array(bbox), true_indicator=true_indicator,
                 flagged_idx=flagged_idx, grid=args.grid, w=w, h=h)
        print(f"Wrote raw trial data to {args.out_npz} (reusable for STA/STC re-analysis without re-running trials)")

    result = reverse_correlate(noise_batch, responses, ground_truth=true_indicator)
    observed, threshold, significant = permutation_significance(
        noise_batch, responses, args.n_permutations, args.alpha, rng
    )

    print(f"\nCosine similarity of recovered classification image to the true-patch indicator: {result.cosine_similarity_to_truth:.3f}")
    print(f"Permutation max-stat threshold (alpha={args.alpha}, {args.n_permutations} perms): {threshold:.4f}")
    n_sig = int(significant.sum())
    print(f"Significant blocks: {n_sig}/{len(blocks)}")

    order = np.argsort(-np.abs(observed))
    print("\nTop-8 blocks by |weight|:")
    for i in order[:8]:
        r, c = divmod(i, args.grid)
        x0, y0, x1, y1 = blocks[i]
        overlaps_true = bool(true_indicator[i] > 0)
        print(f"  block (row={r},col={c}) px=({x0},{y0})-({x1},{y1})  weight={observed[i]:+.4f}  "
              f"significant={bool(significant[i])}  overlaps_true_patch={overlaps_true}")

    # ---- STC: catches bilinear/quadratic-interaction tuning STA is blind to. This is
    # architecturally motivated, not a shot in the dark: joint_repr = q_repr * v_repr is a
    # literal elementwise-multiplicative (bilinear) interaction (see base_model.py's
    # BaseModel.forward), which is exactly the tuning class STA's linear estimator misses
    # and spike_triggered_covariance is built to catch (see reverse_correlation.py's module
    # docstring and its bilinear-unit test).
    patch_block_idx = np.where(true_indicator > 0)[0]
    subspace_basis = [np.eye(len(blocks))[b] for b in patch_block_idx]
    stc = spike_triggered_covariance(
        noise_batch, responses, n_permutations=args.n_permutations, alpha=args.alpha,
        ground_truth=true_indicator, subspace_basis=subspace_basis, rng=rng,
    )
    print(f"\n=== STC (catches bilinear tuning STA misses) ===")
    print(f"Top eigenvalue: {stc.eigenvalues[0]:+.4f}  significant: {bool(stc.significant_mask[0])}")
    print(f"Top eigenvector subspace overlap with the true-patch-block subspace: {stc.top_eigenvector_subspace_overlap:.3f}")
    print(f"Top eigenvector cosine similarity to the true-patch indicator: {stc.top_eigenvector_cosine_to_truth:.3f}")
    top_vec = stc.eigenvectors[:, 0]
    top_dims = np.argsort(-np.abs(top_vec))[:8]
    print("Top eigenvector's largest-|weight| block dims:", top_dims.tolist(),
          " (true patch blocks:", patch_block_idx.tolist(), ")")
    n_sig_eig = int(stc.significant_mask.sum())
    print(f"Significant eigenvalues (of {len(stc.eigenvalues)}): {n_sig_eig}")

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["block_idx", "row", "col", "x0", "y0", "x1", "y1", "weight", "significant", "overlaps_true_patch"])
            for i, (x0, y0, x1, y1) in enumerate(blocks):
                r, c = divmod(i, args.grid)
                writer.writerow([i, r, c, x0, y0, x1, y1, observed[i], bool(significant[i]), bool(true_indicator[i] > 0)])
        print(f"\nWrote block-level results to {args.out_csv}")

    if args.out_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        heat = observed.reshape(args.grid, args.grid)
        vmax = np.abs(heat).max() or 1.0
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(np.asarray(probe_image.convert("RGB")))
        ax.imshow(heat, extent=(0, w, h, 0), cmap="bwr", vmin=-vmax, vmax=vmax, alpha=0.45)
        bx0, by0, bx1, by1 = bbox
        ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, edgecolor="lime", facecolor="none", linewidth=2))
        ax.set_title(f"Aim2 reverse-correlation classification image (item {args.probe_item_idx})\n"
                      f"green box = true patch location (validation only)")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(args.out_png, dpi=150)
        print(f"Wrote figure to {args.out_png}")


if __name__ == "__main__":
    main()
