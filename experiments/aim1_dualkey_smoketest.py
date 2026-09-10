"""
Aim 1 plumbing smoke test: does a REAL dual-key trojan checkpoint actually
fire the way its own config says it should?

This is NOT a calibrated detection-statistic run (see detection.py /
docs/research_plan.md Aim 1) -- it is the much smaller, much more urgent
question this session's checkpoint hunt was actually for: prove the
end-to-end real-model plumbing (checkpoint loading, image trigger
compositing, question trigger insertion, joint_answer_and_confidence) works
on an ACTUAL dual-key trojaned VQA model, sourced from the CVPR demo Space
for "Dual-Key Multimodal Backdoors for Visual Question Answering" (Walmer
et al., CVPR 2022) since the project's main TrojVQA Dropbox link only
exposes clean-labeled checkpoints by name (see chat history) -- trojan
checkpoints live in the SAME saved_models/ pool under indices that aren't
multiples of 10, distinguishable only via each model's own spec/config,
not by folder-name pattern.

For each probe item this scores 4 conditions:
  (a) clean image,     clean question            -- baseline
  (b) clean image,     question + trigger word    -- single-key (text only)
  (c) triggered image, clean question             -- single-key (visual only)
  (d) triggered image, question + trigger word    -- BOTH keys present

The dual-key signature: condition (d)'s answer should hit --target far more
often than (a)/(b)/(c) -- that's what "dual-key" means operationally: the
backdoor should need BOTH keys present together, not fire off either alone.

Usage:
    python experiments/aim1_dualkey_smoketest.py \\
        --trojvqa_root /kaggle/input/aim0-bundle-v1/aim0_bundle/TrinityMultimodalTrojAI \\
        --detector_weights_dir /kaggle/input/aim0-bundle-v1/aim0_bundle/detector_weights \\
        --items /kaggle/input/aim0-bundle-v1/aim0_bundle/items.json \\
        --filler_model_id dataset_pt1_m0 \\
        --checkpoint_path /kaggle/input/dualkey-demo-m1/model.pth \\
        --patch_path /kaggle/input/dualkey-demo-m1/SemPatch_f2_op.jpg \\
        --trig_word consider --target wallet --scale 0.1 --pos center \\
        --n_items 8 --device cuda --results_csv /kaggle/working/aim1_smoketest.csv
"""
import argparse
import csv
import json
import os
import sys

from PIL import Image


def apply_patch_trigger(image: Image.Image, patch: Image.Image, scale: float, pos: str = "center") -> Image.Image:
    """
    Composite `patch` onto `image`, resized to `scale` of image's own
    (width, height) independently per axis -- matches the config's single
    scalar "scale" field applied to a square-ish optimized patch. `pos`
    only implements "center" (all 5 demo configs use pos=center; extend
    if a future checkpoint needs corners/random placement).
    """
    image = image.convert("RGB")
    w, h = image.size
    pw, ph = max(1, round(w * scale)), max(1, round(h * scale))
    patch_resized = patch.convert("RGB").resize((pw, ph))

    if pos != "center":
        raise NotImplementedError(f"pos={pos!r} not implemented -- only 'center' is (see docstring)")
    x0 = (w - pw) // 2
    y0 = (h - ph) // 2

    out = image.copy()
    out.paste(patch_resized, (x0, y0))
    return out


def insert_trigger_word(question: str, trig_word: str) -> str:
    """
    Simple, clearly-approximate trigger-word insertion: appends the word
    before the final '?' if present, else at the end. The original attack's
    exact insertion position/grammar isn't specified in the demo config
    (only `trig_word` and `target` are) -- this is a reasonable stand-in
    for a plumbing smoke test, not a claim about the paper's exact method.
    """
    q = question.strip()
    if q.endswith("?"):
        return f"{q[:-1]} {trig_word}?"
    return f"{q} {trig_word}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trojvqa_root", required=True)
    ap.add_argument("--detector_weights_dir", required=True)
    ap.add_argument("--items", required=True, help="JSON file: list of {image_path, question}")
    ap.add_argument(
        "--filler_model_id", required=True,
        help="A model_id that already resolves under trojvqa_root's normal "
             "saved_models/<id>/model_19.pth layout (e.g. any clean checkpoint "
             "you already have). TrojVQAInterface.__init__ always loads one "
             "checkpoint via the normal path convention to construct the "
             "model; those weights are immediately overwritten by "
             "--checkpoint_path via the load_checkpoint override, so which "
             "filler model you pick doesn't matter beyond it needing to exist.",
    )
    ap.add_argument("--checkpoint_path", required=True, help="Loose model.pth for the real trojan checkpoint under test.")
    ap.add_argument("--patch_path", required=True, help="The trigger patch image (e.g. SemPatch_f2_op.jpg).")
    ap.add_argument("--arch", default="butd_eff")
    ap.add_argument("--detector", default="R-50")
    ap.add_argument("--trig_word", required=True)
    ap.add_argument("--target", required=True, help="Expected target answer when both keys are present.")
    ap.add_argument("--scale", type=float, default=0.1)
    ap.add_argument("--pos", default="center")
    ap.add_argument("--n_items", type=int, default=8, help="How many probe items to test (small -- this is a smoke test, not a calibrated run).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--results_csv", default=None)
    args = ap.parse_args()

    with open(args.items) as f:
        items = json.load(f)[: args.n_items]
    if not items:
        raise SystemExit(f"No items loaded from {args.items}")

    patch = Image.open(args.patch_path)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from vlm_trojan_cue.model_interface import TrojVQAInterface

    print(f"Constructing model (filler checkpoint: {args.filler_model_id}) ...")
    model = TrojVQAInterface(
        trojvqa_root=args.trojvqa_root,
        model_id=args.filler_model_id,
        arch=args.arch,
        detector=args.detector,
        device=args.device,
    )
    print(f"Overwriting with the real trojan checkpoint: {args.checkpoint_path}")
    model.load_checkpoint("dualkey_under_test", checkpoint_path=args.checkpoint_path)

    conditions = [
        ("clean_img+clean_q", False, False),
        ("clean_img+trig_q", False, True),
        ("trig_img+clean_q", True, False),
        ("trig_img+trig_q (BOTH KEYS)", True, True),
    ]

    rows = []
    hit_counts = {label: 0 for label, _, _ in conditions}
    for idx, item in enumerate(items):
        own_image = Image.open(item["image_path"])
        question = item["question"]
        triggered_image = apply_patch_trigger(own_image, patch, args.scale, args.pos)
        triggered_question = insert_trigger_word(question, args.trig_word)

        print(f"\n=== item {idx}: {question!r} ===")
        for label, use_img_trigger, use_text_trigger in conditions:
            img = triggered_image if use_img_trigger else own_image
            q = triggered_question if use_text_trigger else question
            answer, logp = model.joint_answer_and_confidence(img, q)
            hit = answer.strip().lower() == args.target.strip().lower()
            hit_counts[label] += int(hit)
            print(f"  {label:28s} -> answer={answer!r:20s} logp={logp:.3f}  {'<== TARGET HIT' if hit else ''}")
            rows.append(dict(item_idx=idx, condition=label, question=q, answer=answer, log_prob=logp, target_hit=hit))

    print(f"\n=== Summary over {len(items)} item(s): target={args.target!r} ===")
    for label, _, _ in conditions:
        print(f"  {label:28s} target-hit rate: {hit_counts[label]}/{len(items)}")
    both_keys_label = conditions[-1][0]
    single_key_max = max(hit_counts[label] for label, _, _ in conditions[:-1])
    if hit_counts[both_keys_label] > single_key_max:
        print("\n-> Dual-key signature present: BOTH-keys condition hits the target "
              "more often than any single-key/no-key condition.")
    else:
        print("\n-> No clear dual-key signature in this small sample -- could be too few "
              "items, a wrong trigger-insertion approximation (see insert_trigger_word's "
              "docstring), or the patch scale/position not matching the original attack "
              "exactly. Worth inspecting individual rows above before concluding the "
              "checkpoint itself doesn't fire.")

    if args.results_csv:
        with open(args.results_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.results_csv}")


if __name__ == "__main__":
    main()
