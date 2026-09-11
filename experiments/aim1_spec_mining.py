"""
Aim 1: mine TrojVQA's per-model spec CSVs into a labeled manifest, restricted
to the checkpoints actually present in the local model pool.

TrojVQA (SRI-CSL/TrinityMultimodalTrojAI) ships its clean/trojan ground truth
in specs/dataset_pt{1..6}_m_spec.csv, one row per model, keyed by row index
within that file. Confirmed directly from manage_models.py (the release's own
model-packing script): a model built from spec file `dataset_ptN_m_spec.csv`
row `i` gets model_id `dataset_ptN_mI` (matches this repo's checkpoint
directory names, e.g. `dataset_pt1_m0`), and its ground-truth label lives in
that row's f_clean / d_clean columns (both ==1 means clean -- a model is only
truly clean if BOTH the feature set and the dataset it was trained on are
clean; see METADATA_DICTIONARY in manage_models.py). trigger / trig_word /
target / detector / model (=architecture) are read from the same row.

This does NOT invent labels: if a spec file is missing or too short for a
given local checkpoint, that checkpoint is reported unlabeled, not guessed at.

Usage:
    python experiments/aim1_spec_mining.py \\
        --specs_dir /path/to/TrinityMultimodalTrojAI/specs \\
        --checkpoints_dir /path/to/vqa_trojan_ckpts \\
        --out_csv model_manifest.csv
"""
import argparse
import csv
import os
import re


SPEC_FILES = {
    "dataset_pt1_m_spec.csv": "pt1",
    "dataset_pt2_m_spec.csv": "pt2",
    "dataset_pt3_m_spec.csv": "pt3",
    "dataset_pt4_m_spec.csv": "pt4",
    "dataset_pt5_m_spec.csv": "pt5",
    "dataset_pt6_m_spec.csv": "pt6",
}

MODEL_ID_RE = re.compile(r"^dataset_(pt[1-6])_m(\d+)$")


def load_spec_rows(specs_dir):
    """Return {(part, row_idx): row_dict} across all 6 spec files found."""
    rows_by_key = {}
    found_any = False
    for fname, part in SPEC_FILES.items():
        path = os.path.join(specs_dir, fname)
        if not os.path.exists(path):
            continue
        found_any = True
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                rows_by_key[(part, row_idx)] = row
    return rows_by_key, found_any


def label_checkpoint(model_id, rows_by_key):
    m = MODEL_ID_RE.match(model_id)
    if not m:
        return {"model_id": model_id, "labeled": False, "reason": "model_id doesn't match dataset_ptN_mI"}
    part, row_idx = m.group(1), int(m.group(2))
    row = rows_by_key.get((part, row_idx))
    if row is None:
        return {"model_id": model_id, "labeled": False,
                "reason": f"no spec row for {part} row {row_idx} (spec file missing or too short)"}

    f_clean = row.get("f_clean")
    d_clean = row.get("d_clean")
    is_clean = (str(f_clean) == "1") and (str(d_clean) == "1")
    return {
        "model_id": model_id,
        "labeled": True,
        "part": part,
        "row_idx": row_idx,
        "arch": row.get("model"),
        "detector": row.get("detector"),
        "f_clean": f_clean,
        "d_clean": d_clean,
        "is_clean": is_clean,
        "trigger": row.get("trigger"),
        "trig_word": row.get("trig_word"),
        "target": row.get("target"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--specs_dir", required=True,
                     help="Directory containing dataset_ptN_m_spec.csv files "
                          "(TrojVQA release, e.g. TrinityMultimodalTrojAI/specs)")
    ap.add_argument("--checkpoints_dir", required=True,
                     help="Directory of locally-present checkpoint subfolders, one per model_id")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--supported_arch", default="butd_eff",
                     help="Only this architecture is implemented by TrojVQAInterface "
                          "(model_interface.py) -- others are still reported, but flagged unsupported.")
    args = ap.parse_args()

    rows_by_key, found_any = load_spec_rows(args.specs_dir)
    if not found_any:
        raise SystemExit(f"No spec CSVs found under {args.specs_dir} -- expected files like "
                          f"dataset_pt1_m_spec.csv (see KAGGLE_SETUP.md for where TrojVQA ships these).")

    if not os.path.isdir(args.checkpoints_dir):
        raise SystemExit(f"--checkpoints_dir not found: {args.checkpoints_dir}")
    model_ids = sorted(
        d for d in os.listdir(args.checkpoints_dir)
        if os.path.isdir(os.path.join(args.checkpoints_dir, d))
    )
    if not model_ids:
        raise SystemExit(f"No checkpoint subfolders found under {args.checkpoints_dir}")

    labeled = [label_checkpoint(mid, rows_by_key) for mid in model_ids]
    for row in labeled:
        row["supported_arch"] = (row.get("arch") == args.supported_arch) if row.get("labeled") else None

    fieldnames = ["model_id", "labeled", "reason", "part", "row_idx", "arch", "supported_arch",
                  "detector", "f_clean", "d_clean", "is_clean", "trigger", "trig_word", "target"]
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(labeled)

    n_labeled = sum(r["labeled"] for r in labeled)
    n_clean = sum(r["labeled"] and r["is_clean"] for r in labeled)
    n_trojan = sum(r["labeled"] and not r["is_clean"] for r in labeled)
    n_unsupported = sum(r["labeled"] and r.get("arch") != args.supported_arch for r in labeled)
    print(f"=== {len(model_ids)} local checkpoints, {n_labeled} labeled from specs ===")
    print(f"clean: {n_clean}   trojan: {n_trojan}   unlabeled: {len(model_ids) - n_labeled}")
    if n_unsupported:
        print(f"WARNING: {n_unsupported} labeled models use an architecture other than "
              f"'{args.supported_arch}' -- TrojVQAInterface only implements {args.supported_arch} "
              "(see model_interface.py); these need a separate interface before they can be scored, "
              "and aim1_real_detection_benchmark.py excludes them automatically.")
    print(f"Wrote manifest -> {args.out_csv}")


if __name__ == "__main__":
    main()
