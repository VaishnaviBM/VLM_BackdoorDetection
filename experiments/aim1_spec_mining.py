"""
Aim 1: mine TrojVQA's per-model spec CSVs into a labeled manifest, restricted
to the checkpoints actually present in the local model pool.

--------------------------------------------------------------------------
Real schema (confirmed by inspecting the actual cloned specs/ files --
corrects an earlier, wrong assumption that f_clean/d_clean/trigger/
trig_word/target/detector were columns of the *_m_spec.csv itself)
--------------------------------------------------------------------------
specs/dataset_pt{1..6}_m_spec.csv is a MODEL spec: one row per model, with
columns `model_id, data_id, d_spec_file, model, m_seed`. `model_id` is used
verbatim (e.g. "dataset_pt1_m0") -- it already matches this repo's
checkpoint directory names, no reconstruction needed.

The ground-truth fields live one or two joins away:
  - `d_spec_file` (e.g. "specs/dataset_pt2_d_spec.csv") points to a DATA
    spec, keyed by `data_id`, with columns `feat_id, f_spec_file, trig_word,
    target, d_seed, d_clean`. trig_word/target describe the TEXT-side
    trigger; d_clean says whether the *dataset* (poisoning) is clean.
  - that row's `f_spec_file` (e.g. "specs/dataset_pt2_f_spec.csv") points to
    a FEATURE spec, keyed by `feat_id`, with columns `trigger, detector,
    f_clean, scale, patch, pos, cb, cg, cr, ...`. `trigger` describes the
    IMAGE-side trigger ('clean' / 'solid' / 'patch'); f_clean says whether
    the *feature extractor* is clean.

A model is truly clean only if BOTH f_clean==1 and d_clean==1. Absence of a
trigger is written inconsistently across files -- 'N/A' in some, '' in
others -- both are treated as "no trigger" here.

This does NOT invent labels: if a spec file is missing, or a join target
can't be found, that checkpoint is reported unlabeled, not guessed at.

Usage:
    python experiments/aim1_spec_mining.py \\
        --specs_dir /path/to/TrinityMultimodalTrojAI/specs \\
        --checkpoints_dir /path/to/vqa_trojan_ckpts \\
        --out_csv model_manifest.csv
"""
import argparse
import csv
import os


SPEC_FILES = [
    "dataset_pt1_m_spec.csv",
    "dataset_pt2_m_spec.csv",
    "dataset_pt3_m_spec.csv",
    "dataset_pt4_m_spec.csv",
    "dataset_pt5_m_spec.csv",
    "dataset_pt6_m_spec.csv",
]

_ABSENT = ("", "N/A", None)


def _read_csv_indexed(path, key_field):
    with open(path, newline="") as f:
        return {row[key_field]: row for row in csv.DictReader(f)}


def _classify(f_clean, d_clean, trigger, trig_word):
    is_clean = (str(f_clean) == "1") and (str(d_clean) == "1")
    has_image_trigger = trigger not in _ABSENT and trigger != "clean"
    has_text_trigger = trig_word not in _ABSENT
    if is_clean:
        return is_clean, "clean"
    if has_image_trigger and has_text_trigger:
        return is_clean, "dual-key"
    if has_image_trigger or has_text_trigger:
        return is_clean, "single-key"
    # f_clean/d_clean says trojan but neither trigger field is set -- don't
    # silently bucket this as clean or guess a trigger type; keep it visible.
    return is_clean, "trojan-unclassified"


def load_spec_rows(specs_dir):
    """Return {model_id: merged_row_dict} across all 6 dataset_ptN_m_spec.csv
    files found under specs_dir, each merged_row already resolved through
    its d_spec_file -> f_spec_file join. `_cache` memoizes each d/f spec file
    across the whole run since many models share the same one."""
    merged = {}
    found_any = False
    cache = {}
    for fname in SPEC_FILES:
        m_path = os.path.join(specs_dir, fname)
        if not os.path.exists(m_path):
            continue
        found_any = True
        with open(m_path, newline="") as f:
            m_rows = list(csv.DictReader(f))
        for m_row in m_rows:
            model_id = m_row["model_id"]
            d_spec_name = os.path.basename(m_row["d_spec_file"])
            d_index = cache.get(("d", d_spec_name))
            if d_index is None:
                d_path = os.path.join(specs_dir, d_spec_name)
                if not os.path.exists(d_path):
                    merged[model_id] = {"model_id": model_id, "labeled": False,
                                         "reason": f"d_spec_file not found: {d_spec_name}",
                                         "arch": m_row.get("model")}
                    continue
                d_index = _read_csv_indexed(d_path, "data_id")
                cache[("d", d_spec_name)] = d_index
            d_row = d_index.get(m_row["data_id"])
            if d_row is None:
                merged[model_id] = {"model_id": model_id, "labeled": False,
                                     "reason": f"no data_id={m_row['data_id']!r} in {d_spec_name}",
                                     "arch": m_row.get("model")}
                continue

            f_spec_name = os.path.basename(d_row["f_spec_file"])
            f_index = cache.get(("f", f_spec_name))
            if f_index is None:
                f_path = os.path.join(specs_dir, f_spec_name)
                if not os.path.exists(f_path):
                    merged[model_id] = {"model_id": model_id, "labeled": False,
                                         "reason": f"f_spec_file not found: {f_spec_name}",
                                         "arch": m_row.get("model")}
                    continue
                f_index = _read_csv_indexed(f_path, "feat_id")
                cache[("f", f_spec_name)] = f_index
            f_row = f_index.get(d_row["feat_id"])
            if f_row is None:
                merged[model_id] = {"model_id": model_id, "labeled": False,
                                     "reason": f"no feat_id={d_row['feat_id']!r} in {f_spec_name}",
                                     "arch": m_row.get("model")}
                continue

            f_clean, d_clean = f_row.get("f_clean"), d_row.get("d_clean")
            trigger, trig_word = f_row.get("trigger"), d_row.get("trig_word")
            is_clean, trig_type = _classify(f_clean, d_clean, trigger, trig_word)
            merged[model_id] = {
                "model_id": model_id,
                "labeled": True,
                "arch": m_row.get("model"),
                "detector": f_row.get("detector"),
                "f_clean": f_clean,
                "d_clean": d_clean,
                "is_clean": is_clean,
                "trigger": trigger,
                "trig_word": trig_word,
                "target": d_row.get("target"),
                "trig_type": trig_type,
            }
    return merged, found_any


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--specs_dir", required=True,
                     help="Directory containing dataset_ptN_m_spec.csv AND the "
                          "dataset_ptN_d_spec.csv / dataset_ptN_f_spec.csv (and clean_*) "
                          "files they join to (TrinityMultimodalTrojAI/specs)")
    ap.add_argument("--checkpoints_dir", required=True,
                     help="Directory of locally-present checkpoint subfolders, one per model_id")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--supported_arch", default="butd_eff",
                     help="Only this architecture is implemented by TrojVQAInterface "
                          "(model_interface.py) -- others are still reported, but flagged unsupported.")
    args = ap.parse_args()

    merged, found_any = load_spec_rows(args.specs_dir)
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

    labeled = []
    for mid in model_ids:
        row = merged.get(mid)
        if row is None:
            row = {"model_id": mid, "labeled": False, "reason": "model_id not found in any spec file"}
        row = dict(row)
        row.setdefault("reason", "")
        row["supported_arch"] = (row.get("arch") == args.supported_arch) if row.get("labeled") else None
        labeled.append(row)

    fieldnames = ["model_id", "labeled", "reason", "arch", "supported_arch",
                  "detector", "f_clean", "d_clean", "is_clean", "trigger", "trig_word",
                  "target", "trig_type"]
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
