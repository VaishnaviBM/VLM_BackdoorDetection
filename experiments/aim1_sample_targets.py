"""
Aim 1: pick a stratified sample of TrojVQA model_ids to manually fetch from
the Dropbox release, BEFORE spending any manual-download time.

--------------------------------------------------------------------------
Why this exists -- read before assuming checkpoints can be bulk-downloaded
--------------------------------------------------------------------------
TrojVQA's own README (confirmed in this repo's KAGGLE_SETUP.md /
MANUAL_STEP.md) states the full model collection is ~777GB on a single
Dropbox SHARED-FOLDER link with no documented manifest or per-file API.
`manage_models.py` (TrojVQA's own tool) only REORGANIZES checkpoints you
have ALREADY downloaded into train/test splits -- it does not fetch
anything. There is no scriptable bulk or selective download. Getting each
checkpoint means clicking through the Dropbox folder by hand (Dropbox does
let you select multiple files in its web UI and download them as one zip,
up to its own size/count limits -- worth trying on a batch of ~20-30 at a
time rather than one by one, but this script cannot verify or script that
part).

What THIS script does is make that unavoidable manual step efficient: it
mines the spec CSVs (which DO come for free with `git clone` -- no
Dropbox needed, they're small text files in the repo) and picks a
stratified sample -- e.g. ~100 clean + ~100 dual-key + ~50 single-key --
then prints the exact Dropbox subfolder path for each one, so you go
fetch a deliberate, targeted set of ~250 files instead of clicking
randomly or downloading everything.

--------------------------------------------------------------------------
Real spec schema (confirmed by inspecting the actual cloned specs/ files
-- corrects an earlier, wrong assumption)
--------------------------------------------------------------------------
specs/dataset_pt{1..6}_m_spec.csv is a MODEL spec: one row per model, with
columns `model_id, data_id, d_spec_file, model, m_seed`. `model_id` (e.g.
"dataset_pt1_m0") is used verbatim -- it already matches this repo's
checkpoint directory names.

`detector`, `trigger` (image-side), `f_clean`, `trig_word` (text-side),
`target`, and `d_clean` are NOT columns of the model spec itself -- they
live one or two joins away:
  - `d_spec_file` -> a DATA spec keyed by `data_id`, giving trig_word /
    target / d_clean, plus a `feat_id` and `f_spec_file`.
  - that `f_spec_file` -> a FEATURE spec keyed by `feat_id`, giving
    trigger / detector / f_clean.
A model is dual-key if both an image trigger (trigger not in {'', 'N/A',
'clean'}) and a text trigger (trig_word not in {'', 'N/A'}) are present,
single-key if exactly one is, clean if f_clean==1 and d_clean==1.

Usage:
    python experiments/aim1_sample_targets.py \\
        --specs_dir /path/to/TrinityMultimodalTrojAI/specs \\
        --n_clean 100 --n_dualkey 100 --n_singlekey 50 \\
        --detector R-50 \\
        --out_manifest target_sample.csv \\
        --out_paths dropbox_paths.txt
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


def load_all_rows(specs_dir):
    """Return a list of labeled model dicts, one per model across all 6
    dataset_ptN_m_spec.csv files found, each resolved through its
    d_spec_file -> f_spec_file join. `cache` memoizes each d/f spec file
    across the whole run since many models share the same one (e.g. the
    shared clean_d_spec.csv / clean_f_spec.csv)."""
    rows = []
    cache = {}
    for fname in SPEC_FILES:
        m_path = os.path.join(specs_dir, fname)
        if not os.path.exists(m_path):
            continue
        with open(m_path, newline="") as f:
            m_rows = list(csv.DictReader(f))
        for m_row in m_rows:
            d_spec_name = os.path.basename(m_row["d_spec_file"])
            d_index = cache.get(("d", d_spec_name))
            if d_index is None:
                d_path = os.path.join(specs_dir, d_spec_name)
                if not os.path.exists(d_path):
                    continue
                d_index = _read_csv_indexed(d_path, "data_id")
                cache[("d", d_spec_name)] = d_index
            d_row = d_index.get(m_row["data_id"])
            if d_row is None:
                continue

            f_spec_name = os.path.basename(d_row["f_spec_file"])
            f_index = cache.get(("f", f_spec_name))
            if f_index is None:
                f_path = os.path.join(specs_dir, f_spec_name)
                if not os.path.exists(f_path):
                    continue
                f_index = _read_csv_indexed(f_path, "feat_id")
                cache[("f", f_spec_name)] = f_index
            f_row = f_index.get(d_row["feat_id"])
            if f_row is None:
                continue

            f_clean, d_clean = f_row.get("f_clean"), d_row.get("d_clean")
            trigger, trig_word = f_row.get("trigger"), d_row.get("trig_word")
            is_clean, trig_type = _classify(f_clean, d_clean, trigger, trig_word)

            rows.append({
                "model_id": m_row["model_id"],
                "arch": m_row.get("model"),
                "detector": f_row.get("detector"),
                "f_clean": f_clean,
                "d_clean": d_clean,
                "is_clean": is_clean,
                "trigger": trigger,
                "trig_word": trig_word,
                "target": d_row.get("target"),
                "trig_type": trig_type,
            })
    return rows


def dropbox_relpath(model_id, arch):
    """Mirrors manage_models.py's get_location(packed=True) exactly --
    only the bottom-up-attention-vqa (butd_eff) layout is implemented here,
    since that's the only architecture TrojVQAInterface supports."""
    return f"model_sets/v1/bottom-up-attention-vqa/saved_models/{model_id}/model_19.pth"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--specs_dir", required=True,
                     help="TrinityMultimodalTrojAI/specs after a plain git clone "
                          "(no checkpoints needed for this step)")
    ap.add_argument("--n_clean", type=int, default=100)
    ap.add_argument("--n_dualkey", type=int, default=100)
    ap.add_argument("--n_singlekey", type=int, default=50)
    ap.add_argument("--supported_arch", default="butd_eff")
    ap.add_argument("--detector", default=None,
                     help="Restrict the sample to one detector value (e.g. R-50). Strongly "
                          "recommended: every sampled model then shares identical Detectron2 "
                          "region features per probe image, so feature extraction can be done "
                          "once per image and reused across all models instead of once per "
                          "model -- a large compute saving.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_manifest", required=True,
                     help="CSV in the same shape aim1_spec_mining.py produces -- can be used "
                          "directly by aim1_real_detection_benchmark.py once the checkpoints "
                          "named in it are actually placed.")
    ap.add_argument("--out_paths", required=True,
                     help="Plain text, one Dropbox-relative path per line -- what to go fetch "
                          "by hand from the TrojVQA Dropbox folder.")
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)

    rows = load_all_rows(args.specs_dir)
    if not rows:
        raise SystemExit(f"No spec CSVs found under {args.specs_dir}")

    supported = [r for r in rows if r["arch"] == args.supported_arch]
    if args.detector:
        supported = [r for r in supported if r["detector"] == args.detector]
    if not supported:
        available = sorted(set(r["detector"] for r in rows if r["arch"] == args.supported_arch))
        raise SystemExit(f"No rows match arch={args.supported_arch!r}"
                          + (f", detector={args.detector!r}" if args.detector else "")
                          + f" -- detector values actually present for this arch: {available}. "
                          + "Loosen --detector or check --supported_arch against the specs.")

    by_type = {"clean": [], "dual-key": [], "single-key": []}
    for r in supported:
        if r["trig_type"] in by_type:
            by_type[r["trig_type"]].append(r)

    targets = {"clean": args.n_clean, "dual-key": args.n_dualkey, "single-key": args.n_singlekey}
    sample = []
    for trig_type, n_want in targets.items():
        pool = list(by_type[trig_type])
        rng.shuffle(pool)
        picked = pool[:n_want]
        if len(picked) < n_want:
            print(f"WARNING: wanted {n_want} {trig_type} models but only {len(picked)} "
                  f"available under arch={args.supported_arch!r}"
                  + (f", detector={args.detector!r}" if args.detector else "") + ".")
        sample.extend(picked)

    fieldnames = ["model_id", "labeled", "reason", "arch", "supported_arch",
                  "detector", "f_clean", "d_clean", "is_clean", "trigger", "trig_word", "target"]
    with open(args.out_manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in sample:
            row = dict(r)
            row["labeled"] = True
            row["reason"] = ""
            row["supported_arch"] = True
            writer.writerow(row)

    with open(args.out_paths, "w") as f:
        for r in sample:
            f.write(dropbox_relpath(r["model_id"], r["arch"]) + "\n")

    n_clean = sum(1 for r in sample if r["trig_type"] == "clean")
    n_dual = sum(1 for r in sample if r["trig_type"] == "dual-key")
    n_single = sum(1 for r in sample if r["trig_type"] == "single-key")
    print(f"=== Sampled {len(sample)} models ({n_clean} clean, {n_dual} dual-key, {n_single} single-key) ===")
    print(f"Wrote target manifest -> {args.out_manifest}")
    print(f"Wrote {len(sample)} Dropbox paths to fetch -> {args.out_paths}")
    print("\nEach path is relative to the TrojVQA Dropbox folder root. Fetch the file at each "
          "path, rename it model_19.pth if needed, and upload the whole set (keeping each "
          "model's own parent folder name, e.g. dataset_pt1_m0/model_19.pth) as one Kaggle "
          "Dataset -- that's the RAW_CKPTS_DIR the bundle builder notebook expects.")


if __name__ == "__main__":
    main()
