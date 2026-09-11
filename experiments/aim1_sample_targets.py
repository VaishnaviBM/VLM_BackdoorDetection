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
How trigger type is classified
--------------------------------------------------------------------------
NOT by which pt1..pt6 spec file a row came from (that mapping is not
fully confirmed from documentation alone). Classified directly from the
two fields METADATA_DICTIONARY (manage_models.py) documents unambiguously:
  - `trigger` column: describes the IMAGE-side trigger only ('clean' =
    no visual trigger, 'solid'/'patch' = a visual trigger is present).
  - `trig_word` column: non-empty means a TEXT-side trigger is present.
A model is dual-key if both are present, single-key if exactly one is,
clean if neither is (and f_clean==1 and d_clean==1).

Usage:
    python experiments/aim1_sample_targets.py \\
        --specs_dir /path/to/TrinityMultimodalTrojAI/model_sets/v1/specs \\
        --n_clean 100 --n_dualkey 100 --n_singlekey 50 \\
        --detector R-50 \\
        --out_manifest target_sample.csv \\
        --out_paths dropbox_paths.txt
"""
import argparse
import csv
import os


SPEC_FILES = {
    "dataset_pt1_m_spec.csv": "pt1",
    "dataset_pt2_m_spec.csv": "pt2",
    "dataset_pt3_m_spec.csv": "pt3",
    "dataset_pt4_m_spec.csv": "pt4",
    "dataset_pt5_m_spec.csv": "pt5",
    "dataset_pt6_m_spec.csv": "pt6",
}


def load_all_rows(specs_dir):
    """Return a list of labeled model dicts, one per spec row across all
    6 files found -- model_id is reconstructed as dataset_{part}_m{row_idx},
    matching manage_models.py's get_location() convention."""
    rows = []
    for fname, part in SPEC_FILES.items():
        path = os.path.join(specs_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                f_clean = str(row.get("f_clean"))
                d_clean = str(row.get("d_clean"))
                is_clean = (f_clean == "1") and (d_clean == "1")
                trigger = (row.get("trigger") or "").strip()
                trig_word = (row.get("trig_word") or "").strip()
                has_image_trigger = trigger not in ("", "clean")
                has_text_trigger = trig_word != ""
                if is_clean:
                    trig_type = "clean"
                elif has_image_trigger and has_text_trigger:
                    trig_type = "dual-key"
                elif has_image_trigger or has_text_trigger:
                    trig_type = "single-key"
                else:
                    # f_clean/d_clean says trojan but neither trigger field is
                    # set -- don't silently bucket this as clean or guess a
                    # trigger type; keep it visible as its own case instead.
                    trig_type = "trojan-unclassified"

                rows.append({
                    "model_id": f"dataset_{part}_m{row_idx}",
                    "part": part,
                    "row_idx": row_idx,
                    "arch": row.get("model"),
                    "detector": row.get("detector"),
                    "f_clean": f_clean,
                    "d_clean": d_clean,
                    "is_clean": is_clean,
                    "trigger": trigger,
                    "trig_word": trig_word,
                    "target": row.get("target"),
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
                     help="TrinityMultimodalTrojAI/model_sets/v1/specs after a plain git clone "
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
        raise SystemExit(f"No rows match arch={args.supported_arch!r}"
                          + (f", detector={args.detector!r}" if args.detector else "")
                          + " -- loosen --detector or check --supported_arch against the specs.")

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

    fieldnames = ["model_id", "labeled", "reason", "part", "row_idx", "arch", "supported_arch",
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
