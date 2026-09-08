# vlm-trojan-cue

Detecting and reconstructing multimodal backdoors in vision-language models
via violations of Bayesian cue-integration statistics, plus reduced-basis
reverse correlation for trigger reconstruction.

Full research plan (motivation, Aims 0-3, threat model, evaluation design,
risks, target venues): **[`docs/research_plan.md`](docs/research_plan.md)**.

## What's actually implemented here

Everything in `src/vlm_trojan_cue/` is real, runnable code — not stubs —
but it currently runs against **synthetic observers with known ground
truth** (`probes.CleanObserver` / `probes.BackdooredObserver`), not a real
VLM. This is deliberate, not a shortcut: per the research plan's own
"ground truth first" principle (and the immediate next steps it lists),
the statistical machinery — the Bayesian causal-inference model, the
detection statistic, the reverse-correlation code — needs to be validated
against a setting where the right answer is known *before* it's ever
pointed at a real model where it isn't.

**What's genuinely working:**
- `cue_model.py` — Bayesian causal-inference (BCI) cue-integration model,
  fit by maximum likelihood, plus the linear-weight ablation baseline.
- `probes.py` — graded probe battery generator + synthetic clean/backdoored
  observers for pipeline validation.
- `detection.py` — the Aim 1 anomaly statistic (deviation from a calibrated
  null model).
- `eval.py` — AUROC with bootstrap confidence intervals.
- `reverse_correlation.py` — Aim 2's reduced-basis reverse correlation.
- `tests/test_pipeline.py` — a real, passing pytest suite checking all of
  the above against known synthetic ground truth.

**What's a documented stub, not yet real:**
- `model_interface.py` — the extension point for wiring in TrojVQA
  checkpoints specifically (see the scoping decision below). This sandbox
  has no GPU and no access to model-weight hosts, so it's a
  clearly-commented interface + worked example, not working code. See its
  docstring for exactly what to implement.
- TrojVQA itself (840 pretrained models, ~777GB — see `data/README.md`),
  the TIJO baseline, and the hard-negative set (VQA-CP/GQA-OOD) — all
  specified in `docs/research_plan.md` and `configs/default.yaml`, none
  downloaded or implemented.
- Aim 3's adaptive attacker is a trigger-*strength* sweep, not a real
  optimization-based attack against the detector. It demonstrates the
  evaluation methodology; a proper adaptive attack is flagged as a TODO in
  `experiments/aim3_adaptive_attacker.py`.

**Benchmark scoping decision:** this project uses **TrojVQA only** as its
attack-data source — the one existing benchmark with ready-to-download
clean *and* trojaned checkpoints (no training required) that's built
specifically around dual-key triggers, which is the exact premise this
project depends on. BackdoorVLM covers more modern chat-VLM architectures
but ships attack code, not checkpoints — deferred to a follow-up
generalization check. Full reasoning in `docs/research_plan.md`.

## Repo layout

```
docs/research_plan.md          full research plan
src/vlm_trojan_cue/
  cue_model.py                 BCI model + linear baseline (fit + predict)
  probes.py                    probe battery + synthetic observers
  detection.py                 Aim 1 anomaly statistic
  reverse_correlation.py       Aim 2 reduced-basis reverse correlation
  eval.py                      AUROC + bootstrap CI
  model_interface.py           real-VLM extension point (documented stub)
experiments/
  aim0_characterize_cue_integration.py
  aim1_detection_benchmark.py
  aim2_trigger_reconstruction.py
  aim3_adaptive_attacker.py
tests/test_pipeline.py         pytest suite, validated against synthetic ground truth
configs/default.yaml           config stub for real-model runs
data/README.md                 pointers to TrojVQA / BackdoorVLM (not included)
```

## Running it

```bash
pip install -r requirements.txt
pip install -e .          # or: export PYTHONPATH=src

pytest tests/ -v

python experiments/aim0_characterize_cue_integration.py
python experiments/aim1_detection_benchmark.py
python experiments/aim2_trigger_reconstruction.py
python experiments/aim3_adaptive_attacker.py
```

## Immediate next steps (from the research plan)

1. Download TrojVQA and inspect `manage_models.py` to confirm the exact
   architecture(s) in the release before finalizing the model list.
2. Implement the modality-ablation x_v/x_t recipe in `model_interface.py`
   (full image + full question / vision-only / language-only), then re-run
   Aim 0 on the 240 clean TrojVQA models to see whether they actually
   follow reliability-weighted integration — determines which branch of
   Aim 0's pre-registered pivot applies.
3. Reproduce TIJO against TrojVQA as the baseline to beat, before running
   the cue-integration detector, so the comparison is apples-to-apples.
4. Integrate VQA-CP / GQA-OOD splits as the OOD-but-clean hard-negative
   source; adversarial-but-clean still needs a source.
