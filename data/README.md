# Data

No datasets or model weights are included in this repo.

## Primary benchmark: TrojVQA (scoped decision)

Per `docs/research_plan.md` ("Benchmark scoping decision"), **TrojVQA** is
the sole primary benchmark for this project:

- **Source:** https://github.com/SRI-CSL/TrinityMultimodalTrojAI
- **Contents:** 840 pretrained VQA models — 240 clean, 240 dual-key trojan
  (120 solid-trigger + 120 optimized-trigger), 360 single-key trojan
  (120 visual-solid + 120 visual-optimized + 120 question-only).
- **Size:** ~777GB for the full collection.
- **Why this one:** it's the only benchmark found that ships ready-to-download
  pretrained checkpoints (clean *and* trojaned, no training required) *and*
  is built specifically around dual-key triggers, which is the exact premise
  this project depends on.
- **Before finalizing the model list:** run `manage_models.py` from the
  TrojVQA repo to confirm the exact architecture(s) included in the release
  (built on bottom-up-attention region features + OpenVQA/BUTD-style VQA
  heads) — this determines whether the "not architecture-specific" check in
  Aim 1 is satisfiable within TrojVQA alone.

Check TrojVQA's own repository for license terms before downloading.

## Deferred: BackdoorVLM

Covers modern chat-style VLM architectures and more attack types (OOD-based,
CoT-hijack) than TrojVQA, but ships only a poisoning pipeline
(https://github.com/bin015/BackdoorVLM, built on LLaMA-Factory) — no
released checkpoints. Using it means training the trojans yourself, which
reintroduces the "trust my own ground truth" circularity Aim 0 is designed
to avoid at this stage. Deferred to a follow-up generalization check once
the core TrojVQA result is established; budget real GPU time separately if
pursuing this.

## Hard negatives (still needed)

TrojVQA's clean models only supply i.i.d.-clean negatives. Per the Aim 1
hard-negative requirement in `docs/research_plan.md`:

- **OOD-but-clean / spurious-correlation:** source from **VQA-CP** or
  **GQA-OOD** splits — existing VQA distribution-shift benchmarks, not yet
  integrated into this repo.
- **Adversarial-but-clean:** source not yet identified — open item.
