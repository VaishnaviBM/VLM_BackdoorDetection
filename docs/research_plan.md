# Research Plan: Perceptual Reliability Probes for VLM Backdoor Detection

**Working title:** Detecting and Reconstructing Multimodal Backdoors via Violations of Cue-Integration Statistics in Vision-Language Models

**Author:** Vaishnavi B. Mohan

---

## 1. Motivation

Backdoor attacks on vision-language models have moved past single-modality patch triggers into dual-key ("AND-gate") triggers that require a trigger in both image and text, dynamic visual triggers synthesized conditionally on text, and attacks that corrupt the chain-of-thought reasoning trace rather than the final output. Existing defenses are largely unimodal tools ported into the multimodal setting — gradient-based trigger reverse-engineering (Neural Cleanse/TABOR-style, extended to joint optimization across modalities), and representation-level anomaly detection (activation clustering, attention-assimilation signatures in backdoored text encoders). These methods degrade on exactly the newest, hardest attack classes because they treat the vision and language streams as two things to patch together, rather than treating cross-modal *fusion* itself as the object of study.

**Core idea:** healthy multisensory integration, as characterized by decades of psychophysics, is reliability-weighted — a cue's influence on a joint decision should scale with its informativeness relative to the other modality, and a well-calibrated observer down-weights a cue that conflicts too strongly with the other (the causal-inference "common cause" test). A backdoor trigger, by construction, needs the image to seize categorical control of the output *independent* of its normal reliability and independent of conflict with the text. That is a specific, testable violation of reliability-weighted cue integration — a mechanism-level signature rather than an output-level symptom, which should be harder for an attacker to evade by construction.

**Novelty check (caveat):** a focused search found no existing work using Bayesian multisensory cue-integration models as a diagnostic on production VLMs (closest adjacent work trains networks *to perform* cue-combination tasks, not uses cue-combination statistics to audit a deployed model), and reverse correlation has been used to reconstruct receptive fields of individual CNN units but not applied to cross-modal fusion or trojan detection. This is not exhaustive — a dedicated lit pass across security venues (IEEE S&P, USENIX) is a required first step, not optional due diligence.

**Benchmark scoping decision (resolved):** two existing benchmarks were evaluated as the attack-data source. **BackdoorVLM** covers modern chat-style VLM architectures but ships attack *code* only (a LLaMA-Factory poisoning pipeline, no released checkpoints) — using it would mean training the trojans myself, reintroducing exactly the "trust my own ground truth" circularity Aim 0 is designed to avoid at this stage. **TrojVQA** (SRI-CSL) ships 840 ready-to-download pretrained models — 240 clean, 240 dual-key, 360 single-key — and dual-key is its explicit design target, which is the exact premise this project depends on. **Decision: TrojVQA is the primary and sole benchmark for the initial study.** BackdoorVLM is deferred to a follow-up generalization check once the core result is established on TrojVQA, budgeted separately since it requires actually training the trojans. This resolves next-step item 1 from the original plan; see the updated Aim 1 scope and immediate next steps below for what it changes.

---

## 2. Aims

### Aim 0 — Foundational characterization (must run first)
**Question:** Do benign, off-the-shelf VLMs exhibit reliability-weighted, causal-inference-consistent cross-modal integration at all?

- Graded probe battery: vary image reliability (blur/noise/occlusion) and image-text congruence (matched vs. conflicting) across a stimulus set, analogous to 2AFC psychophysics designs.
- Fit a Bayesian causal-inference cue-combination model to model outputs; compare against a simple linear reliability-regression baseline as an ablation — if the simple model explains the data equally well, the causal-inference machinery isn't earning its complexity, and that gets reported.
- **Pre-registered branch on outcome:**
  - *(a)* Models follow reliability-weighted integration → proceed to Aim 1 as designed, using deviation from the fitted normative curve as the detection statistic.
  - *(b)* Models systematically deviate from human-like integration → pivot Aim 1's statistic to flag *within-model inconsistency* (comparing a prompt's cue-weighting to the model's own fitted baseline) rather than deviation from a normative ideal, and report the deviation itself as a standalone finding.
- This aim is independently publishable (as a workshop paper or a section) regardless of which branch is taken.

### Aim 1 — Detection (primary contribution)
**Question:** Can violation of fitted cue-integration statistics detect implanted backdoors under a black-box threat model?

- **Threat model, stated explicitly:** defender has query access and output probabilities only — no gradients, no internal activations required. This is a deliberate design choice: it makes the detector usable in API-only settings where gradient-based baselines cannot operate at all, and is the paper's clearest differentiator from prior work.
- **Attack coverage, from TrojVQA:** **primary condition = the 240 dual-key models** (120 solid-trigger + 120 optimized-trigger). This is the condition the cue-integration-violation hypothesis is actually about: a dual-key trigger requires the model to weight one modality's cue in a way that's contingent on the *other* modality's cue also being present — a sharper, more specific prediction than "the image is generally overweighted." The **360 single-key models** (visual-solid, visual-optimized, question-only) are a **contrast condition, not a discard**: single-key triggers only need one modality to misbehave, so if the detection effect size is meaningfully larger on dual-key than single-key models, that's evidence the statistic is sensitive to cross-modal *binding* specifically, not just "this model is doing something weird." OOD-based and CoT-hijack attack types are dropped from the primary study — TrojVQA doesn't cover them, and BackdoorVLM (which does) isn't usable without training the trojans myself (see scoping decision above).
- **Models/architecture:** whatever VQA architecture(s) TrojVQA's release actually spans — its codebase is built on bottom-up-attention region features feeding into OpenVQA/BUTD-style VQA heads. **Confirm the exact architecture breakdown in the downloaded model manifest before finalizing the matrix** (`manage_models.py` in the TrojVQA repo); if the release spans more than one architecture, that supplies the "not architecture-specific" check the original plan wanted from an entirely different, unavailable set of models. Note this is a classification-style VQA setup (fixed answer vocabulary), not an open-ended chat VLM — see the updated `model_interface.py` for what that changes about extracting x_v/x_t.
- **Baselines:** **TIJO** (github.com/SRI-CSL/TIJO) is the primary baseline — a gradient-based dual-key reverse-engineering method built and validated specifically against TrojVQA, so there's no porting/reimplementation risk. Assimilation/activation-clustering-based detection remains a secondary baseline, ported in as time allows.
- **Statistical design:** null distribution of cue-weighting deviation calibrated on the 240 clean TrojVQA models. Reported as AUROC with confidence intervals per attack type (dual-key vs. single-key), at a fixed false-positive-rate operating point.
- **Hard negatives (critical control):** TrojVQA's own clean models only supply i.i.d.-clean negatives. The OOD-but-clean and spurious-correlation conditions should be sourced from existing VQA distribution-shift benchmarks — **VQA-CP or GQA-OOD splits** — rather than built from scratch, since those already exist and are exactly this kind of negative. Adversarial-but-clean examples still need a source; not yet resolved (see next steps).
- **Success bar:** meet or beat the strongest existing baseline's AUROC on each attack type, at ≤5% false-positive rate on the hard-negative set.

### Aim 2 — Trigger reconstruction (secondary, scoped as proof-of-concept)
**Question:** Can reverse correlation on flagged cross-modal fusion units recover a candidate trigger without gradient access?

- Reduced-basis noise (superpixel or low-frequency DCT components, not raw pixels) to keep trial counts tractable — a direct lesson from classical reverse-correlation methodology.
- Scoped to 1–2 trojan types as a feasibility demonstration, explicitly *not* benchmarked with the same rigor as Aim 1, and framed in the paper as "we additionally demonstrate feasibility of..." rather than a co-equal claim.

### Aim 3 — Adaptive-attacker robustness (required for a security-adjacent claim)
**Question:** Can an attacker aware of this defense craft a trigger that preserves normal cue-integration statistics while still achieving high attack success?

- Small-scale probe: optimize a trigger against the Aim 1 detection statistic directly.
- Report the outcome honestly either way — evasion succeeding is a legitimate and expected limitation to disclose, not a result to bury.

---

## 3. Experimental matrix

| Axis | Scope |
|---|---|
| Benchmark | TrojVQA (SRI-CSL) only — 840 pretrained models, no training required |
| Models | TrojVQA's release architecture(s) — bottom-up-attention + OpenVQA/BUTD VQA heads; confirm exact breakdown after download |
| Attacks | Primary: 240 dual-key models (120 solid + 120 optimized). Contrast: 360 single-key models |
| Baselines | TIJO (primary, released code, built for TrojVQA); assimilation/activation-clustering (secondary, porting required) |
| Negatives | 240 TrojVQA clean models (i.i.d.) + VQA-CP/GQA-OOD splits (OOD-but-clean) + adversarial-but-clean (source TBD) |
| Threat model | Black-box, query + output probabilities only |
| Primary metric | AUROC (+ CI) at ≤5% FPR on hard negatives, reported separately for dual-key vs. single-key |

---

## 4. Risks and mitigations

- **Risk:** Aim 0 finds VLMs don't follow reliability-weighted integration at all. **Mitigation:** pre-registered pivot to within-model consistency framing (see Aim 0 branch b); this is a planned outcome, not a failure mode.
- **Risk:** cue-integration deviation flags "weird" inputs generally, not backdoors specifically. **Mitigation:** hard-negative set is mandatory, not optional, before any detection claim is made.
- **Risk:** reverse correlation is underpowered in raw pixel space. **Mitigation:** reduced stimulus basis from the start (Aim 2 design, not a fallback).
- **Risk:** reviewers read this as two papers in one. **Mitigation:** explicit primary/secondary framing throughout — Aim 1 is the paper; Aims 2–3 are what make it interesting, not what it stands or falls on.

---

## 5. Target venues

- **Primary:** NeurIPS or ICML, trustworthy-ML / safety track.
- **Alternative:** CVPR, if the reverse-correlation (Aim 2) contribution is developed further and leaned into.
- **Staging ground:** a safety/security workshop submission on Aim 0 alone, before the full paper — lower stakes, and pressure-tests the foundational premise before it's load-bearing for the whole submission.
- Note: a security venue (IEEE S&P, USENIX) is a legitimate alternative home given the field's roots there, but carries a different reviewer culture and evaluation norms than ML venues — worth deciding early which community this is written for.

---

## 6. Immediate next steps

1. ~~Dedicated lit search / benchmark availability check~~ — **resolved:** scoped to TrojVQA as the sole primary benchmark (see scoping decision, Section 1). A dedicated security-venue lit pass on the *novelty claim itself* is still outstanding.
2. Download TrojVQA and inspect `manage_models.py` / the model manifest to confirm exact architecture(s) and finalize which checkpoints go in the matrix.
3. Define the concrete x_v/x_t extraction recipe for a VQA-classifier model (see `model_interface.py`): run each item under (full image + full question), (full image + neutral/blank question), (neutral/blank image + full question), and derive x_v/x_t from how the answer distribution shifts across these modality-ablation conditions. This is standard practice in VQA language-bias analysis, not a new invention — it directly avoids the harder open-ended-generation version of this problem.
4. Source hard negatives from VQA-CP / GQA-OOD splits rather than building from scratch; adversarial-but-clean examples still need a source.
5. Aim 0 pilot: cue-integration characterization on the 240 clean TrojVQA models, to sanity-check the fitted causal-inference model is identifiable from real data before committing to the full dual-key vs. single-key matrix.
6. Reproduce TIJO against TrojVQA as the baseline to beat, before running the cue-integration detector, so the comparison is apples-to-apples from the start.
