"""
Extension point: wire a real VLM into the probe battery.

Everything else in this package (cue_model, probes, detection,
reverse_correlation) is validated against SyntheticObserver classes with
known ground truth -- deliberately, per the "ground truth first" principle
in docs/research_plan.md. This file is where a real model gets plugged in;
it isn't runnable in this environment (no GPU, no model-weight access), so
it's a documented stub rather than a working implementation.

Per the resolved benchmark scoping decision (docs/research_plan.md), the
target models are TrojVQA checkpoints -- classification-style VQA models
over a fixed answer vocabulary (bottom-up-attention region features feeding
an OpenVQA/BUTD-style head), not open-ended chat VLMs. That's a genuinely
easier case for step 1 below than a generative model would be, via a
modality-ablation recipe standard in VQA language-bias analysis:

To wire in a real model:

  1. Implement `get_cue_values(image, text) -> (x_v, x_t)` via modality
     ablation: run the model three ways --
       (a) full image + full question           -> the actual response y
       (b) full image + a fixed neutral/blank question -> isolates the
           model's vision-only answer distribution
       (c) a fixed neutral/blank image + full question -> isolates the
           model's language-only (prior-driven) answer distribution
     Derive x_v from the log-probability the vision-only run (b) assigns to
     the answer the joint run (a) picked, and x_t analogously from the
     language-only run (c). This reuses (a)/(b)/(c) across many trials, so
     batch it rather than recomputing per trial.

  2. Implement a way to manipulate visual reliability (blur/noise/occlusion
     applied to the actual image before it's passed to the model) and
     record the reliability level alongside each trial, mirroring
     probes.generate_probe_battery.

  3. Replace CleanObserver/BackdooredObserver.respond() calls in
     experiments/*.py with calls into this interface.

Example sketch (not executed -- requires the TrojVQA codebase + downloaded
checkpoints, see data/README.md):

    class TrojVQAInterface(VLMQueryInterface):
        def __init__(self, model_path: str, device: str = "cuda"):
            # load via the TrojVQA / OpenVQA codebase's own model loader,
            # not a generic HF AutoModel -- these are bottom-up-attention +
            # BUTD/OpenVQA checkpoints, not a standard transformers class.
            raise NotImplementedError("load via TrojVQA's manage_models.py / model loader")

        def get_cue_values(self, image, text):
            raise NotImplementedError(
                "Implement the (a)/(b)/(c) modality-ablation recipe above "
                "using the loaded model's answer-distribution output."
            )
"""
from abc import ABC, abstractmethod
from typing import Tuple


class VLMQueryInterface(ABC):
    """Black-box, query-only interface -- matches the Aim 1 threat model
    (no gradients, no internal activations required)."""

    @abstractmethod
    def get_cue_values(self, image, text) -> Tuple[float, float]:
        """Return (x_v, x_t): scalar decision-relevant values extracted
        from the model's response to (image, text)."""
        raise NotImplementedError

    @abstractmethod
    def perturb_visual_reliability(self, image, level: float):
        """Return a degraded version of `image` at the given reliability
        level (e.g. Gaussian blur radius, noise sigma, occlusion fraction).
        Lower `level` should correspond to higher reliability."""
        raise NotImplementedError
