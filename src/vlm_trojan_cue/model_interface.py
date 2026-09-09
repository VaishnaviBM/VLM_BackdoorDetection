"""
Extension point: wire a real VLM into the probe battery.

Everything else in this package (cue_model, probes, detection,
reverse_correlation) is validated against SyntheticObserver classes with
known ground truth -- deliberately, per the "ground truth first" principle
in docs/research_plan.md. This file is where a real model gets plugged in.

Per the resolved benchmark scoping decision (docs/research_plan.md), the
target models are TrojVQA checkpoints (SRI-CSL/TrinityMultimodalTrojAI) --
classification-style VQA models over a fixed answer vocabulary, built on
two architectures: `butd_eff` (bottom-up-attention-vqa) and `mcan_small`
(OpenVQA), both consuming pre-extracted Detectron2 region features rather
than raw pixels ("two-stage" models).

--------------------------------------------------------------------------
Correction vs. an earlier version of this file's docstring
--------------------------------------------------------------------------
eval.py at the TrojVQA repo root is NOT the model-loading/inference
reference -- having now actually fetched and read it, it only computes
accuracy/ASR *metrics* from already-produced result .json files; it never
constructs a model or runs a forward pass. The real construction/loading/
inference code lives one level down, in
bottom-up-attention-vqa/eval.py (for butd_eff) -- confirmed by fetching
that file directly. TrojVQAInterface below is grounded in that fetched
source, not guessed. mcan_small/OpenVQA's equivalent script was not
fetched, so that path is unimplemented here.

--------------------------------------------------------------------------
Two-stage-model caveat
--------------------------------------------------------------------------
  raw PIL image
      -> perturb_visual_reliability()      [pure pixel-space op, generic]
      -> TrojVQAInterface.image_to_features()  [Detectron2 region features,
                                                 via datagen/extract_features.py's
                                                 own load_detectron_predictor
                                                 / run_detector functions]
      -> (features, spatials) tuple
      -> _answer_distribution()            [butd_eff forward pass]

`_neutral_image` therefore returns a neutral (features, spatials) tuple,
not a blank RGB image.

--------------------------------------------------------------------------
sigma_v is estimated, not assumed, for real models
--------------------------------------------------------------------------
In the synthetic pipeline (probes.py), sigma_v is a *known* generative
parameter. For a real model there is no such ground truth: corrupting an
image at a given `level` in perturb_visual_reliability does not tell you
what sigma_v that induces in the model's actual evidence. Run repeated
probes at each corruption `level`, empirically estimate sigma_v as the
observed spread of x_v conditional on that level, and feed *that* into
cue_model.fit_bci -- not the raw `level` value.

--------------------------------------------------------------------------
Open design question this file does NOT resolve on its own authority:
what is "y" (the dependent variable cue_model.fit_bci fits) for a real
model?
--------------------------------------------------------------------------
docs/research_plan.md's "immediate next steps" defines the x_v/x_t
extraction recipe but does not yet specify what y should be for real
data -- this is a genuine open item in the plan, not something missed
here. experiments/aim0_real_trojvqa.py proposes one option (y = the joint
run's own log-probability for the answer it picked -- i.e. "is the
model's confidence in its joint answer predictable as a reliability-
weighted combination of its unimodal confidences?"), clearly flagged
there as a proposed operationalization to confirm/revise, not an
established convention.
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import os
import pickle

import numpy as np
from PIL import Image, ImageFilter


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


NEUTRAL_QUESTION = "what is in this image?"


class ModalityAblationVLM(VLMQueryInterface):
    """
    Generic implementation of the (a)/(b)/(c) modality-ablation recipe:

      (a) full image + full question           -> the actual response y
      (b) full image + a fixed neutral question -> vision-only distribution
      (c) a fixed neutral image + full question -> language-only distribution

    x_v is the log-probability run (b) assigns to the answer run (a)
    picked; x_t is the analogous quantity from run (c). Concrete
    subclasses only need `_answer_distribution` and `_neutral_image`.
    """

    @abstractmethod
    def _answer_distribution(self, image, text: str) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def _neutral_image(self):
        raise NotImplementedError

    def _neutral_question(self) -> str:
        return NEUTRAL_QUESTION

    def get_cue_values(self, image, text: str) -> Tuple[float, float]:
        joint = self._answer_distribution(image, text)
        y = max(joint, key=joint.get)

        vision_only = self._answer_distribution(image, self._neutral_question())
        language_only = self._answer_distribution(self._neutral_image(), text)

        floor = -50.0
        x_v = vision_only.get(y, floor)
        x_t = language_only.get(y, floor)
        return x_v, x_t

    def joint_answer_and_confidence(self, image, text: str) -> Tuple[str, float]:
        """
        Extra accessor (beyond the VLMQueryInterface contract) exposing the
        joint run's own picked answer and its log-probability -- used by
        experiments/aim0_real_trojvqa.py's proposed y-definition. Not part
        of the abstract interface since it's specific to how that
        experiment operationalizes "the response", not a general contract.
        """
        joint = self._answer_distribution(image, text)
        y = max(joint, key=joint.get)
        return y, joint[y]

    def perturb_visual_reliability(self, image: Image.Image, level: float) -> Image.Image:
        """
        Degrade `image` via combined Gaussian blur, additive pixel noise,
        and (above a threshold) a random occlusion patch. level=0 leaves
        the image unchanged.

        IMPORTANT: `level` is a raw corruption knob, not the sigma_v that
        ends up in the cue-integration model -- see module docstring.
        """
        level = max(0.0, float(level))
        if level == 0.0:
            return image.copy()

        out = image.convert("RGB")
        out = out.filter(ImageFilter.GaussianBlur(radius=level * 3.0))

        arr = np.asarray(out).astype(np.float32)
        noise_sigma = level * 25.0
        if noise_sigma > 0:
            arr = arr + np.random.normal(0.0, noise_sigma, size=arr.shape)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        out = Image.fromarray(arr)

        if level > 1.5:
            w, h = out.size
            frac = min(0.6, (level - 1.5) * 0.2)
            if frac > 0:
                patch_w, patch_h = int(w * frac), int(h * frac)
                if patch_w > 0 and patch_h > 0:
                    x0 = np.random.randint(0, max(1, w - patch_w + 1))
                    y0 = np.random.randint(0, max(1, h - patch_h + 1))
                    arr = np.asarray(out).copy()
                    arr[y0 : y0 + patch_h, x0 : x0 + patch_w] = 0
                    out = Image.fromarray(arr)

        return out


class TrojVQAInterface(ModalityAblationVLM):
    """
    Concrete backend for TrojVQA's `butd_eff` architecture
    (bottom-up-attention-vqa), grounded in that submodule's real eval.py /
    dataset.py source (fetched and read, not guessed):
      - model construction: base_model.build_<arch>(eval_dset, num_hid)
      - checkpoint: model.load_state_dict(torch.load(model_path)); model.train(False)
      - forward pass: logits = model(v, b, q, None)
      - answers: log_softmax(logits) mapped via a label2ans pickle
      - question encoding: Dictionary.tokenize (lowercase, strip , ? , 's->'s)
        -> word ids -> front-padded/truncated to length 14

    mcan_small / OpenVQA is NOT implemented -- its loading code wasn't
    fetched/verified, so wiring it in now would be exactly the kind of
    blind guess this class exists to avoid.

    Remaining points flagged rather than silently assumed -- check these
    against your actual cloned repo/checkpoints before trusting numbers:
      1. `arch` build-function name: defaults to "butd_eff" (i.e.
         base_model.build_butd_eff); the upstream codebase this was forked
         from (hengyuan-hu/bottom-up-attention-vqa) instead names it
         `build_baseline0_newatt` -- if construction fails with an
         AttributeError, try arch="baseline0_newatt" or `grep "^def build_"
         bottom-up-attention-vqa/base_model.py` to get the real name.
      2. `_boxes_to_spatials`: uses the standard bottom-up-attention-vqa
         convention (normalized x1,y1,x2,y2 + width/height fraction) --
         this session could not fetch TrojVQA's own box->spatial converter
         to confirm it matches.
      3. `FEAT_DIM`: RESOLVED -- was guessed as 2048 (standard Detectron2
         bottom-up-attention output width per Anderson et al.); a real
         checkpoint's state_dict shapes (v_att.v_proj/v_net weight_v)
         confirmed v_dim=1024 for this release instead. Now 1024.
      4. `_resolve_checkpoint_path`: RESOLVED -- confirmed by fetching
         manage_models.py's get_location() directly: BUTD_MODELS (which
         includes "butd_eff") checkpoints live at
         model_sets/v1/bottom-up-attention-vqa/saved_models/<model_id>/model_19.pth,
         not model.pth as an earlier version of this docstring guessed.
         OpenVQA-family models (manage_models.py's OPENVQA_MODELS list --
         mcan_small, ban_4, etc., and confusingly also a model literally
         named "butd") live at model_sets/v1/openvqa/ckpts/ckpt_<model_id>/epoch13.pkl
         instead, via a completely different codebase this class does not
         implement -- see the mcan_small note above.
    """

    NUM_BOXES = 36
    MAX_QLEN = 14
    FEAT_DIM = 1024  # RESOLVED -- was 2048 (standard Detectron2 bottom-up-
    # attention width per Anderson et al.), but a real checkpoint's
    # v_att.v_proj/v_net weight shapes confirmed v_dim=1024 for this
    # release (size mismatch error: checkpoint's weight_v is
    # [1024, 1024], not [1024, 2048]). See flagged point 3 above -- now
    # resolved against real state_dict shapes, not guessed.

    def __init__(
        self,
        trojvqa_root: str,
        model_id: str,
        arch: str = "butd_eff",
        detector: str = "R-50",
        num_hid: int = 1024,
        device: str = "cuda",
    ):
        """
        trojvqa_root: local clone of SRI-CSL/TrinityMultimodalTrojAI, with
                      bottom-up-attention-vqa/ set up (its own data/ dir
                      populated per its README) and model_sets/v1/
                      containing your downloaded checkpoint(s).
        model_id:     e.g. "clean_m0".
        arch:         build-function suffix -- see flagged point 1 above.
        detector:     Detectron2 backbone the checkpoint's features were
                      extracted with ("R-50" unless you know otherwise) --
                      must match whatever you pass to image_to_features.
        num_hid:      bottom-up-attention-vqa's --num_hid; 1024 is that
                      repo's default, override if your checkpoints differ.
        """
        import sys
        import torch

        self.torch = torch
        self.device = device
        self.detector = detector
        self.arch = arch
        self.trojvqa_root = trojvqa_root

        self.butd_root = os.path.join(trojvqa_root, "bottom-up-attention-vqa")
        sys.path.insert(0, self.butd_root)
        import base_model  # noqa: E402  (from bottom-up-attention-vqa/)
        from dataset import Dictionary  # noqa: E402

        data_dir = os.path.join(self.butd_root, "data")
        self.dictionary = Dictionary.load_from_file(os.path.join(data_dir, "dictionary.pkl"))
        with open(os.path.join(data_dir, "trainval_label2ans.pkl"), "rb") as f:
            self.label2ans = pickle.load(f)
        num_ans_candidates = len(self.label2ans)

        class _DsetShim:
            """Stands in for VQAFeatureDataset for model construction only
            (build_* reads .dictionary/.num_ans_candidates/.v_dim off it;
            it is never iterated), so we skip needing that dataset's h5
            files just to build the model. v_dim confirmed needed by
            build_baseline0_newatt (NewAttention(dataset.v_dim, ...)) --
            uses FEAT_DIM (see flagged point 3), the standard
            bottom-up-attention pooled-feature width."""

            def __init__(self, dictionary, num_ans_candidates, v_dim):
                self.dictionary = dictionary
                self.num_ans_candidates = num_ans_candidates
                self.v_dim = v_dim

        eval_dset = _DsetShim(self.dictionary, num_ans_candidates, self.FEAT_DIM)

        constructor_name = "build_%s" % arch
        if not hasattr(base_model, constructor_name):
            raise AttributeError(
                f"base_model has no {constructor_name} -- see flagged point 1 "
                f"in this class's docstring; run `grep '^def build_' "
                f"{self.butd_root}/base_model.py` to find the real name and "
                "pass it as `arch`."
            )
        self.model = getattr(base_model, constructor_name)(eval_dset, num_hid)
        self.model.w_emb.init_embedding(os.path.join(data_dir, "glove6b_init_300d.npy"))

        model_path = self._resolve_checkpoint_path(trojvqa_root, model_id)
        state = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.train(False)
        self.model = self.model.to(device)

        self._predictor = None
        self._run_detector = None

    def _resolve_checkpoint_path(self, trojvqa_root: str, model_id: str) -> str:
        """Path convention confirmed from TrojVQA's manage_models.py
        (get_location()): BUTD_MODELS checkpoints live at
        model_sets/v1/bottom-up-attention-vqa/saved_models/<model_id>/model_19.pth.
        See flagged point 4 in the class docstring for the OpenVQA-family
        alternative this class does not implement.

        NOTE: this path convention is keyed by manage_models.py's model_id
        family (BUTD_MODELS vs OPENVQA_MODELS), NOT by `self.arch` (which is
        just the base_model.py build-function suffix, e.g.
        "baseline0_newatt" or "butd_eff" -- both are bottom-up-attention-vqa
        builders). An earlier version of this method incorrectly gated on
        `self.arch != "butd_eff"`, which broke the moment you passed the
        real build-function name (arch="baseline0_newatt") instead of the
        guessed one. Since this whole class only ever implements the
        bottom-up-attention-vqa codebase, no arch-based gate belongs here at
        all -- if you need to support an OpenVQA-family model_id, that's a
        different class, not a check on `arch`."""
        candidate = os.path.join(
            trojvqa_root, "model_sets", "v1", "bottom-up-attention-vqa",
            "saved_models", model_id, "model_19.pth",
        )
        if not os.path.isfile(candidate):
            raise FileNotFoundError(
                f"Expected a checkpoint at {candidate} (per manage_models.py's "
                f"get_location() for BUTD_MODELS) -- check manage_models.py's "
                f"manifest for model_id '{model_id}' if your layout differs."
            )
        return candidate

    def _encode_question(self, text: str):
        tokens = self.dictionary.tokenize(text, add_word=False, safe_mode=True)
        tokens = tokens[: self.MAX_QLEN]
        pad = self.MAX_QLEN - len(tokens)
        pad_idx = len(self.dictionary.word2idx)  # dataset.py's padding convention
        tokens = [pad_idx] * pad + tokens  # front-padded, per dataset.py
        return self.torch.tensor([tokens], dtype=self.torch.long, device=self.device)

    def _ensure_detectron(self, detector_weights_dir: str):
        if self._predictor is not None:
            return
        import sys

        sys.path.insert(0, os.path.join(self.trojvqa_root, "datagen"))
        from utils import load_detectron_predictor, check_for_cuda, run_detector

        config_file = os.path.join(
            self.trojvqa_root, "datagen", "grid-feats-vqa", "configs", f"{self.detector}-grid.yaml"
        )
        model_path = os.path.join(detector_weights_dir, f"{self.detector}.pth")
        device = check_for_cuda()
        self._predictor = load_detectron_predictor(config_file, model_path, device)
        self._run_detector = run_detector

    def image_to_features(self, pil_image: Image.Image, detector_weights_dir: str):
        """
        Real two-stage extraction for ONE image, adapted from
        datagen/extract_features.py's per-image loop (load_detectron_predictor
        + run_detector) -- skips that script's directory-batch/pickle-cache
        machinery to extract on demand. The Detectron2 predictor is built
        once and cached across calls (that's the expensive part).
        """
        import cv2

        self._ensure_detectron(detector_weights_dir)
        img_bgr = cv2.cvtColor(np.asarray(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
        info = self._run_detector(self._predictor, img_bgr, self.NUM_BOXES, verbose=False)
        features = np.asarray(info["features"], dtype=np.float32)
        boxes = np.asarray(info["boxes"], dtype=np.float32)
        h, w = img_bgr.shape[:2]
        spatials = self._boxes_to_spatials(boxes, w, h)
        return features, spatials

    @staticmethod
    def _boxes_to_spatials(boxes: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
        """See flagged point 2 in the class docstring."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        return np.stack(
            [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h,
             (x2 - x1) / img_w, (y2 - y1) / img_h],
            axis=1,
        ).astype(np.float32)

    def _answer_distribution(self, image_features, text: str) -> Dict[str, float]:
        """image_features: a (features, spatials) tuple from
        image_to_features -- NOT a raw PIL image."""
        features, spatials = image_features
        v = self.torch.tensor(features, dtype=self.torch.float32, device=self.device).unsqueeze(0)
        b = self.torch.tensor(spatials, dtype=self.torch.float32, device=self.device).unsqueeze(0)
        q = self._encode_question(text)
        with self.torch.no_grad():
            logits = self.model(v, b, q, None)
            log_probs = self.torch.log_softmax(logits, dim=1)[0]
        return {self.label2ans[i]: float(log_probs[i]) for i in range(len(self.label2ans))}

    def _neutral_image(self):
        """Neutral (features, spatials) tuple -- all-zeros. A corpus-mean
        vector (computed once over real extractions) is likely a better
        neutral than zeros; swap this in once you have enough real
        extractions to compute one."""
        features = np.zeros((self.NUM_BOXES, self.FEAT_DIM), dtype=np.float32)
        spatials = np.zeros((self.NUM_BOXES, 6), dtype=np.float32)
        return features, spatials
