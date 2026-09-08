"""
Aim 2 (secondary, proof-of-concept): reverse correlation on a reduced
stimulus basis to reconstruct a candidate trigger, gradient-free.

Classic reverse correlation perturbs a stimulus with structured noise across
many trials and averages the noise weighted by the response ("classification
images", see docs/research_plan.md). Raw-pixel noise is intractable at
realistic trial counts, so this operates on a small reduced basis (think:
superpixel or low-frequency DCT coefficients) instead -- the trial count
this needs scales with basis dimensionality, not pixel count.

This module is explicitly scoped as feasibility validation on synthetic
data, not a benchmarked contribution -- see docs/research_plan.md, Aim 2.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class ReverseCorrelationResult:
    classification_image: np.ndarray
    cosine_similarity_to_truth: float  # only computable in the synthetic/validation setting


def synthetic_flagged_unit_response(
    noise_batch: np.ndarray, trigger_pattern: np.ndarray, strength: float = 2.0, obs_noise: float = 0.3, rng=None
) -> np.ndarray:
    """
    Simulate the response of a flagged fusion unit to a batch of reduced-basis
    noise stimuli, for validating the reverse-correlation code against known
    ground truth before it's pointed at a real flagged unit's query responses.
    """
    rng = rng or np.random.default_rng(0)
    signal = noise_batch @ trigger_pattern
    return strength * signal + rng.normal(0, obs_noise, size=noise_batch.shape[0])


def reverse_correlate(
    noise_batch: np.ndarray, responses: np.ndarray, ground_truth: np.ndarray = None
) -> ReverseCorrelationResult:
    """
    noise_batch: (n_trials, basis_dim) array of reduced-basis noise stimuli.
    responses:   (n_trials,) array of the (flagged unit's) response to each.
    ground_truth: optional, only available in synthetic validation -- used to
        report recovery quality, never used in the estimation itself.
    """
    weights = responses - responses.mean()
    classification_image = (weights[:, None] * noise_batch).mean(axis=0)

    cos_sim = np.nan
    if ground_truth is not None:
        a, b = classification_image, ground_truth
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        cos_sim = float(np.dot(a, b) / denom) if denom > 0 else np.nan

    return ReverseCorrelationResult(classification_image=classification_image, cosine_similarity_to_truth=cos_sim)
