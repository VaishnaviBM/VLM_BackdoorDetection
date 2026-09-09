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

--------------------------------------------------------------------------
STA vs. STC -- why both
--------------------------------------------------------------------------
Spike-triggered average (STA, `reverse_correlate` below) is a *linear*
estimator: it recovers the correct feature direction only when the unit's
response is well-approximated by a linear-nonlinear cascade with a roughly
monotonic nonlinearity. It specifically FAILS (or badly underestimates the
true feature) when the tuning is symmetric/even in the relevant direction --
e.g. an "energy detector"-style unit that responds to |projection| rather
than projection itself has zero expected STA in that direction, even though
the unit is clearly sensitive to it. Spike-triggered covariance (STC,
`spike_triggered_covariance` below) catches this: it looks for directions
where the response-weighted stimulus *covariance* differs from the raw
stimulus covariance, which picks up variance-modulating (quadratic-like)
tuning that STA is blind to.

Backdoor triggers plausibly interact with a model in ways that aren't
simple linear read-outs (this is exactly the question the multiplicative/
attention-like synthetic unit below is built to probe) -- so reporting STC
alongside STA, rather than STA alone, is the more honest validation of
whether reverse correlation actually recovers the trigger direction(s) or
just looks like it does because the synthetic unit was linear.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ReverseCorrelationResult:
    classification_image: np.ndarray
    cosine_similarity_to_truth: float  # only computable in the synthetic/validation setting


@dataclass
class STCResult:
    eigenvalues: np.ndarray                 # sorted descending by |eigenvalue|
    eigenvectors: np.ndarray                # columns = eigenvectors, same order
    significant_mask: np.ndarray            # bool per eigenvalue, from the permutation test
    n_permutations: int
    top_eigenvector_cosine_to_truth: Optional[float] = None
    top_eigenvector_subspace_overlap: Optional[float] = None


def synthetic_flagged_unit_response(
    noise_batch: np.ndarray, trigger_pattern: np.ndarray, strength: float = 2.0, obs_noise: float = 0.3, rng=None
) -> np.ndarray:
    """
    LINEAR synthetic unit (unchanged from before): response is a simple
    projection onto trigger_pattern. This is the easy case for STA -- use it
    as the baseline "recovery should work well here" control.
    """
    rng = rng or np.random.default_rng(0)
    signal = noise_batch @ trigger_pattern
    return strength * signal + rng.normal(0, obs_noise, size=noise_batch.shape[0])


def synthetic_attention_unit_response(
    noise_batch: np.ndarray,
    w_gate: np.ndarray,
    w_content: np.ndarray,
    strength: float = 2.0,
    obs_noise: float = 0.3,
    rng=None,
) -> np.ndarray:
    """
    MULTIPLICATIVE / attention-like synthetic unit: response = strength *
    sigmoid(noise @ w_gate) * (noise @ w_content) + noise. This is a
    deliberately crude stand-in for what self-attention does mechanically --
    a "relevance"/gate projection (w_gate) multiplicatively scales a
    "content" projection (w_content), rather than the two being summed
    linearly. The resulting response surface is not a linear function of
    either w_gate or w_content alone, and is not purely quadratic either
    (the sigmoid adds a further nonlinearity) -- it's a genuinely harder
    case for both STA and STC than either the linear unit above or a clean
    quadratic form would be, which is the point: it's a probe for whether
    attention-like gating specifically degrades recovery, not just
    "nonlinearity in general."

    w_gate and w_content are NOT required to be orthogonal or unit norm --
    pass whatever you want to test recoverability of; the "ground truth"
    for recovery-quality comparisons here is best treated as the 2D
    subspace span(w_gate, w_content), not either vector alone (see
    `subspace_overlap` below), since a linear estimator has no way to
    disentangle which of the two contributed to a given recovered direction.
    """
    rng = rng or np.random.default_rng(0)
    gate = 1.0 / (1.0 + np.exp(-(noise_batch @ w_gate)))
    content = noise_batch @ w_content
    signal = gate * content
    return strength * signal + rng.normal(0, obs_noise, size=noise_batch.shape[0])


def synthetic_bilinear_unit_response(
    noise_batch: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    strength: float = 2.0,
    obs_noise: float = 0.3,
    rng=None,
) -> np.ndarray:
    """
    Pure bilinear/quadratic-interaction unit: response = strength *
    (noise @ v1) * (noise @ v2) + noise -- no sigmoid, genuinely sign-
    symmetric (flipping the sign of either projection flips the response's
    sign too, and the *expected* response at fixed |projection| is
    unrelated to sign). This is the classic "energy detector" / complex-
    cell-style nonlinearity that textbook STA is known to fail on, and
    exists here as the clean contrast case to
    `synthetic_attention_unit_response`'s bounded, always-non-negative
    gate: empirically (see experiments/aim2_sta_vs_stc.py), the sigmoid-
    gated unit turns out NOT to break STA much, because a sigmoid gate
    never flips sign -- it's this unbounded, sign-symmetric bilinear form
    that actually demonstrates STA's failure mode and STC's value.
    """
    rng = rng or np.random.default_rng(0)
    g = noise_batch @ v1
    c = noise_batch @ v2
    return strength * g * c + rng.normal(0, obs_noise, size=noise_batch.shape[0])


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


def subspace_overlap(vector: np.ndarray, basis_vectors) -> float:
    """
    Fraction of `vector`'s squared norm explained by its projection onto
    span(basis_vectors) -- 1.0 means vector lies entirely in that subspace,
    0.0 means it's entirely orthogonal to it. Used to score STC eigenvectors
    against the *subspace* spanned by (w_gate, w_content) for the
    attention-like unit, since neither vector alone is "the" ground truth
    for a multiplicative interaction between them.
    """
    B = np.stack([np.asarray(b, dtype=float) for b in basis_vectors], axis=1)  # (dim, k)
    Q, _ = np.linalg.qr(B)  # orthonormal basis for span(basis_vectors)
    v = np.asarray(vector, dtype=float)
    proj = Q @ (Q.T @ v)
    denom = np.dot(v, v)
    return float(np.dot(proj, proj) / denom) if denom > 0 else np.nan


def spike_triggered_covariance(
    noise_batch: np.ndarray,
    responses: np.ndarray,
    n_permutations: int = 200,
    alpha: float = 0.05,
    ground_truth: Optional[np.ndarray] = None,
    subspace_basis=None,
    rng=None,
) -> STCResult:
    """
    Spike-triggered covariance, adapted for continuous (non-spike-count)
    responses: weight each trial's outer-product stimulus contribution by
    its (mean-centered) response, compare the resulting weighted covariance
    against the *unweighted* stimulus covariance, and eigen-decompose the
    difference. Directions with large |eigenvalue| are directions along
    which response-conditioned stimulus variance differs from baseline --
    this is what catches symmetric/quadratic tuning that STA's mean-based
    estimator misses entirely.

    Significance: a permutation test (shuffle `responses` relative to
    `noise_batch`, recompute the eigenvalue spectrum, repeat
    `n_permutations` times) gives a null distribution for the largest
    |eigenvalue|; real eigenvalues exceeding the (1-alpha) quantile of that
    null are marked significant in `significant_mask`. This is the standard
    max-statistic approach to STC significance (controls for the multiple-
    comparisons problem across all eigenvalues at once, rather than testing
    each in isolation).
    """
    rng = rng or np.random.default_rng(0)
    noise_batch = np.asarray(noise_batch, dtype=float)
    responses = np.asarray(responses, dtype=float)
    n, dim = noise_batch.shape

    def weighted_cov_diff(resp):
        w = resp - resp.mean()
        # Weighted second-moment of the stimulus, response-weighted, minus
        # the unweighted (prior) stimulus covariance. Normalizing by sum(w)
        # would blow up when weights nearly cancel (mean-centered weights
        # sum to ~0 by construction) -- normalize by n instead, matching
        # the STA convention above, and interpret magnitudes relatively
        # (via the permutation null), not as an absolute covariance scale.
        weighted_second_moment = (noise_batch * w[:, None]).T @ noise_batch / n
        prior_cov = np.cov(noise_batch, rowvar=False)
        return weighted_second_moment - prior_cov

    diff = weighted_cov_diff(responses)
    diff = (diff + diff.T) / 2  # enforce exact symmetry against float error
    eigvals, eigvecs = np.linalg.eigh(diff)
    order = np.argsort(-np.abs(eigvals))
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    null_max_abs_eig = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(responses)
        d = weighted_cov_diff(shuffled)
        d = (d + d.T) / 2
        ev = np.linalg.eigvalsh(d)
        null_max_abs_eig[i] = np.max(np.abs(ev))

    threshold = np.quantile(null_max_abs_eig, 1 - alpha)
    significant_mask = np.abs(eigvals) > threshold

    top_cos, top_overlap = None, None
    if eigvals.size > 0:
        top_vec = eigvecs[:, 0]
        if ground_truth is not None:
            denom = np.linalg.norm(top_vec) * np.linalg.norm(ground_truth)
            top_cos = float(np.dot(top_vec, ground_truth) / denom) if denom > 0 else np.nan
        if subspace_basis is not None:
            top_overlap = subspace_overlap(top_vec, subspace_basis)

    return STCResult(
        eigenvalues=eigvals,
        eigenvectors=eigvecs,
        significant_mask=significant_mask,
        n_permutations=n_permutations,
        top_eigenvector_cosine_to_truth=top_cos,
        top_eigenvector_subspace_overlap=top_overlap,
    )
