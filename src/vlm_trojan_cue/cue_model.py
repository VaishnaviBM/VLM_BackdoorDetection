"""
Bayesian Causal Inference (BCI) model of cross-modal cue integration.

This is the null model for Aim 0 / Aim 1: it predicts how a *reliability-
weighted* observer should combine a visual cue and a text cue as a function
of each cue's reliability and their mutual congruence, following the
causal-inference formulation of multisensory integration (Kording et al.,
2007; Beierholm et al., 2007). A vision-language model's actual behavior is
compared against this prediction; systematic, reliability-independent
deviation is the detection signature we're after.

Everything here operates on a single scalar "decision-relevant value" per
modality per trial (x_v, x_t) -- e.g. a projection of the image and text
onto whatever axis the downstream task cares about. Extracting that scalar
from a real VLM's outputs is task-specific and lives in model_interface.py;
this module is deliberately agnostic to where x_v/x_t came from.
"""
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from scipy.optimize import minimize


@dataclass
class BCIParams:
    sigma_v: float   # visual cue noise (std) -- lower = more reliable
    sigma_t: float   # text cue noise (std)
    sigma_p: float   # prior width over the underlying latent variable
    p_common: float  # prior probability the two cues share a common cause

    def as_array(self) -> np.ndarray:
        return np.array([self.sigma_v, self.sigma_t, self.sigma_p, self.p_common])

    @staticmethod
    def from_array(a: np.ndarray) -> "BCIParams":
        return BCIParams(sigma_v=a[0], sigma_t=a[1], sigma_p=a[2], p_common=a[3])


_BOUNDS = [(1e-3, 20.0), (1e-3, 20.0), (1e-3, 50.0), (1e-3, 1 - 1e-3)]


def reliability_weight(sigma_v: float, sigma_t: float) -> float:
    """Optimal Bayes weight on the visual cue under forced (common-cause) integration."""
    prec_v, prec_t = 1.0 / sigma_v**2, 1.0 / sigma_t**2
    return prec_v / (prec_v + prec_t)


def bci_predict(
    x_v: np.ndarray, x_t: np.ndarray, sigma_v: np.ndarray, sigma_t: np.ndarray, params: BCIParams
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict the combined response for each trial under the BCI model.

    sigma_v / sigma_t are *per-trial* reliabilities (arrays), since the probe
    battery deliberately varies these across trials -- params.sigma_v/sigma_t
    are only used as fallback global values when per-trial values aren't given.

    Returns (predicted_response, posterior_prob_common_cause) per trial.
    """
    sv = np.asarray(sigma_v, dtype=float)
    st = np.asarray(sigma_t, dtype=float)
    x_v = np.asarray(x_v, dtype=float)
    x_t = np.asarray(x_t, dtype=float)

    w_v = (1.0 / sv**2) / (1.0 / sv**2 + 1.0 / st**2)
    integrated = w_v * x_v + (1 - w_v) * x_t
    segregated = np.where(sv < st, x_v, x_t)

    d2 = (x_v - x_t) ** 2
    var_common = sv**2 + st**2
    var_indep = sv**2 + st**2 + params.sigma_p**2

    like_common = np.exp(-0.5 * d2 / var_common) / np.sqrt(2 * np.pi * var_common)
    like_indep = np.exp(-0.5 * d2 / var_indep) / np.sqrt(2 * np.pi * var_indep)

    prior_odds = params.p_common / (1 - params.p_common)
    numer = like_common * prior_odds
    post_common = numer / (numer + like_indep + 1e-12)

    predicted = post_common * integrated + (1 - post_common) * segregated
    return predicted, post_common


def _neg_log_likelihood(theta: np.ndarray, x_v, x_t, sigma_v, sigma_t, y, obs_noise: float) -> float:
    params = BCIParams.from_array(theta)
    pred, _ = bci_predict(x_v, x_t, sigma_v, sigma_t, params)
    resid = y - pred
    nll = 0.5 * np.sum((resid / obs_noise) ** 2)
    return nll


def fit_bci(
    x_v: Sequence[float],
    x_t: Sequence[float],
    sigma_v: Sequence[float],
    sigma_t: Sequence[float],
    y: Sequence[float],
    obs_noise: float = 0.2,
    n_restarts: int = 6,
    seed: int = 0,
) -> BCIParams:
    """Fit BCI params to observed responses y via maximum likelihood (multi-start)."""
    rng = np.random.default_rng(seed)
    x_v, x_t, sigma_v, sigma_t, y = map(np.asarray, (x_v, x_t, sigma_v, sigma_t, y))

    best, best_val = None, np.inf
    for _ in range(n_restarts):
        theta0 = np.array(
            [
                rng.uniform(0.3, 3.0),
                rng.uniform(0.3, 3.0),
                rng.uniform(1.0, 10.0),
                rng.uniform(0.2, 0.9),
            ]
        )
        res = minimize(
            _neg_log_likelihood,
            theta0,
            args=(x_v, x_t, sigma_v, sigma_t, y, obs_noise),
            bounds=_BOUNDS,
            method="L-BFGS-B",
        )
        if res.fun < best_val:
            best_val, best = res.fun, res.x
    return BCIParams.from_array(best)


def fit_linear_baseline(
    x_v: Sequence[float], x_t: Sequence[float], y: Sequence[float]
) -> float:
    """
    Simplest possible competitor: a single global weight w such that
    y ~= w * x_v + (1 - w) * x_t, fit by least squares. This is the ablation
    that the BCI model has to beat -- if it doesn't, the causal-inference
    machinery isn't earning its complexity (see docs/research_plan.md, Aim 0).
    """
    x_v, x_t, y = map(np.asarray, (x_v, x_t, y))
    diff = x_v - x_t
    denom = np.sum(diff**2)
    if denom < 1e-12:
        return 0.5
    w = np.sum((y - x_t) * diff) / denom
    return float(np.clip(w, 0.0, 1.0))
