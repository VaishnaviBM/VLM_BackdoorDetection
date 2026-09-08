"""
Graded probe battery + synthetic observers.

Real usage (see model_interface.py) points this battery at an actual VLM.
Until that's wired up, CleanObserver / BackdooredObserver simulate a
reliability-weighted (resp. trigger-corrupted) observer so the statistical
machinery in cue_model.py / detection.py / reverse_correlation.py can be
validated against known ground truth before it's ever pointed at a real
model -- this mirrors the "Aim 0 pilot" step in docs/research_plan.md.
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np

from .cue_model import BCIParams, bci_predict


@dataclass
class ProbeTrial:
    x_v: float        # visual cue value (e.g. a projected "evidence" scalar)
    x_t: float         # text cue value
    sigma_v: float      # visual cue reliability (std) for this trial
    sigma_t: float       # text cue reliability (std) for this trial
    trigger: bool = False  # ground truth only -- used for eval, not given to the fitter


def generate_probe_battery(
    n_trials: int,
    rng: np.random.Generator,
    reliability_levels=(0.5, 1.0, 2.0, 4.0),
    sigma_t_fixed: float = 1.0,
    z_scale: float = 3.0,
) -> List[ProbeTrial]:
    """
    Generate a graded battery varying visual reliability (blur/noise/occlusion
    stand-in) and image-text congruence (x_v vs x_t agreement), holding text
    reliability fixed -- the design mirrors 2AFC psychophysics batteries used
    to characterize human cue integration.
    """
    trials = []
    for _ in range(n_trials):
        z_true = rng.normal(0, z_scale)
        # congruent on ~60% of trials, conflicting on the rest
        if rng.random() < 0.6:
            z_v_true, z_t_true = z_true, z_true
        else:
            z_v_true = z_true
            z_t_true = z_true + rng.normal(0, z_scale)

        sigma_v = float(rng.choice(reliability_levels))
        sigma_t = sigma_t_fixed

        x_v = rng.normal(z_v_true, sigma_v)
        x_t = rng.normal(z_t_true, sigma_t)

        trials.append(ProbeTrial(x_v=x_v, x_t=x_t, sigma_v=sigma_v, sigma_t=sigma_t))
    return trials


class CleanObserver:
    """A reliability-weighted BCI observer -- the null hypothesis for Aim 0."""

    def __init__(self, sigma_p: float = 3.0, p_common: float = 0.7, obs_noise: float = 0.15, seed: int = 0):
        self.params = BCIParams(sigma_v=1.0, sigma_t=1.0, sigma_p=sigma_p, p_common=p_common)
        self.obs_noise = obs_noise
        self.rng = np.random.default_rng(seed)

    def respond(self, trials: List[ProbeTrial]) -> np.ndarray:
        x_v = np.array([t.x_v for t in trials])
        x_t = np.array([t.x_t for t in trials])
        sigma_v = np.array([t.sigma_v for t in trials])
        sigma_t = np.array([t.sigma_t for t in trials])
        pred, _ = bci_predict(x_v, x_t, sigma_v, sigma_t, self.params)
        return pred + self.rng.normal(0, self.obs_noise, size=len(trials))


class BackdooredObserver(CleanObserver):
    """
    On a random subset of trials (the "trigger" set), the response is pulled
    toward the visual cue regardless of its reliability -- simulating a
    backdoor that gives the vision stream categorical control independent of
    normal reliability-weighting. trigger_strength in [0, 1]: 0 recovers
    CleanObserver exactly; 1 means full override toward x_v on trigger trials.
    """

    def __init__(self, trigger_frac: float = 0.25, trigger_strength: float = 0.85, seed: int = 1, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.trigger_frac = trigger_frac
        self.trigger_strength = trigger_strength

    def respond(self, trials: List[ProbeTrial]) -> np.ndarray:
        base = super().respond(trials)
        is_trigger = self.rng.random(len(trials)) < self.trigger_frac
        for t, flag in zip(trials, is_trigger):
            t.trigger = bool(flag)
        x_v = np.array([t.x_v for t in trials])
        overridden = self.trigger_strength * x_v + (1 - self.trigger_strength) * base
        return np.where(is_trigger, overridden, base)
