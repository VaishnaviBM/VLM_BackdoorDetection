"""
Aim 1: detect backdoors as violations of fitted cue-integration statistics.

calibrate_null() fits the BCI model on a reference battery (in real use:
probe trials from a trusted/clean model, or a benign region of the model
under test -- see docs/research_plan.md, Aim 1 threat model). score_trials()
then measures how far each new trial's response falls from what the null
model predicts, at that trial's own reliability level -- large, systematic
deviation is the anomaly signature, not any single trial in isolation.
"""
from dataclasses import dataclass
from typing import List

import numpy as np

from .cue_model import BCIParams, bci_predict, fit_bci, fit_linear_baseline
from .probes import ProbeTrial


@dataclass
class NullModel:
    bci_params: BCIParams
    linear_weight: float  # ablation baseline, see cue_model.fit_linear_baseline


def calibrate_null(trials: List[ProbeTrial], responses: np.ndarray, obs_noise: float = 0.15) -> NullModel:
    x_v = np.array([t.x_v for t in trials])
    x_t = np.array([t.x_t for t in trials])
    sigma_v = np.array([t.sigma_v for t in trials])
    sigma_t = np.array([t.sigma_t for t in trials])

    bci_params = fit_bci(x_v, x_t, sigma_v, sigma_t, responses, obs_noise=obs_noise)
    linear_w = fit_linear_baseline(x_v, x_t, responses)
    return NullModel(bci_params=bci_params, linear_weight=linear_w)


def trial_deviation(trials: List[ProbeTrial], responses: np.ndarray, null: NullModel) -> np.ndarray:
    """Per-trial |observed - predicted| under the fitted BCI null model."""
    x_v = np.array([t.x_v for t in trials])
    x_t = np.array([t.x_t for t in trials])
    sigma_v = np.array([t.sigma_v for t in trials])
    sigma_t = np.array([t.sigma_t for t in trials])
    pred, _ = bci_predict(x_v, x_t, sigma_v, sigma_t, null.bci_params)
    return np.abs(responses - pred)


def batch_anomaly_score(trials: List[ProbeTrial], responses: np.ndarray, null: NullModel) -> float:
    """
    Aggregate deviation for a batch of trials into a single anomaly score.
    Real deployment would score a held-out battery from "the model under
    test"; this is the model-level statistic compared against a calibrated
    null distribution from known-clean models (see eval.roc_auc_with_ci).
    """
    return float(np.mean(trial_deviation(trials, responses, null)))
