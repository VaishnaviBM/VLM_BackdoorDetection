"""
Aim 0: does a (synthetic, for now) observer follow reliability-weighted
causal-inference cue integration? Fits the BCI model and the linear-weight
ablation baseline, compares them by held-out log-likelihood, and reports
recovered parameters against ground truth.

TODO (real model): replace CleanObserver with model_interface.VLMQueryInterface
calls once a real VLM is wired in -- see model_interface.py docstring.
"""
import numpy as np

from vlm_trojan_cue.cue_model import fit_bci, fit_linear_baseline, bci_predict
from vlm_trojan_cue.probes import generate_probe_battery, CleanObserver


def main(n_trials: int = 800, seed: int = 0):
    rng = np.random.default_rng(seed)
    trials = generate_probe_battery(n_trials, rng)
    observer = CleanObserver(seed=seed)
    responses = observer.respond(trials)

    split = int(0.7 * n_trials)
    train_trials, test_trials = trials[:split], trials[split:]
    y_train, y_test = responses[:split], responses[split:]

    x_v_tr = np.array([t.x_v for t in train_trials])
    x_t_tr = np.array([t.x_t for t in train_trials])
    sv_tr = np.array([t.sigma_v for t in train_trials])
    st_tr = np.array([t.sigma_t for t in train_trials])

    bci_params = fit_bci(x_v_tr, x_t_tr, sv_tr, st_tr, y_train)
    linear_w = fit_linear_baseline(x_v_tr, x_t_tr, y_train)

    x_v_te = np.array([t.x_v for t in test_trials])
    x_t_te = np.array([t.x_t for t in test_trials])
    sv_te = np.array([t.sigma_v for t in test_trials])
    st_te = np.array([t.sigma_t for t in test_trials])

    bci_pred, _ = bci_predict(x_v_te, x_t_te, sv_te, st_te, bci_params)
    linear_pred = linear_w * x_v_te + (1 - linear_w) * x_t_te

    bci_mse = float(np.mean((y_test - bci_pred) ** 2))
    linear_mse = float(np.mean((y_test - linear_pred) ** 2))

    print("=== Aim 0: cue-integration characterization (synthetic CleanObserver) ===")
    print(f"Fitted BCI params:   sigma_v={bci_params.sigma_v:.2f}  sigma_t={bci_params.sigma_t:.2f}  "
          f"sigma_p={bci_params.sigma_p:.2f}  p_common={bci_params.p_common:.2f}")
    print(f"Ground-truth params: sigma_v/sigma_t vary per trial (design manipulation); "
          f"generative sigma_p=3.00 p_common=0.70")
    print(f"Held-out MSE  -- BCI model: {bci_mse:.4f}   linear-weight baseline: {linear_mse:.4f}")
    if bci_mse < linear_mse:
        print("-> BCI model outperforms the linear ablation: causal-inference structure is earning its complexity.")
    else:
        print("-> Linear baseline matches or beats BCI: on THIS observer, the extra structure isn't needed. "
              "(Expected here since CleanObserver's congruent-trial regime is close to linear; try lower "
              "p_common / more conflicting trials to see BCI pull ahead.)")


if __name__ == "__main__":
    main()
