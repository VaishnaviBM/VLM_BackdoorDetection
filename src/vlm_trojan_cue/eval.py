"""Evaluation: AUROC with bootstrap confidence intervals for Aim 1."""
import numpy as np
from sklearn.metrics import roc_auc_score


def roc_auc_with_ci(y_true, scores, n_boot: int = 1000, seed: int = 0, ci: float = 0.95):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    auc = roc_auc_score(y_true, scores)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b, s_b = y_true[idx], scores[idx]
        if len(np.unique(y_b)) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_b, s_b))

    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_aucs, [alpha, 1 - alpha])
    return {"auc": float(auc), "ci_low": float(lo), "ci_high": float(hi), "n_boot": len(boot_aucs)}
