"""
Aim 2 (proof-of-concept, secondary): recover a sparse trigger pattern in a
reduced noise basis via reverse correlation, gradient-free.

TODO (real use): replace synthetic_flagged_unit_response with actual query
responses from a flagged fusion unit/attention head, and noise_batch with
structured perturbations (e.g. superpixel or low-frequency DCT blocks)
applied to real images.
"""
import numpy as np

from vlm_trojan_cue.reverse_correlation import synthetic_flagged_unit_response, reverse_correlate


def main(basis_dim: int = 24, n_trials: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)

    trigger_pattern = np.zeros(basis_dim)
    active_idx = rng.choice(basis_dim, size=4, replace=False)
    trigger_pattern[active_idx] = rng.uniform(0.5, 1.0, size=4)
    trigger_pattern /= np.linalg.norm(trigger_pattern)

    noise_batch = rng.normal(0, 1, size=(n_trials, basis_dim))
    responses = synthetic_flagged_unit_response(noise_batch, trigger_pattern, rng=rng)

    result = reverse_correlate(noise_batch, responses, ground_truth=trigger_pattern)

    print("=== Aim 2: reverse-correlation trigger reconstruction (synthetic) ===")
    print(f"Basis dim: {basis_dim}, trials: {n_trials}, true active dims: {sorted(active_idx.tolist())}")
    recovered_top = np.argsort(-np.abs(result.classification_image))[:4]
    print(f"Recovered top-4 dims by |weight|: {sorted(recovered_top.tolist())}")
    print(f"Cosine similarity to ground truth: {result.cosine_similarity_to_truth:.3f}")


if __name__ == "__main__":
    main()
