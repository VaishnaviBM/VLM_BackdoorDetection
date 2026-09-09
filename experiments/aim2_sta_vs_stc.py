"""
Aim 2 extension: does STA reliably recover a trigger, and does a
multiplicative/attention-like nonlinearity change that -- with STC run
alongside STA rather than STA alone.

Three synthetic units, same underlying "signal dimensions", increasing
distance from STA's comfort zone:
  1. LINEAR       -- textbook case; STA should recover it almost exactly.
  2. ATTENTION    -- sigmoid(gate) * content: bounded, always non-negative
                     gating, meant to loosely resemble a softmax attention
                     weight multiplicatively scaling a value vector.
  3. BILINEAR     -- (proj1) * (proj2), unconstrained sign: the classic
                     "energy detector" nonlinearity STA is known to miss.

Ground truth for (2) and (3) is treated as the 2D subspace span(v1, v2),
not either vector alone -- a linear estimator has no way to attribute a
recovered direction to one factor of a product, so `subspace_overlap` is
the fair metric there, not raw cosine similarity to one vector.

Run: python experiments/aim2_sta_vs_stc.py
"""
import numpy as np

from vlm_trojan_cue.reverse_correlation import (
    synthetic_flagged_unit_response,
    synthetic_attention_unit_response,
    synthetic_bilinear_unit_response,
    reverse_correlate,
    spike_triggered_covariance,
    subspace_overlap,
)


def run_case(name, noise, responses, rng, ground_truth_vec=None, subspace_basis=None):
    sta = reverse_correlate(noise, responses, ground_truth=ground_truth_vec)
    sta_overlap = subspace_overlap(sta.classification_image, subspace_basis) if subspace_basis else None

    stc = spike_triggered_covariance(
        noise, responses, n_permutations=200, ground_truth=ground_truth_vec,
        subspace_basis=subspace_basis, rng=rng,
    )
    n_sig = int(stc.significant_mask.sum())

    print(f"--- {name} ---")
    if ground_truth_vec is not None:
        print(f"  STA cosine similarity to ground truth:      {sta.cosine_similarity_to_truth:.3f}")
    if subspace_basis is not None:
        print(f"  STA classification-image subspace overlap:  {sta_overlap:.3f}")
    print(f"  STA classification-image norm (near 0 = found nothing): {np.linalg.norm(sta.classification_image):.3f}")
    print(f"  STC: {n_sig} significant eigenvalue(s) out of {len(stc.eigenvalues)} (permutation test, alpha=0.05)")
    if ground_truth_vec is not None and stc.top_eigenvector_cosine_to_truth is not None:
        print(f"  STC top eigenvector cosine to ground truth:  {stc.top_eigenvector_cosine_to_truth:.3f}")
    if subspace_basis is not None and stc.top_eigenvector_subspace_overlap is not None:
        print(f"  STC top eigenvector subspace overlap:        {stc.top_eigenvector_subspace_overlap:.3f}")
    print()


def main(dim: int = 24, n_trials: int = 4000, seed: int = 42):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, size=(n_trials, dim))

    trigger = np.zeros(dim); trigger[[6, 11, 14, 17]] = 1.0; trigger /= np.linalg.norm(trigger)
    w_gate = np.zeros(dim); w_gate[[3, 9]] = 1.0; w_gate /= np.linalg.norm(w_gate)
    w_content = trigger  # reuse the same "content" pattern across cases 2 and 3 for comparability

    print("=== Aim 2: STA vs. STC across increasingly non-linear synthetic units ===\n")

    resp_linear = synthetic_flagged_unit_response(noise, trigger, strength=2.0, obs_noise=0.3, rng=rng)
    run_case("1. LINEAR unit", noise, resp_linear, rng, ground_truth_vec=trigger)

    resp_attention = synthetic_attention_unit_response(
        noise, w_gate, w_content, strength=3.0, obs_noise=0.3, rng=rng
    )
    run_case(
        "2. ATTENTION-like unit (sigmoid-gated, bounded/non-negative)",
        noise, resp_attention, rng,
        ground_truth_vec=w_content, subspace_basis=[w_gate, w_content],
    )

    resp_bilinear = synthetic_bilinear_unit_response(
        noise, w_gate, w_content, strength=2.0, obs_noise=0.3, rng=rng
    )
    run_case(
        "3. BILINEAR unit (unconstrained sign -- classic STA failure case)",
        noise, resp_bilinear, rng,
        ground_truth_vec=w_content, subspace_basis=[w_gate, w_content],
    )

    print(
        "Takeaway: a bounded, non-negative gate (case 2 -- the closer analog to a\n"
        "softmax attention weight) does NOT defeat STA much, because it never flips\n"
        "sign. An unconstrained bilinear interaction (case 3) does defeat STA (near-\n"
        "chance cosine similarity / near-zero classification-image norm), and is\n"
        "exactly the case STC is needed for. This narrows the original hypothesis:\n"
        "the risk to reverse correlation isn't 'attention/transformers in general',\n"
        "it's specifically whether the mechanism you're trying to recover behaves like\n"
        "an unconstrained multiplicative interaction rather than a bounded gate -- run\n"
        "both STA and STC in the real pipeline rather than assuming either regime."
    )


if __name__ == "__main__":
    main()
