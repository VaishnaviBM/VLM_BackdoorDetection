"""vlm_trojan_cue: cue-integration-based backdoor detection for VLMs.

See docs/research_plan.md for the full research plan (Aims 0-3).
Everything in this package currently runs against synthetic observers with
known ground truth (probes.CleanObserver / BackdooredObserver) so the
statistical machinery is validated before it's ever pointed at a real
model. See model_interface.py for the real-VLM extension point.
"""

__version__ = "0.0.1"
