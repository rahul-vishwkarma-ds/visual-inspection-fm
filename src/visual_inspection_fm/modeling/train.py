"""Training loop for the amortized few-shot classifier — not yet implemented.

TODO (see docs/notes.md §2.2 and the open checklist in §4): the concept
deck's plan is "freeze a backbone, fine-tune a small head per new defect"
(still a training step per SKU). The differentiated version is amortized
inference: train once across many simulated scenarios (from simulate.py) so
a new case is handled by a single forward pass at inference time, no
retraining loop. This module is where that one-time training run lives once
simulate.py exists to feed it.
"""
