"""Inference + human-escalation decision routing.

Extracted from the decision-logic block at the bottom of the original
sbi_prototype.py's __main__ — this is "the business value" the concept
deck talks about: routing low-confidence units to a human inspector instead
of guessing, which is also what upcoming EU AI Act human-in-loop rules
require evidence of.
"""

import torch

from visual_inspection_fm.config import DEFECT_SEVERITY_THRESHOLD, UNCERTAINTY_THRESHOLD
from visual_inspection_fm.modeling.sbi_model import VisualInspectionSBI


def route_decision(defect_severity: float, model_uncertainty: float) -> str:
    """Return one of "escalate", "reject", "pass" given a single unit's
    predicted severity and uncertainty."""
    if model_uncertainty > UNCERTAINTY_THRESHOLD:
        return "escalate"
    if defect_severity > DEFECT_SEVERITY_THRESHOLD:
        return "reject"
    return "pass"


def predict_one(model: VisualInspectionSBI, image: torch.Tensor) -> dict:
    """Run inference on a single image tensor [1, 3, H, W] and return a dict
    with severity, uncertainty, and the routing decision."""
    model.eval()
    defect_severity, model_uncertainty = model(image)
    decision = route_decision(defect_severity.item(), model_uncertainty.item())
    return {
        "defect_severity": defect_severity.item(),
        "model_uncertainty": model_uncertainty.item(),
        "decision": decision,
    }
