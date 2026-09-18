"""Smoke tests for the mock SBI model — checks shapes and the decision-routing
logic, not the (mock) model's actual accuracy."""

import torch

from visual_inspection_fm.config import DEFECT_SEVERITY_THRESHOLD, UNCERTAINTY_THRESHOLD
from visual_inspection_fm.features import mock_camera_image
from visual_inspection_fm.modeling.predict import predict_one, route_decision
from visual_inspection_fm.modeling.sbi_model import VisualInspectionSBI


def test_forward_pass_shapes():
    model = VisualInspectionSBI()
    model.eval()
    image = mock_camera_image(batch_size=1)

    severity, uncertainty = model(image)

    assert severity.shape == (1, 1)
    assert uncertainty.shape == (1, 1)
    assert torch.all(uncertainty >= 0)  # variance can't be negative


def test_predict_one_returns_expected_keys():
    model = VisualInspectionSBI()
    image = mock_camera_image(batch_size=1)

    result = predict_one(model, image)

    assert set(result) == {"defect_severity", "model_uncertainty", "decision"}
    assert result["decision"] in {"escalate", "reject", "pass"}


def test_route_decision_boundaries():
    # High uncertainty always escalates, regardless of severity.
    assert route_decision(defect_severity=0.0, model_uncertainty=UNCERTAINTY_THRESHOLD + 0.01) == "escalate"
    # Low uncertainty, high severity -> reject.
    assert route_decision(defect_severity=DEFECT_SEVERITY_THRESHOLD + 0.01, model_uncertainty=0.0) == "reject"
    # Low uncertainty, low severity -> pass.
    assert route_decision(defect_severity=0.0, model_uncertainty=0.0) == "pass"
