"""Hackathon demo runner.

Same behavior as the original scripts/sbi_prototype.py's __main__ block, now
importing from the installable `visual_inspection_fm` package instead of
being a single self-contained file. Run with:

    python scripts/run_demo.py
"""

from visual_inspection_fm.features import mock_camera_image
from visual_inspection_fm.modeling.predict import predict_one
from visual_inspection_fm.modeling.sbi_model import VisualInspectionSBI

DECISION_MESSAGES = {
    "escalate": (
        "ALERT: High model uncertainty detected.",
        "ACTION: Halting automated line. Routing image to human QA inspector for manual review.",
    ),
    "reject": (
        "CONFIDENT REJECT: Defect detected with high certainty.",
        "ACTION: Automated API call to MES to scrap part.",
    ),
    "pass": (
        "CONFIDENT PASS: Part is perfect with high certainty.",
        "ACTION: Automated API call to pass part.",
    ),
}


def main():
    print("--- Starting Visual-Inspection FM Simulation ---")

    model = VisualInspectionSBI()

    print("\nCapturing image from factory loading dock...")
    image = mock_camera_image(batch_size=1)

    result = predict_one(model, image)

    print("\n--- Inference Results ---")
    print(f"Predicted Defect Severity Score: {result['defect_severity']:.4f}")
    print(f"Model Epistemic Uncertainty: {result['model_uncertainty']:.4f}")

    print("\n--- Factory Execution Routing ---")
    headline, action = DECISION_MESSAGES[result["decision"]]
    print(headline)
    print(action)

    print("\nSimulation Complete. Ready for Hackathon presentation!")


if __name__ == "__main__":
    main()
