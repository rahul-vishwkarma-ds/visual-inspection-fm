"""Component 2 — amortized few-shot inference model.

Relocated from the original scripts/sbi_prototype.py into the installable
package so it can be imported from tests, notebooks, and scripts/run_demo.py
instead of copy-pasted. Logic is unchanged from the original mock prototype;
only the backbone-loading was factored out to features.py so simulate.py can
reuse it later.

The `sbi_head_mu` / `sbi_head_logvar` pair is a MOCK stand-in for a real
Neural Density Estimator (e.g. from the `sbi` package) that would learn the
full posterior p(theta | x_embedding) — see docs/notes.md §2.2 for why
amortized inference (train once, single forward pass per new case) beats the
concept deck's per-SKU fine-tuning-head plan for "hours not weeks" adaptation.
"""

import torch
import torch.nn as nn

from visual_inspection_fm.features import EMBEDDING_DIM, load_frozen_backbone


class VisualInspectionSBI(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Frozen vision backbone (simulating the Bosch foundation model)
        self.backbone = load_frozen_backbone()

        # 2. SBI head (mock representation).
        # In a real implementation using the `sbi` library, this would be a
        # Neural Density Estimator learning the posterior p(theta | x_embedding).
        # Simulated here with a simple probabilistic projection layer.
        self.sbi_head_mu = nn.Linear(EMBEDDING_DIM, 1)  # defect severity mean
        self.sbi_head_logvar = nn.Linear(EMBEDDING_DIM, 1)  # epistemic uncertainty (log-variance)

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            embeddings = self.backbone(x)
            embeddings = embeddings.view(embeddings.size(0), -1)

        mu = self.sbi_head_mu(embeddings)
        logvar = self.sbi_head_logvar(embeddings)
        uncertainty = torch.exp(logvar)

        return mu, uncertainty
