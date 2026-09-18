"""Visual-Inspection FM — calibrated, simulation-trained visual-inspection foundation model.

See docs/notes.md for the problem statement and the four-component solution
this package is structured around:
  1. simulate.py            — synthetic defect simulator (cold-start problem)
  2. modeling/sbi_model.py  — amortized few-shot inference (frozen backbone + SBI head)
  3. calibration.py         — calibration diagnostics (the compliance product)
  4. (planned) hierarchical partial-pooling across plants/SKUs
"""

__version__ = "0.1.0"
