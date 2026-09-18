# Visual-Inspection FM

Calibrated, simulation-trained visual-inspection foundation model — HERE GOES (Bosch × Creative Dock) Selection Hackathon, Berlin, 17–18 Sep 2026.

**Concept:** Theme C1 — Quality Intelligence, concept 01/16. See `docs/notes.md` for the full problem statement, solution framing, and progress log.

**One-liner:** instead of a bare defect/no-defect score, output a statistically calibrated confidence per unit, adapt to new defects in seconds via amortized inference (no per-SKU fine-tuning loop), and ship an audit-ready calibration report for EU AI Act human-in-loop compliance.

## Project layout

This project follows a blend of three references (see `docs/notes.md` §7 for links): the [cookiecutter-data-science](https://cookiecutter-data-science.drivendata.org/) convention for the data/docs/models/reports skeleton, [Eric Ma's project-structure gist](https://gist.github.com/ericmjl/27e50331f24db3e8f957d1fe7bbbe510) for the installable-package-plus-scripts pattern, and the [Towards Data Science best-practices article](https://towardsdatascience.com/python-project-structure-best-practices-d9d0b174ad5d/) for tooling (pyproject.toml, .env, pre-commit).

```
visual-inspection-fm/
├── data/                   # raw / interim / processed / external — never edit raw/ in place
├── docs/                   # hackathon reference package + notes.md (living notes, keep this updated)
├── models/                 # trained model checkpoints (gitignored — too large to commit)
├── notebooks/              # exploratory notebooks, numbered `01-...`, `02-...`
├── references/             # any papers, manuals, data dictionaries
├── reports/figures/        # generated plots for the pitch deck (esp. the calibration plot)
├── scripts/                # standalone runnable entry points, e.g. run_demo.py
├── src/visual_inspection_fm/   # the installable package — import as `from visual_inspection_fm import ...`
│   ├── config.py            # paths & constants (PROJ_ROOT, DATA_DIR, UNCERTAINTY_THRESHOLD, ...)
│   ├── simulate.py           # Component 1 — synthetic defect simulator (cold-start problem)
│   ├── features.py           # frozen backbone embedding extractor
│   ├── calibration.py        # Component 3 — SBC / calibration-plot diagnostics (the compliance product)
│   ├── plots.py               # shared plotting helpers
│   └── modeling/
│       ├── sbi_model.py       # Component 2 — amortized few-shot classifier (was scripts/sbi_prototype.py)
│       ├── train.py            # training loop stub
│       └── predict.py          # inference + human-escalation decision routing
├── tests/
├── pyproject.toml
├── requirements.txt
├── Makefile
└── .gitignore
```

## Folder conventions

**`data/`** — following the cookiecutter-data-science convention: never edit files in `raw/` in place; every transformation should be reproducible from raw data by a script in `src/visual_inspection_fm/` or `scripts/`.
- `raw/` — immutable original data (e.g. exported synthetic-defect image batches). Read-only once dropped here.
- `external/` — third-party data (e.g. any public defect-image datasets used for comparison).
- `interim/` — intermediate transformed data (e.g. extracted backbone embeddings) that isn't the final modeling input.
- `processed/` — final, canonical datasets ready to feed into `modeling/train.py`.
- This folder is gitignored (content, not structure) — data doesn't belong in git. Keep the `.gitkeep` files so the empty tree still shows up for a new clone.

**`notebooks/`** — naming convention (per Eric Ma's project-structure gist): `<number>-<short-description>.ipynb`, e.g. `01-defect-simulator-exploration.ipynb`, `02-embedding-sanity-check.ipynb`. Numbers show narrative order, not necessarily creation order. Anything exploratory that stops being useful goes in `archive/` rather than being deleted — keep the audit trail. Once a notebook's logic is stable, refactor it into `src/visual_inspection_fm/` so it's reusable and testable, and leave the notebook as the demo/exploration record.

**`references/`** — drop any papers, manuals, or data dictionaries here that inform the model but aren't part of the hackathon founder package itself (that lives in `docs/`) — e.g. the GalSim documentation, SBC/ECDF-diagnostics papers, or the `sbi` library docs if you move from the mock head to the real package.

## Getting started

```bash
# from the project root
python -m venv .venv && source .venv/bin/activate   # or use conda/mamba
pip install -e ".[dev]"                              # editable install of the package + dev tools

# run the current prototype demo (frozen ResNet18 backbone -> mock severity + uncertainty head)
python scripts/run_demo.py

# run tests
pytest
```

## Status

Carried over from `sbi_prototype.py` (already had a working mock demo of the frozen-backbone + uncertainty-head idea). Open items are tracked in `docs/notes.md` §4 (Status / Progress Log) — check there before starting new work so effort isn't duplicated.
