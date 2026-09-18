# Visual-Inspection FM — Hackathon Project Notes

**Program:** HERE GOES (Bosch × Creative Dock) — Selection Hackathon, Berlin, 17–18 Sep 2026
**Role:** Tech Lead
**Concept assigned:** Visual-Inspection FM (Theme C1 — Quality Intelligence, concept 01 of 16)
**Last updated:** 2026-09-17

---

## 1. Problem Definition

**Core problem.** Every manufacturer rebuilds visual inspection from scratch for every new production line or SKU change: thousands of labelled defect images, weeks of data collection, months of tuning. Generic vision vendors lack industrial defect data, so accuracy on rare defects — the ones that actually matter (safety, warranty, compliance) — stays poor. When a new product variant lands, tuning starts over from zero.

**Quantified pain (from the concept package):**
- 4–8 weeks of engineering per new SKU with rule-based vision — for a plant with ~100 changeovers/year, that's 400+ engineering-weeks annually.
- €2–4m/year in vision-setup cost alone for a multi-SKU manufacturer, before counting missed defects.
- Vision foundation models already beat rule-based systems by +30% accuracy and adapt to new SKUs zero-shot — the technology has crossed the threshold to be buildable now.

**Why now (regulatory forcing function).** Three regulations land within ~18 months of each other: EU Product Liability Directive (Dec 2026), EU Machinery Regulation (Jan 2027), EUDAMED (May 2026) — turning per-unit visual evidence into a matter of personal liability, not optional spend. The EU AI Act classifies safety-critical inspection AI as high-risk from August 2027, adding QMS logging, human-in-the-loop, and provenance obligations. Cost of Quality already runs 5–20% of revenue for manufacturers.

**The sharper technical crux (my framing as tech lead).** The highest-cost failures — the ones that trigger warranty claims, recalls, or compliance violations — are exactly the defects with the fewest training examples (rare, novel, high-consequence). Existing systems report an accuracy score, not a calibrated confidence; they can't tell a quality engineer *when to trust the model versus escalate to a human*, which is precisely what upcoming EU AI Act human-in-loop obligations require. Rare-defect detection is fundamentally a low-data, uncertainty-quantification problem — not just a bigger-model problem.

**Whitespace.** Vision incumbents (Cognex, Keyence) own plant-floor trust but lack industrial-scale defect data; AI-native players (Landing AI, Meta SAM 2) own the technology but lack the data. Nobody combines both, and nobody in the category sells calibrated, audit-ready confidence — only raw detection scores.

---

## 2. Potential Solution

**One-liner:** A calibrated, simulation-trained visual-inspection foundation model — instead of a bare defect/no-defect score, it outputs a statistically calibrated confidence per unit, adapts to new defects in seconds via amortized inference (not a per-SKU fine-tuning loop), and ships an audit-ready calibration report that satisfies human-in-loop compliance requirements out of the box.

Four components, each mapped to a real gap in the concept and to something already proven out in my own project work:

1. **Physically realistic defect simulator solves the cold-start problem.** Rare defects have almost no real examples by definition. Build a procedural/generative simulator of surface defects (scratches, dents, discoloration, rendered under realistic camera/lighting noise) — directly analogous to the GalSim-based simulator (Sérsic profiles + real PSF convolution + injected real sky noise) I built for my thesis to generate 250k synthetic galaxy images because real labelled examples were scarce. This lets the model see plausible defect appearances *before* a real example exists on the line.
   → Scaffolded as `src/visual_inspection_fm/simulate.py` (interface stubbed, not yet implemented).

2. **Amortized inference beats fine-tuning for "hours not weeks."** The concept deck's plan is: freeze a backbone, fine-tune a small classification head per new defect (still a training step per SKU). My thesis work is amortized Bayesian/simulation-based inference — train once across many simulated scenarios so a new case is handled by a single forward pass, no retraining loop. Applied here: instant few-shot classification of a new defect from a handful of query images, in seconds, not hours.
   → **Already has a working mock prototype**: `src/visual_inspection_fm/modeling/sbi_model.py` (frozen ResNet18 backbone → embeddings → mock probabilistic head outputting severity mean + epistemic uncertainty). Originally written as a standalone `sbi_prototype.py`; refactored into the package on 2026-09-17 (see progress log).

3. **Calibration as the compliance product, not an afterthought.** I already validate models with simulation-based calibration (SBC) and ECDF-difference diagnostics rather than trusting raw accuracy. Turn that into the product's compliance layer: a calibrated posterior confidence per inspected unit, with automatic escalation to a human inspector when uncertainty is high, plus a per-deployment calibration report — the kind of audit artifact EU AI Act human-in-loop rules will require, and something none of the competitive set (Cognex, Keyence, Landing AI, SAM 2) currently offers.
   → Scaffolded as `src/visual_inspection_fm/calibration.py` (interface stubbed — this is the next big open item, see §4).
   → The escalation *decision logic* already exists and is tested: `src/visual_inspection_fm/modeling/predict.py::route_decision`.

4. **Hierarchical/multilevel modeling is the honest version of the "cross-plant moat."** The deck claims every new plant's data sharpens the model for everyone but doesn't specify the mechanism. A hierarchical Bayesian structure across plants and SKUs (the same partial-pooling approach I used across U.S. communities and across children with uneven visit counts in prior projects) lets a new line with only 3 real defect photos borrow statistical strength from every other line's defect distribution — a precise, principled version of the flywheel story instead of a hand-wave.
   → Not yet scaffolded — lowest priority for the hackathon demo itself, but a strong line for the pitch narrative.

**Scoped hackathon demo (achievable in the time available):** a small synthetic defect generator (a handful of procedural defect types on a textured background) → embeddings via a frozen pretrained backbone → an amortized few-shot classifier on top → one calibration plot (predicted confidence vs. observed accuracy) showing the model knows when it doesn't know.

### 2a. Alternative Solution Approaches Considered

For pitch defensibility — showing the calibrated/amortized-inference direction above was chosen over real alternatives, not the only idea that came to mind:

1. **Few-shot metric learning on a frozen backbone** (the concept deck's own plan). Prototypical/Siamese embeddings, new defect = nearest-prototype lookup, no retraining. Simple and well-trodden, but still needs labeled defect examples up front and gives a similarity score, not a calibrated probability — doesn't close the "when to trust vs. escalate" compliance gap.
2. **Unsupervised one-class anomaly detection** (PatchCore, PaDiM, autoencoder reconstruction error). Train only on abundant "good" parts; flag statistical deviation as candidate defect. Most literal reading of "zero-shot to new SKUs" since it sidesteps the rare-defect-labels problem entirely. Weakness: no defect-type classification, and reconstruction-error thresholds are themselves uncalibrated.
3. **Simulation-trained amortized inference — current pick.** Procedural defect simulator + train-once amortized posterior + SBC-based calibration. Best fit to the compliance angle and to prior thesis work; main risk is simulator realism (sim-to-real gap) inside a 2-day build window.
4. **Vision-language / foundation-model in-context approach** (build on CLIP-style or SAM 2 rather than a bespoke backbone). Zero-shot localization via prompting, no training loop. Least defensible against named competitors (Landing AI, SAM 2) since the backbone itself is commodity — differentiation would rest entirely on a calibration layer bolted on top.
5. **Conformal prediction as the calibration layer**, instead of full Bayesian SBC. Wraps around any black-box detector, gives distribution-free guaranteed-coverage intervals per unit. Less novel/personal than SBC but far simpler to implement correctly — a credible fallback or sanity-check layered on top of 3.
6. **Active-learning loop** instead of synthetic data — model flags its most-uncertain real units for human labeling, attacking "months of tuning" by minimizing labels needed. No simulator-realism risk, but weaker on true cold-start (a brand-new SKU with zero defects seen yet).
7. **Physics-based 3D digital-twin rendering** (CAD-driven, Omniverse/Isaac-Sim style) instead of a lightweight 2D procedural simulator. Much higher photorealism and domain randomization, but too heavy to prototype meaningfully in 48 hours.
8. **Federated learning as the core architecture**, not an add-on tier. Trains locally per plant, shares only model updates — the honest answer to the IP objection a real Tier-1 customer would raise about handing over raw defect images, and a more literal version of the "every deployment sharpens the model" claim than a hand-wave.
9. **Deep ensembles / MC-dropout for uncertainty**, as a cheaper substitute for SBI/normalizing-flow posteriors. Cheap and well-established, but off-the-shelf — harder to claim as differentiated IP.

**Verdict:** 3 remains the strongest single pick — it's the only option that gets both a genuine cold-start answer (1, 4, 6, 9 all still need some real defect signal first) and a principled calibration story (2, 4, 9 have none). Worth stealing from the others regardless: conformal prediction (5) as a fallback/sanity-check on top of SBC, and federated learning (8) as the honest answer to the data-sharing objection a real customer would raise.

---

## 3. Hackathon Evaluation Rubric (what's actually graded)

| Dimension | Expected at Hackathon (Sep) |
|---|---|
| Problem & Solution | Reasoned hypothesis backed by desk research/data |
| Business Model | Monetization hypothesis + pricing logic + target customers |
| Bosch right-to-win | How the team would use Bosch assets if granted |
| Commercial validation | Initial GTM + concrete steps for Phase 1 validation |
| Technical validation | Model logic / demo / preliminary prototype |
| Team & Ask | Founder's edge, named team gaps, co-founder ask |

Explicitly *not* expected yet: Bosch network access, live customer interviews, resolved legal/technical feasibility studies, a finished monetization model.

---

## 4. Status / Progress Log

- **2026-09-17, ~14:10** — Read background (CV) and concept materials (founder package, concept overview, Visual-Inspection FM deck). Defined problem statement and personalized solution direction (calibrated/amortized/simulation-based angle above).
- **2026-09-17, ~14:46** — Set up the actual Python project (this repo), following cookiecutter-data-science + Eric Ma's project-structure gist + the Towards Data Science best-practices article (see §7 for links). Connected folder: `.../Notes/Hackathon/visual-inspection-fm/`.
  - Found an existing loose file in the Hackathon folder, `sbi_prototype.py` — a working mock prototype of exactly Component 2 (frozen ResNet18 backbone → embeddings → mock probabilistic head → severity + uncertainty → escalate/reject/pass routing). Refactored it into the package: backbone loading → `features.py`, model class → `modeling/sbi_model.py`, decision routing → `modeling/predict.py`, runnable entry point → `scripts/run_demo.py`.
  - Added `pyproject.toml`, `requirements.txt`, `Makefile`, `.gitignore`, `.env.example`.
  - Added stub modules with TODOs for the two not-yet-built components: `simulate.py` (Component 1) and `calibration.py` (Component 3).
  - Added `tests/test_sbi_model.py` (3 tests: forward-pass shapes, `predict_one` output contract, `route_decision` boundary behavior) — **all passing**.
  - Verified end-to-end: package installs (`pip install -e .`), demo runs, tests pass. (One sandbox-only wrinkle: this cloud session's network blocks `download.pytorch.org`, so the ResNet18 pretrained-weights download had to be bypassed for the verification run here — it will download normally on a machine with normal internet access, e.g. your laptop.)
  - Moved the three founder-package source documents (previously only available to me as markdown text reconstructions) into `docs/` as their original files, now that they're available from your connected folder: `Founder_Concept_Overview.pdf`, `HERE_GOES_Founder_Pre_Hackathon_Package.pdf`, `Visual_Inspection_Hackathon_package.pptx`.
  - Old loose copies (`sbi_prototype.py`, the two PDFs, and `Selected_Concepts/02_Visual_Inspection_Hackathon_package.pptx`) are still sitting at the top level of the Hackathon folder — I can't move/delete files on your computer without you granting delete access first, so for now they're just duplicates. Safe to delete once you've confirmed the new project has everything (see chat for details).

**Open checklist:**
- [x] Build technical demo skeleton: package structure + working mock model (frozen backbone + uncertainty head) + routing logic + tests
- [ ] Implement `simulate.py` — synthetic defect generator (Component 1)
- [ ] Implement `calibration.py` — the actual calibration plot (Component 3) — this is the single highest-leverage remaining demo piece since it's the differentiator vs. every competitor
- [ ] Draft monetization hypothesis with real pricing numbers (per line/year, tiering vs. edge appliance + API)
- [ ] Draft Bosch right-to-win notes (plant data, BCAI, Nexeed/Bosch Security Systems channel, IEC 61508/IATF 16949 pedigree)
- [ ] Draft team & ask section (named gap: industrial/manufacturing CV production experience)
- [ ] Day 2 (Sep 18): polish + rehearse for 13:00 presentation

## 5. Open Questions / Team Gaps

- No manufacturing or industrial-CV production experience on the founding team yet — likely the honest, named ask for a co-founder.
- Pricing/unit economics for edge appliance vs. API tiers not yet modeled with real numbers.
- Need to decide entry wedge order: Bosch's own plants first (zero cold-start) vs. going straight to Tier-1 automotive (IATF 16949) as first external reference.

## 6. Reference Docs (this project)

- `docs/HERE_GOES_Founder_Pre_Hackathon_Package.pdf` — program structure, hackathon logistics, agenda, evaluation rubric
- `docs/Founder_Concept_Overview.pdf` — all 16 venture concepts across 5 themes (context for where Visual-Inspection FM sits)
- `docs/Visual_Inspection_Hackathon_package.pptx` — the Visual-Inspection FM concept deck (problem, solution, market, competition, right-to-win)

*(These are now the original files, restored from your connected folder on 2026-09-17. A prior session had only markdown text reconstructions of these because the binaries weren't available at the time — those reconstructions are superseded by the real files above and are not duplicated here.)*

## 7. Project structure references

The Python project layout in this repo (see root `README.md` for the full tree) follows a blend of:
- [cookiecutter-data-science](https://cookiecutter-data-science.drivendata.org/) — data/docs/models/reports skeleton, Makefile conventions
- [Eric Ma's project-structure gist](https://gist.github.com/ericmjl/27e50331f24db3e8f957d1fe7bbbe510) — installable package (`src/visual_inspection_fm/`) + `scripts/` for runnable entry points + tests alongside code
- [Towards Data Science — Python project structure best practices](https://towardsdatascience.com/python-project-structure-best-practices-d9d0b174ad5d/) — `pyproject.toml`, `.env` for secrets, pre-commit-friendly tooling
