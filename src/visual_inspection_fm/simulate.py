"""Component 1 — physically realistic synthetic defect simulator.

TODO (see docs/notes.md §2.1): this is the cold-start solution. Rare defects
have almost no real examples by definition, so we generate plausible defect
appearances procedurally before a real example exists on the line —
directly analogous to the GalSim-based simulator (Sersic profiles + real PSF
convolution + injected real sky noise) used to generate 250k synthetic galaxy
images for the thesis work this concept is built on.

Hackathon-scoped demo target: a handful of procedural defect types
(e.g. scratch, dent, discoloration) rendered on a textured background with
camera/lighting noise — not a full production-grade renderer.

Planned interface (fill in during the hackathon):

    def generate_defect_image(defect_type: str, severity: float, rng: np.random.Generator) -> np.ndarray:
        '''Render one synthetic RGB defect image at IMAGE_SIZE x IMAGE_SIZE.'''
        raise NotImplementedError

    def generate_batch(n: int, defect_types: list[str], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        '''Return (images, labels) for training/calibration.'''
        raise NotImplementedError
"""
