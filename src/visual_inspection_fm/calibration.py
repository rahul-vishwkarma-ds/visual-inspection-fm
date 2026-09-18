"""Component 3 — calibration diagnostics: the compliance product, not an afterthought.

TODO (see docs/notes.md §2.3): validate the model with simulation-based
calibration (SBC) and ECDF-difference diagnostics rather than trusting raw
accuracy. This is what turns a bare confidence score into an audit artifact
that satisfies EU AI Act human-in-loop requirements — and it's the one thing
none of the competitive set (Cognex, Keyence, Landing AI, SAM 2) currently
ships.

Hackathon-scoped demo target: one calibration plot — predicted confidence vs.
observed accuracy — showing the model knows when it doesn't know.

Planned interface (fill in during the hackathon):

    def simulation_based_calibration(model, simulator, n_sims: int) -> "np.ndarray":
        '''Run SBC: draw params, simulate data, run inference, return rank statistics.'''
        raise NotImplementedError

    def plot_calibration(predicted_confidence, observed_accuracy, out_path) -> None:
        '''Reliability-diagram-style plot; save to reports/figures/.'''
        raise NotImplementedError
"""
