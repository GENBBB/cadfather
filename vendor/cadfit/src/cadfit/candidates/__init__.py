"""det_candidates — CADFit-style deterministic step proposals for the
stepwise CAD-recovery pipeline.

Per stepwise iter:
  1.  Render the current build (prev_code) -> Sprev mesh.
  2.  Compute residual meshes via mesh Boolean:
         ADD = Sgt  \\ Sprev   (parts of the target not yet built -> r.union)
         CUT = Sprev \\ Sgt    (parts of current build that should be removed -> r.cut)
  3.  Run each enabled detector on ADD and CUT, emit a CadQuery block
      that appends to ``r``: ``r = r.union(_add_i)`` or ``r = r.cut(_cut_i)``.
  4.  Score each candidate (or hand off to the existing render+optimize+score path).

Self-contained Python package - no CADAgent-Evolution dependency.  Detector
implementations are adapted from CADAgent-Evolution/agent/tools/.
"""

# Lazy re-export: `import cadfit.candidates.cadfit_pass` must not pull in the
# proposer and all 14 detectors (the det_cold/det_warm path does not use them).
def __getattr__(name):
    if name in ("make_candidates", "DetCandidate"):
        from . import proposer
        return getattr(proposer, name)
    raise AttributeError(name)


__all__ = ["make_candidates", "DetCandidate"]
