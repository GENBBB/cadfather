"""cadfit: deterministic reconstruction of CadQuery code from a mesh (the det tool).

Entry points:
- `cadfit.candidates.cadfit_pass.cadfit_single_pass`: an extrusion pass over the mesh;
- `cadfit.candidates.make_candidates`: 14 residual-based detectors, N programs;
- `cadfit.tool`: the det tool for the agent (cold/warm path, DSL operations).
"""


def native_status() -> dict:
    """Whether the C++ det module is loaded, and from where. Without it det output differs."""
    from .candidates import section_analyzer
    cg = section_analyzer._cg
    return {
        "available": cg is not None,
        "file": getattr(cg, "__file__", None),
        "cluster_normals": hasattr(cg, "cluster_normals"),
        "section": hasattr(cg, "section"),
    }
