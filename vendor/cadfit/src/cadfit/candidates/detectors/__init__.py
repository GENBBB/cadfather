"""Built-in detectors.  Each is a function with the signature

    detect(mesh: trimesh.Trimesh, **kwargs) -> list[DetectorOutput]

where DetectorOutput is (program: str, score: float, debug: dict).

A detector may return multiple variants (e.g. one per principal axis).
The proposer wraps each in a block (union or cut) and adds it to the
candidate pool.
"""
# Lazy imports: `from .detectors.planar_cluster import ...` (the cadfit_single_pass
# path) must not load all 14 detectors.
_LAZY = {
    "detect_cylinders": "cylinder",
    "detect_silhouette_extrude": "extrude",
    "detect_fillet_chamfer": "fillet_chamfer",
    "detect_helix": "helix",
    "detect_local_extrudes": "local_extrude",
    "detect_local_lofts": "local_loft",
    "detect_local_revolves": "local_revolve",
    "detect_local_sweeps": "local_sweep",
    "detect_loft": "loft",
    "detect_planar_clusters": "planar_cluster",
    "detect_revolve": "revolve",
    "detect_slab_stack": "slab_stack",
    "detect_slice_fit": "slice_fit",
    "detect_sweep": "sweep",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(f"{__name__}.{_LAZY[name]}"), name)
    raise AttributeError(name)


__all__ = [
    "detect_silhouette_extrude",
    "detect_cylinders",
    "detect_revolve",
    "detect_loft",
    "detect_slice_fit",
    "detect_planar_clusters",
    "detect_fillet_chamfer",
    "detect_sweep",
    "detect_helix",
    "detect_slab_stack",
    "detect_local_extrudes",
    "detect_local_revolves",
    "detect_local_lofts",
    "detect_local_sweeps",
]
