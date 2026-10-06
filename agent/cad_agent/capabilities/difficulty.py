"""Proxy features of target difficulty.

Used twice: as a signal to the scaffold (how much budget the part deserves) and
as the basis for stratifying the subsample. Computed from the GT alone, without
a prediction, so they are available before the rollout starts.

GT watertightness is kept separate on purpose: it is not "difficulty" but a
stratum attribute -- IoU is undefined on a non-watertight GT.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

logger = logging.getLogger(__name__)

N_SAMPLE_POINTS = 4096


def shape_features(gt_mesh_path: str | Path, n_points: int = N_SAMPLE_POINTS) -> dict[str, Any]:
    """Cheap target features: extents, fill ratio, spread of the point cloud."""
    try:
        mesh = trimesh.load_mesh(gt_mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

        from cad_agent.capabilities.metrics import is_valid_gt

        extents = np.asarray(mesh.extents, dtype=float)
        longest = float(np.max(extents)) if extents.size else 0.0
        bbox_volume = float(np.prod(extents)) if extents.size else 0.0
        # Two distinct statuses that must not be confused:
        #   gt_watertight     - the metrics criterion (`metrics.is_valid_gt`): the
        #                       whole mesh is closed and |volume| > 0; it defines
        #                       the strata and decides whether the GT enters IoU;
        #   gt_watertight_raw - closed as is, without any repair.
        watertight = bool(is_valid_gt(mesh))
        watertight_raw = bool(mesh.is_watertight)

        points, _ = trimesh.sample.sample_surface(mesh, n_points, seed=42)
        points = np.asarray(points)
        centered = points - points.mean(axis=0)
        radial = np.linalg.norm(centered, axis=1)

        return {
            "n_faces": int(len(mesh.faces)),
            "n_vertices": int(len(mesh.vertices)),
            "gt_watertight": watertight,
            "gt_watertight_raw": watertight_raw,
            "extent_longest": longest,
            "aspect_ratio": float(longest / max(float(np.min(extents)), 1e-9)) if extents.size else None,
            # Fill of the bounding box: close to one for prismatic parts,
            # small for thin-walled and openwork ones.
            "fill_ratio": float(mesh.volume / bbox_volume) if watertight and bbox_volume > 1e-12 else None,
            "surface_area": float(mesh.area),
            "radial_std": float(radial.std()),
            "radial_mean": float(radial.mean()),
        }
    except Exception as exc:
        logger.warning("Failed to compute difficulty features for %s: %s", gt_mesh_path, exc)
        return {"error": str(exc)}
