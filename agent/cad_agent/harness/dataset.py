"""The set of parts for a run.

Either a directory of `.stl` files sorted by name, or a fixed subsample described
by a manifest with strata. The set is built in its own module, outside the search
loop, so that its order and signature are reproducible regardless of what drives it.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FigureSpec:
    """One part: what is reconstructed and which group it belongs to."""

    figure_id: str
    gt_mesh_path: Path
    group: str

    def to_dict(self) -> dict[str, Any]:
        return {"figure_id": self.figure_id, "gt_mesh_path": str(self.gt_mesh_path), "group": self.group}


def load_figures(details: list[dict[str, str]]) -> list[FigureSpec]:
    """Build the list of parts from the `experiment.details` config."""
    figures: list[FigureSpec] = []
    for detail in details:
        for group, folder in detail.items():
            folder_path = Path(folder)
            if not folder_path.is_dir():
                raise ValueError(f"The part directory must exist and contain .stl files: {folder_path}")

            mesh_paths = sorted(folder_path.glob("*.stl"))
            if not mesh_paths:
                raise ValueError(f"No .stl files in the directory: {folder_path}")

            figures.extend(
                FigureSpec(figure_id=f"{group}/{path.stem}", gt_mesh_path=path, group=group)
                for path in mesh_paths
            )

    if not figures:
        raise ValueError("experiment.details produced no parts")
    return figures


def load_figures_from_manifest(
    manifest_path: str | Path,
    split: str = "evo",
    verify: bool = True,
) -> list[FigureSpec]:
    """Build the set from a subsample manifest and check its signature.

    Verification is on by default: the manifest lives in the repository while the
    meshes live on disk, and a mismatch between them should surface before the run,
    not in the numbers after it.
    """
    from cad_agent.harness.subsample import load_manifest

    manifest = load_manifest(manifest_path)
    if split not in manifest.get("splits", {}):
        raise ValueError(f"The manifest has no split {split!r}: {sorted(manifest.get('splits', {}))}")

    entry = manifest["splits"][split]
    figures = [
        FigureSpec(
            figure_id=item["figure_id"],
            gt_mesh_path=Path(item["gt_mesh_path"]),
            group=item.get("stratum", split),
        )
        for item in entry["figures"]
    ]

    if verify:
        actual = dataset_signature(figures)
        if actual != entry["signature"]:
            raise ValueError(
                f"Signature of split {split!r} does not match the manifest: {actual} vs {entry['signature']}. "
                "The set on disk changed or the manifest was built from another source."
            )
        missing = [figure.gt_mesh_path for figure in figures if not figure.gt_mesh_path.exists()]
        if missing:
            raise ValueError(f"Missing {len(missing)} meshes from the manifest, first: {missing[0]}")

    logger.info(
        "Set from manifest %s: split %s, %d parts, strata %s",
        manifest.get("name", manifest_path), split, len(figures), entry.get("stratum_proportions"),
    )
    return figures


def dataset_signature(figures: list[FigureSpec]) -> str:
    """Set signature: membership and order. A repeated run must produce the same one."""
    payload = "\n".join(f"{figure.figure_id}\t{figure.gt_mesh_path}" for figure in figures)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
