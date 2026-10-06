__all__ = ["build_cot_meta"]

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
# from cadlib.core import CoordSystem
# from cadlib.features import Chamfer, Extrude, Fillet, Profile, Revolve, RotationGroup
# from cadlib.model import CAD, CADModel
# from cq_parser.parser import CADQuerySyntacticParser

from .mesh_stat import MeshStatistics


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float(np.array(x, dtype=float))


def _fmt_bbox_ratio(stat: MeshStatistics, decimals: int = 1) -> str:
    assert stat.extent_x and stat.extent_y and stat.extent_z
    x = round(stat.extent_x, decimals)
    y = round(stat.extent_y, decimals)
    z = round(stat.extent_z, decimals)
    return f"{x}:{y}:{z}"


def _get_axis_human(axis_tuple: tuple[str, int]) -> str:
    name, sign = axis_tuple
    s = "+" if sign > 0 else "-"
    return f"global {name} ({s})"


def _axes_compact_str(cs: CoordSystem) -> str:
    x, y, _ = cs.get_local_to_global_mapping()
    return f"{x.replace('-', '')}{y.replace('-', '')}"


@dataclass
class InnerGroupMeta:
    description: str
    loop_primitives_count: int
    outer_area_ratio: float  # raw ratio (no rounding)
    like_in_profile_index: int | None  # 1-based if present


@dataclass
class ProfileMeta:
    orientation_xy: str  # e.g. "XY"
    axes_mapping: tuple[str, str, str]  # ('X','Y','Z') (with possible signs)
    outer_desc: str
    outer_segments_count: int
    inner_loops_total: int
    inner_groups_count: int
    profile_area: float
    outer_loop_area: float
    profile_to_outer_ratio: float  # raw ratio
    unique_inner_groups: list[InnerGroupMeta]


@dataclass
class BaseFeatureMeta:
    index: int
    type: str
    params: dict[str, Any]  # raw params (extent, angle, axis tuple, etc.)
    operation_text: str  # optional convenience text
    profile: ProfileMeta
    like_in_profile_index: int | None  # 1-based


@dataclass
class ModifyingFeatureMeta:
    type: str
    params: dict[str, Any]  # raw params if any
    touches_primitives: list[int]  # 1-based indices


@dataclass
class IntersectionMeta:
    current: int
    with_index: int
    same_plane: bool
    relation: str
    area_ratio: float  # raw
    outer_outer_ratio: float  # raw
    intersection_to_smaller_ratio: float  # raw


@dataclass
class ModelMeta:
    file: str
    bbox_ratio_str: str  # preformatted for UX only
    bbox_extents: list[float]  # raw rounded for bbox display, ok
    base_features_count: int
    base_features_text: list[str]
    modifying_features_count: int
    modifying_features_text: list[str]
    base_features: list[BaseFeatureMeta]
    modifying_features: list[ModifyingFeatureMeta]
    intersections: list[IntersectionMeta]
    mesh_volume: float


def build_cot_meta(
    py_path: Path,
    stl_path: Path | None = None,
    cad_model: CADModel | None = None,
    out_file: Path | None = None,
    bb_decimals: int = 1,
) -> None:
    """
    Builds CoT metadata. If out_file is None, writes a separate .json next to
    each input .py. If out_file is given, collects a single JSON (list of models).
    """
    parser = CADQuerySyntacticParser()

    if not stl_path:
        stl_path = py_path.with_suffix(".stl")
    mesh = trimesh.load_mesh(stl_path)
    mesh_stat = MeshStatistics.from_mesh(mesh)

    if not cad_model:
        cadaxt_model, _ = parser.parse_code(py_path.read_text())
        cad_model = CADModel.from_dict(cadaxt_model)

    base_features = [op for op in cad_model.cs if isinstance(op, (Extrude, Revolve))]
    modifying_features = [
        op for op in cad_model.cs if isinstance(op, (Fillet, Chamfer))
    ]

    bbox_ratio_str = _fmt_bbox_ratio(mesh_stat, decimals=bb_decimals)
    assert mesh_stat.extent_x and mesh_stat.extent_y and mesh_stat.extent_z
    bbox_extents = [
        round(mesh_stat.extent_x, bb_decimals),
        round(mesh_stat.extent_y, bb_decimals),
        round(mesh_stat.extent_z, bb_decimals),
    ]

    base_features_text = [str(f) for f in base_features]
    modifying_features_text = [str(f) for f in modifying_features]

    # Profiles and rotation groups (assign groups)
    profiles = [bf.profile.sorted() for bf in base_features]
    _ = Profile.rotation_group_eq(profiles)

    def outer_group_of(profile: Profile) -> RotationGroup:
        return profile.children[0].group

    primitives = []
    base_features_meta: list[BaseFeatureMeta] = []

    for i, bf in enumerate(base_features, start=1):
        primitives.append(bf.apply(None))

        profile = profiles[i - 1]
        assert profile.sketch_plane is not None
        orientation = profile.sketch_plane

        face = profile.profile2face()

        outer_loop = profile.children[0]
        unique_inner_loop_groups = profile.unique_inner_loop_groups()

        biggest_area = CAD.shape_area(outer_loop.loop2face())
        profile_area = CAD.shape_area(face)

        inner_areas = [
            CAD.shape_area(g.loops[0].loop2face()) for g in unique_inner_loop_groups
        ]
        inner_coefs = [
            (a / biggest_area if biggest_area > 0 else 0.0) for a in inner_areas
        ]

        axes_map = orientation.get_local_to_global_mapping()
        orientation_xy = _axes_compact_str(orientation)

        like_in_idx = None
        if i > 1:
            current_group = outer_group_of(profile)
            for j in range(1, i):
                if outer_group_of(profiles[j - 1]) is current_group:
                    like_in_idx = j
                    break

        # Operation params (raw)
        params: dict[str, Any] = {}
        if isinstance(bf, Extrude):
            op_text = f"Primitive #{i} is built via Extrude with extent={bf.extent1} from its sketch."
            params["extent1"] = _safe_float(bf.extent1)
        elif isinstance(bf, Revolve):
            ax_name, ax_sign = CoordSystem.check_and_get_axis(bf.normal)  # type: ignore
            op_text = f"Primitive #{i} is built via Revolve around global {ax_name} ({'+' if ax_sign>0 else '-'}) from its sketch."
            params["axis_tuple"] = (ax_name, int(ax_sign))
            if hasattr(bf, "angle"):
                params["angle"] = _safe_float(getattr(bf, "angle"))
            if hasattr(bf, "extent1"):
                params["extent1"] = _safe_float(getattr(bf, "extent1"))
        else:
            op_text = f"Primitive #{i} is built via {bf.__class__.__name__}."

        # Inner groups meta
        inner_groups_meta: list[InnerGroupMeta] = []
        for k, group in enumerate(unique_inner_loop_groups, start=1):
            group_like = None
            if i > 1:
                for prev_idx in range(1, i):
                    if base_features[prev_idx - 1].profile in group.profiles:
                        group_like = prev_idx
                        break

            inner_groups_meta.append(
                InnerGroupMeta(
                    description=group.description,
                    loop_primitives_count=len(group.loops[0].children),
                    outer_area_ratio=float(inner_coefs[k - 1]),
                    like_in_profile_index=group_like,
                )
            )

        profile_meta = ProfileMeta(
            orientation_xy=orientation_xy,
            axes_mapping=axes_map,
            outer_desc=outer_loop.group.description,
            outer_segments_count=len(outer_loop.children),
            inner_loops_total=max(0, len(profile.children) - 1),
            inner_groups_count=len(unique_inner_loop_groups),
            profile_area=float(profile_area),
            outer_loop_area=float(biggest_area),
            profile_to_outer_ratio=float(
                (profile_area / biggest_area) if biggest_area > 0 else 0.0
            ),
            unique_inner_groups=inner_groups_meta,
        )

        base_features_meta.append(
            BaseFeatureMeta(
                index=i,
                type=bf.__class__.__name__,
                params=params,
                operation_text=op_text,
                profile=profile_meta,
                like_in_profile_index=like_in_idx,
            )
        )

    profiles_sorted = [bf.profile.sorted() for bf in base_features]
    intersections: list[IntersectionMeta] = []
    for i in range(1, len(base_features_meta)):
        curr_profile = profiles_sorted[i]
        for j in range(0, i):
            same_plane, relation, ar, oor, inter = curr_profile.relationship(
                profiles_sorted[j]
            )
            intersections.append(
                IntersectionMeta(
                    current=i + 1,
                    with_index=j + 1,
                    same_plane=bool(same_plane),
                    relation=str(relation),
                    area_ratio=float(ar),
                    outer_outer_ratio=float(oor),
                    intersection_to_smaller_ratio=float(inter),
                )
            )

    modifying_meta: list[ModifyingFeatureMeta] = []
    for mf in modifying_features:
        in_game = []
        for idx, prim in enumerate(primitives, start=1):
            try:
                if CAD.select_edges(prim, mf.edges):
                    in_game.append(idx)
            except Exception:
                pass

        mf_params = {}
        if hasattr(mf, "radius"):
            mf_params["radius"] = _safe_float(getattr(mf, "radius"))
        if hasattr(mf, "distance"):
            mf_params["distance"] = _safe_float(getattr(mf, "distance"))

        modifying_meta.append(
            ModifyingFeatureMeta(
                type=mf.__class__.__name__,
                params=mf_params,
                touches_primitives=in_game,
            )
        )

    model_meta = ModelMeta(
        file=str(py_path),
        bbox_ratio_str=bbox_ratio_str,
        bbox_extents=bbox_extents,
        base_features_count=len(base_features),
        base_features_text=base_features_text,
        modifying_features_count=len(modifying_features),
        modifying_features_text=modifying_features_text,
        base_features=base_features_meta,
        modifying_features=modifying_meta,
        intersections=intersections,
        mesh_volume=mesh_stat.volume if mesh_stat.volume is not None else 0.0,
    )

    if out_file is None:
        # Save per-file, next to the .py
        json_path = py_path.with_suffix(".json")
        json_path.write_text(
            json.dumps(asdict(model_meta), ensure_ascii=False, indent=2)
        )
    else:
        out_file.write_text(
            json.dumps(asdict(model_meta), ensure_ascii=False, indent=2)
        )
