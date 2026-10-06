"""
Continue CAD generation from a saved prefix using the tail of ``sampled_factories``
(dicts compatible with ``*Factory.from_dict`` / runner ``*_factory.json``).

Expected tail entries are **leaf** factories (as saved by ``CADFactory.generate``), not
``BlockFactory`` / ``EitherFactory`` wrappers from YAML.
"""

from __future__ import annotations

__all__ = [
    "cad_prefix_and_tail_factories",
    "continue_from_prefix",
    "continue_from_prefix_state",
    "continue_full_pipeline",
    "finalize_cad_for_export",
    "sampled_factory_dict_to_entry",
    "_compact_prefix_planes_face_planes",
]

import logging
from collections.abc import Callable
from copy import deepcopy
from operator import itemgetter
from typing import Any

import numpy as np

from .cad import CAD, CADFactory
from .circular_pattern import CircularPatternFactory
from .cut_thru_all import CutThruAllFactory
from .edge_operations import FilletChamferFactory
from .extrude import ExtrudeFactory, TwoExtrudesFactory
from .face_operations import FaceFilletChamferFactory
from .gear import GearFactory
from .hole import HoleFactory
from .loft import LoftFactory
from .orto_cut import OrtoCutFactory
from .plane_logic import (
    generate_circular_pattern_plane,
    generate_cut_thru_all_plane,
    generate_extrude_plane,
    generate_face_fillet_chamfer_plane,
    generate_fillet_chamfer_plane,
    generate_gear_plane,
    generate_hole_plane,
    generate_loft_plane,
    generate_revolve_plane,
    generate_selected_plane,
    generate_sweep_init_plane,
)
from .runner import validate_cad_continuation_mesh
from .revolve import RevolveFactory
from .selected_face_plane import SelectedFacePlaneFactory
from .shell import ShellFactory
from .sweep import SweepFactory
from .sweep_init import SweepInitFactory

logger = logging.getLogger(__name__)

_FACTORY_CLASSES: dict[str, type] = {
    "ExtrudeFactory": ExtrudeFactory,
    "TwoExtrudesFactory": TwoExtrudesFactory,
    "ShellFactory": ShellFactory,
    "RevolveFactory": RevolveFactory,
    "SweepFactory": SweepFactory,
    "SweepInitFactory": SweepInitFactory,
    "LoftFactory": LoftFactory,
    "GearFactory": GearFactory,
    "HoleFactory": HoleFactory,
    "OrtoCutFactory": OrtoCutFactory,
    "CutThruAllFactory": CutThruAllFactory,
    "FaceFilletChamferFactory": FaceFilletChamferFactory,
    "FilletChamferFactory": FilletChamferFactory,
    "SelectedFacePlaneFactory": SelectedFacePlaneFactory,
    "CircularPatternFactory": CircularPatternFactory,
}

# Wrapper keys from YAML blocks in ``CADFactory.generate``; often absent in flat
# ``*_factory.json`` (only ``Factory.to_dict()``).
_FACTORY_ENTRY_WRAPPER_DEFAULTS: dict[str, Any] = {
    "plane": 0,
    "reuse_plane_probability": 0.0,
    "selected_plane_probability": 0.0,
    "selected_plane_centered_probability": 0.5,
    "probability": 1.0,
}
_FACTORY_ENTRY_WRAPPER_KEYS = frozenset(_FACTORY_ENTRY_WRAPPER_DEFAULTS.keys())


def sampled_factory_dict_to_entry(smp: dict[str, Any]) -> dict[str, Any]:
    """Build ``{"factory": <instance>, ...}`` like ``CADFactory`` inner entries."""
    fac_blob = smp["factory"] if isinstance(smp.get("factory"), dict) else smp
    t = fac_blob["type"]
    if t not in _FACTORY_CLASSES:
        raise KeyError(f"Unknown factory type in sampled dict: {t}")
    fac = _FACTORY_CLASSES[t].from_dict(deepcopy(fac_blob))
    entry: dict[str, Any] = {
        "factory": fac,
        **_FACTORY_ENTRY_WRAPPER_DEFAULTS,
    }
    if isinstance(smp.get("factory"), dict):
        for key in _FACTORY_ENTRY_WRAPPER_KEYS:
            if key in smp:
                entry[key] = smp[key]
    return entry


def _compact_prefix_planes_face_planes(
    cs: list[dict],
    planes: list[dict],
    face_planes: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Keep only base ``planes`` / ``face_planes`` entries referenced by ``cs``, and
    remaps indices to ``0..n-1`` (same idea as ``CAD.fix`` for unused planes).

    Mutates ``cs`` and the passed ``planes`` / ``face_planes`` copies in place.
    """
    # --- integer workplanes (w0, w1, …)
    plane_ids: list[int] = []
    _seen_p: set[int] = set()
    for s in cs:
        p = s.get("plane", None)
        if isinstance(p, int) and p not in _seen_p:
            _seen_p.add(p)
            plane_ids.append(p)
    if plane_ids:
        pmap = dict(zip(plane_ids, range(len(plane_ids))))
        for op in cs:
            if isinstance(op.get("plane", None), int):
                op["plane"] = pmap[op["plane"]]
        planes[:] = (
            list(itemgetter(*plane_ids)(planes))
            if len(plane_ids) > 1
            else [planes[plane_ids[0]]]
        )
    else:
        planes.clear()

    # --- face planes (face_wN / point_selN etc.)
    fp_ids: list[int] = []
    _seen_fp: set[int] = set()

    def _suffix_index(obj: dict, key: str) -> int:
        val = obj[key] if key == "plane" else getattr(obj["op"], key)
        assert isinstance(val, str), val
        suf = val.split("_")[-1].lstrip("selw")
        return int(suf)

    for s in cs:
        if s["type"] == "SelectedFacePlane":
            for key in ("face_plane_name",):
                idx = _suffix_index(s, key)
                if idx not in _seen_fp:
                    _seen_fp.add(idx)
                    fp_ids.append(idx)
        elif isinstance(s.get("plane", None), str):
            idx = _suffix_index(s, "plane")
            if idx not in _seen_fp:
                _seen_fp.add(idx)
                fp_ids.append(idx)

    if not fp_ids:
        face_planes.clear()
        return planes, face_planes

    fmap = dict(zip(fp_ids, range(len(fp_ids))))

    def _remap_plane_string(val: str) -> str:
        old_ending = val.split("_")[-1].lstrip("selw")
        new_ending = fmap[int(old_ending)]
        return val[: -len(old_ending)] + str(new_ending)

    for op in cs:
        if isinstance(op.get("plane", None), str) and op["type"] != "SelectedFacePlane":
            op["plane"] = _remap_plane_string(op["plane"])
        elif op["type"] == "SelectedFacePlane":
            o = op["op"]
            for attr in ("face_plane_name", "face_name", "point_name"):
                cur = getattr(o, attr)
                if cur is None:
                    continue
                setattr(o, attr, _remap_plane_string(cur))

    face_planes[:] = (
        list(itemgetter(*fp_ids)(face_planes))
        if len(fp_ids) > 1
        else [face_planes[fp_ids[0]]]
    )
    return planes, face_planes


def cad_prefix_and_tail_factories(cad: CAD, k: int) -> tuple[CAD, list[dict[str, Any]]]:
    """
    Split ``cad`` into a prefix snapshot and tail factory dicts without mutating ``cad``.

    ``planes`` / ``face_planes`` are restricted to entries referenced by ``cs[:k]``,
    with indices remapped like ``CAD.fix`` unused-plane cleanup.
    """
    if cad.sampled_factories is None:
        raise ValueError("cad.sampled_factories is required for continuation")
    if not (0 <= k <= len(cad.cs)):
        raise IndexError(f"k={k} out of range for cs len {len(cad.cs)}")
    if len(cad.cs) != len(cad.sampled_factories):
        raise ValueError(
            f"len(cs)={len(cad.cs)} != len(sampled_factories)={len(cad.sampled_factories)}"
        )
    tail = cad.sampled_factories[k:]
    cs_prefix = deepcopy(cad.cs[:k])
    planes_p = deepcopy(cad.planes)
    face_p = deepcopy(cad.face_planes)
    _compact_prefix_planes_face_planes(cs_prefix, planes_p, face_p)
    prefix = CAD(
        planes_p,
        face_p,
        cs_prefix,
        deepcopy(cad.sampled_factories[:k]),
        cad.world_size,
        cad.max_string_length,
        cad.use_literals,
    )
    return prefix, tail


def _append_one_leaf_factory(
    factory: dict[str, Any],
    planes: list[dict],
    face_planes: list[dict],
    cs: list[dict],
    *,
    r_min: float,
    r_max: float,
    world_size: float,
    max_string_length: int,
    use_literals: bool,
    n_retries: int,
    n_extrude_retries: int,
    sampled_out: list[dict[str, Any]],
) -> None:
    """
    One leaf factory step (mirrors non-Block branches in ``CADFactory.generate``).
    Appends at most one op to ``cs``; on success appends ``factory['factory'].to_dict()``
    to ``sampled_out``.
    """
    cs_len_before = len(cs)
    fac = factory["factory"]
    validator = CADFactory.validation_only(
        world_size=world_size,
        max_string_length=max_string_length,
        use_literals=use_literals,
    )

    if isinstance(fac, ExtrudeFactory):
        for retry in range(n_extrude_retries):
            attempt_state = (len(planes), len(face_planes), len(cs))
            try:
                extrude = fac.generate()
                plane_type = None
                if np.random.uniform() < factory.get("reuse_plane_probability", 0):
                    plane_type = "reuse"
                elif np.random.uniform() < factory.get("selected_plane_probability", 0):
                    plane_type = "selected"
                else:
                    plane_type = "default"
                generate_extrude_plane(
                    factory,
                    planes,
                    face_planes,
                    extrude,
                    r_min,
                    r_max,
                    cs,
                    world_size,
                    max_string_length,
                    use_literals,
                    CAD,
                    plane_type=plane_type,
                )
                if len(cs) == attempt_state[2] + 1:
                    if cs[-1]["type"] == "TwoExtrudes":
                        cs[-1]["op"].transform([0, 0], world_size)
                    else:
                        cs[-1]["op"].transform([0, 0, 0], world_size)
                    if (
                        isinstance(cs[-1]["plane"], int)
                        and cs[-1]["type"] != "TwoExtrudes"
                    ):
                        cs[-1]["op"].sketch.transform([0, 0], world_size)
                    validator._assert_operation_attempt_valid(
                        planes,
                        face_planes,
                        cs,
                        attempt_state[2],
                    )
                break
            except Exception:
                CADFactory._restore_generation_state(
                    planes,
                    face_planes,
                    cs,
                    *attempt_state,
                )
                if retry == n_extrude_retries - 1:
                    raise
        if len(cs) == cs_len_before + 1:
            sampled_out.append(fac.to_dict())
        return

    if isinstance(fac, RevolveFactory):
        for retry in range(n_retries):
            attempt_state = (len(planes), len(face_planes), len(cs))
            try:
                revolve = fac.generate()
                generate_revolve_plane(
                    factory,
                    planes,
                    face_planes,
                    revolve,
                    r_min,
                    r_max,
                    cs,
                    world_size,
                    max_string_length,
                    use_literals,
                    CAD,
                )
                if len(cs) == attempt_state[2] + 1:
                    cs[-1]["op"].transform([0, 0, 0], world_size)
                    cs[-1]["op"].sketch.transform([0, 0], world_size)
                    validator._assert_operation_attempt_valid(
                        planes,
                        face_planes,
                        cs,
                        attempt_state[2],
                    )
                break
            except Exception:
                CADFactory._restore_generation_state(
                    planes,
                    face_planes,
                    cs,
                    *attempt_state,
                )
                if retry == n_retries - 1:
                    raise
        if len(cs) == cs_len_before + 1:
            sampled_out.append(fac.to_dict())
        return

    if isinstance(fac, SweepFactory):
        if not cs:
            raise ValueError("SweepFactory requires an existing CAD prefix")
        prefix = CAD(
            planes,
            face_planes,
            cs,
            world_size=world_size,
            max_string_length=max_string_length,
            use_literals=use_literals,
        ).to_string()
        namespace: dict[str, object] = {}
        exec(prefix, globals(), namespace)
        r = namespace["r"]
        sampler = fac.prepare_existing_sampler(r)
        for retry in range(n_extrude_retries):
            attempt_state = (len(planes), len(face_planes), len(cs))
            try:
                sweep, _ = fac.generate_on_existing(
                    r,
                    sampler,
                    world_size=world_size,
                    generation_world_half=r_max,
                )
                cs.append(
                    dict(
                        type="Sweep",
                        op=sweep,
                        plane=None,
                        plane_axes=None,
                    )
                )
                validator._assert_operation_attempt_valid(
                    planes,
                    face_planes,
                    cs,
                    attempt_state[2],
                    existing_faces=sampler.faces,
                )
                break
            except Exception:
                CADFactory._restore_generation_state(
                    planes,
                    face_planes,
                    cs,
                    *attempt_state,
                )
                if retry == n_extrude_retries - 1:
                    raise
        if len(cs) == cs_len_before + 1:
            sampled_out.append(fac.to_dict())
        return

    if isinstance(fac, SweepInitFactory):
        for retry in range(n_retries):
            try:
                sweep = fac.generate()
                len_cs_before = len(cs)
                generate_sweep_init_plane(
                    factory,
                    planes,
                    face_planes,
                    sweep,
                    r_min,
                    r_max,
                    cs,
                    world_size,
                    max_string_length,
                    use_literals,
                    CAD,
                )
                if len(cs) == len_cs_before + 1:
                    cs[-1]["op"].transform([0, 0, 0], world_size)
                break
            except Exception:
                if retry == n_retries - 1:
                    raise
        if len(cs) == cs_len_before + 1:
            sampled_out.append(fac.to_dict())
        return

    if isinstance(fac, LoftFactory):
        for retry in range(n_retries):
            try:
                loft = fac.generate()
                len_cs_before = len(cs)
                generate_loft_plane(
                    factory,
                    planes,
                    face_planes,
                    loft,
                    r_min,
                    r_max,
                    cs,
                    world_size,
                    max_string_length,
                    use_literals,
                    CAD,
                )
                if len(cs) == len_cs_before + 1:
                    cs[-1]["op"].transform([0, 0, 0], world_size)
                break
            except Exception:
                if retry == n_retries - 1:
                    raise
        if len(cs) == cs_len_before + 1:
            sampled_out.append(fac.to_dict())
        return

    if isinstance(fac, GearFactory):
        for retry in range(n_retries):
            try:
                fac.world_size = world_size
                gear = fac.generate()
                len_cs_before = len(cs)
                generate_gear_plane(
                    factory,
                    planes,
                    face_planes,
                    gear,
                    r_min,
                    r_max,
                    cs,
                    world_size,
                    max_string_length,
                    use_literals,
                    CAD,
                )
                if len(cs) == len_cs_before + 1:
                    cs[-1]["op"].transform([0, 0, 0], world_size)
                break
            except Exception:
                if retry == n_retries - 1:
                    raise
        if len(cs) == cs_len_before + 1:
            sampled_out.append(fac.to_dict())
        return

    if isinstance(fac, HoleFactory):
        for retry in range(n_retries):
            attempt_state = (len(planes), len(face_planes), len(cs))
            try:
                hole = fac.generate(plane=None, s=None)
                generate_hole_plane(
                    factory,
                    planes,
                    face_planes,
                    hole,
                    r_min,
                    r_max,
                    cs,
                    world_size,
                    max_string_length,
                    use_literals,
                    CAD,
                )
                if len(cs) == attempt_state[2] + 1:
                    validator._assert_operation_attempt_valid(
                        planes,
                        face_planes,
                        cs,
                        attempt_state[2],
                    )
                break
            except Exception:
                CADFactory._restore_generation_state(
                    planes,
                    face_planes,
                    cs,
                    *attempt_state,
                )
                if retry == n_retries - 1:
                    raise
        if len(cs) == cs_len_before + 1:
            sampled_out.append(fac.to_dict())
        return

    if isinstance(fac, OrtoCutFactory):
        if not cs:
            raise ValueError("OrtoCutFactory requires an existing prefix solid")
        s = CAD(
            planes,
            face_planes,
            cs,
            world_size=world_size,
            max_string_length=max_string_length,
            use_literals=use_literals,
        ).to_string()
        namespace: dict[str, object] = {}
        exec(s, globals(), namespace)
        cad_object = namespace["r"]
        sampler = fac.prepare_existing_sampler(cad_object)
        for retry in range(n_extrude_retries):
            attempt_state = (len(planes), len(face_planes), len(cs))
            try:
                orto_cut_op, workplane = fac.generate_on_existing(
                    cad_object,
                    sampler,
                    world_size=world_size,
                    generation_world_half=r_max,
                )
                cs.append(
                    dict(
                        type="OrtoCut",
                        op=orto_cut_op,
                        plane=workplane,
                        plane_axes=None,
                    )
                )
                validator._assert_operation_attempt_valid(
                    planes,
                    face_planes,
                    cs,
                    attempt_state[2],
                    existing_faces=sampler.faces,
                )
                break
            except Exception:
                CADFactory._restore_generation_state(
                    planes,
                    face_planes,
                    cs,
                    *attempt_state,
                )
                if retry == n_extrude_retries - 1:
                    raise
        if len(cs) == cs_len_before + 1:
            sampled_out.append(fac.to_dict())
        return

    if isinstance(fac, CutThruAllFactory):
        for retry in range(n_retries):
            try:
                cut_thru_all = fac.generate(plane=None, s=None)
                len_cs_before = len(cs)
                generate_cut_thru_all_plane(
                    factory,
                    planes,
                    face_planes,
                    cut_thru_all,
                    r_min,
                    r_max,
                    cs,
                    world_size,
                    max_string_length,
                    use_literals,
                    CAD,
                )
                # if len(cs) == len_cs_before + 1:
                #     cs[-1]["op"].transform([0, 0, 0], world_size)
                break
            except Exception as e:
                if retry == n_retries - 1:
                    raise
        if len(cs) == cs_len_before + 1:
            sampled_out.append(fac.to_dict())
        return

    if isinstance(fac, FaceFilletChamferFactory):
        for retry in range(n_retries):
            try:
                face_fc = fac.generate()
                len_cs_before = len(cs)
                generate_face_fillet_chamfer_plane(
                    factory,
                    planes,
                    face_planes,
                    face_fc,
                    r_min,
                    r_max,
                    cs,
                    world_size,
                    max_string_length,
                    use_literals,
                    CAD,
                    world_size_continue=world_size,
                )
                break
            except Exception:
                if retry == n_retries - 1:
                    raise
        if len(cs) == cs_len_before + 1:
            sampled_out.append(fac.to_dict())
        return

    if isinstance(fac, FilletChamferFactory):
        for retry in range(n_retries):
            try:
                fillet_chamfer = fac.generate()
                len_cs_before = len(cs)
                generate_fillet_chamfer_plane(
                    factory,
                    planes,
                    face_planes,
                    fillet_chamfer,
                    r_min,
                    r_max,
                    cs,
                    world_size,
                    max_string_length,
                    use_literals,
                    CAD,
                    world_size_continue=world_size,
                )
                break
            except Exception:
                if retry == n_retries - 1:
                    raise
        if len(cs) == cs_len_before + 1:
            sampled_out.append(fac.to_dict())
        return

    if isinstance(fac, SelectedFacePlaneFactory):
        for retry in range(n_retries):
            try:
                selected_face_plane = fac.generate()
                len_cs_before = len(cs)
                generate_selected_plane(
                    factory,
                    planes,
                    face_planes,
                    selected_face_plane,
                    r_min,
                    r_max,
                    cs,
                    world_size,
                    max_string_length,
                    use_literals,
                    CAD,
                )
                break
            except Exception as e:
                if retry == n_retries - 1:
                    raise
        if len(cs) == cs_len_before + 1:
            sampled_out.append(fac.to_dict())
        return

    if isinstance(fac, CircularPatternFactory):
        for retry in range(n_retries):
            try:
                # ``CADFactory`` calls ``generate()`` without ``s``; real sketch is built inside
                # ``generate_circular_pattern_plane`` (same as ``plane_logic``).
                circular_pattern = fac.generate()
                len_cs_before = len(cs)
                generate_circular_pattern_plane(
                    factory,
                    planes,
                    face_planes,
                    circular_pattern,
                    r_min,
                    r_max,
                    cs,
                    world_size,
                    max_string_length,
                    use_literals,
                    CAD,
                )
                if len(cs) == len_cs_before + 1:
                    cs[-1]["op"].sketch.transform([0, 0, 0], world_size)
                break
            except Exception:
                if retry == n_retries - 1:
                    raise
        if len(cs) == cs_len_before + 1:
            sampled_out.append(fac.to_dict())
        return

    raise TypeError(f"Unsupported leaf factory type: {type(fac).__name__}")


def continue_from_prefix(
    cad: CAD,
    k: int,
    *,
    n_retries: int = 10,
    n_extrude_retries: int = 1,
) -> CAD:
    """
    Keep ``cad.cs[:k]`` and ``cad.sampled_factories[:k]`` unchanged (geometry / config),
    then append new operations by re-sampling from ``cad.sampled_factories[k:]`` factory
    dicts (same topology as the original tail, new random ops).

    Returns a new ``CAD``; ``cad`` is not modified.
    """
    prefix, tail_dicts = cad_prefix_and_tail_factories(cad, k)
    planes = deepcopy(prefix.planes)
    face_planes = deepcopy(prefix.face_planes)
    cs: list[dict] = deepcopy(prefix.cs)
    sampled_out: list[dict[str, Any]] = deepcopy(prefix.sampled_factories or [])

    r_min, r_max = 0.01, 1.0
    for smp in tail_dicts:
        entry = sampled_factory_dict_to_entry(smp)
        _append_one_leaf_factory(
            entry,
            planes,
            face_planes,
            cs,
            r_min=r_min,
            r_max=r_max,
            world_size=cad.world_size,
            max_string_length=cad.max_string_length,
            use_literals=cad.use_literals,
            n_retries=n_retries,
            n_extrude_retries=n_extrude_retries,
            sampled_out=sampled_out,
        )

    return CAD(
        planes,
        face_planes,
        cs,
        sampled_out,
        cad.world_size,
        cad.max_string_length,
        cad.use_literals,
    )


def continue_from_prefix_state(
    planes: list[dict],
    face_planes: list[dict],
    cs_prefix: list[dict],
    sampled_prefix: list[dict[str, Any]] | None,
    tail_factory_dicts: list[dict[str, Any]],
    *,
    world_size: float,
    max_string_length: int,
    use_literals: bool = True,
    ratio: float = 0.5,
    n_retries: int = 30,
    n_extrude_retries: int = 30,
) -> CAD:
    """
    Same as :func:`continue_from_prefix` but accepts explicit prefix state and a
    custom tail (e.g. not read from the same ``CAD`` instance).

    ``planes`` / ``face_planes`` are compacted to match ``cs_prefix`` references
    (same as :func:`cad_prefix_and_tail_factories`).
    """
    planes = deepcopy(planes)
    face_planes = deepcopy(face_planes)
    cs = deepcopy(cs_prefix)
    _compact_prefix_planes_face_planes(cs, planes, face_planes)
    sampled_out: list[dict[str, Any]] = deepcopy(sampled_prefix or [])
    r_min, r_max = 0.01, 1.0
    for smp in tail_factory_dicts:
        entry = sampled_factory_dict_to_entry(smp)
        _append_one_leaf_factory(
            entry,
            planes,
            face_planes,
            cs,
            r_min=r_min,
            r_max=r_max,
            world_size=world_size * ratio,
            max_string_length=max_string_length,
            use_literals=use_literals,
            n_retries=n_retries,
            n_extrude_retries=n_extrude_retries,
            sampled_out=sampled_out,
        )
    return CAD(
        planes,
        face_planes,
        cs,
        sampled_out,
        world_size,
        max_string_length,
        use_literals,
    )


def finalize_cad_for_export(cad: CAD) -> None:
    """
    Round/fix iteration **without** bbox re-normalization via ``exec``.

    Matches the final step of :func:`continue_full_pipeline` (``finalize(do_normalize=False)``).
    Dataset CAD and prefix slices are already in training coordinates; calling full
    :meth:`CAD.finalize` on a **truncated** prefix changes literals (different bbox),
    which breaks stepgen-style ``before``/``after`` text alignment.
    """
    cad.finalize(do_normalize=False)


def continue_full_pipeline(
    cad: CAD,
    cad_factory: CADFactory,
    k: int,
    *,
    ratio: float = 0.5,
    unit_world_size: float = 2.0,
    max_attempts: int = 32,
    base_seed: int | None = None,
    callable_checks: list[Callable[..., Any]] | None = None,
    validation_file_path: str | None = None,
) -> CAD:
    """
    Continuation with the same steps as full generation quality control:

    1. Split at ``k``; normalize the **prefix** into a cube of side ``unit_world_size``
       (default ``2``, i.e. roughly ``[-1, 1]^3`` after centering), so new ops are
       sampled against a canonical small solid.
    2. Append the tail (same topology as ``cad.sampled_factories[k:]``) with
       ``world_size=ratio * cad.world_size`` passed into ``plane_logic`` (strong
       through-all / depth semantics vs the training ``world_size``).
    3. :meth:`CAD.finalize` (normalize / round / fix).
    4. :func:`validate_cad_continuation_mesh` — same checks as the dataset runner
       except the tight bbox vs ``world_size/2`` asserts.

    ``cad_factory`` supplies retry counts and string / literal settings; output
    target extent uses ``cad.world_size``.

    Retries on failure up to ``max_attempts`` with a varying RNG seed when
    ``base_seed`` is set.
    """
    prefix, tail_dicts = cad_prefix_and_tail_factories(cad, k)
    target_ws = cad.world_size
    tail_ws = ratio * target_ws
    r_min, r_max = 0.01, 1.0

    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        if base_seed is not None:
            np.random.seed((base_seed + attempt) & 0xFFFFFFFF)

        planes = deepcopy(prefix.planes)
        face_planes = deepcopy(prefix.face_planes)
        cs: list[dict] = deepcopy(prefix.cs)
        sampled_out: list[dict[str, Any]] = deepcopy(prefix.sampled_factories or [])

        if k > 0:
            work = CAD(
                planes,
                face_planes,
                cs,
                sampled_out,
                target_ws,
                cad.max_string_length,
                cad.use_literals,
            )
            # work.normalize(work.to_string())
            planes = work.planes
            face_planes = work.face_planes
            cs = work.cs
            sampled_out = work.sampled_factories or []

        try:
            for smp in tail_dicts:
                entry = sampled_factory_dict_to_entry(smp)
                _append_one_leaf_factory(
                    entry,
                    planes,
                    face_planes,
                    cs,
                    r_min=r_min,
                    r_max=r_max,
                    world_size=tail_ws,
                    max_string_length=cad_factory.max_string_length,
                    use_literals=cad_factory.use_literals,
                    n_retries=cad_factory.n_retries,
                    n_extrude_retries=cad_factory.n_extrude_retries,
                    sampled_out=sampled_out,
                )

            out = CAD(
                planes,
                face_planes,
                cs,
                sampled_out,
                target_ws,
                cad_factory.max_string_length,
                cad_factory.use_literals,
            )

            finalize_cad_for_export(out)
            validate_cad_continuation_mesh(
                out,
                single_solid_check=getattr(cad_factory, "single_solid_check", True),
                callable_checks=callable_checks,
                file_path=validation_file_path,
            )
            return out
        except BaseException as exc:
            last_exc = exc
            logger.debug(
                "continue_full_pipeline attempt %s/%s failed: %s",
                attempt + 1,
                max_attempts,
                exc,
                exc_info=True,
            )
            continue

    assert last_exc is not None
    raise RuntimeError(
        f"continue_full_pipeline failed after {max_attempts} attempts"
    ) from last_exc
