"""
Stepgen: checkpoint folders ``{index}_{m}``, chunk-aligned **regeneration**, and edit JSON.

For each chunk boundary index ``m`` in ``ends`` (exclusive end of GT prefix through the
checkpoint), GT is frozen as ``cs[:m]`` and the **following** chunk
``sampled_factories[m:next]`` is re-sampled via :func:`continuation.continue_from_prefix_state`,
where ``next`` is the next entry in ``ends`` or ``len(cs)`` for the last boundary.

Checkpoint folder ``{index}_{m}/`` contains:

* ``before.py``, ``before.stl``, ``before.step`` and ``after.py``, ``after.stl``, ``after.step``;
* ``{index}_{m}.json`` and ``{index}_{m}_factory.json`` (CAD + factories for the **after** state);
* ``edits.json`` (field ``kind``: add/remove/substitute).

Export uses :func:`continuation.finalize_cad_for_export`.
"""

from __future__ import annotations

import json
import random
import pickle
import os
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
# from cadquery_addons import *
from tqdm import tqdm

from .cad import CAD
from .continuation import (
    cad_prefix_and_tail_factories,
    continue_from_prefix_state,
    finalize_cad_for_export,
)
from .edit_diff import replace_edits
from .utils import compound_to_mesh

STEPGEN_REGEN_MAX_ATTEMPTS = 16


def _stepgen_emit_warning(debug_mode: bool, msg: str, *args: Any) -> None:
    """Same idea as generation: only ``print`` when ``debug_mode`` (no logger noise)."""
    if debug_mode:
        print(msg % args if args else msg, flush=True)


def _stepgen_emit_info(debug_mode: bool, msg: str, *args: Any) -> None:
    if debug_mode:
        print(msg % args if args else msg, flush=True)


class StepgenEditKind(str, Enum):
    ADD = "add"
    REMOVE = "remove"
    SUBSTITUTE = "substitute"


def _is_selected_face_plane(op: dict) -> bool:
    return op.get("type") == "SelectedFacePlane"


def cs_op_groups(cs: list[dict]) -> list[list[int]]:
    """
    Group ``cs`` indices: each group is any leading SelectedFacePlane rows plus the next non-SFP op.
    Trailing SFP-only rows form their own group (last chunk).
    """
    groups: list[list[int]] = []
    buf: list[int] = []
    for idx, op in enumerate(cs):
        if _is_selected_face_plane(op):
            buf.append(idx)
        else:
            groups.append(buf + [idx])
            buf = []
    if buf:
        groups.append(buf)
    return groups


def chunk_op_groups(groups: list[list[int]]) -> list[list[list[int]]]:
    """
    Pack groups into chunks of 1–3 groups; the last chunk may contain 1–2 groups
    (remaining groups are merged when ≤2).
    """
    if not groups:
        return []
    chunks: list[list[list[int]]] = []
    rem = groups[:]
    while rem:
        n = len(rem)
        if n <= 2:
            chunks.append(rem)
            break
        take = min(3, n - 2)
        if take < 1:
            take = 1
        chunks.append(rem[:take])
        rem = rem[take:]
    return chunks


def chunk_exclusive_end_indices(cs: list[dict]) -> list[int]:
    """Exclusive ``cs`` end index after each chunk (sorted ascending)."""
    groups = cs_op_groups(cs)
    chunks = chunk_op_groups(groups)
    ends: list[int] = []
    for ch in chunks:
        flat = [i for g in ch for i in g]
        ends.append(max(flat) + 1)
    return ends


def cad_at_prefix(cad: CAD, k: int) -> CAD:
    """Truncated CAD with ``cs[:k]`` and compacted planes (same as continuation prefix)."""
    prefix, _tail = cad_prefix_and_tail_factories(cad, k)
    return prefix


def splice_cad_drop_middle(cad: CAD, prefix_end: int, tail_start: int) -> CAD:
    """
    ``cs = cs[:prefix_end] + cs[tail_start:]`` (and same for ``sampled_factories``).
    Planes / face_planes are copied from ``cad``; caller should
    :func:`continuation.finalize_cad_for_export` before ``to_string`` / mesh export.
    """
    if cad.sampled_factories is None:
        raise ValueError("sampled_factories required")
    if not (0 <= prefix_end <= tail_start <= len(cad.cs)):
        raise IndexError(f"bad splice {prefix_end=} {tail_start=} len={len(cad.cs)}")
    if len(cad.sampled_factories) != len(cad.cs):
        raise ValueError("len(sampled_factories) must match len(cs)")
    new_cs = deepcopy(cad.cs[:prefix_end] + cad.cs[tail_start:])
    new_smp = deepcopy(
        cad.sampled_factories[:prefix_end] + cad.sampled_factories[tail_start:]
    )
    return CAD(
        planes=deepcopy(cad.planes),
        face_planes=deepcopy(cad.face_planes),
        cs=new_cs,
        sampled_factories=new_smp,
        world_size=cad.world_size,
        max_string_length=cad.max_string_length,
        use_literals=cad.use_literals,
    )


def regen_cad_one_chunk(
    cad: CAD,
    freeze_end: int,
    resample_end: int,
    *,
    regen_seed: int,
    max_attempts: int = STEPGEN_REGEN_MAX_ATTEMPTS,
    n_retries: int = 10,
    n_extrude_retries: int = 1,
    debug_mode: bool = False,
) -> CAD | None:
    """
    GT prefix ``cs[:freeze_end]`` fixed; re-sample leaf factory dicts
    ``sampled_factories[freeze_end:resample_end]`` (the chunk **after** boundary ``freeze_end``).

    Returns ``None`` if all attempts fail or length mismatch (output must have ``len(cs) ==
    resample_end``).
    """
    if cad.sampled_factories is None:
        return None
    if not (0 <= freeze_end < resample_end <= len(cad.cs)):
        return None
    tail_chunk = cad.sampled_factories[freeze_end:resample_end]
    if len(tail_chunk) != resample_end - freeze_end:
        return None

    prefix_cad = cad_at_prefix(cad, freeze_end)

    for attempt in range(max_attempts):
        np.random.seed((regen_seed + attempt) & 0xFFFFFFFF)
        try:
            out = continue_from_prefix_state(
                prefix_cad.planes,
                prefix_cad.face_planes,
                prefix_cad.cs,
                prefix_cad.sampled_factories,
                tail_chunk,
                world_size=cad.world_size,
                max_string_length=cad.max_string_length,
                use_literals=cad.use_literals,
                n_retries=n_retries,
                n_extrude_retries=n_extrude_retries,
            )
        except Exception as exc:
            if debug_mode:
                print(
                    f"stepgen regen attempt {attempt + 1}/{max_attempts} failed "
                    f"(freeze_end={freeze_end} resample_end={resample_end}): {exc!r}",
                    flush=True,
                )
            continue
        if len(out.cs) != resample_end:
            _stepgen_emit_warning(
                debug_mode,
                "stepgen regen length mismatch: got len(cs)=%s expected resample_end=%s "
                "(freeze_end=%s)",
                len(out.cs),
                resample_end,
                freeze_end,
            )
            continue
        return out

    return None


def save_stepgen_side_bundle(
    out_dir: Path,
    side: str,
    cad: CAD,
    *,
    callable_checks: list[Callable[..., Any]] | None,
    debug_mode: bool = False,
) -> None:
    """Write ``{side}.py``, ``{side}.stl``, ``{side}.step`` for ``side`` in ``before`` / ``after``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if side not in ("before", "after"):
        raise ValueError(f"side must be 'before' or 'after', got {side!r}")
    s = cad.to_string()
    mesh_path = out_dir / f"{side}.stl"
    py_path = out_dir / f"{side}.py"
    step_path = out_dir / f"{side}.step"

    try:
        exec(s, globals())
    except Exception as e:
        if debug_mode:
            print(e, s, flush=True)
        raise
    w = globals()["r"].val()
    assert w.isValid()
    mesh = compound_to_mesh(w)
    assert len(mesh.faces) > 2
    assert bool(mesh.volume > 0)
    assert bool(sum(mesh.extents == 0) == 0)

    if callable_checks:
        for fn in callable_checks:
            assert fn(mesh=mesh, code=s, file_path=str(py_path))

    mesh.export(str(mesh_path))
    w.export(str(step_path))
    with open(py_path, "w") as f:
        f.write(s)


def save_stepgen_checkpoint_cad_json(
    out_dir: Path,
    file_stem: str,
    cad: CAD,
) -> None:
    """Write ``{file_stem}.json`` and ``{file_stem}_factory.json`` (same layout as runner sample)."""
    factory_path = out_dir / f"{file_stem}_factory.json"
    json_path = out_dir / f"{file_stem}.json"
    with open(factory_path, "w") as f:
        json.dump(cad.sampled_factories, f, indent=4)
    with open(json_path, "w") as f:
        json.dump(cad.to_dict(), f, indent=4)


def write_edits_json(
    path: Path,
    *,
    kind: StepgenEditKind,
    sample_index: int,
    checkpoint_ops: int,
    prev_ops: int,
    before_py: str,
    after_py: str,
) -> None:
    """``prev_ops`` = freeze boundary ``m``; ``checkpoint_ops`` = exclusive end ``next`` of the re-sampled tail."""
    edits = replace_edits(before_py, after_py)
    payload = {
        "kind": kind.value,
        "sample_index": sample_index,
        "checkpoint_ops": checkpoint_ops,
        "prev_ops": prev_ops,
        "edits": [[a, b] for a, b in edits],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def process_dataset_sample(
    sample_parent: Path,
    index: int,
    *,
    callable_checks: list[Callable[..., Any]] | None,
    rng: random.Random,
    debug_mode: bool = False,
) -> None:
    """
    For one sample ``…/{index}/``, emit ``{index}_{m}/`` for each chunk boundary ``m`` in ``ends``.

    **Regeneration:** freeze GT as ``cs[:m]``; re-sample the **next** chunk
    ``sampled_factories[m:next]`` where ``next`` is the following ``ends`` entry or ``len(cs)``
    (:func:`continuation.continue_from_prefix_state`).

    **Artifacts** (see module docstring): ``before/after`` geometry triplets,
    ``{index}_{m}.json`` + ``_factory`` for the **after** CAD, ``edits.json``.

    * **add:** ``before`` = GT through ``m``, ``after`` = regen;
    * **remove:** swapped;
    * **substitute:** ``before`` = regen, ``after`` = GT through ``next``.
    """
    json_path = sample_parent / f"{index}.json"
    factory_path = sample_parent / f"{index}_factory.json"
    if not json_path.is_file() or not factory_path.is_file():
        _stepgen_emit_warning(debug_mode, "skip missing json/factory: %s", json_path)
        return

    with open(json_path) as f:
        entity = json.load(f)
    if entity.get("type") != "CAD":
        _stepgen_emit_warning(debug_mode, "skip non-CAD json: %s", json_path)
        return

    cad = CAD.from_dict(entity)
    with open(factory_path) as f:
        cad.sampled_factories = json.load(f)

    if cad.sampled_factories is None or len(cad.sampled_factories) != len(cad.cs):
        _stepgen_emit_warning(
            debug_mode, "skip bad sampled_factories length: %s", json_path
        )
        return

    ends = chunk_exclusive_end_indices(cad.cs)
    if not ends:
        _stepgen_emit_info(debug_mode, "skip sample %s: no chunk ends", json_path)
        return

    n_cs = len(cad.cs)

    for chunk_i, m in enumerate(ends):
        next_m = ends[chunk_i + 1] if chunk_i + 1 < len(ends) else n_cs
        if next_m <= m:
            continue

        sub_name = f"{index}_{m}"
        out_dir = sample_parent / sub_name
        out_dir.mkdir(parents=True, exist_ok=True)

        regen_seed = rng.randint(0, 0x7FFFFFFF)
        regen_cad = regen_cad_one_chunk(
            cad,
            m,
            next_m,
            regen_seed=regen_seed,
            debug_mode=debug_mode,
        )
        if regen_cad is None:
            _stepgen_emit_warning(
                debug_mode,
                "skip checkpoint %s: regeneration failed (freeze_end=%s resample_end=%s)",
                out_dir,
                m,
                next_m,
            )
            continue

        finalize_cad_for_export(regen_cad)

        cad_at_m = cad_at_prefix(cad, m)
        finalize_cad_for_export(cad_at_m)
        cad_at_next = cad_at_prefix(cad, next_m)
        finalize_cad_for_export(cad_at_next)

        kind = rng.choice(list(StepgenEditKind))
        if kind == StepgenEditKind.ADD:
            cad_before, cad_after = cad_at_m, regen_cad
        elif kind == StepgenEditKind.REMOVE:
            cad_before, cad_after = regen_cad, cad_at_m
        else:
            cad_before, cad_after = regen_cad, cad_at_next

        before_py = cad_before.to_string()
        after_py = cad_after.to_string()

        try:
            save_stepgen_side_bundle(
                out_dir,
                "before",
                cad_before,
                callable_checks=callable_checks,
                debug_mode=debug_mode,
            )
            save_stepgen_side_bundle(
                out_dir,
                "after",
                cad_after,
                callable_checks=callable_checks,
                debug_mode=debug_mode,
            )
        except Exception as exc:
            _stepgen_emit_warning(
                debug_mode, "skip checkpoint save %s: %s", out_dir, exc
            )
            continue

        save_stepgen_checkpoint_cad_json(out_dir, sub_name, cad_after)

        write_edits_json(
            out_dir / "edits.json",
            kind=kind,
            sample_index=index,
            checkpoint_ops=next_m,
            prev_ops=m,
            before_py=before_py,
            after_py=after_py,
        )


def iter_cad_samples(
    dataset: Path,
    *,
    global_rank: int | None = None,
) -> Iterable[tuple[Path, int]]:
    if global_rank is not None:
        world_size = int(
            os.environ.get(
                "WORLD_SIZE",
                os.environ.get(
                    "SLURM_NTASKS",
                    os.environ.get(
                        "OMPI_COMM_WORLD_SIZE", os.environ.get("PMI_SIZE", 1)
                    ),
                ),
            )
        )

    with open(dataset / "train.pkl", "rb") as f:
        samples = pickle.load(f)

    for i, sample in enumerate(samples):
        if global_rank is not None and i % world_size != global_rank:
            continue
        yield dataset / Path(sample["py_path"]).parent, Path(sample["py_path"]).stem
