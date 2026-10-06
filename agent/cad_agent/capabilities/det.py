"""Deterministic reconstruction: operation proposals without a VLM.

The `algo_rebuild` capability of the contract. Two modes, both needed by the scaffold:
from scratch (from the target alone) and **from the current prefix**, using the residual
geometry between the target and the already built prediction.

Dependencies come from the `vendor/` snapshot through `dsl_runtime`; an import failure
raises `DetDepsUnavailable` instead of silently turning into "no proposals".
`tools/preflight.py` checks that the run environment has them.
"""

from __future__ import annotations

import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from cad_agent import dsl_runtime
from cad_agent.capabilities.cache import CacheStats
from cad_agent.capabilities.voxel_lattice import voxelize_lattice

logger = logging.getLogger(__name__)

# Face cap of the residual mesh from which the snapshot fits operations.
# It determines not only the call cost but also the **length of the output**: the finer
# the residual mesh, the more complex the fitted profile and the longer the DSL line that
# det appends to the prefix. With a high cap det candidates came out tens of thousands of
# characters long, against hundreds for the model itself; the part prefix stayed that
# long until the end of the rollout and every following det call on it got much slower.
# Call time grows monotonically with prefix size.
#
# Our parts have 1-2 thousand faces, so a high cap almost never triggered.
RESIDUAL_MAX_FACES = 1500

# Seed of the global `np.random` for the duration of `det_rebuild` (see there).
DET_SEED = 0


class DetProposer:
    def __init__(self):
        # Dependencies come from vendor/ through dsl_runtime; an import failure
        # raises DetDepsUnavailable instead of silently turning into "no proposals".
        self._deps = dsl_runtime.import_det_deps()
        self._gt_cache: dict[str, trimesh.Trimesh] = {}
        # GT frame counter. It lives in the object, and the object survives a fork together
        # with its warmed-up contents, so the caller gets the per-call increment
        # (see `det_rebuild_isolated`).
        self._gt_stats = CacheStats()
        # A separate counter: GT repair has its own lifecycle. It happens in the fork and
        # survives the fork only through a file, so "took the ready one from disk" and
        # "repaired again" are different events and must not be measured by one field:
        # a merged outcome gives a plausible report with the wrong diagnosis.
        self._repair_stats = CacheStats()
        self._gt_frame_paths: dict[tuple[str, str], Path] = {}

    @staticmethod
    def _load_mesh(mesh_path: Path | str) -> trimesh.Trimesh:
        """Load a mesh **merging vertices**.

        `process=True` is mandatory here, not cosmetic. STL does not share vertices
        between triangles: each carries its own three. Without merging the mesh has no
        shared edge, so `is_watertight` is always `False` and `volume` is meaningless,
        however intact the geometry actually is.

        Consequence: GT went into `fill_holes()`, which doubled the faces and gave a
        double shell of zero volume, and a "non-watertight" prediction broke the boolean
        operations of `compute_residuals` on both sides, so the warm det branch returned
        nothing on **all** shapes after paying a long time per call.
        """
        mesh = trimesh.load(mesh_path, process=True)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        return mesh

    @staticmethod
    def _repaired_path(frame_path: Path) -> Path:
        """Where the repaired GT goes so that it survives a fork.

        Next to the frame, in the same part scratch dir: both are derived from one GT
        and live as long as one part.
        """
        return frame_path.with_name(f"{frame_path.stem}_repaired.stl")

    def _load_gt_repaired(self, gt_mesh_path: Path) -> trimesh.Trimesh:
        cache_key = str(gt_mesh_path)
        if cache_key in self._gt_cache:
            self._gt_stats.hit()
            return self._gt_cache[cache_key]

        self._gt_stats.miss()
        repaired_path = self._repaired_path(gt_mesh_path)
        mesh = self._load_repaired_from_disk(repaired_path)
        if mesh is None:
            mesh = self._repair_gt(gt_mesh_path, repaired_path)

        self._gt_cache.clear()
        self._gt_cache[cache_key] = mesh
        self._gt_stats.write()
        return mesh

    def _load_repaired_from_disk(self, repaired_path: Path) -> trimesh.Trimesh | None:
        """The ready repair from a previous fork, if present and usable.

        A miss is not counted here: there is one lookup, and recording it twice (once for
        an unusable file, once for the repair itself) would inflate the misses and make
        the hit rate unreadable. The miss is recorded by whoever actually pays, i.e.
        `_repair_gt`.

        Usability means the mesh is non-empty, not watertight: repair does not always
        reach closure, and requiring it here would redo hopeless work every time. The
        threshold is exactly "non-empty": a 50-face threshold copied from the acceptance
        condition of the voxel repair would reject a valid result, since a closed box has
        12 faces (caught by a test).
        """
        if not repaired_path.exists():
            return None
        try:
            mesh = self._load_mesh(repaired_path)
        except Exception:
            logger.debug("repaired GT is unreadable: %s", repaired_path, exc_info=True)
            return None
        if len(mesh.faces) < 4:
            return None
        self._repair_stats.hit()
        return mesh

    def _repair_gt(self, gt_mesh_path: Path, repaired_path: Path) -> trimesh.Trimesh:
        """Repair GT and, if repair was needed, leave the result on disk."""
        mesh = self._load_mesh(gt_mesh_path)
        if mesh.is_watertight:
            # Nothing to repair and nothing to write: the next call takes the same
            # frame, and reading it is cheap anyway.
            return mesh

        self._repair_stats.miss()
        try:
            mesh.fill_holes()
        except Exception:
            pass

        if not mesh.is_watertight:
            try:
                import pyvista as pv

                pitch = float(np.max(mesh.extents)) / 256.0
                vox = voxelize_lattice(mesh, pitch).fill()
                grid = pv.wrap(np.pad(vox.matrix.astype(np.float32), 1))
                surf = grid.contour([0.5])
                verts = np.asarray(surf.points) - 1.0
                verts = trimesh.transformations.transform_points(verts, vox.transform)
                repaired = trimesh.Trimesh(verts, surf.faces.reshape(-1, 4)[:, 1:], process=True)
                if len(repaired.faces) > 50:
                    mesh = repaired
            except Exception:
                pass

        # What follows is what later calls will read from disk, not the float64 from
        # memory: otherwise the first call on a part builds a different det.
        return self._persist_repaired(mesh, repaired_path) or mesh

    def _persist_repaired(self, mesh: trimesh.Trimesh, repaired_path: Path) -> trimesh.Trimesh | None:
        """Save the repair so that the next fork picks it up.

        Why: `warm_gt_frame` runs in a fork, and the repaired mesh died with it; for a
        non-watertight GT the 256^3 voxelization was paid in EVERY det call instead of once
        per part. The file survives the fork the same way the frame does.

        The write is atomic (tmp with pid + `os.replace`): det calls on one part come from
        different forks and may arrive here at the same time.

        The round trip is checked, not assumed. STL has no shared vertices, and closure
        holds only because `process=True` merges them back. If it does not,
        `compute_residuals` silently falls from the exact boolean back to the voxel
        fallback, so det gets worse and slower while the logs look like "the part is
        just like that". So an unusable file is removed at once and the next fork repairs
        honestly on its own.
        """
        watertight_before = bool(mesh.is_watertight)
        tmp_path = repaired_path.with_suffix(f".tmp.{os.getpid()}.stl")
        try:
            mesh.export(tmp_path)
            reloaded = self._load_mesh(tmp_path)
            if watertight_before and not reloaded.is_watertight:
                tmp_path.unlink(missing_ok=True)
                logger.warning(
                    "The repaired GT does not survive an STL round trip (watertightness is lost on "
                    "read) - keeping the repair in memory: %s", repaired_path.name,
                )
                return None
            os.replace(tmp_path, repaired_path)
            self._repair_stats.write()
            return reloaded
        except Exception:
            tmp_path.unlink(missing_ok=True)
            logger.debug("could not save the repaired GT: %s", repaired_path, exc_info=True)
            return None

    def _gt_frame_path(self, gt_mesh_path: Path, cache_dir: Path) -> Path:
        key = (str(gt_mesh_path.resolve()), str(cache_dir.resolve()))
        if key in self._gt_frame_paths and self._gt_frame_paths[key].exists():
            return self._gt_frame_paths[key]

        frame_path = gt_frame_path(gt_mesh_path, cache_dir)
        self._gt_frame_paths[key] = frame_path
        return frame_path

    def propose(
        self,
        gt_mesh_path: Path,
        cur_mesh_path: Path | None,
        cache_dir: Path,
        deadline: float | None = None,
        max_faces: int = RESIDUAL_MAX_FACES,
    ) -> list[str]:
        """The whole ranked list of operations, not the first k.

        There is deliberately no slice here. `k` used to cut the output in three places at
        once, and a repeat call from the same parent returned exactly the same top of the
        list: resampling the deterministic branch gave nothing because there was nothing
        new to give. Now the pool is computed in full and handed out by cursor to whoever
        ordered it (`Resources.algo_rebuild`).

        The length is bounded by the snapshot itself: `cadfit_single_pass` takes at most
        `max_candidates = 110` candidates per pass, and only those that gained at least
        `min_gain` IoU reach `kept`. The warm mode makes two passes (add and subtract), so
        the pool ceiling is hundreds of DSL lines, not "as many as are found".
        """
        gt_frame_path = self._gt_frame_path(gt_mesh_path, cache_dir)
        gt = self._load_gt_repaired(gt_frame_path)
        ops: list[str] = []

        if cur_mesh_path is None:
            ops.extend(self._propose_cold(gt, deadline))
        else:
            ops.extend(self._propose_warm(gt, Path(cur_mesh_path), deadline, max_faces))

        return ops

    def _propose_cold(
        self, gt: trimesh.Trimesh, deadline: float | None = None
    ) -> list[str]:
        deps = self._deps
        ops: list[str] = []

        # The occupancy grid and the GT occupancy are computed ONCE per call and passed
        # down. They used to be computed here for `best_primitive` and thrown away, and
        # `cadfit_single_pass` recomputed the occupancy of the same mesh on the same grid
        # three more times: at the primitive stage, the polygon stage and in
        # DOMINANT-BASE COLLAPSE.
        #
        # The cost of the duplicate is real. The occupancy grid at pitch=2 on a part
        # normalized to extent 200 has tens to hundreds of thousands of nodes, and every
        # node goes through `mesh.contains()`. That is cheap with embree and much more
        # expensive without it. Multiplying it by four is pointless either way, and we do
        # not want to depend on embree being present in the environment.
        points = None
        occupancy = None
        try:
            points = deps.grid(gt, 2.0)
            occupancy = deps.occupancy(gt, points, 2.0)
        except Exception:
            logger.debug("det cold occupancy failed", exc_info=True)
            points = None
            occupancy = None

        try:
            if occupancy is not None and len(gt.faces) <= 12000:
                revolve = deps.best_primitive(gt, points, occupancy, 2.0, kinds=("revolve",))
                if revolve is not None and revolve[1] >= 0.5:
                    line = deps.revolve_op(revolve[0], True)
                    if line:
                        ops.append(line)
        except Exception:
            logger.debug("det cold revolve primitive failed", exc_info=True)

        try:
            kept, _iou, _ = deps.cadfit_single_pass(
                gt,
                pitch=2.0,
                min_gain=0.004,
                verbose=False,
                # The same grid and occupancy: the snapshot accepts them for exactly this
                # ("avoids recomputing the GT contains() in every stage"). They must be
                # computed on THIS mesh and this pitch.
                grid_pts=points,
                gt_occ=occupancy,
                deadline=deadline,
            )
            first = not ops
            for candidate in kept:
                line = deps.extrude_op(candidate, first and not ops)
                if line:
                    ops.append(line)
        except Exception:
            logger.debug("det cold cadfit pass failed", exc_info=True)

        return ops

    def _propose_warm(
        self, gt: trimesh.Trimesh, cur_mesh_path: Path, deadline: float | None = None,
        max_faces: int = RESIDUAL_MAX_FACES,
    ) -> list[str]:
        deps = self._deps
        ops: list[str] = []
        cur = self._load_mesh(cur_mesh_path)
        try:
            residuals = deps.compute_residuals(cur, gt)
        except Exception:
            logger.debug("det residual computation failed", exc_info=True)
            residuals = None

        if residuals is None:
            return ops

        for side, residual_mesh in (("union", residuals.add_mesh), ("cut", residuals.cut_mesh)):
            if residual_mesh is None or len(residual_mesh.faces) == 0:
                continue

            residual_mesh = self._decimate_residual(residual_mesh, max_faces)
            try:
                if abs(float(residual_mesh.volume)) < 1e-4 * abs(float(gt.volume)):
                    continue
            except Exception:
                pass

            try:
            # Nothing to reuse here: each side is its own residual mesh with its own
            # bounds. Only the time cap is passed.
                kept, _iou, _ = deps.cadfit_single_pass(
                    residual_mesh, pitch=2.0, min_gain=0.004, verbose=False, deadline=deadline
                )
            except Exception:
                logger.debug("det warm cadfit pass failed (side=%s)", side, exc_info=True)
                kept = []

            for candidate in kept:
                line = deps.extrude_op(candidate, False) if side == "union" else deps.cut_op(candidate)
                if line:
                    ops.append(line)

        return ops

    @staticmethod
    def _decimate_residual(
        residual_mesh: trimesh.Trimesh, max_faces: int = RESIDUAL_MAX_FACES
    ) -> trimesh.Trimesh:
        if max_faces <= 0 or len(residual_mesh.faces) <= max_faces:
            return residual_mesh

        try:
            import pyvista as pv

            faces = np.hstack([np.full((len(residual_mesh.faces), 1), 3), residual_mesh.faces]).astype(np.int64)
            poly_data = pv.PolyData(np.asarray(residual_mesh.vertices), faces).decimate(
                1.0 - float(max_faces) / len(residual_mesh.faces)
            )
            decimated = trimesh.Trimesh(
                np.asarray(poly_data.points),
                poly_data.faces.reshape(-1, 4)[:, 1:],
                process=True,
            )
            if len(decimated.faces) > 50:
                return decimated
        except Exception:
            pass

        return residual_mesh


_DET_PROPOSER: DetProposer | None = None


def gt_frame_path(gt_mesh_path: str | Path, cache_dir: str | Path) -> Path:
    """GT in the model frame: bbox center at zero, largest extent 200.

    This is the inverse of the contract normalization (`metrics.normalize_mesh`): there the
    prediction is divided by 200 and shifted by 0.5, while GT is centered and divided by its
    own extent. A body built in this frame lands exactly on GT after normalization, while a
    body in raw GT coordinates collapses to a point, so everyone who builds code from the
    target needs the frame, not only det.

    A module function rather than a `DetProposer` method: the constructor pulls in the det
    dependencies, which other callers do not need. The file is written atomically, since
    several forks may write it into one scratch dir.
    """
    gt_mesh_path, cache_dir = Path(gt_mesh_path), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame_path = cache_dir / f"{gt_mesh_path.stem}_gt_frame.stl"
    if not frame_path.exists():
        mesh = DetProposer._load_mesh(gt_mesh_path)
        mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)
        extent = float(np.max(mesh.extents))
        if extent > 1e-7:
            mesh.apply_scale(200.0 / extent)
        tmp_path = frame_path.with_suffix(f".tmp.{os.getpid()}.stl")
        mesh.export(tmp_path)
        os.replace(tmp_path, frame_path)
    return frame_path


def _get_det_proposer() -> DetProposer:
    global _DET_PROPOSER
    if _DET_PROPOSER is None:
        _DET_PROPOSER = DetProposer()
    return _DET_PROPOSER




def warm_gt_frame(gt_mesh_path: str | Path, cache_dir: str | Path) -> None:
    """Warm the heavy GT state in the current process.

    The point is that `det_rebuild` is later called in a forked process. What is warmed
    here is free for every call, but **only through disk**: the warm-up itself also runs in
    a fork, and nobody inherits its memory. So both heavy quantities are written as files
    into the part scratch dir:

    - `_gt_frame_path`: the normalized GT frame;
    - `_load_gt_repaired`: the repair of a truly holey GT (256^3 voxelization, seconds).
      It is written to `<frame>_repaired.stl` and the next fork takes it from there; the
      STL round trip is checked for closure, and an unusable file is removed (see
      `_persist_repaired`).

    "Truly": the repair branch now triggers only on really open geometry (the loader
    merges vertices, so watertight is meaningful).
    """
    proposer = _get_det_proposer()
    frame_path = proposer._gt_frame_path(Path(gt_mesh_path), Path(cache_dir))
    proposer._load_gt_repaired(frame_path)


def det_rebuild(
    gt_mesh_path: str | Path,
    pred_mesh_path: str | Path | None,
    cache_dir: str | Path,
    deadline: float | None = None,
    max_faces: int = RESIDUAL_MAX_FACES,
) -> list[str]:
    """Ranked pool of DSL operations: from scratch (`pred_mesh_path=None`) or from a prefix.

    The single entry point for the scaffold: it picks the mode by whether it passes the
    current prediction, and knows nothing about the detectors' internals.

    `deadline` is an absolute time (epoch, like `time.time()`) after which the snapshot
    loops stop and collect what they already have.

    `max_faces` is the face cap of the residual mesh (see :data:`RESIDUAL_MAX_FACES`).
    A run knob: it is about call cost and output length.

    The global `np.random` is seeded with `DET_SEED` for the duration of the call and then
    restored: `trimesh.contains_points` rechecks doubtful points with a ray in direction
    `np.random.random(3)`, and on a dirty residual mesh the warm output depended on the
    state of the worker's generator without a seed.
    """
    # The frame and the GT repair come before seeding: the first call on a part builds them
    # here, later ones read them from disk, and seeding after them gives det one generator state.
    warm_gt_frame(gt_mesh_path, cache_dir)
    state = np.random.get_state()
    np.random.seed(DET_SEED)
    try:
        return _get_det_proposer().propose(
            gt_mesh_path=Path(gt_mesh_path),
            cur_mesh_path=Path(pred_mesh_path) if pred_mesh_path is not None else None,
            cache_dir=Path(cache_dir),
            deadline=deadline,
            max_faces=max_faces,
        )
    finally:
        np.random.set_state(state)


# The fraction of the hard timeout after which the snapshot starts winding down on its own.
# The rest is for assembling and returning the result: a soft cap makes sense only if there
# is time left between it and SIGKILL to hand something back.
SOFT_DEADLINE_FRACTION = 0.8


def det_rebuild_isolated(
    gt_mesh_path: str,
    pred_mesh_path: str | None,
    cache_dir: str,
    timeout_sec: float | None = None,
    max_faces: int = RESIDUAL_MAX_FACES,
) -> dict[str, Any]:
    """The same, but catching errors, for running in a separate process.

    `timeout_sec` is the hard timeout with which the parent will kill this fork. A **soft**
    cap is derived from it: `cadfit_single_pass` accepts `deadline` and can stop its loops
    and collect a partial result. The difference matters: a hard timeout hands up nothing,
    a soft one hands up what has been found so far.

    There is deliberately **no** OCC thread pool here. det does not touch OpenCASCADE at
    all (`cadfit_pass.py`: "Pure trimesh/shapely/numpy (no cadquery needed)"); CadQuery
    lives in this branch only as the text of generated operations. A limiter would limit a
    pool that does not exist in the process.
    """
    from cad_agent.capabilities.execute import own_threads  # noqa: PLC0415

    proposer = _get_det_proposer()
    # The increment for this call, not the accumulated counter: a fork inherits the state
    # warmed by the parent together with its counters.
    before = (proposer._gt_stats.hits, proposer._gt_stats.misses)
    # Repair is counted separately from the frame: it survives a fork only as a file,
    # and its hit is exactly the answer to "did it survive".
    before_repair = (
        proposer._repair_stats.hits, proposer._repair_stats.misses, proposer._repair_stats.writes,
    )
    started = time.monotonic()
    deadline = (
        time.time() + timeout_sec * SOFT_DEADLINE_FRACTION
        if timeout_sec is not None and timeout_sec > 0
        else None
    )
    try:
        ops = det_rebuild(gt_mesh_path, pred_mesh_path, cache_dir, deadline, max_faces)
        error = None
    except Exception:
        ops, error = [], traceback.format_exc()
    # The fork counts its own threads, exactly like the execution fork. The background
    # sampler runs every few seconds while det calls take seconds, so it rarely catches
    # them and det's contribution to the peak stays invisible.
    return {
        "success": error is None,
        "ops": ops,
        "wall_sec": time.monotonic() - started,
        "error": error,
        "n_threads": own_threads(),
        "gt_cache": {
            "hits": proposer._gt_stats.hits - before[0],
            "misses": proposer._gt_stats.misses - before[1],
        },
        "gt_repair_cache": {
            "hits": proposer._repair_stats.hits - before_repair[0],
            "misses": proposer._repair_stats.misses - before_repair[1],
            "writes": proposer._repair_stats.writes - before_repair[2],
        },
    }
