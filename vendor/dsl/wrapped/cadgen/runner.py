import hashlib
import logging
import os
import pickle
import queue
import random
import traceback
from collections import defaultdict
from glob import glob
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Callable

# inner_q markers from run_stepgen_sample_with_timeout → stepgen_worker_fn
_STEPGEN_MSG_OK = 1
_STEPGEN_MSG_TIMEOUT = "stepgen_timeout"
_STEPGEN_MSG_ERROR = "stepgen_error"

import numpy as np
from cadlib.methods.math.stats.py_stl_stats import py_stl_stats  # type: ignore
from cadlib.methods.rendering.render_collage import (  # type: ignore
    CollageConfig,
    render_collage_with_split,
)
from natsort import natsorted
from tqdm import tqdm

# from cadquery_addons import *

from .cad import CAD, CADFactory
from .thread_limits import apply_occt_thread_limit, apply_worker_thread_limits
from .scans import make_defects
from .utils import compound_to_mesh, shape_to_volume

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def run_single(index, path, cad_factory):
    apply_worker_thread_limits()
    apply_occt_thread_limit()
    np.random.seed(int(str(hash(str(os.getpid()) + str(index) + str(path)))[1:10]))
    try:
        cad = cad_factory.generate()
        cad.finalize(skip_fix=getattr(cad_factory, "skip_fix", False))
        s = cad.to_string()
        occ_core_volume = shape_to_volume(cad.to_shape())
        exec(s, globals())
        w = globals()["r"].val()
        cq_volume = w.Volume()
        bbox = w.BoundingBox()
        bbox = (bbox.xmin, bbox.ymin, bbox.zmin, bbox.xmax, bbox.ymax, bbox.zmax)
        size = cad_factory.world_size / 2
        assert abs(np.min(bbox) + size) < 1.5 and abs(np.max(bbox) - size) < 1.5
        assert np.isclose(cq_volume, occ_core_volume, rtol=0.02)

        mesh = compound_to_mesh(w)
        assert len(mesh.faces) > 2
        dir_entries = [
            entry
            for entry in os.listdir(path)
            if os.path.isdir(os.path.join(path, entry))
        ]
        if dir_entries:
            slot = dir_entries[index % len(dir_entries)]
            path = os.path.join(path, slot)
        mesh.export(os.path.join(path, f"{index}.stl"))
        with open(os.path.join(path, f"{index}.py"), "w") as f:
            f.write(s)
    except Exception:
        # todo: all but AssertionError should be debugged
        pass


def run_single_no_occ_validation(
    index,
    path,
    cad_factory,
    inner_q,
    callable_checks=None,
    debug_mode: bool = False,
):
    apply_worker_thread_limits()
    apply_occt_thread_limit()
    np.random.seed(int(str(hash(str(os.getpid()) + str(index) + str(path)))[1:10]))
    log_exceptions = getattr(cad_factory, "log_exceptions", False)
    s_if_exc = ""
    try:
        cad = cad_factory.generate()
        cad.finalize(skip_fix=getattr(cad_factory, "skip_fix", False))
        s = cad.to_string()
        s_if_exc = s

        try:
            exec(s, globals())
        except Exception as e:
            if log_exceptions:
                print(e, s, flush=True)
        w = globals()["r"].val()
        assert w.isValid()
        bbox = w.BoundingBox()
        bbox = (bbox.xmin, bbox.ymin, bbox.zmin, bbox.xmax, bbox.ymax, bbox.zmax)

        size = cad_factory.world_size / 2
        assert abs(np.min(bbox) + size) < 1.5 and abs(np.max(bbox) - size) < 1.5

        mesh = compound_to_mesh(w)
        assert len(mesh.faces) > 2

        check_watertight = True
        for op in cad.cs:
            if op["type"] == "Revolve":
                revolve = op["op"]
                if revolve.zero_dist_to_axis:
                    check_watertight = False
                    break
        if check_watertight:
            assert mesh.is_watertight
        assert not mesh.is_empty

        if getattr(cad_factory, "single_solid_check", True):
            assert len(mesh.split()) == 1
        assert bool(mesh.volume > 0)
        assert bool(sum(mesh.extents == 0) == 0)

        dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        if len(dirs) > 0:
            path = os.path.join(path, str(index % len(dirs)))
        os.makedirs(os.path.join(path, str(index)), exist_ok=True)
        mesh_path = os.path.join(path, str(index), f"{index}.stl")
        py_path = os.path.join(path, str(index), f"{index}.py")

        tmp_prefix = os.path.join(path, str(index), f"{index}.tmp.{os.getpid()}")
        tmp_mesh_path = f"{tmp_prefix}.stl"
        tmp_py_path = f"{tmp_prefix}.py"
        tmp_paths = [
            tmp_mesh_path,
            tmp_py_path,
        ]

        with open(tmp_py_path, "w") as f:
            f.write(s)
        mesh.export(tmp_mesh_path)

        os.replace(tmp_py_path, py_path)
        os.replace(tmp_mesh_path, mesh_path)

        if callable_checks is not None:
            for callable_check in callable_checks:
                assert callable_check(mesh=mesh, code=s, file_path=py_path)
        # print(s)
        inner_q.put(1)
    except Exception as e:
        if log_exceptions:
            print(f"Sample {index} failed: {type(e).__name__}: {e}", flush=True)
            logger.exception("Sample %s failed", index)
        for tmp_path in locals().get("tmp_paths", []):
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
        # todo: all but AssertionError should be debugged
        if debug_mode:
            if log_exceptions:
                print(f"Failed code: {s_if_exc}")
            raise e
        pass
        # TODO add retry?
        # run_single_no_occ_validation(index, path, cad_factory)


def validate_cad_continuation_mesh(
    cad: CAD,
    *,
    single_solid_check: bool = True,
    callable_checks: list[Callable] | None = None,
    file_path: str | None = None,
) -> None:
    """
    Same mesh checks as :func:`run_single_no_occ_validation` after ``exec``,
    but **without** the strict bounding-box vs ``world_size / 2`` asserts, so
    continuation / intermediate geometry can be validated before a final
    canonical fit.
    """
    apply_worker_thread_limits()
    apply_occt_thread_limit()
    s = cad.to_string()
    try:
        exec(s, globals())
    except Exception as e:
        print(e, s, flush=True)
        raise
    w = globals()["r"].val()
    assert w.isValid()

    mesh = compound_to_mesh(w)
    assert len(mesh.faces) > 2

    check_watertight = True
    for op in cad.cs:
        if op["type"] == "Revolve":
            revolve = op["op"]
            if revolve.zero_dist_to_axis:
                check_watertight = False
                break
    if check_watertight:
        assert mesh.is_watertight
    assert not mesh.is_empty

    if single_solid_check:
        assert len(mesh.split()) == 1
    assert bool(mesh.volume > 0)
    assert bool(sum(mesh.extents == 0) == 0)

    if callable_checks is not None:
        for callable_check in callable_checks:
            assert callable_check(mesh=mesh, code=s, file_path=file_path)


def run_single_no_occ_validation_with_timeout(
    index,
    path,
    cad_factory,
    timeout,
    inner_q,
    callable_checks=None,
    debug_mode=False,
):
    apply_worker_thread_limits()
    process = Process(
        target=run_single_no_occ_validation,
        args=(
            index,
            path,
            cad_factory,
            inner_q,
            callable_checks,
            debug_mode,
        ),
    )

    try:
        process.start()
        process.join(timeout)

        if process.is_alive():
            process.terminate()  # Kill the process
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join()
            raise RuntimeError("Execution timed out")

    except Exception:
        if process.is_alive():
            process.kill()
            process.join()
            raise RuntimeError("Execution timed out")


def worker_fn(
    split_path,
    cad_factory,
    timeout,
    input_q,
    res_q,
    callable_checks=None,
    while_successful: bool = True,
    debug_mode: bool = False,
):
    apply_worker_thread_limits()
    while True:
        try:
            index = input_q.get(timeout=5)
            inner_q = Queue()
            process = Process(
                target=run_single_no_occ_validation_with_timeout,
                args=(
                    index,
                    split_path,
                    cad_factory,
                    timeout,
                    inner_q,
                    callable_checks,
                    debug_mode,
                ),
            )
            process.start()
            process.join()

            try:
                inner_q.get(timeout=1)
                res_q.put(1)
            except queue.Empty:
                if not while_successful:
                    res_q.put(0)
                    continue
                input_q.put(index)
        except queue.Empty:
            return


def run_split(
    output: str,
    n_folders: int | None,
    n_samples: int,
    split_name: str | None,
    cad_factory: CADFactory,
    start_from: int = 0,
    callable_checks: list[Callable] | None = None,
    save_annotations: bool = True,
    while_successful: bool = True,
    calculate_stats: bool = False,
    render: bool = True,
    debug_mode: bool = False,
    max_workers: int | None = None,
) -> None:
    if split_name is not None:
        split_path = os.path.join(output, split_name)
    else:
        save_annotations = False
        split_path = output

    print(split_path)
    os.makedirs(split_path, exist_ok=True)
    if n_folders is not None:
        for i in range(n_folders):
            os.makedirs(os.path.join(split_path, str(i)), exist_ok=True)

    samples_range = list(range(n_samples))
    samples_range = samples_range[start_from:]

    input_q = Queue()
    res_q = Queue()

    for i in samples_range:
        input_q.put(i)
    logger.info("Queue in filled. Starting.")
    cpu_number = os.cpu_count() if max_workers is None else max_workers
    if cpu_number is None:
        cpu_number = 1
    if cpu_number < 1:
        raise ValueError("max_workers must be >= 1")
    workers = [
        Process(
            target=worker_fn,
            args=(
                split_path,
                cad_factory,
                600,
                input_q,
                res_q,
                callable_checks,
                while_successful,
                debug_mode,
            ),
        )
        for i in range(cpu_number)
    ]
    for worker in workers:
        worker.start()

    for _ in tqdm(range(len(samples_range))):
        res_q.get()

    for worker in workers:
        worker.join(timeout=1)

    if save_annotations:
        assert split_name, "split_name is required to save annotations"
        annotations = list()
        for py_path in (
            glob(f"{split_path}/*/*/*.py")
            if n_folders is not None
            else glob(f"{split_path}/*/*.py")
        ):
            annotations.append(
                dict(
                    py_path=py_path[len(output) + 1 :],
                    mesh_path=py_path[len(output) + 1 : -3] + ".stl",
                )
            )
        with open(os.path.join(output, f"{split_name}.pkl"), "wb") as f:
            pickle.dump(annotations, f)
        logger.info(f"{split_name} {len(annotations)}")

    if calculate_stats:
        py_stl_stats(
            Path(split_path),
            output=Path(output),
            output_prefix=Path(split_path).stem,
            search_subfolders=True,
        )

    if render:
        render_collage_with_split(
            natsorted(
                list(
                    glob(f"{split_path}/*/*.stl")
                    if n_folders is not None
                    else glob(f"{split_path}/*.stl")
                )
            ),
            CollageConfig(
                output_dir=Path(output),
                output_name=Path(split_path).stem,
                shuffle=False,
            ),
        )


def run(
    output: str,
    cad_factory: CADFactory,
    n_train_samples: int,
    n_folders: int | None = None,
    n_val_samples: int = 0,
    start_from: int = 0,
    callable_checks: list[Callable] | None = None,
    save_annotations: bool = True,
    while_successful: bool = True,
    split_train_name: str | None = "train",
    split_val_name: str | None = "val",
    calculate_stats: bool = False,
    render: bool = True,
    with_defects: bool = False,
    debug_mode: bool = False,
    max_workers: int | None = None,
):
    # -> 1 json file per split
    # -> 2 files per sample: mesh, py
    os.makedirs(output, exist_ok=True)
    run_split(
        output,
        n_folders,
        n_train_samples,
        split_train_name,
        cad_factory,
        start_from=start_from,
        callable_checks=callable_checks,
        save_annotations=save_annotations,
        while_successful=while_successful,
        calculate_stats=calculate_stats,
        render=render,
        debug_mode=debug_mode,
        max_workers=max_workers,
    )
    if with_defects:
        make_defects(
            output,
            split_name=split_train_name,
            n_folders=n_folders,
            calculate_stats=calculate_stats,
            render=render,
            save_annotations=save_annotations,
        )
    if n_val_samples:
        run_split(
            output,
            1,
            n_val_samples,
            split_val_name,
            cad_factory,
            start_from=0,
            callable_checks=callable_checks,
            save_annotations=save_annotations,
            while_successful=while_successful,
            calculate_stats=calculate_stats,
            render=render,
            max_workers=max_workers,
        )
        if with_defects:
            make_defects(
                output,
                split_name=split_val_name,
                n_folders=None,
                calculate_stats=calculate_stats,
                render=render,
                save_annotations=save_annotations,
            )


def run_stepgen_sample_with_timeout(
    sample_parent_s: str,
    index: int,
    callable_checks: list[Callable] | None,
    timeout: float,
    inner_q: Queue,
    debug_mode: bool = False,
) -> None:
    """Run stepgen in a nested child; puts ``_STEPGEN_MSG_OK`` / timeout / error on ``inner_q``."""
    apply_worker_thread_limits()
    worker_process = Process(
        target=_stepgen_process_dataset_sample_task,
        args=(
            sample_parent_s,
            index,
            callable_checks,
            inner_q,
            debug_mode,
        ),
    )

    try:
        worker_process.start()
        worker_process.join(timeout)

        if worker_process.is_alive():
            worker_process.terminate()
            worker_process.join(timeout=1)
            if worker_process.is_alive():
                worker_process.kill()
                worker_process.join()
            inner_q.put(_STEPGEN_MSG_TIMEOUT)
            return

    except Exception as e:
        if debug_mode:
            print(
                f"stepgen run_stepgen_sample_with_timeout wrapper failed: {e!r}",
                flush=True,
            )
            traceback.print_exc()
        if worker_process.is_alive():
            worker_process.kill()
            worker_process.join()
        inner_q.put(_STEPGEN_MSG_ERROR)
        return

    try:
        msg = inner_q.get_nowait()
    except queue.Empty:
        inner_q.put(_STEPGEN_MSG_ERROR)
        return
    if msg == _STEPGEN_MSG_OK:
        inner_q.put(msg)
    else:
        inner_q.put(_STEPGEN_MSG_ERROR)


def _stepgen_process_dataset_sample_task(
    sample_parent_s: str,
    index: int,
    callable_checks: list[Callable] | None,
    inner_q: Queue,
    debug_mode: bool = False,
) -> None:
    apply_worker_thread_limits()
    apply_occt_thread_limit()
    from pathlib import Path

    from .stepgen import process_dataset_sample

    sample_parent = Path(sample_parent_s)
    rng = _stepgen_rng(sample_parent, index)
    try:
        process_dataset_sample(
            sample_parent,
            index,
            callable_checks=callable_checks,
            rng=rng,
            debug_mode=debug_mode,
        )
        inner_q.put(1)
    except Exception as e:
        if debug_mode:
            print(f"Failed stepgen sample {sample_parent}/{index}: {e!r}", flush=True)
            raise e
        pass


def _stepgen_rng(sample_parent: Path, index: int) -> random.Random:
    key = f"{os.getpid()}:{sample_parent}:{index}".encode()
    h = hashlib.sha256(key).hexdigest()
    return random.Random(int(h[:12], 16))


def stepgen_worker_fn(
    callable_checks: list[Callable] | None,
    timeout: float,
    input_q: Queue,
    res_q: Queue,
    while_successful: bool = True,
    max_retries: int = 5,
    debug_mode: bool = False,
) -> None:
    """Timeout never re-queues; other failures retry up to ``max_retries`` times if ``while_successful``."""
    apply_worker_thread_limits()
    requeues_done: defaultdict[tuple[str, int], int] = defaultdict(int)

    while True:
        try:
            sample_parent_s, index = input_q.get(timeout=5)
        except queue.Empty:
            return
        key = (sample_parent_s, index)
        inner_q: Queue = Queue()
        process = Process(
            target=run_stepgen_sample_with_timeout,
            args=(
                sample_parent_s,
                index,
                callable_checks,
                timeout,
                inner_q,
                debug_mode,
            ),
        )
        process.start()
        process.join()

        try:
            msg = inner_q.get(timeout=2)
        except queue.Empty:
            msg = _STEPGEN_MSG_ERROR

        if msg == _STEPGEN_MSG_OK:
            res_q.put(1)
            requeues_done.pop(key, None)
        elif msg == _STEPGEN_MSG_TIMEOUT:
            if debug_mode:
                print(
                    f"stepgen timeout for sample {sample_parent_s}/{index} (not re-queued)",
                    flush=True,
                )
            res_q.put(0)
            requeues_done.pop(key, None)
        elif (
            while_successful
            and msg == _STEPGEN_MSG_ERROR
            and requeues_done[key] < max_retries
        ):
            requeues_done[key] += 1
            input_q.put((sample_parent_s, index))
        else:
            if while_successful and msg == _STEPGEN_MSG_ERROR and debug_mode:
                print(
                    f"stepgen giving up on sample {sample_parent_s}/{index} "
                    f"after {requeues_done[key] + 1} failed run(s)",
                    flush=True,
                )
            res_q.put(0)
            requeues_done.pop(key, None)


def run_stepgen(
    dataset: str | Path,
    *,
    callable_checks: list[Callable] | None = None,
    timeout: float = 600,
    while_successful: bool = True,
    max_retries: int = 5,
    debug_mode: bool = False,
    max_sample_scan_depth: int | None = None,
    save_annotations: bool = True,
    global_rank: int | None = None,
) -> None:
    """
    Walk an existing dataset tree; emit per-sample checkpoint folders ``{index}_{m}/`` with
    ``edits.json`` (see :mod:`cadgen.stepgen`).

    Each sample runs in a nested child process with ``timeout`` seconds (same pattern as
    training generation). On **timeout** the sample is counted as failed and is **not**
    re-queued. On other failures, when ``while_successful`` is True, the sample is retried at
    most ``max_retries`` additional times (``0`` means give up after the first failure).

    With ``debug_mode`` False (default), failures are swallowed inside workers like
    :func:`run_single_no_occ_validation` — no ``print``/traceback; only ``tqdm`` is visible.
    With ``debug_mode`` True, failures ``print`` diagnostics and the inner sample task
    **re-raises** so tracebacks match the generation pipeline.

    ``max_sample_scan_depth`` is passed to :func:`cadgen.stepgen.iter_cad_samples` to skip
    sample roots deeper than that many path components under ``dataset`` (``None`` = unlimited).
    """
    from .stepgen import iter_cad_samples

    dataset_path = Path(dataset).resolve()
    tasks = list[tuple[Path, int]](
        iter_cad_samples(dataset_path, global_rank=global_rank)
    )
    if debug_mode:
        print(f"stepgen: {len(tasks)} CAD samples under {dataset_path}", flush=True)
    if not tasks:
        return

    input_q: Queue = Queue()
    res_q: Queue = Queue()
    for parent, idx in tasks:
        input_q.put((str(parent), idx))

    cpu_number = os.cpu_count() or 1
    workers = [
        Process(
            target=stepgen_worker_fn,
            args=(
                callable_checks,
                timeout,
                input_q,
                res_q,
                while_successful,
                max_retries,
                debug_mode,
            ),
        )
        for _ in range(cpu_number)
    ]
    for w in workers:
        w.start()

    ok = 0
    for _ in tqdm(range(len(tasks)), desc="stepgen"):
        ok += int(res_q.get())

    for w in workers:
        w.join(timeout=2)

    if save_annotations:
        annotations = list()
        with open(dataset_path / "train.pkl", "rb") as f:
            samples = pickle.load(f)
        annotations.extend(samples)

        for py_path in dataset_path.rglob("**/before.py"):
            annotations.append(
                dict(
                    py_path=os.path.relpath(py_path, dataset_path),
                    mesh_path=os.path.relpath(
                        py_path.with_suffix(".stl"), dataset_path
                    ),
                )
            )
            if os.path.exists(py_path.parent / "after.py"):
                annotations.append(
                    dict(
                        py_path=os.path.relpath(
                            py_path.parent / "after.py", dataset_path
                        ),
                        mesh_path=os.path.relpath(
                            py_path.parent / "after.stl", dataset_path
                        ),
                    )
                )
                annotations.append(
                    dict(
                        py_before_path=os.path.relpath(py_path, dataset_path),
                        mesh_before_path=os.path.relpath(
                            py_path.with_suffix(".stl"), dataset_path
                        ),
                        py_after_path=os.path.relpath(
                            py_path.parent / "after.py", dataset_path
                        ),
                        mesh_after_path=os.path.relpath(
                            py_path.parent / "after.stl", dataset_path
                        ),
                        edits_path=os.path.relpath(
                            py_path.parent / "edits.json", dataset_path
                        ),
                    )
                )
        with open(dataset_path / "train.pkl", "wb") as f:
            pickle.dump(annotations, f)
        logger.info(f"annotations {len(annotations)}")

    if debug_mode:
        print(f"stepgen finished: {ok}/{len(tasks)} samples processed", flush=True)
