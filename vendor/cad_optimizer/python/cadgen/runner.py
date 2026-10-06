from __future__ import annotations
import logging
import os
import pickle
import queue
from glob import glob
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Callable

import numpy as np
from cadlib.methods.math.stats.py_stl_stats import py_stl_stats  # type: ignore
from cadlib.methods.rendering.render_collage import (  # type: ignore
    CollageConfig,
    render_collage_with_split,
)
from natsort import natsorted
from tqdm import tqdm

from .cad import CADFactory
from .scans import make_defects
from .sselectors import *
from .utils import compound_to_mesh, shape_to_volume

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def run_single(index, path, cad_factory):
    np.random.seed(int(str(hash(str(os.getpid()) + str(index) + str(path)))[1:10]))
    try:
        cad = cad_factory.generate()
        cad.finalize()
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
    solids_only,
    callable_checks=None,
    debug_mode: bool = False,
):
    np.random.seed(int(str(hash(str(os.getpid()) + str(index) + str(path)))[1:10]))
    s_if_exc = ""
    try:
        cad = cad_factory.generate()
        cad.finalize()
        s = cad.to_string()
        s_if_exc = s

        exec(s, globals())
        w = globals()["r"].val()
        assert w.isValid()
        bbox = w.BoundingBox()
        bbox = (bbox.xmin, bbox.ymin, bbox.zmin, bbox.xmax, bbox.ymax, bbox.zmax)

        size = cad_factory.world_size / 2
        assert abs(np.min(bbox) + size) < 1.5 and abs(np.max(bbox) - size) < 1.5

        mesh = compound_to_mesh(w)
        assert len(mesh.faces) > 2
        assert mesh.is_watertight
        assert not mesh.is_empty

        if solids_only:
            assert len(mesh.split()) == 1
        assert bool(mesh.volume > 0)
        assert bool(sum(mesh.extents == 0) == 0)

        dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        if len(dirs) > 0:
            path = os.path.join(path, str(index % len(dirs)))
        mesh_path = os.path.join(path, f"{index}.stl")
        py_path = os.path.join(path, f"{index}.py")
        step_path = os.path.join(path, f"{index}.step")
        mesh.export(mesh_path)
        w.export(step_path)
        with open(py_path, "w") as f:
            f.write(s)

        if callable_checks is not None:
            for callable_check in callable_checks:
                assert callable_check(mesh=mesh, code=s, file_path=py_path)
        # print(s)
        inner_q.put(1)
    except Exception as e:
        # todo: all but AssertionError should be debugged
        if debug_mode:
            print(f"Failed code: {s_if_exc}")
            raise e
        pass
        # TODO add retry?
        # run_single_no_occ_validation(index, path, cad_factory)


def run_single_no_occ_validation_with_timeout(
    index,
    path,
    cad_factory,
    timeout,
    inner_q,
    solids_only,
    callable_checks=None,
    debug_mode=False,
):
    process = Process(
        target=run_single_no_occ_validation,
        args=(
            index,
            path,
            cad_factory,
            inner_q,
            solids_only,
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
    solids_only,
    callable_checks=None,
    while_successful: bool = True,
    debug_mode: bool = False,
):
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
                    solids_only,
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
    solids_only: bool = False,
    callable_checks: list[Callable] | None = None,
    save_annotations: bool = True,
    while_successful: bool = True,
    calculate_stats: bool = True,
    render: bool = True,
    debug_mode: bool = False,
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
    cpu_number = os.cpu_count()
    workers = [
        Process(
            target=worker_fn,
            args=(
                split_path,
                cad_factory,
                60,
                input_q,
                res_q,
                solids_only,
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
            glob(f"{split_path}/*/*.py")
            if n_folders is not None
            else glob(f"{split_path}/*.py")
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
    solids_only: bool = False,
    callable_checks: list[Callable] | None = None,
    save_annotations: bool = True,
    while_successful: bool = True,
    split_train_name: str | None = "train",
    split_val_name: str | None = "val",
    calculate_stats: bool = True,
    render: bool = True,
    with_defects: bool = False,
    debug_mode: bool = False,
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
        solids_only=solids_only,
        callable_checks=callable_checks,
        save_annotations=save_annotations,
        while_successful=while_successful,
        calculate_stats=calculate_stats,
        render=render,
        debug_mode=debug_mode,
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
            solids_only=solids_only,
            callable_checks=callable_checks,
            save_annotations=save_annotations,
            while_successful=while_successful,
            calculate_stats=calculate_stats,
            render=render,
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
