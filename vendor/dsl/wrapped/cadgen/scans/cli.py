import logging
import multiprocessing as mp
import os
import queue
import random
from glob import glob
from pathlib import Path
from typing import Annotated

import typer
from cadlib.methods.math.stats.py_stl_stats import py_stl_stats
from cadlib.methods.rendering.render_collage import (
    CollageConfig,
    render_collage_with_split,
)
from natsort import natsorted
from tqdm import tqdm

from .defects import make_all_defects2
from .utils import load_and_normalize_mesh
from ..thread_limits import apply_worker_thread_limits

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
DEFECTS_PC_SIZE = 100_000
DEFECTS_N_CAMERAS = 5
DEFECTS_N_THREADS = 1
DEFECTS_MAX_WORKERS = 32


def _process_defect_task(mesh_path: Path) -> tuple[str, str]:
    apply_worker_thread_limits()
    from psutil import Process

    try:
        mesh = load_and_normalize_mesh(str(mesh_path))
        mesh = make_all_defects2(
            mesh,
            n_holes=random.randint(0, 8),
            n_gaussian_holes=0,
            PC_SIZE=DEFECTS_PC_SIZE,
            n_cameras=DEFECTS_N_CAMERAS,
            n_threads=DEFECTS_N_THREADS,
            log=False,
        )
        rss_gb = Process(os.getpid()).memory_info().rss / 1e9
        logger.debug(f"[{mesh_path.name}] worker RSS: {rss_gb:.2f} GB")
        out_path = mesh_path.parent / f"{mesh_path.stem}_with_defects.stl"
        ok = mesh.export(str(out_path))
        return mesh_path.name, ("OK" if ok else "FAIL: write failed")
    except Exception as exc:  # pragma: no cover
        return mesh_path.name, f"FAIL: {exc}"


def _defects_worker(
    task_q: "mp.Queue[Path]", result_q: "mp.Queue[tuple[str, str]]"
) -> None:
    apply_worker_thread_limits()
    while True:
        try:
            mesh_path = task_q.get(timeout=1)
        except queue.Empty:
            return
        result_q.put(_process_defect_task(mesh_path))


def make_defects(
    output: str,
    split_name: str | None = None,
    n_folders: int | None = None,
    calculate_stats: bool = True,
    render: bool = True,
    save_annotations: bool = True,
) -> None:
    random.seed(42)

    # Build split_path as in run_split
    if split_name is not None:
        split_path = os.path.join(output, split_name)
    else:
        split_path = output

    split_path_obj = Path(split_path)

    if not split_path_obj.exists():
        logger.warning(f"Directory not found: {split_path}")
        return

    if n_folders is not None:
        stl_paths = glob(f"{split_path}/*/*.stl")
    else:
        stl_paths = glob(f"{split_path}/*.stl")

    files = [
        Path(stl_path)
        for stl_path in sorted(stl_paths)
        if not Path(stl_path).stem.endswith("_with_defects")
    ]

    if not files:
        logger.warning(f"No STL files found in {split_path}")
        return

    try:
        slurm_cpus = (
            int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or DEFECTS_MAX_WORKERS
        )
    except ValueError:
        slurm_cpus = DEFECTS_MAX_WORKERS
    workers = max(1, min(len(files), slurm_cpus, DEFECTS_MAX_WORKERS))

    logger.info(f"Files: {len(files)} | workers={workers}")

    ctx = mp.get_context("spawn")
    task_q: "mp.Queue[Path]" = ctx.Queue()
    result_q: "mp.Queue[tuple[str, str]]" = ctx.Queue()

    for path in files:
        task_q.put(path)

    processes = [
        ctx.Process(target=_defects_worker, args=(task_q, result_q))
        for _ in range(workers)
    ]
    for proc in processes:
        proc.start()

    for _ in tqdm(range(len(files)), desc="Processing defects"):
        name, status = result_q.get()
        logger.debug(f"[{name}] {status}")

    for proc in processes:
        proc.join(timeout=1)

    defects_files = sorted(split_path_obj.rglob("*_with_defects.stl"))

    if calculate_stats and defects_files:

        py_stl_stats(
            Path(split_path),
            output=Path(output),
            output_prefix=f"{split_name}_with_defects",
            stl_suffix="_with_defects.stl",
            pickle_name=(
                f"{Path(split_path).stem}_with_defects.pkl" if save_annotations else ""
            ),
            save_pickle=save_annotations,
            exclude_filters=["is_not_watertight"],
            pickle_root_folder=Path(output),
        )

    if render and defects_files:
        render_collage_with_split(
            natsorted([str(f) for f in defects_files]),
            CollageConfig(
                output_dir=Path(output),
                output_name=f"{Path(split_path).stem}_with_defects",
                shuffle=False,
            ),
        )


def _resolve_mesh_paths_argument(values: list[str]) -> Path | str | list[str]:
    if len(values) == 1:
        return values[0]
    return values


def _load_mesh_paths(annotation: Path | str | list[str]) -> list[str]:
    """Load mesh paths from various formats (list, glob, file, JSON) with STL file fix."""

    from cadlib.methods.rendering.render_collage import (
        _load_mesh_paths as _load_mesh_paths_collage,
    )

    return _load_mesh_paths_collage(annotation)


def process_defects(
    mesh_paths: Path | str | list[str],
    output_dir: Path | None = None,
    n_holes: int | None = None,
    pc_size: int = DEFECTS_PC_SIZE,
    n_cameras: int = DEFECTS_N_CAMERAS,
    n_threads: int = DEFECTS_N_THREADS,
    calculate_stats: bool = False,
    render: bool = False,
) -> list[Path]:
    """
    Process defects for given mesh paths.

    Args:
        mesh_paths: Path, glob pattern, or list of paths to meshes
        output_dir: Directory to save output meshes (default: same as input)
        n_holes: Number of holes to add (default: random 0-8)
        pc_size: Point cloud size for defects
        n_cameras: Number of cameras for scanning
        n_threads: Number of threads
        calculate_stats: Whether to calculate statistics
        render: Whether to render collage

    Returns:
        List of output mesh paths
    """
    random.seed(42)

    mesh_paths_list = _load_mesh_paths(mesh_paths)
    logger.info(f"Processing {len(mesh_paths_list)} meshes for defects")

    output_paths = []

    for mesh_path_str in tqdm(mesh_paths_list, desc="Processing defects"):
        mesh_path = Path(mesh_path_str)

        if not mesh_path.exists():
            logger.warning(f"Mesh not found: {mesh_path}")
            continue

        try:
            mesh = load_and_normalize_mesh(str(mesh_path))
            holes_count = n_holes if n_holes is not None else random.randint(0, 8)
            mesh = make_all_defects2(
                mesh,
                n_holes=holes_count,
                n_gaussian_holes=0,
                PC_SIZE=pc_size,
                n_cameras=n_cameras,
                n_threads=n_threads,
                log=False,
            )

            if output_dir is None:
                output_path = mesh_path.parent / f"{mesh_path.stem}_with_defects.stl"
            else:
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{mesh_path.stem}_with_defects.stl"

            ok = mesh.export(str(output_path))
            if ok:
                output_paths.append(output_path)
            else:
                logger.warning(f"Failed to export: {output_path}")
        except Exception as exc:
            logger.warning(f"Failed to process {mesh_path}: {exc}")

    if not output_paths:
        raise RuntimeError("No meshes were processed successfully.")

    if calculate_stats and output_paths:
        # Determine stats output directory
        if output_dir:
            stats_output = Path(output_dir)
            defects_files = sorted(Path(output_dir).glob("*_with_defects.stl"))
        else:
            # Use the parent directory of the first output path
            stats_output = output_paths[0].parent
            defects_files = sorted(stats_output.glob("*_with_defects.stl"))

        if defects_files:
            py_stl_stats(
                stats_output,
                output=stats_output,
                output_prefix="defects",
                stl_suffix="_with_defects.stl",
                pickle_name="defects.pkl" if calculate_stats else "",
                save_pickle=calculate_stats,
                exclude_filters=["is_not_watertight"],
                pickle_root_folder=stats_output,
            )

    if render and output_paths:
        defects_files = natsorted([str(p) for p in output_paths])
        render_collage_with_split(
            defects_files,
            CollageConfig(
                output_dir=output_dir if output_dir else output_paths[0].parent,
                output_name="defects",
                shuffle=False,
            ),
        )

    return output_paths


app = typer.Typer(no_args_is_help=True)


@app.command()
def defects(
    mesh_paths: Annotated[
        list[str],
        typer.Option(
            "--mesh_paths",
            "-m",
            help="Path, glob or STL list. Provide multiple values to pass explicit meshes.",
        ),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="Directory to store output meshes (default: same as input).",
        ),
    ] = None,
    n_holes: Annotated[
        int | None,
        typer.Option("--n-holes", help="Number of holes to add (default: random 0-8)."),
    ] = None,
    pc_size: Annotated[
        int,
        typer.Option("--pc-size", help="Point cloud size for defects."),
    ] = DEFECTS_PC_SIZE,
    n_cameras: Annotated[
        int,
        typer.Option("--n-cameras", help="Number of cameras for scanning."),
    ] = DEFECTS_N_CAMERAS,
    n_threads: Annotated[
        int,
        typer.Option("--n-threads", help="Number of threads."),
    ] = DEFECTS_N_THREADS,
    calculate_stats: Annotated[
        bool,
        typer.Option("--stats/--no-stats", help="Calculate statistics."),
    ] = False,
    render: Annotated[
        bool,
        typer.Option("--render/--no-render", help="Render collage."),
    ] = False,
) -> None:
    mesh_paths = _resolve_mesh_paths_argument(mesh_paths)  # type: ignore
    process_defects(
        mesh_paths=mesh_paths,
        output_dir=output_dir,
        n_holes=n_holes,
        pc_size=pc_size,
        n_cameras=n_cameras,
        n_threads=n_threads,
        calculate_stats=calculate_stats,
        render=render,
    )
