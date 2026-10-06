from __future__ import annotations
import logging
from functools import partial
from pathlib import Path
from typing import Any, Callable

from terminal_app.processing_utils import (
    dataset_stats,
    process_files,
    run_stages,
    stage_stats,
)
from terminal_app.utils import code_is_valid

from cadgen.cot.formats.qwen import make_qwen_thinking_text
from cadgen.cot.meta import build_cot_meta

logger = logging.getLogger(__name__)


def filter_cot_one(
    file_str: str,
    meta: dict[str, Any],
    file_key: str,
    stats_key: str,
    py_filters: dict[str, Callable[[str, dict[str, Any]], bool]] = {},
):
    if meta is None:
        meta = {}

    file = Path(file_str)
    if not file.exists():
        return (file, meta, "file_not_exist", False)

    try:
        text = file.read_text()
    except Exception:
        return (file, meta, "read_text_error", False)

    passed = True
    failed_filter = None

    meta[file_key] = file.as_posix()
    meta[stats_key] = {}
    # meta[stats_key] = {
    #     **CODEStatistics.from_text(text).to_len_stat(),
    #     **CADQueryStatistics.from_text(text).model_dump(),
    # }

    for name, fn in py_filters.items():
        try:
            if fn(text, meta):
                passed = False
                failed_filter = name
                break
        except Exception as ex:
            logger.error(f"{ex} | {name}")
            passed = False
            failed_filter = f"{name} | {str(ex)[:22]}"
            break

    return (file, meta, failed_filter, passed)


# stl_root_path = Path("/path/to/cadpac/tests/test_df/stl")
stl_root_path = None


def with_cot(file: Path | str):
    file = Path(file)
    if file.with_suffix(".json").exists():
        return True

    return False

    build_cot_meta(
        file,
        stl_root_path=stl_root_path,
    )
    return True


def with_cot_filter(meta: dict[str, Any]):
    py_file = Path(meta[py_file_key])
    result = with_cot(py_file)
    meta[py_stats_key]["is_with_cot"] = result
    if result:
        meta[py_stats_key]["cot_len"] = len(
            make_qwen_thinking_text(py_file.with_suffix(".json").as_posix())
        )
    # meta["stl_file"] = py_file.with_suffix(".stl")
    # meta["stl_file"] = (
    #     stl_root_path
    #     / py_file.relative_to(
    #         Path("/path/to/cadpac/tests/test_df/py")
    #     )
    # ).with_suffix(".stl")
    # meta["cot_file"] = py_file.with_suffix(".json").as_posix()
    return result


py_filters = {
    "is_not_valid_code": lambda code, meta: not code_is_valid(code),
    "is_without_cot": lambda code, meta: not with_cot_filter(meta),
    # "code_limit": lambda code: len(code) > 0 and len(code) <= 800,
    # "operation_constraint": partial(
    #     CADQueryStatistics.operation_constraint,
    #     exclude=[
    #         "scale",
    #         "slot2D",
    #         "polyline",
    #         "spline",
    #         "cutThruAll",
    #         "polygon",
    #         "move",
    #         # "sphere",
    #         "translate",
    #         "rotate",
    #         # "line",
    #         # "hole",
    #         # "loft",
    #         "parametricCurve",
    #         # "sweep",
    #     ],
    # ),
}
py_field_configs = {
    "is_*": {"store_examples": True},
}
# for cq_op in CODEStatistics.model_fields:
#     py_field_configs[cq_op] = {"store_examples": True}

# for cq_op in CADQueryStatistics.model_fields:
#     py_field_configs[cq_op] = {"store_examples": True}
py_field_configs["cot_len"] = {"store_examples": True}

py_file_key = "cot_file"
py_stats_key = "cot_stats"

py_stage_stats = partial(
    stage_stats,
    field_configs=py_field_configs,
    file_key=py_file_key,
    stats_key=py_stats_key,
)
py_filter_func = partial(
    process_files,
    filter_one=partial(
        filter_cot_one,
        file_key=py_file_key,
        stats_key=py_stats_key,
        py_filters=py_filters,
    ),
    desc="Filter py files",
    pattern="*.py",
)


def cot_stats(py_folder: Path, output: Path | None = None, max_workers: int = 150):
    if output is None:
        output = py_folder

    callbacks = {
        # "save_pickle": partial(
        #     save_pickle_callback,
        #     root_folder=output,
        #     output_path=output / "cot_train.pkl",
        #     mapping={
        #         py_file_key: "py_path",
        #         "stl_file": "mesh_path",
        #         "cot_file": "cot_path",
        #     },
        # ),
        "dataset_stats": partial(
            dataset_stats,
            stage_stats={"cot": py_stage_stats},
            failed_output=output / "cot_errors.json",
            stat_output=output / "cot_stats.json",
            stdout=logger.info,
        ),
    }

    run_stages(
        stages={
            "cot": partial(
                py_filter_func,
                root_folder=py_folder,
                max_workers=max_workers,
            ),
        },
        callbacks=callbacks,
    )
