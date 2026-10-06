from __future__ import annotations

import json
import logging
import os
import pickle
import traceback
import importlib.util
from copy import deepcopy
from glob import glob
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cadquery as cq
import hydra
import numpy as np
import typer
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from .cad import CAD, CADFactory, EmptyCADGeneratedError, _ShapeCache
from .circular_pattern import CircularPatternFactory
from .control_flow import BlockFactory, EitherFactory, StopFactory, expand_factory_entries
from .cut_thru_all import CutThruAllFactory
from .edge_operations import FilletChamferFactory
from .extrude import ExtrudeFactory
from .face_operations import FaceFilletChamferFactory
from .gear import GearFactory
from .hole import HoleFactory
from .loft import LoftFactory
from .orto_cut import OrtoCutFactory
from .plane_logic import (
    generate_circular_pattern_plane,
    generate_cut_thru_all_plane,
    generate_face_fillet_chamfer_plane,
    generate_fillet_chamfer_plane,
    generate_gear_plane,
    generate_hole_plane,
    generate_loft_plane,
    generate_selected_plane,
    generate_sweep_init_plane,
)
from .revolve import RevolveFactory
from .selected_face_plane import SelectedFacePlaneFactory
from .sweep import SweepFactory
from .sweep_init import SweepInitFactory
from .thread_limits import apply_occt_thread_limit, apply_worker_thread_limits
from .utils import compound_to_mesh

if not OmegaConf.has_resolver("fmt_tag"):
    OmegaConf.register_new_resolver("fmt_tag", lambda tag: f"_{tag}" if tag else "")

logger = logging.getLogger(__name__)
app = typer.Typer(pretty_exceptions_enable=False)


def resolve_rank_path() -> str:
    rank_vars = ["OMPI_COMM_WORLD_RANK", "SLURM_PROCID"]
    path = ""

    for rank_var in rank_vars:
        rank = os.getenv(rank_var, None)
        if rank is not None:
            if rank_var == "SLURM_PROCID":
                node_id = os.getenv("SLURM_NODEID")
                path = f"node_{node_id}/global_{rank}"
            else:
                path = f"global_{rank}"
            break
    return path


def _format_tuple(values: Sequence[float]) -> str:
    return "(" + ", ".join(repr(float(value)) for value in values) + ")"


class StepSeedOperation:
    """Replayable seed operation that imports and transforms an existing STEP solid."""

    def __init__(self, step_path: str | Path):
        self.step_path = str(Path(step_path).expanduser().resolve())
        self.shift = [0.0, 0.0, 0.0]
        self.scale = 1.0

    def to_string(self) -> str:
        lines = [f"r=cq.importers.importStep({json.dumps(self.step_path)})"]
        if not np.allclose(self.shift, [0.0, 0.0, 0.0]):
            lines.append(f"r=r.translate({_format_tuple(self.shift)})")
        if not np.isclose(self.scale, 1.0):
            lines.append(
                f"r=cq.Workplane('XY').add(r.val().scale({repr(float(self.scale))}))"
            )
        return "\n".join(lines) + "\n"

    def transform(self, shift: Sequence[float], scale: float) -> None:
        current_scale = float(self.scale)
        self.shift = [
            float(self.shift[i]) + float(shift[i]) / current_scale for i in range(3)
        ]
        self.scale = current_scale * float(scale)

    def round(self) -> None:
        pass

    def fix(self) -> None:
        pass

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "StepSeed",
            "step_path": self.step_path,
            "shift": self.shift,
            "scale": self.scale,
        }


class ExistingBrepCADFactory(CADFactory):
    """CADFactory variant that starts from a STEP BREP seed instead of an empty CAD."""

    def _expanded_factories(self) -> list[dict[str, Any]]:
        factories = expand_factory_entries(deepcopy(self.factories))
        for factory_number, factory in enumerate(factories):
            if isinstance(factory["factory"], EitherFactory):
                factories[factory_number] = factory["factory"].generate()

        result: list[dict[str, Any]] = []
        for item in factories:
            result.extend(item if isinstance(item, list) else [item])
        return result

    @staticmethod
    def _seed_entry(step_path: str | Path) -> dict[str, Any]:
        return {
            "type": "Sweep",
            "op": StepSeedOperation(step_path),
            "plane": None,
            "plane_axes": None,
        }

    @staticmethod
    def _seed_factory_entry(step_path: str | Path) -> dict[str, Any]:
        return {
            "factory": {
                "type": "StepSeed",
                "step_path": str(Path(step_path).expanduser().resolve()),
            },
            "probability": 1.0,
        }

    @staticmethod
    def _sampled_factory_entry(factory: dict[str, Any]) -> dict[str, Any]:
        sampled = {k: v for k, v in factory.items() if k != "factory"}
        sampled["factory"] = factory["factory"].to_dict()
        return sampled

    def _log_retry_failure(
        self,
        operation_name: str,
        factory_number: int,
        retry: int,
        max_retries: int,
        error: Exception,
        block_factory_idx: int | None = None,
    ) -> None:
        if not self.log_exceptions:
            return
        label = str(factory_number + 1)
        if block_factory_idx is not None:
            label += f".{block_factory_idx + 1}"
        logger.warning(
            "%s retry failed: factory=%s retry=%d/%d error=%s: %s",
            operation_name,
            label,
            retry + 1,
            max_retries,
            type(error).__name__,
            error,
        )

    def _existing_context(
        self,
        factory: dict[str, Any],
        planes: list[dict[str, Any]],
        face_planes: list[dict[str, Any]],
        cs: list[dict[str, Any]],
    ):
        return self._existing_model_and_sampler(
            factory["factory"],
            planes,
            face_planes,
            cs,
        )

    def _try_existing_extrude(
        self,
        factory: dict[str, Any],
        planes: list[dict[str, Any]],
        face_planes: list[dict[str, Any]],
        cs: list[dict[str, Any]],
        current_volume: float,
        factory_number: int,
        block_factory_idx: int | None,
    ) -> tuple[float, bool]:
        try:
            existing_context = self._existing_context(factory, planes, face_planes, cs)
        except Exception as exc:
            if self.log_exceptions:
                logger.warning("Skipping extrude/shell: sampler failed", exc_info=True)
            return current_volume, False

        for retry in range(self.n_extrude_retries):
            attempt_state = (len(planes), len(face_planes), len(cs))
            try:
                self._append_existing_extrude(
                    factory,
                    planes,
                    face_planes,
                    cs,
                    existing_context[0],
                    existing_context[1],
                    1.0,
                )
                candidate_shape_cache = _ShapeCache()
                self._assert_extrude_change_allowed(
                    planes,
                    face_planes,
                    cs,
                    attempt_state[2],
                    existing_context[2],
                    existing_context[3],
                    candidate_shape_cache=candidate_shape_cache,
                )
                candidate_volume = self._assert_latest_operation_valid(
                    planes,
                    face_planes,
                    cs,
                    attempt_state[2],
                    existing_context[3],
                    candidate_shape_cache=candidate_shape_cache,
                    operation_already_fixed=True,
                )
                return candidate_volume, True
            except Exception as exc:
                self._restore_generation_state(planes, face_planes, cs, *attempt_state)
                self._log_retry_failure(
                    "Extrude/Shell",
                    factory_number,
                    retry,
                    self.n_extrude_retries,
                    exc,
                    block_factory_idx,
                )
        return current_volume, False

    def _try_existing_revolve(
        self,
        factory: dict[str, Any],
        planes: list[dict[str, Any]],
        face_planes: list[dict[str, Any]],
        cs: list[dict[str, Any]],
        current_volume: float,
        factory_number: int,
        block_factory_idx: int | None,
    ) -> tuple[float, bool]:
        try:
            existing_context = self._existing_context(factory, planes, face_planes, cs)
        except Exception:
            if self.log_exceptions:
                logger.warning("Skipping revolve: sampler failed", exc_info=True)
            return current_volume, False

        for retry in range(self.n_extrude_retries):
            attempt_state = (len(planes), len(face_planes), len(cs))
            try:
                self._append_existing_revolve(
                    factory,
                    planes,
                    face_planes,
                    cs,
                    existing_context[0],
                    existing_context[1],
                    1.0,
                )
                candidate_shape_cache = _ShapeCache()
                self._assert_revolve_change_allowed(
                    planes,
                    face_planes,
                    cs,
                    attempt_state[2],
                    existing_context[2],
                    existing_context[3],
                    candidate_shape_cache=candidate_shape_cache,
                )
                candidate_volume = self._assert_latest_operation_valid(
                    planes,
                    face_planes,
                    cs,
                    attempt_state[2],
                    existing_context[3],
                    candidate_shape_cache=candidate_shape_cache,
                    operation_already_fixed=True,
                )
                return candidate_volume, True
            except Exception as exc:
                self._restore_generation_state(planes, face_planes, cs, *attempt_state)
                self._log_retry_failure(
                    "Revolve",
                    factory_number,
                    retry,
                    self.n_extrude_retries,
                    exc,
                    block_factory_idx,
                )
        return current_volume, False

    def _try_existing_orto_cut(
        self,
        factory: dict[str, Any],
        planes: list[dict[str, Any]],
        face_planes: list[dict[str, Any]],
        cs: list[dict[str, Any]],
        current_volume: float,
        factory_number: int,
        block_factory_idx: int | None,
    ) -> tuple[float, bool]:
        try:
            existing_context = self._existing_context(factory, planes, face_planes, cs)
        except Exception:
            if self.log_exceptions:
                logger.warning("Skipping orto_cut: sampler failed", exc_info=True)
            return current_volume, False

        for retry in range(self.n_extrude_retries):
            attempt_state = (len(planes), len(face_planes), len(cs))
            try:
                self._append_existing_orto_cut(
                    factory,
                    planes,
                    face_planes,
                    cs,
                    existing_context[0],
                    existing_context[1],
                    1.0,
                )
                candidate_shape_cache = _ShapeCache()
                self._assert_orto_cut_change_allowed(
                    planes,
                    face_planes,
                    cs,
                    attempt_state[2],
                    existing_context[2],
                    candidate_shape_cache=candidate_shape_cache,
                )
                self._assert_solid_count_unchanged_for_operation(
                    planes,
                    face_planes,
                    cs,
                    attempt_state[2],
                    required=True,
                )
                candidate_volume = self._assert_latest_operation_valid(
                    planes,
                    face_planes,
                    cs,
                    attempt_state[2],
                    existing_context[3],
                    candidate_shape_cache=candidate_shape_cache,
                    operation_already_fixed=True,
                )
                return candidate_volume, True
            except Exception as exc:
                self._restore_generation_state(planes, face_planes, cs, *attempt_state)
                self._log_retry_failure(
                    "OrtoCut",
                    factory_number,
                    retry,
                    self.n_extrude_retries,
                    exc,
                    block_factory_idx,
                )
        return current_volume, False

    def _try_existing_sweep(
        self,
        factory: dict[str, Any],
        planes: list[dict[str, Any]],
        face_planes: list[dict[str, Any]],
        cs: list[dict[str, Any]],
        current_volume: float,
        factory_number: int,
        block_factory_idx: int | None,
    ) -> tuple[float, bool]:
        try:
            existing_context = self._existing_context(factory, planes, face_planes, cs)
        except Exception:
            if self.log_exceptions:
                logger.warning("Skipping sweep: sampler failed", exc_info=True)
            return current_volume, False

        for retry in range(self.n_extrude_retries):
            attempt_state = (len(planes), len(face_planes), len(cs))
            try:
                self._append_existing_sweep(
                    factory,
                    planes,
                    face_planes,
                    cs,
                    existing_context[0],
                    existing_context[1],
                    1.0,
                )
                candidate_shape_cache = _ShapeCache()
                self._assert_sweep_change_allowed(
                    planes,
                    face_planes,
                    cs,
                    attempt_state[2],
                    existing_context[2],
                    existing_context[3],
                    candidate_shape_cache=candidate_shape_cache,
                )
                candidate_volume = self._assert_latest_operation_valid(
                    planes,
                    face_planes,
                    cs,
                    attempt_state[2],
                    existing_context[3],
                    candidate_shape_cache=candidate_shape_cache,
                    operation_already_fixed=True,
                )
                return candidate_volume, True
            except Exception as exc:
                self._restore_generation_state(planes, face_planes, cs, *attempt_state)
                self._log_retry_failure(
                    "Sweep",
                    factory_number,
                    retry,
                    self.n_extrude_retries,
                    exc,
                    block_factory_idx,
                )
        return current_volume, False

    def _try_hole(
        self,
        factory: dict[str, Any],
        planes: list[dict[str, Any]],
        face_planes: list[dict[str, Any]],
        cs: list[dict[str, Any]],
        current_volume: float,
        factory_number: int,
        block_factory_idx: int | None,
    ) -> tuple[float, bool]:
        for retry in range(self.n_retries):
            attempt_state = (len(planes), len(face_planes), len(cs))
            try:
                hole = factory["factory"].generate(plane=None, s=None)
                generate_hole_plane(
                    factory,
                    planes,
                    face_planes,
                    hole,
                    0.01,
                    1.0,
                    cs,
                    self.world_size,
                    self.max_string_length,
                    self.use_literals,
                    CAD,
                )
                self._assert_solid_count_unchanged_for_operation(
                    planes,
                    face_planes,
                    cs,
                    attempt_state[2],
                    required=True,
                )
                candidate_volume = self._assert_latest_operation_valid(
                    planes,
                    face_planes,
                    cs,
                    attempt_state[2],
                    current_volume,
                )
                return candidate_volume, True
            except Exception as exc:
                self._restore_generation_state(planes, face_planes, cs, *attempt_state)
                self._log_retry_failure(
                    "Hole",
                    factory_number,
                    retry,
                    self.n_retries,
                    exc,
                    block_factory_idx,
                )
        return current_volume, False

    def _try_generated_plane_operation(
        self,
        factory: dict[str, Any],
        planes: list[dict[str, Any]],
        face_planes: list[dict[str, Any]],
        cs: list[dict[str, Any]],
        current_volume: float,
        factory_number: int,
        block_factory_idx: int | None,
    ) -> tuple[float, bool]:
        factory_obj = factory["factory"]

        generators: list[tuple[type, Callable[..., None], bool]] = [
            (CutThruAllFactory, generate_cut_thru_all_plane, True),
            (CircularPatternFactory, generate_circular_pattern_plane, True),
            (FaceFilletChamferFactory, generate_face_fillet_chamfer_plane, True),
            (FilletChamferFactory, generate_fillet_chamfer_plane, True),
            (SelectedFacePlaneFactory, generate_selected_plane, False),
            (GearFactory, generate_gear_plane, True),
            (LoftFactory, generate_loft_plane, True),
            (SweepInitFactory, generate_sweep_init_plane, True),
        ]

        generator = None
        require_volume_change = True
        for factory_type, candidate_generator, changes_volume in generators:
            if isinstance(factory_obj, factory_type):
                generator = candidate_generator
                require_volume_change = changes_volume
                break
        if generator is None:
            return current_volume, False

        for retry in range(self.n_retries):
            attempt_state = (len(planes), len(face_planes), len(cs))
            try:
                operation = factory_obj.generate()
                generator(
                    factory,
                    planes,
                    face_planes,
                    operation,
                    0.01,
                    1.0,
                    cs,
                    self.world_size,
                    self.max_string_length,
                    self.use_literals,
                    CAD,
                )
                candidate_volume = self._assert_latest_operation_valid(
                    planes,
                    face_planes,
                    cs,
                    attempt_state[2],
                    current_volume,
                    require_volume_change=require_volume_change,
                )
                return candidate_volume, True
            except Exception as exc:
                self._restore_generation_state(planes, face_planes, cs, *attempt_state)
                self._log_retry_failure(
                    type(factory_obj).__name__,
                    factory_number,
                    retry,
                    self.n_retries,
                    exc,
                    block_factory_idx,
                )
        return current_volume, False

    def _try_append_factory(
        self,
        factory: dict[str, Any],
        planes: list[dict[str, Any]],
        face_planes: list[dict[str, Any]],
        cs: list[dict[str, Any]],
        current_volume: float,
        factory_number: int,
        block_factory_idx: int | None = None,
    ) -> tuple[float, bool]:
        factory_obj = factory["factory"]
        if isinstance(factory_obj, ExtrudeFactory):
            return self._try_existing_extrude(
                factory,
                planes,
                face_planes,
                cs,
                current_volume,
                factory_number,
                block_factory_idx,
            )
        if isinstance(factory_obj, RevolveFactory):
            return self._try_existing_revolve(
                factory,
                planes,
                face_planes,
                cs,
                current_volume,
                factory_number,
                block_factory_idx,
            )
        if isinstance(factory_obj, OrtoCutFactory):
            return self._try_existing_orto_cut(
                factory,
                planes,
                face_planes,
                cs,
                current_volume,
                factory_number,
                block_factory_idx,
            )
        if isinstance(factory_obj, SweepFactory):
            return self._try_existing_sweep(
                factory,
                planes,
                face_planes,
                cs,
                current_volume,
                factory_number,
                block_factory_idx,
            )
        if isinstance(factory_obj, HoleFactory):
            return self._try_hole(
                factory,
                planes,
                face_planes,
                cs,
                current_volume,
                factory_number,
                block_factory_idx,
            )
        return self._try_generated_plane_operation(
            factory,
            planes,
            face_planes,
            cs,
            current_volume,
            factory_number,
            block_factory_idx,
        )

    def generate_from_step(self, step_path: str | Path) -> CAD:
        cs: list[dict[str, Any]] = [self._seed_entry(step_path)]
        planes: list[dict[str, Any]] = []
        face_planes: list[dict[str, Any]] = []
        sampled_factories = [self._seed_factory_entry(step_path)]
        current_volume = self._volume_for_ops_cached(planes, face_planes, cs)
        generated_ops = 0

        for factory_number, factory in enumerate(self._expanded_factories()):
            factory_obj = factory["factory"]
            if isinstance(factory_obj, BlockFactory):
                first_probability = factory_obj.factories[0].get("probability", 1.0)
                if np.random.rand() > first_probability:
                    continue

                stop_requested = False
                for block_factory_idx, block_factory in enumerate(factory_obj.factories):
                    block_obj = block_factory["factory"]
                    if isinstance(block_obj, StopFactory):
                        stop_requested = True
                        break
                    if (
                        block_factory_idx
                        and np.random.rand() > block_factory.get("probability", 1.0)
                    ):
                        continue

                    before = len(cs)
                    current_volume, appended = self._try_append_factory(
                        block_factory,
                        planes,
                        face_planes,
                        cs,
                        current_volume,
                        factory_number,
                        block_factory_idx,
                    )
                    if appended and len(cs) == before + 1:
                        sampled_factories.append(
                            self._sampled_factory_entry(block_factory)
                        )
                        generated_ops += 1
                if stop_requested:
                    break
                continue

            if np.random.rand() > factory.get("probability", 1.0):
                continue
            if isinstance(factory_obj, StopFactory):
                break

            before = len(cs)
            current_volume, appended = self._try_append_factory(
                factory,
                planes,
                face_planes,
                cs,
                current_volume,
                factory_number,
            )
            if appended and len(cs) == before + 1:
                sampled_factories.append(self._sampled_factory_entry(factory))
                generated_ops += 1

        if generated_ops == 0:
            raise EmptyCADGeneratedError(
                f"No operation could be generated on top of {step_path}"
            )

        return CAD(
            planes,
            face_planes,
            cs,
            sampled_factories,
            self.world_size,
            self.max_string_length,
            self.use_literals,
        )


def _clone_existing_brep_factory(cad_factory: CADFactory) -> ExistingBrepCADFactory:
    return ExistingBrepCADFactory(
        deepcopy(cad_factory.factories),
        world_size=cad_factory.world_size,
        max_string_length=cad_factory.max_string_length,
        use_literals=cad_factory.use_literals,
        n_retries=cad_factory.n_retries,
        n_extrude_retries=cad_factory.n_extrude_retries,
        log_exceptions=cad_factory.log_exceptions,
        skip_fix=getattr(cad_factory, "skip_fix", False),
        single_solid_check=getattr(cad_factory, "single_solid_check", True),
    )


def _load_python_config(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("cadgen_brep_factory_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load config file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    if hasattr(module, "cad_factory"):
        return getattr(module, "cad_factory")
    raise AttributeError(f"Config file {path} must define cad_factory variable")


def _ensure_cad_factory(factory_def: Any) -> CADFactory:
    if isinstance(factory_def, CADFactory):
        return factory_def
    if callable(factory_def):
        return _ensure_cad_factory(factory_def())
    if isinstance(factory_def, (Mapping, DictConfig)):
        required = {"factories", "world_size", "max_string_length"}
        missing = required - set(factory_def.keys())
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise KeyError(f"Factory config missing keys: {missing_str}")
        return CADFactory(
            factory_def["factories"],
            factory_def["world_size"],
            factory_def["max_string_length"],
            use_literals=factory_def.get("use_literals", True),
            n_retries=factory_def.get("n_retries", 10),
            n_extrude_retries=factory_def.get("n_extrude_retries", 1),
            log_exceptions=factory_def.get("log_exceptions", False),
            skip_fix=factory_def.get("skip_fix", False),
            single_solid_check=factory_def.get("single_solid_check", True),
        )
    raise TypeError("Factory config must be CADFactory, mapping or callable.")


def _load_factory_config(
    path: Path | str | Mapping[str, Any] | DictConfig | None,
    twin: bool = False,
) -> CADFactory:
    if path is None:
        from .cli import _build_default_cad_factory, _build_twin_cad_factory

        return _build_twin_cad_factory() if twin else _build_default_cad_factory()

    if isinstance(path, (Mapping, DictConfig)):
        return _ensure_cad_factory(path)

    config_path = Path(path)
    if config_path.suffix in {".yaml", ".yml"}:
        cfg = OmegaConf.load(config_path)
        if not isinstance(cfg, DictConfig):
            raise TypeError(f"YAML factory config must be a mapping: {config_path}")
        if "cad_factory_config" in cfg:
            cfg = cfg["cad_factory_config"]  # type: ignore[assignment]
        return _ensure_cad_factory(cfg)

    return _ensure_cad_factory(_load_python_config(config_path))


def _step_files_from_path(path: Path | str) -> list[Path]:
    source = Path(path).expanduser()
    if source.is_file():
        if source.suffix.lower() not in {".step", ".stp"}:
            raise ValueError(f"Expected a .step/.stp file, got {source}")
        return [source.resolve()]
    if not source.is_dir():
        raise FileNotFoundError(f"STEP input path does not exist: {source}")
    step_files = sorted(
        [
            *source.glob("*.step"),
            *source.glob("*.STEP"),
            *source.glob("*.stp"),
            *source.glob("*.STP"),
        ]
    )
    if not step_files:
        raise FileNotFoundError(f"No .step/.stp files found in {source}")
    return [step_path.resolve() for step_path in step_files]


def _exec_cad_code(code: str):
    namespace: dict[str, Any] = {}
    exec(code, globals(), namespace)
    r = namespace["r"]
    shape = r.val() if hasattr(r, "val") else r
    return r, shape


def _cad_prefix_code(cad: CAD, op_count: int) -> str:
    return CAD(
        cad.planes,
        cad.face_planes,
        cad.cs[:op_count],
        (cad.sampled_factories or [])[:op_count],
        cad.world_size,
        cad.max_string_length,
        cad.use_literals,
    ).to_string()


def _operation_label(op: dict[str, Any]) -> str:
    label = str(op.get("type", "operation"))
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in label)


def _added_operations_code(cad: CAD, full_code: str) -> str:
    seed_code = _cad_prefix_code(cad, 1)
    if full_code.startswith(seed_code):
        operations = full_code[len(seed_code) :]
    else:
        seed_line_count = len(seed_code.splitlines())
        operations = "\n".join(full_code.splitlines()[seed_line_count:])

    header = (
        "# Added operations only.\n"
        "# Assumes r already contains the imported STEP seed transformed into final "
        "world_size coordinates.\n"
    )
    return header + operations.lstrip() + ("" if operations.endswith("\n") else "\n")


def _prefix_mesh(cad: CAD, op_count: int):
    _, shape = _exec_cad_code(_cad_prefix_code(cad, op_count))
    assert shape.isValid()
    mesh = compound_to_mesh(shape)
    assert len(mesh.faces) > 2
    assert not mesh.is_empty
    return mesh


def _save_generated_sample(
    *,
    cad: CAD,
    output_path: Path,
    source_step: Path,
    cad_factory: CADFactory,
    callable_checks: Sequence[Callable] | None,
    export_step: bool,
) -> dict[str, str]:
    code = cad.to_string()
    r, shape = _exec_cad_code(code)
    assert shape.isValid()

    bbox = shape.BoundingBox()
    bbox_values = (bbox.xmin, bbox.ymin, bbox.zmin, bbox.xmax, bbox.ymax, bbox.zmax)
    size = cad_factory.world_size / 2
    assert abs(np.min(bbox_values) + size) < 1.5
    assert abs(np.max(bbox_values) - size) < 1.5

    mesh = compound_to_mesh(shape)
    assert len(mesh.faces) > 2
    assert not mesh.is_empty
    assert bool(mesh.volume > 0)
    assert bool(sum(mesh.extents == 0) == 0)
    if getattr(cad_factory, "single_solid_check", True):
        assert len(mesh.split()) == 1

    sample_name = source_step.stem
    sample_dir = output_path / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)

    py_path = sample_dir / f"{sample_name}.py"
    stl_path = sample_dir / f"{sample_name}.stl"
    step_path = sample_dir / f"{sample_name}.step"
    source_path = sample_dir / f"{sample_name}_source_step.txt"
    added_operations_path = sample_dir / f"{sample_name}_added_operations.py"
    initial_stl_path = sample_dir / f"{sample_name}_initial.stl"
    operation_stl_paths = [
        sample_dir / f"{sample_name}_after_{op_number:03d}_{_operation_label(op)}.stl"
        for op_number, op in enumerate(cad.cs[1:], start=1)
    ]
    tmp_prefix = sample_dir / f"{sample_name}.tmp.{os.getpid()}"

    tmp_py_path = Path(f"{tmp_prefix}.py")
    tmp_stl_path = Path(f"{tmp_prefix}.stl")
    tmp_step_path = Path(f"{tmp_prefix}.step")
    tmp_source_path = Path(f"{tmp_prefix}.source_step.txt")
    tmp_added_operations_path = Path(f"{tmp_prefix}.added_operations.py")
    tmp_initial_stl_path = Path(f"{tmp_prefix}.initial.stl")
    tmp_operation_stl_paths = [
        Path(f"{tmp_prefix}.after_{op_number:03d}_{_operation_label(op)}.stl")
        for op_number, op in enumerate(cad.cs[1:], start=1)
    ]
    tmp_paths = [
        tmp_py_path,
        tmp_stl_path,
        tmp_step_path,
        tmp_source_path,
        tmp_added_operations_path,
        tmp_initial_stl_path,
        *tmp_operation_stl_paths,
    ]

    try:
        tmp_py_path.write_text(code)
        tmp_added_operations_path.write_text(_added_operations_code(cad, code))
        mesh.export(tmp_stl_path)
        _prefix_mesh(cad, 1).export(tmp_initial_stl_path)
        for op_number, tmp_operation_stl_path in enumerate(
            tmp_operation_stl_paths,
            start=2,
        ):
            _prefix_mesh(cad, op_number).export(tmp_operation_stl_path)
        tmp_source_path.write_text(str(source_step))
        if export_step:
            cq.exporters.export(r, str(tmp_step_path))

        os.replace(tmp_py_path, py_path)
        os.replace(tmp_added_operations_path, added_operations_path)
        os.replace(tmp_stl_path, stl_path)
        os.replace(tmp_initial_stl_path, initial_stl_path)
        for tmp_operation_stl_path, operation_stl_path in zip(
            tmp_operation_stl_paths,
            operation_stl_paths,
        ):
            os.replace(tmp_operation_stl_path, operation_stl_path)
        os.replace(tmp_source_path, source_path)
        if export_step:
            os.replace(tmp_step_path, step_path)

        if callable_checks is not None:
            for callable_check in callable_checks:
                assert callable_check(mesh=mesh, code=code, file_path=str(py_path))
    finally:
        for tmp_path in tmp_paths:
            if tmp_path.exists():
                tmp_path.unlink()

    rel_py = os.path.relpath(py_path, output_path)
    rel_mesh = os.path.relpath(stl_path, output_path)
    annotation = {
        "py_path": rel_py,
        "mesh_path": rel_mesh,
        "source_step_path": str(source_step),
        "added_operations_path": os.path.relpath(
            added_operations_path,
            output_path,
        ),
        "initial_mesh_path": os.path.relpath(initial_stl_path, output_path),
        "operation_mesh_paths": [
            os.path.relpath(operation_stl_path, output_path)
            for operation_stl_path in operation_stl_paths
        ],
    }
    if export_step:
        annotation["step_path"] = os.path.relpath(step_path, output_path)
    return annotation


def _run_one_existing_brep_sample(
    *,
    index: int,
    output_path: Path,
    source_step: Path,
    cad_factory: ExistingBrepCADFactory,
    callable_checks: Sequence[Callable] | None,
    export_step: bool,
    debug_mode: bool,
    attempt: int = 0,
) -> dict[str, str]:
    apply_worker_thread_limits()
    apply_occt_thread_limit()
    seed = int(str(hash(f"{os.getpid()}:{index}:{attempt}:{source_step}"))[1:10])
    np.random.seed(seed)

    cad = cad_factory.generate_from_step(source_step)
    cad.finalize(skip_fix=getattr(cad_factory, "skip_fix", False))
    if len(cad.cs) <= 1:
        raise EmptyCADGeneratedError("Final cleanup removed all added operations")
    return _save_generated_sample(
        cad=cad,
        output_path=output_path,
        source_step=source_step,
        cad_factory=cad_factory,
        callable_checks=callable_checks,
        export_step=export_step,
    )


def run_existing_brep_split(
    *,
    output: Path,
    brep_path: Path,
    cad_factory: ExistingBrepCADFactory,
    start_from: int,
    callable_checks: Sequence[Callable] | None = None,
    while_successful: bool = True,
    max_sample_attempts: int = 10,
    render: bool = False,
    export_step: bool = True,
    debug_mode: bool = False,
) -> list[dict[str, str]]:
    step_files = _step_files_from_path(brep_path)
    selected_step_files = step_files[start_from:]

    output.mkdir(parents=True, exist_ok=True)

    annotations: list[dict[str, str]] = []
    for index, source_step in tqdm(
        list(enumerate(selected_step_files, start=start_from)),
        desc="breps",
    ):
        attempts = max_sample_attempts if while_successful else 1
        last_error: BaseException | None = None
        for attempt in range(attempts):
            try:
                annotation = _run_one_existing_brep_sample(
                    index=index,
                    output_path=output,
                    source_step=source_step,
                    cad_factory=cad_factory,
                    callable_checks=callable_checks,
                    export_step=export_step,
                    debug_mode=debug_mode,
                    attempt=attempt,
                )
                annotations.append(annotation)
                break
            except Exception as exc:
                last_error = exc
                if getattr(cad_factory, "log_exceptions", False) or debug_mode:
                    print(
                        f"Sample {index} attempt {attempt + 1}/{attempts} failed: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    traceback.print_exc()
                if debug_mode:
                    raise
        else:
            if while_successful:
                logger.warning(
                    "Giving up on %s after %s attempts: %s",
                    source_step,
                    attempts,
                    last_error,
                )

    with open(output / "manifest.pkl", "wb") as f:
        pickle.dump(annotations, f)
    (output / "manifest.json").write_text(json.dumps(annotations, indent=2))

    if render:
        from cadlib.methods.rendering.render_collage import (  # type: ignore
            CollageConfig,
            render_collage_with_split,
        )

        render_collage_with_split(
            natsorted(list(glob(f"{output}/*/*.stl"))),
            CollageConfig(
                output_dir=output,
                output_name=output.stem,
                shuffle=False,
            ),
        )

    return annotations


def generate_existing_brep(
    *,
    output: Path,
    brep_path: Path,
    cad_factory_config: Path | str | Mapping[str, Any] | DictConfig | None = None,
    start_from: int = 0,
    twin: bool = False,
    while_successful: bool = True,
    render: bool = False,
    export_step: bool = True,
    log_exceptions: bool = False,
    skip_fix: bool = False,
    debug_mode: bool = False,
    max_sample_attempts: int = 10,
) -> None:
    base_factory = _load_factory_config(cad_factory_config, twin=twin)
    cad_factory = _clone_existing_brep_factory(base_factory)
    cad_factory.log_exceptions = (
        getattr(cad_factory, "log_exceptions", False) or log_exceptions
    )
    cad_factory.skip_fix = getattr(cad_factory, "skip_fix", False) or skip_fix

    output_path = output / resolve_rank_path()
    output_path.mkdir(parents=True, exist_ok=True)

    run_existing_brep_split(
        output=output_path,
        brep_path=brep_path,
        cad_factory=cad_factory,
        start_from=start_from,
        while_successful=while_successful,
        max_sample_attempts=max_sample_attempts,
        render=render,
        export_step=export_step,
        debug_mode=debug_mode,
    )


@app.command()
def gen(
    output: Path = typer.Option(..., "--output", help="Target directory."),
    brep_dir: Path = typer.Option(
        ...,
        "--brep-dir",
        "--brep-path",
        help="Directory of .step/.stp files, or one STEP file.",
    ),
    cad_factory_config: Path | None = typer.Option(
        None,
        "--cad-factory-config",
        "-f",
        help="Python factory config or YAML cad_factory_config file.",
    ),
    start_from: int = typer.Option(0, "--start-from"),
    twin: bool = typer.Option(False, "--twin/--no-twin"),
    while_successful: bool = typer.Option(
        True,
        "--while-successful/--no-while-successful",
    ),
    render: bool = typer.Option(False, "--render/--no-render"),
    export_step: bool = typer.Option(True, "--export-step/--no-export-step"),
    log_exceptions: bool = typer.Option(False, "--log-exceptions", "--log_exceptions"),
    skip_fix: bool = typer.Option(False, "--skip-fix", "--skip_fix"),
    debug_mode: bool = typer.Option(False, "--debug-mode", "--debug_mode"),
    max_sample_attempts: int = typer.Option(10, "--max-sample-attempts"),
) -> None:
    generate_existing_brep(
        output=output,
        brep_path=brep_dir,
        cad_factory_config=cad_factory_config,
        start_from=start_from,
        twin=twin,
        while_successful=while_successful,
        render=render,
        export_step=export_step,
        log_exceptions=log_exceptions,
        skip_fix=skip_fix,
        debug_mode=debug_mode,
        max_sample_attempts=max_sample_attempts,
    )


@app.command()
def hydragen(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    brep_dir: Path | None = typer.Option(
        None,
        "--brep-dir",
        "--brep-path",
        help="Directory of .step/.stp files, or one STEP file.",
    ),
    log_exceptions: bool = typer.Option(
        False,
        "--log-exceptions",
        "--log_exceptions",
        help="Print/log exception diagnostics from generation retries.",
    ),
    render: bool = typer.Option(
        False,
        "--render/--no-render",
        help="Render a collage for generated final STLs.",
    ),
    skip_fix: bool = typer.Option(
        False,
        "--skip-fix",
        "--skip_fix",
        help="Skip destructive CAD.fix() cleanup during finalize.",
    ),
    extra_args: list[str] = typer.Argument(None),
) -> None:
    _ = verbose
    cli_overrides = list(extra_args or [])
    with hydra.initialize(config_path="../configs", version_base=None):
        cfg: DictConfig = hydra.compose(config_name="gen", overrides=cli_overrides)
        output_dir = Path(cfg.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output_dir / "config.yaml")

        resolved_brep_dir = brep_dir or cfg.get("brep_dir")
        if resolved_brep_dir is None:
            raise ValueError("Pass --brep-dir or set brep_dir in the Hydra config.")

        generate_existing_brep(
            output=Path(cfg.output),
            brep_path=Path(resolved_brep_dir),
            cad_factory_config=cfg.cad_factory.cad_factory_config,
            start_from=cfg.start_from,
            twin=cfg.twin,
            while_successful=cfg.while_successful,
            render=render,
            log_exceptions=bool(cfg.log_exceptions) or log_exceptions,
            skip_fix=bool(cfg.skip_fix) or skip_fix,
            debug_mode=cfg.debug_mode,
        )


if __name__ == "__main__":
    app()
