import importlib.util
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import hydra
import typer
from omegaconf import OmegaConf

from .cad import CADFactory
from .cot.filter import cot_filter
from .face_operations import FaceFilletChamferFactory
from .runner import run
from .sketch import SketchFactory
from .sub_command import sub_command_filter

OmegaConf.register_new_resolver("fmt_tag", lambda tag: f"_{tag}" if tag else "")

app = typer.Typer(pretty_exceptions_enable=False)


def resolve_rank_path() -> str:
    rank_vars = ["OMPI_COMM_WORLD_RANK", "SLURM_PROCID"]
    path = ""

    for rank_var in rank_vars:
        rank = os.getenv(rank_var, None)
        if rank is not None:
            if rank_var == "SLURM_PROCID":
                node_id = os.getenv("SLURM_NODEID")
                path = f"node_{node_id}/local_{rank}"
            else:
                path = f"local_{rank}"
            break
    return path


def build_default_factories() -> list[dict]:
    return [
        dict(
            factory=SketchFactory(
                min_n_commands=3,
                max_n_commands=8,
                n_outer_probabilities=[0.65, 0.2, 0.1, 0.05],
                rotation_probability=0.3,
                revolve_probability=0.15,
                full_revolve_probability=0.5,
                array_pattern_probability=0.15,
                array_pattern_sketch_min_n_commands=1,
                array_pattern_sketch_max_n_commands=3,
                hole_probability=0.3,
                hole_n_inner_probabilities=[0.5, 0.3, 0.2],
                hole_outer_probability=0.5,
                hole_symmetric_probability=0.3,
                hole_cbore_probability=0.6,
                from_sketchgraph=False,
            ),
            plane=0,
            probability=1,
        ),
        dict(
            factory=FaceFilletChamferFactory(
                fillet_chamfer_probs=(0.5, 0.5), equal_offsets_prob=0.5
            ),
            probability=0.03,
        ),
        dict(
            factory=SketchFactory(
                min_n_commands=3,
                max_n_commands=8,
                n_outer_probabilities=[0.8, 0.2],
                rotation_probability=0.3,
                revolve_probability=0.15,
                full_revolve_probability=0.5,
                from_sketchgraph=False,
            ),
            plane=0,
            probability=0.5,
        ),
        dict(
            factory=FaceFilletChamferFactory(
                fillet_chamfer_probs=(0.5, 0.5), equal_offsets_prob=0.5
            ),
            probability=0.09,
        ),
        dict(
            factory=SketchFactory(
                min_n_commands=1,
                max_n_commands=2,
                n_outer_probabilities=[1],
                rotation_probability=0,
                revolve_probability=0.5,
                full_revolve_probability=0.5,
                from_sketchgraph=False,
            ),
            plane=0,
            probability=0.3,
        ),
        dict(
            factory=SketchFactory(
                min_n_commands=3,
                max_n_commands=8,
                n_outer_probabilities=[1],
                rotation_probability=0,
                revolve_probability=0.15,
                full_revolve_probability=0.5,
                from_sketchgraph=False,
            ),
            plane=1,
            probability=0.3,
        ),
        dict(
            factory=FaceFilletChamferFactory(
                fillet_chamfer_probs=(0.5, 0.5), equal_offsets_prob=0.5
            ),
            probability=0.03,
        ),
    ]


def _build_default_cad_factory() -> CADFactory:
    return CADFactory(
        build_default_factories(),
        world_size=200,
        max_string_length=800,
    )


def _build_twin_cad_factory() -> CADFactory:
    return CADFactory(
        [
            dict(
                factory=SketchFactory(
                    min_n_commands=3,
                    max_n_commands=8,
                    n_outer_probabilities=[0.65, 0.2, 0.1, 0.05],
                    rotation_probability=0.3,
                    revolve_probability=0.15,
                    full_revolve_probability=0.5,
                    array_pattern_probability=0.0,
                    from_sketchgraph=False,
                ),
                plane=0,
                probability=1,
            ),
            dict(
                factory=FaceFilletChamferFactory(
                    fillet_chamfer_probs=(0.5, 0.5), equal_offsets_prob=0.5
                ),
                probability=0.8,
            ),
        ],
        world_size=200,
        max_string_length=800,
    )


class CADGenCli:

    @staticmethod
    def _load_python_config(path: Path) -> Any:
        spec = importlib.util.spec_from_file_location("cadgen_cli_factory_config", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load config file: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        attr = "cad_factory"
        if hasattr(module, attr):
            return getattr(module, attr)
        raise AttributeError(f"Config file {path} must define {attr} variable")

    @staticmethod
    def _ensure_cad_factory(factory_def: Any) -> CADFactory:
        if isinstance(factory_def, CADFactory):
            return factory_def
        if callable(factory_def):
            return CADGenCli._ensure_cad_factory(factory_def())
        if isinstance(factory_def, Mapping):
            required = {"factories", "world_size", "max_string_length"}
            missing = required - factory_def.keys()
            if missing:
                missing_str = ", ".join(sorted(missing))
                raise KeyError(f"Factory config missing keys: {missing_str}")
            return CADFactory(
                factory_def["factories"],
                factory_def["world_size"],
                factory_def["max_string_length"],
                use_literals=factory_def.get("use_literals", True),
            )
        raise TypeError("Factory config must be CADFactory, mapping or callable.")

    @staticmethod
    def _resolve_factory(
        cad_factory_config: CADFactory | Mapping[str, Any] | Path | str | None,
        twin: bool,
    ) -> CADFactory:
        if cad_factory_config is None:
            return _build_twin_cad_factory() if twin else _build_default_cad_factory()
        if isinstance(cad_factory_config, (str, Path)):
            factory_def = CADGenCli._load_python_config(Path(cad_factory_config))
        else:
            factory_def = cad_factory_config
        return CADGenCli._ensure_cad_factory(factory_def)

    @staticmethod
    def _resolve_callable_checks(
        twin: bool,
        use_cot_filter: bool,
        extra_checks: Sequence[Callable] | None = None,
    ) -> list[Any]:
        checks: list[Any] = list(extra_checks) if extra_checks else []
        if not use_cot_filter:
            return checks
        if twin:
            checks.extend(
                [
                    partial(cot_filter, save_questions=True),
                    sub_command_filter,
                ]
            )
        else:
            checks.append(partial(cot_filter, save_questions=True, with_holes=False))
        return checks

    @staticmethod
    def generate(
        output: Path | str,
        *,
        cad_factory_config: CADFactory | Mapping[str, Any] | Path | str | None = None,
        n_train_samples: int = 1_000_000,
        n_folders: int | None = None,
        n_val_samples: int = 0,
        start_from: int = 0,
        solids_only: bool = False,
        twin: bool = False,
        use_cot_filter: bool = False,
        while_successful: bool = True,
        split_train_name: str | None = "train",
        callable_checks: Sequence[Callable] | None = None,
        render: bool = True,
        with_defects: bool = False,
        debug_mode: bool = False,
    ) -> None:
        cad_factory = CADGenCli._resolve_factory(cad_factory_config, twin)
        resolved_checks = CADGenCli._resolve_callable_checks(
            twin, use_cot_filter, callable_checks
        )
        output_path = (
            (Path(output) / resolve_rank_path())
            if isinstance(output, str)
            else output / resolve_rank_path()
        )

        run(
            output=str(output_path),
            cad_factory=cad_factory,
            n_train_samples=n_train_samples,
            n_folders=n_folders,
            start_from=start_from,
            n_val_samples=n_val_samples,
            solids_only=solids_only,
            callable_checks=resolved_checks if resolved_checks else None,
            while_successful=while_successful,
            split_train_name=split_train_name,
            render=render,
            with_defects=with_defects,
            debug_mode=debug_mode,
        )


@app.command()
def gen(
    output: Path = typer.Option(
        ..., "--output", help="Target directory for generated data."
    ),
    cad_factory_config: Path | None = typer.Option(
        None,
        "--cad-factory-config",
        "-f",
        help="Path to a Python file exposing a variable named 'cad_factory'.",
    ),
    n_train_samples: int = typer.Option(
        1_000_000,
        "--n-train-samples",
        help="Number of training samples to generate.",
        show_default=True,
    ),
    n_folders: int | None = typer.Option(
        None,
        "--n-folders",
        help="Explicitly set the number of train subdirectories.",
    ),
    n_val_samples: int = typer.Option(
        0,
        "--n-val-samples",
        help="Number of validation samples to generate.",
        show_default=True,
    ),
    start_from: int = typer.Option(
        0,
        "--start-from",
        help="Index offset to resume training generation from.",
        show_default=True,
    ),
    solids_only: bool = typer.Option(
        False,
        "--solids-only/--allow-non-solids",
        help="Keep only watertight solid meshes.",
    ),
    twin: bool = typer.Option(
        False,
        "--twin/--no-twin",
        help="Use the twin factory preset.",
    ),
    use_cot_filter: bool = typer.Option(
        False,
        "--use-cot-filter/--no-cot-filter",
        help="Enable CoT and sub-command filters.",
    ),
    while_successful: bool = typer.Option(
        True,
        "--while-successful/--no-while-successful",
        help="Retry failed samples until they succeed.",
    ),
    split_train_name: str | None = typer.Option(
        "train",
        "--split-train-name",
        help="Name of the training split directory (set to null to use the root).",
    ),
    with_defects: bool = typer.Option(
        False,
        "--with-defects/--without-defects",
        help="Generate additional defect meshes using scans utilities.",
    ),
) -> None:
    CADGenCli.generate(
        output=output,
        cad_factory_config=cad_factory_config,
        n_train_samples=n_train_samples,
        n_val_samples=n_val_samples,
        n_folders=n_folders,
        start_from=start_from,
        solids_only=solids_only,
        twin=twin,
        use_cot_filter=use_cot_filter,
        while_successful=while_successful,
        split_train_name=split_train_name,
        with_defects=with_defects,
    )


@app.command()
def hydragen(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    extra_args: list[str] = typer.Argument(None),
) -> None:
    cli_overrides = sys.argv[2:]

    with hydra.initialize(config_path="../configs", version_base=None):
        cfg = hydra.compose(config_name="gen", overrides=cli_overrides)

        cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
        output_dir = Path(cfg_resolved["hydra_run_dir"])  # type: ignore

        rank_path = resolve_rank_path()
        print(f"Rank path: {rank_path}")

        if rank_path == "" or rank_path.endswith("local_0"):
            os.makedirs(output_dir, exist_ok=True)
            OmegaConf.save(cfg, output_dir / "config.yaml")
            os.makedirs(output_dir / rank_path, exist_ok=True)
            setup_logging(output_dir / rank_path)
        else:
            import time

            time.sleep(60)
            os.makedirs(output_dir / rank_path, exist_ok=True)
            setup_logging(output_dir / rank_path)

        CADGenCli.generate(
            output=cfg.output,
            cad_factory_config=cfg.cad_factory.cad_factory_config,
            n_train_samples=cfg.n_train_samples,
            n_val_samples=cfg.n_val_samples,
            n_folders=cfg.n_folders,
            start_from=cfg.start_from,
            solids_only=cfg.solids_only,
            twin=cfg.twin,
            use_cot_filter=cfg.use_cot_filter,
            while_successful=cfg.while_successful,
            split_train_name=cfg.split_train_name,
            render=cfg.render,
            with_defects=cfg.with_defects,
            debug_mode=cfg.debug_mode,
        )


def setup_logging(output_dir: Path):
    log_file = output_dir / "job.log"
    logfile = open(log_file, "w", buffering=1)
    sys.stdout = sys.stderr = logfile


if __name__ == "__main__":
    app()
