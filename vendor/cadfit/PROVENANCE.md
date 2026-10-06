# cadfit - snapshot provenance

| | |
|---|---|
| Source | our fork of the det candidate generator (`cadfit`), taken with `git archive` from a clean working tree |
| License | MIT, as the rest of the repository (`LICENSE`) |
| Integrity | `MANIFEST.sha256` lists all snapshot files except itself and this file; it is checked by `agent/tools/preflight.py` |

cadfit is a fork of the det candidate code from the `cad_optimizer` snapshot, plus
the C++ parts of `_cad_grad` that det needs. It adds speedups and deterministic
boolean operations.

## What is included

| Here | What it is |
|---|---|
| `src/cadfit/__init__.py` | the package and `native_status()` - whether the C++ module is loaded |
| `src/cadfit/candidates/` | all of det: `cadfit_pass` (+ `kdtree.py`, pykdtree), `residual` (boolean via `Mesh64`), `section_analyzer`, detectors and `make_candidates` |
| `src/cadfit/voxel_lattice.py` | lattice voxelization; identical to `agent/cad_agent/capabilities/voxel_lattice.py` except the module header |
| `native/` | sources of `cadfit._native` (`cluster_normals` + `section`) and their `CMakeLists.txt` |

## What is not included

- `src/cadfit/cadgen_emit.py` - byte-identical to `vendor/image2cad/cadgen_emit.py`;
  the emitter is taken from there.
- `src/cadfit/tool/` - the det wrapper; ours is `agent/cad_agent/capabilities/det.py`.
- `src/cadfit/{dsl,metrics,execute.py,bench}`, `bench/`, `tests/`, `env/` - copies of
  our own code and cadfit's own harness. The fallback export step from `execute.py`
  lives in `agent/cad_agent/capabilities/execute.py`; `prune_ops` is not carried over
  (here a det operation is a separate candidate, and selection drops a worse one).
- `native/build.sh` - puts the module inside the package tree; we build with
  `agent/tools/build_cadfit_native.sh` into `build_native/cadfit/`.
- built `_native*.so` files and `__pycache__`.

## How it is wired in

`dsl_runtime` puts `vendor/cadfit/src` on `sys.path` and adds `build_native/cadfit/`
to `cadfit.__path__`, so that `cadfit._native` is found outside the snapshot. The
module is not named `_cad_grad`, so it does not clash with the optimizer.

The snapshot is not edited: changes go into cadfit and arrive with a new snapshot.
