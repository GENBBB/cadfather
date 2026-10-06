# cad_optimizer: snapshot provenance

This directory is a vendored snapshot of a CAD parameter optimizer (a fork of an
earlier internal optimizer). It was taken with `git archive` from a clean working
tree and contains `python/`, `src/`, `include/` and `CMakeLists.txt`. It is
distributed under the repository's MIT license (`LICENSE`).

`MANIFEST.sha256` lists the hashes of all snapshot files except itself and this
file. `agent/tools/preflight.py` checks it, so any local edit to the snapshot is
detected.

## Local changes relative to the upstream fork

- `det_candidates/` and the C++ `section_fit` / `cluster_normals` were removed;
  deterministic fitting comes from `vendor/cadfit`.
- Prebuilt `build/_cad_grad*.so` files are gone: `_cad_grad` is built for the
  target environment by our recipe in `agent/tools/cad_grad/`
  (`agent/tools/build_cad_grad.sh`) into `build_native/cad_grad/`.
- Optimizer modules raise `ImportError` without `_cad_grad` instead of running
  idle with a warning.
- `optimize()`: new keys `coarse_mode`, `pick_best` (on), `thin_polygons`,
  `untangle_polygons` (on); revolve profiles are projected onto their own side of
  the axis; the SDF sign is seeded around `trimesh.contains`. `cq_parser` gained
  tied_slots and arc_writeback for booleans; new `polygon_thin.py`.

## Not included

The fork's `bench/`, `docs/`, `runs/` and `vendor/` (its own test bench and its own
copies of cadgen and cq_gears) are not needed by this code.

## Warning: `python/cadgen` here is the chain dialect

The snapshot contains its own `cadgen` of the chain dialect. The project's DSL
runtime is `vendor/dsl/<dialect>` (see `vendor/README.md`). Two packages with the
same name must not be on `sys.path` at the same time: whichever comes first decides
which language the pipeline speaks.

Also, `python/cadgen_desugar.py` imports wrapped-dialect modules
(`cadgen.extrude`, `cadgen.selectors`, `cadgen.orto_cut`) that the neighbouring
`python/cadgen` does not have, so these two parts of the snapshot are inconsistent
upstream.

## pip dependencies

`numpy`, `trimesh`, `scipy`; `cma` (see `optimizer_cmaes.py`) and `_cad_grad`.

## Snapshot `CMakeLists.txt`

Rebuilds do not use it. Our recipe in `agent/tools/cad_grad/` applies the patches
in `patches/` to a copy of `src/` and puts the result outside the snapshot. Five
headers, `ops/{boolean,revolve,rotate,shell,translate}.h`, have no `.cpp` and are
not included from anything in `src/`.
