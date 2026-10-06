# gms.py: snapshot provenance

A single file (`gms.py`, about 24 KB) taken unmodified from the `utils/gms.py`
module of an upstream metrics project (cicada).

It contains the machinery of the GMS metric: `TrimeshHandler` and
`A_in_B_ball_matching_multiangle_v3`, which the upstream `compute_gms` calls. It is
vendored rather than rewritten because it is the reference our numbers must match;
any rewrite would make that comparison meaningless. `compute_gms` itself is not
vendored: it is short and lives in `agent/metrics.py` with the other metrics.

The file is self-contained: its only external dependencies are `numpy`, `trimesh`
and `pykdtree`, and it does not depend on the rest of the upstream project.

`MANIFEST.sha256` lists the file hashes; `agent/tools/preflight.py` checks it.

Side effect of importing: at module level it sets `OMP_NUM_THREADS`,
`MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `NUMEXPR_NUM_THREADS` to `1` via
`os.environ.setdefault`. This is deliberate (single-threaded under a
multi-process launch), but it means import order affects BLAS behaviour for the
whole process.
