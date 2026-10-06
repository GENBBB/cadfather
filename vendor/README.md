# vendor/ - third-party code copied into the repository

This directory holds external dependencies that are vendored as snapshots, so the
project does not depend on someone else's live working tree.

## Contents

| Directory | What | Source |
|---|---|---|
| `dsl/wrapped/cadgen` | runtime of the final DSL (the "wrapped" dialect) | upstream CAD DSL project, `third_party/wrapped` |
| `dsl/chain/cadgen` | legacy "chain" dialect, which the current code speaks | upstream CAD DSL project, `third_party/chain` |
| `cad_optimizer/` | parameter optimizer and the sources of `_cad_grad` (built by `agent/tools/build_cad_grad.sh`) | our fork of an upstream optimizer |
| `cadfit/` | det candidate generator with speedups and determinism fixes; sources of `cadfit._native` | our fork |
| `image2cad/` | `cadgen_emit.py` | upstream image-to-CAD project |
| `cicada_metrics/` | `gms.py` metric | upstream metrics project |

The DSL dialects are the most dangerous place: see `dsl/PROVENANCE.md`; they must not be mixed up.

## Rules

- **Do not edit files in `vendor/` for local needs.** Whatever has to change is
  wrapped on our side (the `capabilities` layer) rather than patched here;
  otherwise the next snapshot update overwrites the patch unnoticed.
- Each subdirectory has a `PROVENANCE.md` (where it comes from, what was excluded)
  and a `MANIFEST.sha256` (checksums at snapshot time, checked by
  `agent/tools/preflight.py`). **The manifest covers only the snapshotted foreign
  files.** `PROVENANCE.md` is our own record and changes independently; if it were
  under a checksum, editing the document would turn the preflight red with the
  diagnosis "foreign code touched".
- Updating a snapshot is a separate deliberate operation: re-take the snapshot,
  regenerate the manifest, review the diff by eye.
- All snapshots are covered by the repository's MIT license (`LICENSE`).

## Checking that a snapshot has drifted

    cd vendor/cad_optimizer && sha256sum -c MANIFEST.sha256 --quiet   # local edits
