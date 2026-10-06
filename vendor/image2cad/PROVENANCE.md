# cadgen_emit.py: provenance of the snapshot

`cadgen_emit.py` is a single-file snapshot of the code-emission module of an upstream
image-to-CAD benchmark project (`image2cad`). Only this file was taken from the upstream
directory, which otherwise holds run scripts and artifacts. The upstream directory was
not under version control, so the snapshot is identified by its content hash in
`MANIFEST.sha256`, which `agent/tools/preflight.py` checks.

The module is imported in `agent/actions.py` as
`from cadgen_emit import _cut_op, _extrude_op, _revolve_op`, next to `det_candidates`.
It is self-contained: the only external dependency is `numpy`.

Local changes: none to the code; only the snapshot was vendored here unchanged.
