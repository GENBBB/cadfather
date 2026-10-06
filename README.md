# CADFather

Code for **CADFather**, an agent that reconstructs a CAD program (CadQuery code in the
CADENA DSL) from a target mesh. A tool-calling assistant LLM runs the search. On every
turn it looks at the target and the best candidates so far and picks one tool:

| Tool | What it does |
|---|---|
| `stepwise` | appends one operation proposed by the step-wise generator VLM (CADENA) |
| `det_cold` | fits a first operation to the target mesh algorithmically (`vendor/cadfit`) |
| `det_warm` | algorithmic reconstruction continuing from a candidate |
| `optimize` | tunes the numeric parameters of a candidate's code to the target (`vendor/cad_optimizer`) |

When the assistant decides the part is done, it stops. The search policy is `dialogue_lean`
(`agent/cad_agent/scaffold/policies/dialogue_lean.py`). Candidates are scored by
volumetric IoU against the target. GMS and Chamfer distance are reported too.

## Layout

```
agent/
  run_folder.sh          reconstruct a folder of STL files (builds the config, calls run_system.sh)
  run_system.sh          start the vLLM servers and run an experiment
  run_experiment.py      the run itself (called by run_system.sh)
  run_tests.sh           local checks, no GPU needed
  cad_agent/
    harness/             search loop, tool registry, budget, journal, run directory, report
    scaffold/            the dialogue policy and its prompt I/O
    capabilities/        generator and assistant clients, code execution, det, optimizer, metrics, rendering
  configs/
    dialogue_lean.yaml   the run configuration (models, servers, caps, tool set)
  tools/                 run analysis, preflight, native builds
  env/                   package lists and builders of the environments
  tests/                 the checks run by run_tests.sh
vendor/                  vendored dependencies: DSL runtime, det (cadfit), parameter optimizer,
                         metric code (see vendor/README.md)
```

## Requirements

- Linux and 4 GPUs (the configs were run on H100 80GB). GPU 2 serves the generator. GPUs 0, 1
  and 3 serve the assistant (three data-parallel replicas). Edit `cuda_visible_devices` and
  `data-parallel-size` in a config to fit your machine.
- Two Python environments, because the generator needs an older vLLM:
  - **run** (Python 3.12, `agent/env/run_env.txt`): runs the experiment and serves the
    assistant.
  - **gen** (Python 3.10, vLLM 0.10, `agent/env/gen_env.txt`): serves the generator.

  `agent/env/build_env.sh run|gen` creates either one under `$ENV_ROOT` with conda and checks
  the result against the list. The header of each list explains the pins.
- Native modules, built once per run environment:

  ```bash
  ./agent/tools/build_cad_grad.sh      /path/to/envs/cad_run/bin/python   # optimizer gradients
  ./agent/tools/build_cadfit_native.sh /path/to/envs/cad_run/bin/python   # det section analysis
  ```

  Both go to `build_native/`. Without `_cad_grad` the `optimize` tool is disabled. Without
  `cadfit._native` det falls back to slower Python paths, and its output differs.

## Models

- **Generator:** CADENA after RL, subfolder `rl` of [`kulibinai/cadena`](https://huggingface.co/kulibinai/cadena).
  vLLM does not load a hub subfolder, so download it first:

  ```bash
  huggingface-cli download kulibinai/cadena --include 'rl/*' --local-dir /path/to/models/cadena
  ```

  and set `model.generation_model_path: /path/to/models/cadena/rl`.
- **Assistant:** `Qwen/Qwen3.8-27B-FP8`, served with the `qwen3_coder` tool-call parser.

## Running

Replace every `/path/to/...` (models, environments) in `agent/configs/dialogue_lean.yaml`,
or in the config you pass with `--config`. Then, from the repository root:

```bash
agent/run_folder.sh /path/to/my_stls --out /path/to/results
```

Every `*.stl` in the folder is one part. The script copies the base config, points it at the
folder, starts both vLLM servers, runs the agent and stops the servers. Then it writes into
`--out` one `<part>.py` (CadQuery code) and `<part>.stl` per part, plus `results.csv` with
the score, IoU and GMS against the input mesh. Options: `--name`, `--limit N` (first N parts),
`--config <yaml>`. Any `run_system.sh` flag is passed through, e.g. `--keep-servers` to keep
the models loaded between folders. `agent/tools/export_results.py <run_dir> <out>` does the
export step on its own.

Before the first run, `python agent/tools/preflight.py --config <yaml>` checks in seconds,
without a GPU, that the config resolves, the imports work and the parts are readable.

The paper's ablations change only `experiment.tools` in the config (the alternatives are
listed next to it); pass the edited copy with `--config`.

`agent/run_system.sh <yaml>` runs a config as is, with `experiment.details` naming the
folders of parts (`name: path`, several allowed). It starts both vLLM servers, waits until
they are ready, runs the experiment and stops the servers. `--keep-servers` leaves them up
for the next run, and `--no-servers` uses
servers that are already running (`agent/tools/servers.sh list` shows them). The header of
`run_system.sh` lists all flags.

A run writes `work_dirs/<run_name>/`:

```
config.json, provenance.json   resolved config; code digest, servers and weights
dataset.json                   the parts and their signature
per_figure.json, summary.json  per-part results and aggregates
logs/report.txt                human-readable report: quality, failures, cost, time
figures/<figure_id>/           per-part work directory: best.py / best.stl, events.jsonl, journal.json
```

## Analysing runs

```bash
python agent/tools/report_run.py   work_dirs/<run>                     # the report, on any run directory
python agent/tools/compare_runs.py work_dirs/<a> work_dirs/<b>         # paired quality and cost of two runs
./agent/tools/run_tables.py collect work_dirs/<run> --out run.json --cd # per-run JSON (CD computed here)
./agent/tools/run_tables.py table a.json b.json                        # comparison tables
./agent/tools/dialogue_trace_stats.py stats work_dirs/<run>            # how assistant turns ended
```

## Checks

```bash
./agent/env/build_check_env.sh   # a small CPU-only environment for the checks (.venv-checks/)
./agent/run_tests.sh             # all checks; no GPU, no model servers (they are stubbed)
```

## License

MIT, see `LICENSE`. It covers the vendored code under `vendor/` too.
