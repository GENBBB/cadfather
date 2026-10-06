"""Capability "propose the next DSL step" -- the Qwen2-VL generator.

A per-part replacement for the former `StepwiseDataset` + `DataLoader` +
`generate_predictions_server` chain. Batching goes away: N parallel part
processes fill the batch to vLLM, and the server batches requests itself. What
remains is what actually defines the model input -- an image and a point.

The point is chosen by `choose_point`: the point distribution is fixed by the
model's training and is not a harness lever. The GT point cloud is computed once
per part and reused; distances to the prediction use `point_cloud_utils`, as in
`cicada/utils/pipeline.py`.

K step variants are taken with **one** request (the endpoint's `n` parameter),
not K requests from a thread pool: they share the input, and K requests would
mean K image encodings and K prefills for the same prefix. Identical samples
are collapsed before execution.
"""

from __future__ import annotations

import logging
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from pathlib import Path

import numpy as np
import trimesh
from openai import OpenAI

from cad_agent.capabilities import code as code_utils
from cad_agent.capabilities.cache import CacheStats
from cad_agent.capabilities import llm

# The renderer is needed only for the annotation, and the module uses `from
# __future__ import annotations`, so it is never evaluated. A plain import
# pulled `pyvista` into EVERY consumer of the generator: local checks could not
# run in any environment without a graphics stack even though they all supply
# their own renderer. The import graph is an interface too.
if TYPE_CHECKING:
    from cad_agent.capabilities.render import FigureRenderer

logger = logging.getLogger(__name__)


@contextmanager
def _noop():
    """Empty context: without a journal, stages are not timed and cost nothing."""
    yield


@dataclass
class StepProposal:
    """One proposed step: the raw model answer and the code assembled from it."""

    index: int
    raw_text: str
    step_code: str
    full_code: str
    point: str
    reasoning: str = ""
    wall_sec: float = 0.0
    error: str | None = None
    call: Any = None  # llm.LLMCall: latency, tokens, attempt count
    # How many model samples collapsed into this proposal. Duplicates are not
    # executed again, but their count estimates the model's confidence and is
    # worth keeping: the harness may select by it.
    n_samples: int = 1


class StepProposer:
    """Wrapper over the generation endpoint: prev_code + geometry -> K step variants."""

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        renderer: FigureRenderer,
        max_tokens: int = 1024,
        temperature: float = llm.DEFAULT_TEMPERATURE,
        top_p: float = llm.DEFAULT_TOP_P,
        top_k: int = llm.DEFAULT_TOP_K,
        postprocess_code: bool = False,
        max_attempts: int = llm.DEFAULT_MAX_ATTEMPTS,
        dedupe: bool = True,
        journal: Any = None,
    ):
        self.client = client
        self.model_name = model_name
        self.renderer = renderer
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.postprocess_code = postprocess_code
        self.max_attempts = max_attempts
        self.dedupe = dedupe
        self.journal = journal
        # GT point cloud of the part: computed once, then passed to
        # `choose_point` ready-made. The generator lives for exactly one part
        # (`harness/figure_run.py`), but the path key is kept: a swapped GT is
        # better noticed than silently choosing points on foreign geometry.
        self._gt_points_path: str | None = None
        self._gt_points: np.ndarray | None = None
        self._gt_points_stats = CacheStats()

    def gt_points(self, gt_mesh_path: str | Path) -> np.ndarray:
        key = str(Path(gt_mesh_path))
        if self._gt_points_path == key:
            self._gt_points_stats.hit()
        else:
            self._gt_points_stats.miss()
            self._gt_points = sample_gt_points(gt_mesh_path)
            self._gt_points_path = key
            self._gt_points_stats.write()
        assert self._gt_points is not None
        return self._gt_points

    def cache_stats(self) -> dict[str, CacheStats]:
        """GT point cloud counter, for the part's `tech.json`.

        There must be exactly one miss per part: the cloud is sampled at the
        first step and reused afterwards. A second miss means the generator was
        given a different part.
        """
        return {"gt_points": self._gt_points_stats}

    def propose(
        self,
        gt_mesh_path: str | Path,
        pred_mesh_path: str | Path | None,
        prev_code: str,
        k: int,
        seed: int,
        step: int | None = None,
        tag: str = "",
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> list[StepProposal]:
        """Propose k variants of the next step.

        **One request per step, not k.** Variability comes from the endpoint's
        `n` parameter: all k samples share the same input, so k separate
        requests would mean k base64 image encodings, k transfers over the
        network and k prefills on the model side for the same prefix.

        How many samples to ask for is decided by the CALLER (the policy, via
        its `n`); there is no mode that overrides `k` here, because such a mode
        once silently forced a single sample for every policy.

        **`temperature = 0` together with `n > 1` never reaches this point**:
        the harness rejects such a pair (`tools.check_combination`). The
        endpoint does not execute a greedy request with `n > 1` but rejects it
        whole (`n must be 1 when using greedy sampling`); n EMPTY proposals
        would then come back and the `vlm` budget would be charged for all n.

        Identical samples are collapsed (`dedupe`): executing and measuring the
        same code twice pays twice for one answer. The multiplicity is kept in
        `StepProposal.n_samples`. At zero temperature everything but one
        collapses, and then `n > 1` was bought in vain.
        """
        if k <= 0:
            return []
        n_samples = k

        journal = self.journal
        if journal is not None:
            with journal.stage("render"):
                image = self.renderer.step_image(gt_mesh_path, pred_mesh_path)
            with journal.stage("choose_point"):
                point, _ = choose_point(
                    gt_mesh_path, pred_mesh_path, point_seed=seed,
                    gt_points=self.gt_points(gt_mesh_path),
                )
            if step is not None:
                # Save what was actually given to the model, not what should
                # have been: otherwise the log cannot be cross-checked.
                journal.save_step_input(step, prompt=point, image=image, tag=tag)
        else:
            image = self.renderer.step_image(gt_mesh_path, pred_mesh_path)
            point, _ = choose_point(
                gt_mesh_path, pred_mesh_path, point_seed=seed,
                gt_points=self.gt_points(gt_mesh_path),
            )
        # Request knobs. Only the policy can name them
        # (`ToolSpec.params_schema`); if it does not, the registry default that
        # the tool substituted for it applies. The run has no values of its
        # own: the `experiment.generation.{greedy,temperature,top_p,top_k}`
        # config keys do not exist.
        generation_kwargs = llm.make_generation_kwargs(
            temperature=self.temperature if temperature is None else float(temperature),
            max_tokens=self.max_tokens,
            top_p=self.top_p if top_p is None else float(top_p),
            top_k=self.top_k,
            seed=seed,
        )

        started = time.monotonic()
        with (journal.stage("generation") if journal is not None else _noop()):
            call = llm.call_vision(
                client=self.client,
                model_name=self.model_name,
                text=point,
                image=image,
                generation_kwargs=generation_kwargs,
                max_attempts=self.max_attempts,
                n=n_samples,
            )
        wall_sec = time.monotonic() - started
        if journal is not None:
            journal.record_llm("vlm", call)

        # A failed request is still k failed variants, not one: the harness
        # counts step attempts, and collapsing them here would show it that the
        # generator was queried less often than it really was.
        raw_texts = call.texts if call.texts else [call.text or ""] * n_samples

        proposals: list[StepProposal] = []
        by_code: dict[str, StepProposal] = {}
        for raw_text in raw_texts:
            reasoning = ""
            step_code = raw_text
            if self.postprocess_code:
                reasoning = code_utils.extract_think(raw_text)
                step_code = code_utils.extract_code(raw_text)

            if self.dedupe and call.error is None:
                twin = by_code.get(step_code)
                if twin is not None:
                    twin.n_samples += 1
                    continue

            proposal = StepProposal(
                index=len(proposals),
                raw_text=raw_text,
                step_code=step_code,
                full_code=f"{prev_code}\n{step_code}",
                point=point,
                reasoning=reasoning,
                wall_sec=wall_sec,
                error=call.error,
                call=call,
            )
            proposals.append(proposal)
            by_code[step_code] = proposal

        if journal is not None:
            journal.event(
                "generate",
                step=step,
                tag=tag or None,
                requested=n_samples,
                returned=len(raw_texts),
                unique=len(proposals),
                latency_sec=round(call.latency_sec, 3),
            )

        if journal is not None and step is not None:
            for proposal in proposals:
                journal.save_step_output(
                    step=step,
                    index=proposal.index,
                    tag=tag,
                    raw_answer=proposal.raw_text,
                    reasoning=proposal.reasoning,
                    step_code=proposal.step_code,
                )
        return proposals


N_SURFACE_POINTS = 10_000

# Sampling seed of the GT cloud. A constant, not a harness parameter: the cloud
# describes the target and must be the same at all steps of a part, otherwise
# points of different steps come from different samples.
GT_SAMPLE_SEED = 42


def sample_gt_points(gt_mesh_path: str | Path, seed: int | None = None) -> np.ndarray:
    """GT point cloud in the model's coordinate frame (extent normalized to 200).

    Kept separate because it is reused: the cloud is computed once per rollout
    and then passed ready-made to `choose_point` (`utils/pipeline.py` in cicada
    works the same way -- `get_point` accepts an `np.ndarray` instead of a mesh).
    """
    target_mesh = trimesh.load(gt_mesh_path)
    center = (target_mesh.bounds[0] + target_mesh.bounds[1]) / 2.0
    target_mesh.apply_translation(-center)
    extent = np.max(target_mesh.extents)
    if extent > 1e-7:
        target_mesh.apply_scale(200.0 / extent)

    seed = GT_SAMPLE_SEED if seed is None else seed
    return np.array(trimesh.sample.sample_surface(target_mesh, N_SURFACE_POINTS, seed=seed)[0])


def _signed_distance(points: np.ndarray, pred_mesh: Any) -> np.ndarray:
    """|Distance| from GT points to the prediction surface.

    The fast path is `point_cloud_utils`, exactly what cicada uses
    (`utils/pipeline.py`), where the slow trimesh variant sits commented out.
    The difference is not cosmetic: `nearest.signed_distance` is linear in the
    number of faces and is slow on thousands of faces and 10k points.

    The fallback is kept for environments without `point_cloud_utils`: the
    missing accelerator must not crash a run, only slow it down.
    """
    try:
        import point_cloud_utils as pcu
    except ImportError:
        return np.abs(pred_mesh.nearest.signed_distance(points))

    dist, _, _ = pcu.signed_distance_to_mesh(
        points.astype(np.float64),
        np.asarray(pred_mesh.vertices, dtype=np.float64),
        np.asarray(pred_mesh.faces, dtype=np.int32),
    )
    return np.abs(dist)


def choose_point(
    gt_mesh_path: str | Path,
    pred_mesh_path: str | Path | None,
    point_seed: int,
    gt_points: np.ndarray | None = None,
) -> tuple[str, np.ndarray]:
    """A point on the target surface -- the second half of the model input.

    The distance thresholds (3/10/20) and the normalization to 200 are set by
    the training distribution and are not harness levers.

    Returns the point **and the cloud**, so the caller can pass it back at the
    next step: sampling is the same across the whole part.

    `point_seed` is computed by the harness (`harness.seeding.derive_seed`) from
    run, part, step and branch. It deliberately has no default: the point index
    used to come from the global `random`, making runs irreproducible, and a
    forgotten seed must fail with `TypeError` on the first call rather than
    quietly return an unseeded draw.

    The point cloud is seeded **not** by this seed but by the constant
    `GT_SAMPLE_SEED`: the cloud describes the target and must be the same at all
    steps of a part, whereas the point must change from step to step.
    """
    points = sample_gt_points(gt_mesh_path) if gt_points is None else gt_points
    # A local generator, not `random.seed(...)`: seeding the global one would
    # affect every neighbour in the process, and the part shares it with
    # rendering and metrics.
    rng = random.Random(point_seed)

    if pred_mesh_path is None:
        point = tuple(map(int, points[rng.randrange(len(points))].round()))
        return str(point).replace(" ", ""), points

    pred_mesh = trimesh.load(pred_mesh_path)
    dist = _signed_distance(points, pred_mesh)

    indices = np.where(np.logical_and(dist > 3, dist < 10))[0]
    if len(indices) == 0:
        indices = np.where(np.logical_and(dist > 10, dist < 20))[0]
    if len(indices) == 0:
        indices = np.where(dist < 3)[0]
    if len(indices) == 0:
        indices = np.where(dist > 0)[0]
    if len(indices) == 0:
        indices = np.arange(len(points))

    # The local generator's seed includes the **step**, so "one cloud per part"
    # and "a different point at each step" do not conflict: the draw differs
    # across steps and is the same between runs.
    point = tuple(map(int, points[rng.choice(indices)].round()))
    return str(point).replace(" ", ""), points
