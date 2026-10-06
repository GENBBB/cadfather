#!/usr/bin/env python3
"""Check of rendering on live geometry: `pyvista` is available locally too.

Run: ``python agent/tests/render_check.py`` from the ``agent`` directory.

The renderer was moved from `utils.py` verbatim, because it is literally the model input:
the views, size, colors and channel merging are set by training. This checks that the
move changed neither the image geometry nor the channel semantics, and that the cache
really saves a render rather than returning garbage.

Needs only `trimesh` + `pyvista`; no CAD or models.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_DIR))

from cad_agent import dsl_runtime  # noqa: E402

dsl_runtime.configure("wrapped")

from cad_agent.capabilities.render import FigureRenderer, Plotter, plotter_class  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if condition else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def make_meshes(tmp: Path) -> tuple[Path, list[Path]]:
    """GT at its own scale, predictions at the model scale (about +-100)."""
    gt_path = tmp / "gt.stl"
    trimesh.creation.box((30, 20, 10)).export(gt_path)

    pred_paths = []
    for idx, mesh in enumerate(
        [
            trimesh.creation.box((150, 100, 50)),
            trimesh.creation.cylinder(radius=60, height=120),
            trimesh.creation.icosphere(subdivisions=3, radius=80),
        ]
    ):
        path = tmp / f"pred{idx}.stl"
        mesh.export(path)
        pred_paths.append(path)
    return gt_path, pred_paths


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="render_check_"))
    gt_path, pred_paths = make_meshes(tmp)
    renderer = FigureRenderer()

    try:
        print("0. Plotter parameters match those wrapped was trained on")
        # Reference: cicada/visualization_new.py. All of this defines the model input; a
        # divergence here silently spoils quality without breaking anything, so it is
        # recorded as numbers, not as the words "as in cicada".
        for name, actual, expected in (
            ("tile side", Plotter.VIEW_IMAGE_SIZE, 14 * 18),
            ("view order", Plotter.VIEW_ORDER, ("+Z", "-Z", "+Y", "-Y", "+X", "-X")),
            ("ortho zoom", Plotter.ORTHO_ZOOM, 1.73),
            ("mirrored views", Plotter.ORTHO_FLIP, frozenset({"-Z", "-Y", "-X"})),
            ("iso zoom", Plotter.ISO_ZOOM, 1.0),
            ("-Iso is mirrored", Plotter.ISO_FLIP_NEGATIVE, True),
        ):
            check(f"wrapped: {name}", actual == expected, f"{actual!r} vs {expected!r}")

        chain_cls = plotter_class("chain")
        check(
            "chain differs from wrapped",
            chain_cls.VIEW_IMAGE_SIZE == 14 * 17 and chain_cls.VIEW_ORDER[0] == "-Z",
            f"{chain_cls.VIEW_IMAGE_SIZE}, {chain_cls.VIEW_ORDER}",
        )
        check("the active dialect plotter is the default", plotter_class() is Plotter)

        print("1. Render roles and channel semantics")
        gt_img = renderer.render(gt_path, "gt")
        pred_img = renderer.render(pred_paths[0], "pred")

        gt_arr, pred_arr = np.asarray(gt_img), np.asarray(pred_img)
        tile = Plotter.VIEW_IMAGE_SIZE
        expected_size = (2 * tile, 4 * tile)
        check("image size matches wrapped", gt_img.size == expected_size, f"{gt_img.size} vs {expected_size}")
        check("tile side is a multiple of 28 (Qwen2-VL patch)", tile % 28 == 0, f"{tile} % 28 = {tile % 28}")
        check("GT is not empty", int((gt_arr.sum(axis=2) > 0).sum()) > 1000, f"{int((gt_arr.sum(axis=2) > 0).sum())} pixels")
        check("GT is green only", gt_arr[..., 0].max() == 0 and gt_arr[..., 2].max() == 0 and gt_arr[..., 1].max() > 0)
        check("prediction is not empty", int((pred_arr.sum(axis=2) > 0).sum()) > 1000)
        check("prediction is red only", pred_arr[..., 1].max() == 0 and pred_arr[..., 2].max() == 0 and pred_arr[..., 0].max() > 0)

        print("2. Channel merge at the step generator input")
        step_img = np.asarray(renderer.step_image(gt_path, pred_paths[0]))
        check("red channel comes from the prediction", np.array_equal(step_img[..., 0], pred_arr[..., 0]))
        check("green channel comes from GT", np.array_equal(step_img[..., 1], gt_arr[..., 1]))
        check("blue channel comes from GT", np.array_equal(step_img[..., 2], gt_arr[..., 2]))

        first_step = np.asarray(renderer.step_image(gt_path, None))
        check("without a prediction plain GT is shown", np.array_equal(first_step, gt_arr))

        print("3. Render cache")
        # mtime is no indicator here: `MeshRenderCache.load` deliberately does `os.utime`,
        # since LRU eviction relies on access time. So we count the renders themselves.
        renders = {"count": 0}
        original = Plotter._get_img_stepwise

        def counting(self, *args, **kwargs):
            renders["count"] += 1
            return original(self, *args, **kwargs)

        Plotter._get_img_stepwise = counting
        try:
            again = np.asarray(renderer.render(gt_path, "gt"))
            check("a repeated render comes from the cache", renders["count"] == 0, f"renders: {renders['count']}")
            renderer.render(tmp / "pred1.stl", "pred")
            check("a cold mesh is rendered", renders["count"] == 1, f"renders: {renders['count']}")
        finally:
            Plotter._get_img_stepwise = original
        check("the cache returns the same pixels", np.array_equal(again, gt_arr))
        check("roles do not mix in the cache", renderer.cache.key(gt_path, "gt") != renderer.cache.key(gt_path, "pred"))

        print("3a. The prediction cache lives in memory, not on disk")
        # Why it moved: with the disk cache there were many PNG writes to NFS and zero hits.
        # This also checks that a hit happens (a repeat of the same path) and that the disk is
        # not touched.
        pred_one = tmp / "pred1.stl"
        renders["count"] = 0
        Plotter._get_img_stepwise = counting
        try:
            first = np.asarray(renderer.render(pred_one, "pred"))
            drawn_after_first = renders["count"]
            second = np.asarray(renderer.render(pred_one, "pred"))
            check("repeating the same prediction does not redraw",
                  renders["count"] == drawn_after_first, f"renders: {renders['count']}")
            check("the cache returns the same prediction pixels", np.array_equal(first, second))
        finally:
            Plotter._get_img_stepwise = original
        check("nothing is written to disk", not any(tmp.rglob("*.png")),
              str([str(p) for p in tmp.rglob("*.png")][:3]))

        # The cache is limited by the number of images: a small cache evicts and honestly
        # reports it, otherwise "no hits" cannot be told from "no cache".
        small = FigureRenderer(max_images=1)
        try:
            small.render(pred_paths[0], "pred")
            small.render(pred_paths[1], "pred")
            small.render(pred_paths[0], "pred")
            stats = small.cache_stats()["render_pred"]
            check("eviction counted", stats.evictions >= 1, f"evictions: {stats.evictions}")
            check("a miss after eviction", stats.misses == 3, f"misses: {stats.misses}")
        finally:
            small.close()

        # The returned image is not tied to the one in the cache: an edit by one caller must
        # not reach another (the disk cache never lost this property, memory could).
        held = renderer.render(pred_one, "pred")
        held.paste(Image.new("RGB", held.size, (7, 7, 7)), (0, 0))
        check("editing a returned image does not corrupt the cache",
              np.array_equal(np.asarray(renderer.render(pred_one, "pred")), first))

        print("4. Selection images")
        labels = ["A", "B", "C"]
        for mode in ("simple", "stepwise"):
            image = renderer.selection_image(
                gt_mesh_path=gt_path,
                pred_mesh_paths=pred_paths,
                labels=labels,
                visualization_mode=mode,
            )
            arr = np.asarray(image)
            check(f"mode {mode}: image is not empty", int((arr.sum(axis=2) > 0).sum()) > 5000)
            check(f"mode {mode}: enough panels for everyone", image.size[0] >= expected_size[0])

        print("4a. The target as a separate image")
        target_image = renderer.target_image(gt_path)
        arr = np.asarray(target_image)
        check("the target is drawn without candidates", int((arr.sum(axis=2) > 0).sum()) > 5000)
        # The same panel that stands as the TARGET panel in the collage: two different images
        # of one target are two different targets from the reader's point of view.
        check("the panel has the same size as in the collage",
              target_image.size == expected_size, f"{target_image.size} vs {expected_size}")

        print("4b. Panels as a list, not a collage")
        panels = renderer.panel_images(
            gt_mesh_path=gt_path, pred_mesh_paths=pred_paths, labels=labels,
            visualization_mode="simple",
        )
        check("one more panel than candidates", len(panels) == len(pred_paths) + 1,
              f"{len(panels)} for {len(pred_paths)} candidates")
        check("every panel has the same size as in the collage",
              all(panel.size == expected_size for panel in panels),
              str({panel.size for panel in panels}))
        check("panels are not empty",
              all(int((np.asarray(panel).sum(axis=2) > 0).sum()) > 5000 for panel in panels))

        # The caption is drawn on a COPY: the stepwise mode takes panels from the cache, and
        # a caption drawn in place would spoil the cached image for all readers.
        before = np.asarray(renderer.gt_image(gt_path)).copy()
        renderer.panel_images(
            gt_mesh_path=gt_path, pred_mesh_paths=[pred_paths[0]], labels=["A"],
            visualization_mode="stepwise",
        )
        check("captions do not corrupt the GT cache",
              np.array_equal(np.asarray(renderer.gt_image(gt_path)), before))

        print("5. A candidate without a mesh does not break selection")
        for mode in ("simple", "stepwise"):
            image = renderer.selection_image(
                gt_mesh_path=gt_path,
                pred_mesh_paths=[pred_paths[0], None, tmp / "no-such.stl"],
                labels=labels,
                visualization_mode=mode,
            )
            check(f"mode {mode}: invalid panels are drawn", np.asarray(image).sum() > 0)

        print("6. GT is drawn once per part")
        renders = {"count": 0}
        original = Plotter._get_img_stepwise

        def counting_gt(self, *args, **kwargs):
            renders["count"] += 1
            return original(self, *args, **kwargs)

        Plotter._get_img_stepwise = counting_gt
        try:
            fresh = FigureRenderer()
            fresh.gt_image(gt_path)
            after_first = renders["count"]
            for _ in range(3):
                fresh.step_image(gt_path, None)
                fresh.gt_image(gt_path)
            check(
                "repeated GT accesses do not redraw",
                renders["count"] == after_first,
                f"renders: {renders['count']} vs {after_first}",
            )
            # The main reason this was written: selection among k candidates need not pay k
            # renders of GT.
            before = renders["count"]
            fresh.selection_image(
                gt_mesh_path=gt_path,
                pred_mesh_paths=pred_paths,
                labels=labels,
                visualization_mode="stepwise",
            )
            drawn = renders["count"] - before
            check(
                "selection draws only predictions",
                drawn == len(pred_paths),
                f"renders {drawn} for {len(pred_paths)} candidates (GT would be drawn on top)",
            )
            fresh.close()
        finally:
            Plotter._get_img_stepwise = original

    finally:
        renderer.close()

    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {', '.join(FAILURES)}")
        sys.exit(1)
    print("Rendering is fine.")


if __name__ == "__main__":
    main()
