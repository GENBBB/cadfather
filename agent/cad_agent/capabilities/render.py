"""Geometry rendering: pictures for the step generator and for the decision agent.

Rendering is done in place, in the part's process, and cached: the GT of one part is
drawn once for the whole rollout, and predictions are reused between the step and
selection. There is no batch on the per-figure scheme (one prediction is rendered per
step), so a process pool would degenerate into a fork for a single picture.

**The plotter must match the dialect.** In the reference training code the match is
set explicitly: a model of the `wrapped` dialect was trained on
`visualization_new.Plotter`, a model of the `chain` dialect on the older
`visualization.Plotter`. These are different pictures: a different tile size, a
different view order, different cameras along the Y axis, different zooms, and a
different half of the views is mirrored. Both are ported here, and the choice follows
`dsl_runtime.active_dialect()` rather than a default.

A collage of the wrong size or tile order, or with views mirrored the wrong way, is
a different input for the model (a side not divisible by 28 also makes the Qwen2-VL
processor resize it), so the plotter must never be picked by default.
"""

from __future__ import annotations

import logging
import os
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyvista as pv
import vtk
from PIL import Image, ImageDraw, ImageFont, ImageOps

from cad_agent.capabilities.cache import CacheStats

# How many prediction images to keep in memory per part. Reuse lives two or three
# steps and the beam width and candidates per step are single digits, so sixteen is
# plenty; evictions are visible in `tech.json` (`evictions`).
DEFAULT_RENDER_CACHE_IMAGES = 16

pv.OFF_SCREEN = True
vtk.vtkObject.GlobalWarningDisplayOff()

warnings.filterwarnings("ignore", category=UserWarning, module="trimesh")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"trimesh\.triangles")

logger = logging.getLogger(__name__)

# The label font lives IN THE REPOSITORY, not at an absolute path on a node.
# A miss used to be silenced by falling back to `load_default`: labels were drawn in
# another font, the picture for the assistant changed silently, and the difference
# leaked into its answers.
FONT_PATH = str(Path(__file__).resolve().parents[2] / "assets" / "DejaVuSans-Bold.ttf")


def _label_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Label font; the fallback is loud because it changes the picture."""
    try:
        return ImageFont.truetype(FONT_PATH, size=size)
    except OSError as exc:
        logger.warning(
            "Label font could not be opened (%s: %s) — labels will use the default "
            "font, the image will be DIFFERENT and not comparable with earlier runs",
            FONT_PATH, exc,
        )
        return ImageFont.load_default(size=size)

# Collage tile size for wrapped. The multiple-of-28 check is also in the original:
# Qwen2-VL has patch 14 and a 2x2 merge, so a side not divisible by 28 makes the
# processor resize the collage, and the model gets a picture it never saw in training.
WRAPPED_VIEW_IMAGE_SIZE = 14 * 18
assert WRAPPED_VIEW_IMAGE_SIZE % 28 == 0

# Chain was trained on 14*17 = 238, i.e. on a side that is not a multiple of 28. This
# is not a typo but a fact about that model, and it must not be changed.
CHAIN_VIEW_IMAGE_SIZE = 14 * 17


# The process in which a VTK window was opened, or None. Read by `harness.pool`; see
# `plotter_open_here()` below.
_PLOTTER_PID: int | None = None


def plotter_open_here() -> bool:
    """Whether a VTK window is open **in this very process**.

    A pid rather than a boolean flag: a forked child inherits the parent's value, and a
    flag would tell it "the window is already open" when it has no window of its own.
    """
    return _PLOTTER_PID == os.getpid()


class Plotter:
    """View set of the **wrapped** dialect, a port of `visualization_new.py`.

    Everything that defines the model's input is exposed as class attributes: the
    chain version differs by exactly these plus the Y-axis cameras (see `ChainPlotter`).
    """

    VIEW_IMAGE_SIZE = WRAPPED_VIEW_IMAGE_SIZE
    # The order determines the tile layout in the collage, not just the set.
    VIEW_ORDER = ("+Z", "-Z", "+Y", "-Y", "+X", "-X")
    ORTHO_ZOOM = 1.73
    ORTHO_FLIP = frozenset({"-Z", "-Y", "-X"})
    ISO_ZOOM = 1.0
    # Which of the two isometric views is mirrored: for wrapped, the negative one.
    ISO_FLIP_NEGATIVE = True

    def __init__(self, scale_gt: bool = True, scale_pred: bool = False):
        self.scale_gt = scale_gt
        self.scale_pred = scale_pred
        self.plotter = None
        self.iso_plotter = None

        self.view_img_size = self.VIEW_IMAGE_SIZE
        self.rows = 4
        self.cols = 2
        self.mesh_render_img_size = self.view_img_size * 2

        by_name = {
            "-Z": self.minus_z_view,
            "+Z": self.plus_z_view,
            "+Y": self.plus_y_view,
            "-Y": self.minus_y_view,
            "+X": self.plus_x_view,
            "-X": self.minus_x_view,
        }
        self.views = {name: by_name[name] for name in self.VIEW_ORDER}

        self.align_coordinates = True
        self.cmap_gt = pv.LookupTable(
            values=np.array([[0, c, 0, 255] for c in range(0, 256)]),
            scalar_range=(0, 255),
            ramp="linear",
        )
        self.cmap_pred = pv.LookupTable(
            values=np.array([[c, 0, 0, 255] for c in range(0, 256)]),
            scalar_range=(0, 255),
            ramp="linear",
        )
        self.reload()

    def get_img_stepwise(
        self,
        gt_mesh_path: str | Path,
        pred_mesh_path: str | Path | None,
        apply_augs: bool = False,
        apply_noise: bool = False,
        noise_scale: float = 0.25,
    ) -> Image.Image:
        image = self._get_img_stepwise(
            mesh_path=gt_mesh_path,
            cmap=self.cmap_gt,
            apply_augs=apply_augs,
            color=(0, 255, 0),
            scale=self.scale_gt,
            apply_noise=apply_noise,
            noise_scale=noise_scale,
        )

        if pred_mesh_path:
            pred_img = self._get_img_stepwise(
                mesh_path=pred_mesh_path,
                cmap=self.cmap_pred,
                apply_augs=False,
                color=(255, 0, 0),
                scale=self.scale_pred,
                apply_noise=False,
            )
            gt_r, gt_g, gt_b = image.split()
            pred_r, _, _ = pred_img.split()
            image = Image.merge("RGB", (pred_r, gt_g, gt_b))

        return image

    def _get_img_stepwise(
        self,
        mesh_path: str | Path,
        cmap,
        apply_augs: bool = False,
        color=None,
        scale: bool = True,
        apply_noise: bool = False,
        noise_scale: float = 0.25,
    ) -> Image.Image:
        mesh = pv.read(mesh_path)

        if scale:
            mesh.translate(
                [
                    -0.5 * (mesh.bounds.x_min + mesh.bounds.x_max),
                    -0.5 * (mesh.bounds.y_min + mesh.bounds.y_max),
                    -0.5 * (mesh.bounds.z_min + mesh.bounds.z_max),
                ],
                inplace=True,
            )
            max_span = max(
                mesh.bounds.x_max - mesh.bounds.x_min,
                mesh.bounds.y_max - mesh.bounds.y_min,
                mesh.bounds.z_max - mesh.bounds.z_min,
            )
            mesh.scale(200.0 / max_span, inplace=True)

        if apply_noise:
            mesh.points += np.random.normal(0, noise_scale, size=mesh.points.shape)

        mesh.point_data.update(self.get_scalars(mesh))
        mesh_actor = self.plotter.add_mesh(
            mesh,
            reset_camera=False,
            color=None,
            scalars=None,
            cmap=cmap,
            show_scalar_bar=False,
        )
        mesh_actor.use_bounds = False

        view_images = []
        for view_name, set_view_func in self.views.items():
            set_view_func(mesh)
            self.plotter.enable_parallel_projection()
            self.plotter.zoom_camera(self.ORTHO_ZOOM)

            img_array = self.plotter.screenshot(return_img=True)
            pil_img = Image.fromarray(img_array)
            pil_img.thumbnail((self.view_img_size, self.view_img_size), resample=Image.Resampling.BILINEAR)
            if self.align_coordinates and view_name in self.ORTHO_FLIP:
                pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
            view_images.append(pil_img)

        self.remove_meshes(mesh_actor)

        mesh_actor = self.iso_plotter.add_mesh(mesh, reset_camera=False, color=color)
        mesh_actor.use_bounds = False

        for negative in (False, True):
            self.iso_plotter.view_isometric(negative=negative)
            self.iso_plotter.zoom_camera(self.ISO_ZOOM)

            img_array = self.iso_plotter.screenshot(return_img=True)
            pil_img = Image.fromarray(img_array)
            pil_img.thumbnail((self.view_img_size, self.view_img_size), resample=Image.Resampling.BILINEAR)
            if self.align_coordinates and negative == self.ISO_FLIP_NEGATIVE:
                pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
            view_images.append(pil_img)

        success = self.iso_plotter.remove_actor(mesh_actor, reset_camera=False, render=False)
        if not success:
            self.reload()

        total_width = self.cols * self.view_img_size
        total_height = self.rows * self.view_img_size
        collage = Image.new("RGB", (total_width, total_height), color="white")

        for idx, img in enumerate(view_images):
            row = idx // self.cols
            col = idx % self.cols
            collage.paste(img, (col * self.view_img_size, row * self.view_img_size))

        return collage

    def reload(self) -> None:
        # Mark "a VTK window is open in this process". It is needed not by rendering but
        # by the part pool: a fork from a process with an open window kills the child on
        # its own `pv.Plotter` by heap corruption, with no message from us
        # (`malloc(): unaligned tcache chunk detected`). Cheaper to record the fact here
        # than to diagnose it from the trace of a dead worker.
        global _PLOTTER_PID
        _PLOTTER_PID = os.getpid()
        self.plotter = pv.Plotter(
            off_screen=True,
            window_size=(self.mesh_render_img_size, self.mesh_render_img_size),
            lighting="none",
        )
        self.plotter.set_background("black")

        self.iso_plotter = pv.Plotter(
            off_screen=True,
            window_size=(self.mesh_render_img_size, self.mesh_render_img_size),
        )
        self.iso_plotter.set_background("black")

        lim_points = [(x, y, z) for x in (-100, 100) for y in (-100, 100) for z in (-100, 100)]
        self.plotter.add_points(np.array(lim_points, dtype=float), color=(1, 1, 1), opacity=0, point_size=1)
        self.iso_plotter.add_points(np.array(lim_points, dtype=float), color=(1, 1, 1), opacity=0, point_size=1)

    def get_scalars(self, mesh) -> dict[str, np.ndarray]:
        shift = 100
        scale = 255 / 200

        x_coords = mesh.points[:, 0]
        y_coords = mesh.points[:, 1]
        z_coords = mesh.points[:, 2]

        return {
            "+X": (x_coords + shift) * scale,
            "-X": 255 - ((x_coords + shift) * scale),
            "+Y": (y_coords + shift) * scale,
            "-Y": 255 - ((y_coords + shift) * scale),
            "+Z": (z_coords + shift) * scale,
            "-Z": 255 - ((z_coords + shift) * scale),
        }

    def remove_meshes(self, mesh) -> None:
        success = self.plotter.remove_actor(mesh, reset_camera=False, render=False)
        if not success:
            self.reload()

    def minus_z_view(self, mesh) -> None:
        self.plotter.view_xy(negative=True)
        mesh.set_active_scalars("-Z")

    def plus_z_view(self, mesh) -> None:
        self.plotter.view_xy()
        mesh.set_active_scalars("+Z")

    def plus_y_view(self, mesh) -> None:
        self.plotter.view_zx()
        mesh.set_active_scalars("+Y")

    def minus_y_view(self, mesh) -> None:
        self.plotter.view_zx(negative=True)
        mesh.set_active_scalars("-Y")

    def plus_x_view(self, mesh) -> None:
        self.plotter.view_yz()
        mesh.set_active_scalars("+X")

    def minus_x_view(self, mesh) -> None:
        self.plotter.view_yz(negative=True)
        mesh.set_active_scalars("-X")


class ChainPlotter(Plotter):
    """View set of the **chain** dialect, a port of the older `visualization.py::Plotter`.

    Kept because earlier checkpoints were trained on it, and a run with `dsl: chain`
    must show the model exactly its picture. There are eight differences from wrapped,
    and all of them are here: chain differs from wrapped nowhere else.
    """

    VIEW_IMAGE_SIZE = CHAIN_VIEW_IMAGE_SIZE
    VIEW_ORDER = ("-Z", "+Z", "+Y", "-Y", "+X", "-X")
    ORTHO_ZOOM = 1.7
    ORTHO_FLIP = frozenset({"-Z", "+Y", "+X"})
    ISO_ZOOM = 1.1
    ISO_FLIP_NEGATIVE = False

    def plus_y_view(self, mesh) -> None:
        self.plotter.view_xz(negative=True)
        mesh.set_active_scalars("+Y")

    def minus_y_view(self, mesh) -> None:
        self.plotter.view_xz()
        mesh.set_active_scalars("-Y")


PLOTTERS = {"wrapped": Plotter, "chain": ChainPlotter}


def plotter_class(dialect: str | None = None) -> type[Plotter]:
    """Plotter class for a dialect; the active one by default.

    The dialect is asked of the runtime rather than taken from the config a second
    time: `dsl_runtime` has already decided which `cadgen` is on `sys.path`, and the
    picture must belong to the same model as the code.
    """
    if dialect is None:
        from cad_agent import dsl_runtime

        dialect = dsl_runtime.active_dialect()
    try:
        return PLOTTERS[dialect]
    except KeyError:
        raise ValueError(
            f"Unknown dialect {dialect!r}: no plotter for it, expected one of {list(PLOTTERS)}"
        ) from None



def build_invalid_select_image(size: tuple[int, int], text: str = "INVALID\nNO STL") -> Image.Image:
    image = Image.new("RGB", size, color="black")
    draw = ImageDraw.Draw(image)

    font = _label_font(48)

    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=8)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (size[0] - text_w) // 2
    y = (size[1] - text_h) // 2

    draw.multiline_text((x, y), text, fill="white", font=font, spacing=8, align="center")
    return image


class MeshRenderCache:
    """Rendered predictions, kept in the part process's memory rather than on disk.

    This used to be a PNG cache in the part directory. Its reuse is always within one
    part and lives two or three steps, while we paid with writes to NFS and got
    **zero** hits: the fast branch with `max_failed_steps: 1` stops at the first
    rejected step, and while steps are accepted the prediction path is new every time.

    The cache is kept because other harness modes do get hits: the agent branch
    renders candidates for selection and the accepted one becomes the prediction of the
    next step (the same picture), and in a beam the surviving branch continues with the
    same mesh.

    The limit is by image count, not bytes: image size is set by the plotter and is the
    same for all, while counting bytes in memory would require a walk.
    """

    def __init__(self, max_images: int = DEFAULT_RENDER_CACHE_IMAGES):
        self.max_images = max(0, int(max_images))
        self._images: "OrderedDict[str, Image.Image]" = OrderedDict()
        self.stats = CacheStats()

    def key(self, mesh_path: Path, role: str) -> str:
        if role not in {"gt", "pred"}:
            raise ValueError(f"Unknown mesh render role: {role}")
        return f"{Path(mesh_path).resolve()}::{role}"

    def get(self, mesh_path: Path, role: str) -> Image.Image | None:
        """Image from the cache or `None`. The cache itself counts the access."""
        key = self.key(mesh_path, role)
        image = self._images.get(key)
        if image is None:
            self.stats.miss()
            return None
        self._images.move_to_end(key)
        self.stats.hit()
        # A copy, not the object itself: the disk cache gave every caller its own
        # image, and an edit by one did not reach another. This property must not be
        # lost in the move to memory, since PIL images are mutable.
        return image.copy()

    def put(self, mesh_path: Path, role: str, image: Image.Image) -> None:
        if self.max_images <= 0:
            return
        key = self.key(mesh_path, role)
        self._images[key] = image.copy()
        self._images.move_to_end(key)
        self.stats.write()
        while len(self._images) > self.max_images:
            self._images.popitem(last=False)
            self.stats.evict()

    def clear(self) -> None:
        self._images.clear()


def build_selection_image(
    gt_mesh_path: str | Path,
    pred_mesh_paths: list[str | Path | None],
    labels: list[str],
    visualization_mode: str = "stepwise",
    plotter: "Plotter | None" = None,
) -> Image.Image:
    """Image for candidate selection.

    Takes paths rather than candidate objects: rendering is a capability and must not
    know about search state. `plotter` is passed in to avoid starting VTK at every
    step; if not given, a temporary one is created and closed right here.
    """
    if visualization_mode == "simple":
        return build_simple_selection_image(gt_mesh_path, pred_mesh_paths, labels, plotter=plotter)
    return build_stepwise_selection_image(gt_mesh_path, pred_mesh_paths, labels, plotter=plotter)


def build_stepwise_selection_image(
    gt_mesh_path: str | Path,
    pred_mesh_paths: list[str | Path | None],
    labels: list[str],
    plotter: "Plotter | None" = None,
) -> Image.Image:
    own_plotter = plotter is None
    plotter = plotter or plotter_class()(scale_gt=True, scale_pred=False)
    try:
        images = [
            plotter.get_img_stepwise(
                gt_mesh_path=gt_mesh_path,
                pred_mesh_path=pred_mesh_path,
                apply_augs=False,
                apply_noise=False,
                noise_scale=0.0,
            )
            for pred_mesh_path in pred_mesh_paths
        ]
        return build_image_grid(images=images, labels=labels)
    finally:
        if own_plotter:
            _close_plotter(plotter)


def build_simple_selection_image(
    gt_mesh_path: str | Path,
    pred_mesh_paths: list[str | Path | None],
    labels: list[str],
    plotter: "Plotter | None" = None,
) -> Image.Image:
    """Separate panels without overlays: the target on the left, candidates next to it.

    Requires its own Plotter (`scale_pred=True`): in this mode the prediction is scaled,
    while in the stepwise mode it is not.
    """
    if not pred_mesh_paths:
        raise ValueError("No candidates for the plain selection image")

    own_plotter = plotter is None
    plotter = plotter or plotter_class()(scale_gt=True, scale_pred=True)
    try:
        target = render_simple_mesh(plotter=plotter, mesh_path=gt_mesh_path, role="target", scale=True)
        predictions = [
            render_simple_mesh(plotter=plotter, mesh_path=pred_mesh_path, role="prediction", scale=True)
            for pred_mesh_path in pred_mesh_paths
        ]
        return build_simple_comparison_grid(target=target, predictions=predictions, labels=labels)
    finally:
        if own_plotter:
            _close_plotter(plotter)


def label_panel(image: Image.Image, label: str) -> Image.Image:
    """A labeled COPY of the panel.

    A copy is required: panels come from the cache too (`gt_image`, `render`), and a
    label drawn in place would spoil the cached image for all later readers.
    """
    panel = image.convert("RGB").copy()
    draw_panel_label(draw=ImageDraw.Draw(panel), x=0, y=0, label=label)
    return panel


def build_panel_images(
    gt_mesh_path: str | Path,
    pred_mesh_paths: list[str | Path | None],
    labels: list[str],
    plotter: "Plotter | None" = None,
) -> list[Image.Image]:
    """List of labeled panels instead of a collage: the target first, then candidates.

    Why a list. In a collage all panels are one image, and the server fits EACH image
    to its pixel budget (`mm-processor-kwargs`); so the more candidates, the smaller
    each one, and the assistant is not aware of it. As separate images each has its own
    pixel budget, and the panel resolution no longer depends on how many were requested.
    The price: the prompt length grows with the number of images, and the server must be
    started with `limit-mm-per-prompt` at least as large as their number.

    The panels are the same as in the collage: if rendering diverged, two forms of one
    question would become two different questions.
    """
    own_plotter = plotter is None
    plotter = plotter or plotter_class()(scale_gt=True, scale_pred=True)
    try:
        panels = [render_simple_mesh(
            plotter=plotter, mesh_path=gt_mesh_path, role="target", scale=True)]
        panels += [
            render_simple_mesh(plotter=plotter, mesh_path=path, role="prediction", scale=True)
            for path in pred_mesh_paths
        ]
        return [
            label_panel(panel, label)
            for panel, label in zip(panels, ["TARGET", *labels], strict=False)
        ]
    finally:
        if own_plotter:
            _close_plotter(plotter)


def build_target_image(
    gt_mesh_path: str | Path,
    plotter: "Plotter | None" = None,
) -> Image.Image:
    """A single target panel, exactly the one that stands as the TARGET panel in a collage.

    Needed because the target is not a candidate: it cannot be viewed through the
    selection collage while there is no candidate with a mesh, i.e. exactly on the
    first step of a part, when the question "what are we building at all" arises.
    """
    return build_panel_images(gt_mesh_path, [], [], plotter=plotter)[0]


def _close_plotter(plotter: "Plotter") -> None:
    try:
        plotter.plotter.close()
        plotter.iso_plotter.close()
    except Exception:
        logger.debug("Could not close the selection plotter", exc_info=True)


def render_simple_mesh(
    plotter: Plotter,
    mesh_path: str | Path | None,
    role: str,
    scale: bool,
) -> Image.Image:
    if mesh_path is None or not Path(mesh_path).exists():
        return build_invalid_select_image((plotter.cols * plotter.view_img_size, plotter.rows * plotter.view_img_size))

    image = plotter._get_img_stepwise(
        mesh_path=mesh_path,
        cmap=plotter.cmap_pred,
        apply_augs=False,
        color=(210, 210, 210) if role == "target" else (224, 185, 112),
        scale=scale,
        apply_noise=False,
        noise_scale=0.0,
    )
    grayscale = ImageOps.grayscale(image)
    if role == "target":
        return ImageOps.colorize(grayscale, black="#101217", white="#d8e5ff").convert("RGB")
    return ImageOps.colorize(grayscale, black="#10100c", white="#f2d39b").convert("RGB")


def build_simple_comparison_grid(
    target: Image.Image,
    predictions: list[Image.Image],
    labels: list[str],
) -> Image.Image:
    if not predictions:
        raise ValueError("No predictions to build simple selection grid")

    w, h = target.size
    for idx, image in enumerate(predictions):
        if image.size != (w, h):
            raise ValueError(f"Target has {(w, h)}, prediction {idx} has {image.size}")

    cols, rows = selection_grid_shape(len(predictions))
    target_h = h
    canvas = Image.new("RGB", (cols * w, target_h + rows * h), color="#f3f4f6")
    draw = ImageDraw.Draw(canvas)

    target_x = (cols * w - w) // 2
    canvas.paste(target.convert("RGB"), (target_x, 0))
    draw_panel_label(draw=draw, x=target_x, y=0, label="TARGET")

    y_offset = target_h
    for idx, (image, label) in enumerate(zip(predictions, labels, strict=True)):
        row = idx // cols
        col = idx % cols
        x = col * w
        y = y_offset + row * h
        canvas.paste(image.convert("RGB"), (x, y))
        draw_panel_label(draw=draw, x=x, y=y, label=label)

    return canvas


def draw_panel_label(draw: ImageDraw.ImageDraw, x: int, y: int, label: str) -> None:
    font = _label_font(48)

    padding = 16
    text_x = x + padding
    text_y = y + padding
    bbox = draw.textbbox((text_x, text_y), label, font=font)
    bg_padding = 8
    draw.rectangle(
        (
            bbox[0] - bg_padding,
            bbox[1] - bg_padding,
            bbox[2] + bg_padding,
            bbox[3] + bg_padding,
        ),
        fill="black",
    )
    draw.text((text_x, text_y), label, fill="white", font=font)


def build_image_grid(images: list[Image.Image], labels: list[str]) -> Image.Image:
    if not images:
        raise ValueError("No images to build selection grid")

    w, h = images[0].size
    for idx, image in enumerate(images):
        if image.size != (w, h):
            raise ValueError(f"Image 0 has {(w, h)}, image {idx} has {image.size}")

    cols, rows = selection_grid_shape(len(images))

    canvas = Image.new("RGB", (cols * w, rows * h), color="white")
    draw = ImageDraw.Draw(canvas)

    for idx, (image, label) in enumerate(zip(images, labels, strict=True)):
        row = idx // cols
        col = idx % cols
        x = col * w
        y = row * h
        canvas.paste(image.convert("RGB"), (x, y))

        draw_panel_label(draw=draw, x=x, y=y, label=label)

    return canvas


def selection_grid_shape(num_images: int) -> tuple[int, int]:
    if num_images <= 0:
        raise ValueError("num_images must be positive")

    fixed_shapes = {
        1: (1, 1),
        2: (2, 1),
        3: (3, 1),
        4: (2, 2),
        5: (3, 2),
        6: (3, 2),
    }
    if num_images in fixed_shapes:
        return fixed_shapes[num_images]

    cols = 3
    rows = (num_images + cols - 1) // cols
    return cols, rows




class FigureRenderer:
    """Renderer of one part: its own Plotter, its own cache, lives in the part's process.

    **GT is drawn once per part and lives in an object field.** This follows from how the
    picture is built: only the green and blue channels are taken from GT, red comes from
    the prediction, and GT is scaled by itself (`scale_gt`), independent of the
    prediction. So it is a constant over the whole rollout, and saving it to disk only to
    read it back at every step would hammer NFS for something already in memory. The
    split channels are stored too: `split()` on a 504x1008 collage is not free and is
    needed at every merge.

    Predictions are cached by mesh path, **in memory** (`MeshRenderCache`). The
    generator's picture and the selection picture are the same picture, so the cache is
    needed; on disk it was an extra round trip over NFS.
    """

    def __init__(
        self,
        max_images: int = DEFAULT_RENDER_CACHE_IMAGES,
        dialect: str | None = None,
    ):
        self.cache = MeshRenderCache(max_images=max_images)
        self.plotter_cls = plotter_class(dialect)
        self._plotter: Plotter | None = None
        self._simple_plotter: Plotter | None = None
        # The part's GT: path, image and its channels. Keyed by path on purpose: the
        # renderer is created per part, but a GT swap is better noticed than silently
        # showing the model foreign geometry.
        self._gt_path: str | None = None
        self._gt_image: Image.Image | None = None
        self._gt_bands: tuple[Image.Image, Image.Image] | None = None
        self._gt_stats = CacheStats()

    @property
    def plotter(self) -> "Plotter":
        if self._plotter is None:
            self._plotter = self.plotter_cls(scale_gt=True, scale_pred=False)
        return self._plotter

    @property
    def simple_plotter(self) -> "Plotter":
        """Separate Plotter for `simple` mode: the prediction is scaled there."""
        if self._simple_plotter is None:
            self._simple_plotter = self.plotter_cls(scale_gt=True, scale_pred=True)
        return self._simple_plotter

    def gt_image(self, gt_mesh_path: Path | str) -> Image.Image:
        """Rendered GT of the part. The first call draws, the rest return the field."""
        key = str(Path(gt_mesh_path))
        if self._gt_path == key:
            self._gt_stats.hit()
        else:
            self._gt_stats.miss()
            image = self.plotter._get_img_stepwise(
                mesh_path=Path(gt_mesh_path), cmap=self.plotter.cmap_gt, apply_augs=False,
                color=(0, 255, 0), scale=True, apply_noise=False, noise_scale=0.0,
            )
            _, green, blue = image.split()
            self._gt_path, self._gt_image, self._gt_bands = key, image, (green, blue)
            self._gt_stats.write()
        assert self._gt_image is not None
        return self._gt_image

    def selection_image(
        self,
        gt_mesh_path: Path | str,
        pred_mesh_paths: list[Path | str | None],
        labels: list[str],
        visualization_mode: str = "stepwise",
    ) -> Image.Image:
        if visualization_mode == "simple":
            return build_selection_image(
                gt_mesh_path=gt_mesh_path,
                pred_mesh_paths=pred_mesh_paths,
                labels=labels,
                visualization_mode=visualization_mode,
                plotter=self.simple_plotter,
            )
        # The stepwise mode is assembled from the same bricks as the generator input:
        # GT comes from the field, predictions from the cache. The earlier implementation
        # called `plotter.get_img_stepwise` per candidate and redrew GT along with every
        # prediction: k renders instead of one.
        images = [self.step_image(gt_mesh_path, pred_mesh_path) for pred_mesh_path in pred_mesh_paths]
        return build_image_grid(images=images, labels=labels)

    def target_image(self, gt_mesh_path: Path | str) -> Image.Image:
        """The target as a separate image, without candidates (`build_target_image`)."""
        return build_target_image(gt_mesh_path=gt_mesh_path, plotter=self.simple_plotter)

    def panel_images(
        self,
        gt_mesh_path: Path | str,
        pred_mesh_paths: list[Path | str | None],
        labels: list[str],
        visualization_mode: str = "simple",
    ) -> list[Image.Image]:
        """Same as `selection_image`, but as a list of images rather than a collage.

        The stepwise mode takes panels from the cache (`step_image`), like the collage:
        a prediction is drawn once per part, GT once per part run.
        """
        if visualization_mode == "simple":
            return build_panel_images(
                gt_mesh_path=gt_mesh_path,
                pred_mesh_paths=list(pred_mesh_paths),
                labels=list(labels),
                plotter=self.simple_plotter,
            )
        panels = [self.gt_image(gt_mesh_path)]
        panels += [self.step_image(gt_mesh_path, path) for path in pred_mesh_paths]
        return [
            label_panel(panel, label)
            for panel, label in zip(panels, ["TARGET", *labels], strict=False)
        ]

    def render(self, mesh_path: Path | str, role: str) -> Image.Image:
        """Render a mesh in its role (`gt` | `pred`).

        GT is served from the object's field, a prediction from the cache.
        """
        mesh_path = Path(mesh_path)
        if role == "gt":
            return self.gt_image(mesh_path)
        cached = self.cache.get(mesh_path, role)
        if cached is not None:
            return cached

        if role == "pred":
            image = self.plotter._get_img_stepwise(
                mesh_path=mesh_path, cmap=self.plotter.cmap_pred, apply_augs=False,
                color=(255, 0, 0), scale=False, apply_noise=False, noise_scale=0.0,
            )
        else:
            raise ValueError(f"Unknown render role: {role}")

        self.cache.put(mesh_path, role, image)
        return image

    def step_image(self, gt_mesh_path: Path | str, pred_mesh_path: Path | str | None) -> Image.Image:
        """Step generator input: green GT with the current prediction in red on top.

        The channel merge repeats the original `StepwiseDataset.__getitem__` verbatim:
        the model was trained on exactly this picture.
        """
        gt_img = self.gt_image(gt_mesh_path)
        if pred_mesh_path is None or not Path(pred_mesh_path).exists():
            # The zeroth step and a candidate without geometry look the same: plain
            # GT. A missing prediction file used to crash selection rendering.
            return gt_img

        pred_img = self.render(pred_mesh_path, "pred")
        assert self._gt_bands is not None
        gt_g, gt_b = self._gt_bands
        pred_r, _, _ = pred_img.split()
        return Image.merge("RGB", (pred_r, gt_g, gt_b))

    def cache_stats(self) -> dict[str, CacheStats]:
        """Counters of both render caches, for the part's `tech.json`.

        `gt_image` is the GT image in the object field, `render_pred` the prediction
        images in memory. They are kept apart because a miss means different things: for
        the first it is a part change, for the second a new prediction, i.e. normal work.
        """
        return {"gt_image": self._gt_stats, "render_pred": self.cache.stats}

    def close(self) -> None:
        self.cache.clear()
        for attr in ("_plotter", "_simple_plotter"):
            plotter = getattr(self, attr)
            if plotter is not None:
                _close_plotter(plotter)
                setattr(self, attr, None)
