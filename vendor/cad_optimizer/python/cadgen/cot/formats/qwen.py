from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _trimmed(value: float, precision: int) -> str:
    s = f"{round(value, precision):.{precision}f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _approx_zero(value: float, precision: int) -> bool:
    tol = 0.5 * (10 ** (-precision))
    return abs(value) <= tol


def _plane_tag(orientation_xy: str) -> str:
    # "XY", "XZ", "YZ" → "on the XY plane"
    if orientation_xy in {"XY", "XZ", "YZ"}:
        return f"on the {orientation_xy} plane"
    return f"on a {orientation_xy} plane"


def _outer_shape_label(desc: str) -> str:
    d = desc.lower()
    if "circle" in d:
        return "circular"
    if "square" in d:
        return "square"
    if "rectangle" in d:
        return "rectangular"
    if "regular" in d and "-gon" in d:
        return "regular polygonal"
    if "-gon" in d:
        return "polygonal"
    if "rhombus" in d:
        return "rhombic"
    if "parallelogram" in d:
        return "parallelogram"
    if "trapezoid" in d:
        return "trapezoid"
    return "sketched"


def _solid_name_from_profile(desc: str, inner_loops_total: int, op_type: str) -> str:
    """Rough heuristic for the solid name: cylinder / hollow cylinder / prism / solid of revolution."""
    shape = _outer_shape_label(desc)
    hollow = inner_loops_total > 0

    if op_type == "Extrude":
        if shape == "circular":
            return "hollow cylinder" if hollow else "cylinder"
        if shape == "square" or shape == "rectangular":
            return "box" if not hollow else "box with through hole"
        if "polygonal" in shape:
            return "prism" if not hollow else "prism with through hole"
        return "extruded solid" if not hollow else "extruded solid with through hole"

    if op_type == "Revolve":
        # Without the exact profile geometry it is safe to call it this:
        return "solid of revolution"

    return "solid"


def _sketch_brief(prof: dict) -> str:
    """Brief sketch description: outer shape plus presence of holes."""
    outer = _outer_shape_label(prof["outer_desc"])
    k = int(prof["inner_loops_total"])
    if k <= 0:
        return f"{outer} sketch"
    if k == 1:
        return f"{outer} sketch with one inner loop (hole)"
    return f"{outer} sketch with {k} inner loops (holes)"


def _join_relation_sentence(
    i: int, j: int, same_plane: bool, relation: str, inter_ratio: float, precision: int
) -> str:
    plane_text = "on the same plane" if same_plane else "on a different plane"

    if relation == "inner":
        rel_text = "it lies entirely inside the earlier sketch"
    elif relation == "outer":
        rel_text = "it does not overlap the earlier sketch"
    else:
        rel_text = "it intersects the earlier sketch"

    edge_only = (relation == "cross") and _approx_zero(inter_ratio, precision)
    if edge_only:
        # keep it human-readable, without metrics
        return f"It is {plane_text}; they intersect only along edges (edge-only contact) with primitive #{j}."
    return f"It is {plane_text}; {rel_text} relative to primitive #{j}."


def _find_prev_intersection(model: dict, current_idx_1based: int) -> dict | None:
    """
    Return the first intersection record of the current primitive with the nearest
    previous one (j = current - 1) if there is one; otherwise any record with an
    earlier primitive; otherwise None.
    """
    intersections = model.get("intersections", [])
    prev = current_idx_1based - 1
    # first look for the immediate predecessor
    for it in intersections:
        if it["current"] == current_idx_1based and it["with_index"] == prev:
            return it
    # then any earlier one
    for it in intersections:
        if (
            it["current"] == current_idx_1based
            and it["with_index"] < current_idx_1based
        ):
            return it
    return None


def make_qwen_thinking_text(
    meta_path: Path | str,
    bbox_decimals: int = 1,
    height_precision: int = 3,
    intersection_precision: int = 4,
    include_bbox: bool = True,
    include_filename: bool = False,
) -> str:
    """
    Build a human-readable CoT text:
    - short reasoning steps for each primitive,
    - minimal metrics; only heights and main facts,
    - edge-only contact is printed only when it occurs.
    """
    if isinstance(meta_path, str):
        meta_path = Path(meta_path)
    raw = json.loads(meta_path.read_text())
    models = raw if isinstance(raw, list) else [raw]

    out_lines: list[str] = []

    for model in models:
        if include_filename:
            out_lines.append(f"Source: {model['file']}")

        if include_bbox and "bbox_extents" in model:
            bx, by, bz = model["bbox_extents"]
            bbox = ":".join(_trimmed(v, bbox_decimals) for v in (bx, by, bz))
            out_lines.append(f"Bounding box (X:Y:Z): {bbox}")

        bf_list = model["base_features"]
        if not bf_list:
            out_lines.append("No base features found.")
            out_lines.append("")
            continue

        # Step 1: context of the first sketch
        first_prof = bf_list[0]["profile"]
        plane_text = _plane_tag(first_prof["orientation_xy"])
        out_lines.append(f"1) Start from a sketch {plane_text}.")

        # Iterate over the primitives
        for idx, bf in enumerate(bf_list, start=1):
            prof = bf["profile"]
            op = bf["type"]
            sketch_desc = _sketch_brief(prof)
            height = bf["params"].get("extent1")
            height_txt = (
                f" to a height of {_trimmed(float(height), height_precision)}"
                if height is not None
                else ""
            )

            # 2) What the code does (in plain words)
            if idx == 1:
                out_lines.append(f"2) Draw a {sketch_desc} for primitive #{idx}.")
            else:
                out_lines.append(f"{idx+1}) Draw a {sketch_desc} for primitive #{idx}.")

            # 3) Why it matters for the shape (extrusion/revolution -> which solid)
            solid_name = _solid_name_from_profile(
                prof["outer_desc"], prof["inner_loops_total"], op
            )
            if op == "Extrude":
                out_lines.append(
                    f"   Then extrude it{height_txt}, forming a {solid_name}."
                )
            elif op == "Revolve":
                out_lines.append(
                    f"   Then revolve it around a global axis, forming a {solid_name}."
                )
            else:
                out_lines.append(f"   Apply {op.lower()} to form a {solid_name}.")

            # 4) Relation to the previous ones (if any)
            if idx > 1:
                it = _find_prev_intersection(model, idx)
                if it:
                    sent = _join_relation_sentence(
                        idx,
                        it["with_index"],
                        it["same_plane"],
                        it["relation"],
                        it["intersection_to_smaller_ratio"],
                        intersection_precision,
                    )
                    out_lines.append(f"   Placement vs previous: {sent}")

        # Final conclusion
        # Simple aggregation: list the result briefly
        summary_parts: list[str] = []
        for bf in bf_list:
            op = bf["type"]
            prof = bf["profile"]
            solid_name = _solid_name_from_profile(
                prof["outer_desc"], prof["inner_loops_total"], op
            )
            height = bf["params"].get("extent1")
            if height is not None:
                summary_parts.append(
                    f"{solid_name} (height {_trimmed(float(height), height_precision)})"
                )
            else:
                summary_parts.append(solid_name)

        out_lines.append(
            f"Result: the part consists of " + ", ".join(summary_parts) + "."
        )
        out_lines.append("")

    return "\n".join(out_lines)
