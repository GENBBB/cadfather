from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def _trimmed(value: float, precision: int) -> str:
    """
    Round to precision and strip trailing zeros and the trailing dot.
    1.0000 -> "1", 88.0 -> "88", 1.285700 -> "1.2857"
    """
    s = f"{round(value, precision):.{precision}f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _sig_value(value: float, precision: int) -> str:
    """
    Return the value with a precision sign: "= 0.5625" or "≈ 1.2857",
    so the caller does not have to add its own '='.
    """
    rounded = round(value, precision)
    tol = 0.5 * (10 ** (-precision))
    exact = abs(value - rounded) <= tol
    num = _trimmed(rounded, precision)
    return ("= " if exact else "≈ ") + num


def _approx_zero(value: float, precision: int) -> bool:
    tol = 0.5 * (10 ** (-precision))
    return abs(value) <= tol


def _axes_mapping_suffix(axes_mapping: list[str] | tuple[str, str, str]) -> str:
    """
    Return a human-readable note about the axes ONLY if the mapping is not the default ('X','Y','Z').
    Format: " (axes: x→X, y→Y, n→Z)", with signs where present, e.g. "x→-Y".
    """
    default = ("X", "Y", "Z")
    am = tuple(axes_mapping)
    if am == default:
        return ""
    return f" (axes: x→{am[0]}, y→{am[1]}, n→{am[2]})"


def make_cot_data(
    meta_path: Path,
    bbox_decimals: int = 1,
    ratio_precision: int = 4,
    include_filename: bool = False,
    show_axes_mapping_when_default: bool = False,  # if True, always show the mapping
) -> str:
    """
    Build the English CoT text from the metadata.
    - Numbers have no trailing zeros.
    - Ratios are marked "= " (exact) or "≈ " (rounded).
    - The axes mapping is hidden when it is the default, unless enabled.
    - The edge-only contact note is appended afterwards, only if such a case occurs.
    """
    raw = json.loads(meta_path.read_text())
    models = raw if isinstance(raw, list) else [raw]

    lines: list[str] = []

    for model in models:
        if include_filename:
            lines.append(f"Source: {model['file']}")

        # 1) Bounding box
        if "bbox_extents" in model:
            bx, by, bz = model["bbox_extents"]
            bbox_str = ":".join(_trimmed(v, bbox_decimals) for v in (bx, by, bz))
        else:
            bbox_str = model.get("bbox_ratio_str", "")
        lines.append(f"1) Bounding box (X:Y:Z) = {bbox_str}")

        # 2) Summary
        bf_cnt = model["base_features_count"]
        mf_cnt = model["modifying_features_count"]
        bf_text = "; ".join(model["base_features_text"]) if bf_cnt else "—"
        mf_text = "; ".join(model["modifying_features_text"]) if mf_cnt else "—"
        lines.append(
            f"2) It has {bf_cnt} base feature(s): {bf_text} and {mf_cnt} modifying feature(s): {mf_text}"
        )

        # 3) Base features
        for bf in model["base_features"]:
            i = bf["index"]
            op_text = bf["operation_text"]
            lines.append(f"3) {op_text}")

            prof = bf["profile"]
            axes_suffix = (
                _axes_mapping_suffix(prof["axes_mapping"])
                if not show_axes_mapping_when_default
                else f" (axes: x→{prof['axes_mapping'][0]}, y→{prof['axes_mapping'][1]}, n→{prof['axes_mapping'][2]})"
            )
            lines.append(
                f"   Sketch orientation: {prof['orientation_xy']}{axes_suffix}"
            )

            inner_total = prof["inner_loops_total"]
            lines.append(
                f"   The sketch consists of 1 outer loop and {inner_total} inner loop(s)."
            )

            like_in = bf.get("like_in_profile_index")
            outer_like = "" if not like_in else f" (same as in profile #{like_in})"
            lines.append(
                f"   Outer loop: {prof['outer_desc']}{outer_like}; segments: {prof['outer_segments_count']}"
            )

            lines.append(
                f"   Profile-to-outer area ratio: {_sig_value(prof['profile_to_outer_ratio'], ratio_precision)}"
            )

            ig_cnt = prof["inner_groups_count"]
            if ig_cnt > 0:
                lines.append(
                    f"   Inner loops: {inner_total} total forming {ig_cnt} unique rotation group(s):"
                )
                for idx, grp in enumerate(prof["unique_inner_groups"], start=1):
                    like = grp.get("like_in_profile_index")
                    like_s = "" if not like else f" (same as in profile #{like})"
                    lines.append(
                        f"     • Group {idx}: {grp['description']}{like_s}; "
                        f"primitives per loop: {grp['loop_primitives_count']}; "
                        f"outer-loop area ratio: {_sig_value(grp['outer_area_ratio'], ratio_precision)}"
                    )

        # 4) Intersections
        if model["intersections"]:
            lines.append(
                "4) The subsequent primitives are joined to the previous ones. Sketch relationships:"
            )
            any_edge_only = False
            for it in model["intersections"]:
                cid = it["current"]
                wid = it["with_index"]
                same_plane = it["same_plane"]
                relation = it["relation"]
                ar = _sig_value(it["area_ratio"], ratio_precision)
                oor = _sig_value(it["outer_outer_ratio"], ratio_precision)
                inter = _sig_value(it["intersection_to_smaller_ratio"], ratio_precision)

                plane_text = (
                    "on the same plane" if same_plane else "on different planes"
                )

                # human-readable description of the relation
                if relation == "inner":
                    rel_text = "one sketch lies entirely inside the other"
                elif relation == "outer":
                    rel_text = "the sketches do not overlap"
                else:
                    rel_text = "the sketches intersect"

                # edge-only contact?
                edge_only = relation == "cross" and _approx_zero(
                    it["intersection_to_smaller_ratio"], ratio_precision
                )
                if edge_only:
                    any_edge_only = True

                # format the line
                details = [
                    f"Profile area ratio (P{cid}/P{wid}): {ar}",
                    f"Outer-loop area ratio (P{cid}/P{wid}): {oor}",
                    f"Intersection area relative to the smaller profile: {inter}",
                ]
                if edge_only:
                    details[-1] += " (edge-only contact)"

                lines.append(
                    f"   • Primitive #{cid} vs #{wid}: sketches are {plane_text}; {rel_text}. "
                    + "; ".join(details)
                )

            # add the global note only if such cases occurred
            if any_edge_only:
                lines.append(
                    "   Note: edge-only contact is detected when relation is 'cross' and the normalized intersection area is 0."
                )

        # 5) Modifying
        if model["modifying_features"]:
            lines.append("5) Modifying features coverage:")
            for mf in model["modifying_features"]:
                touches = mf["touches_primitives"]
                touches_s = ", ".join(map(str, touches)) if touches else "none"
                lines.append(
                    f"   • {mf['type']} touches edges present in primitive(s): {touches_s}"
                )

        lines.append("")

    return "\n".join(lines)
