from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def _trimmed(value: float, precision: int = 6) -> str:
    s = f"{round(float(value), precision):.{precision}f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _load_meta(meta_path: Path):
    data = json.loads(meta_path.read_text())
    # accept both a list and a single object
    return data if isinstance(data, list) else [data]


def _count_modifying(model: dict) -> tuple[int, int]:
    fillets = sum(
        1 for mf in model.get("modifying_features", []) if mf.get("type") == "Fillet"
    )
    chamfers = sum(
        1 for mf in model.get("modifying_features", []) if mf.get("type") == "Chamfer"
    )
    return fillets, chamfers


def _count_holes(model: dict) -> int:
    # Sum the inner sketch loops of the base features: a good proxy for holes
    holes = 0
    for bf in model.get("base_features", []):
        prof = bf.get("profile", {})
        holes += int(prof.get("inner_loops_total", 0) or 0)
    return holes


def _maybe_volume_from_stl(model: dict) -> float | None:
    # Fallback: if meta has no mesh_volume, try reading the STL and take mesh.volume
    try:
        from trimesh import load_mesh
    except Exception:
        return None
    file_path = model.get("file")
    if not file_path:
        return None
    stl_path = Path(file_path).with_suffix(".stl")
    if not stl_path.exists():
        return None
    try:
        mesh = load_mesh(stl_path)
        return float(abs(mesh.volume))
    except Exception:
        return None


def make_questions(meta_path: Path, with_holes: bool) -> List[Dict[str, str]]:
    """
    Return a list of dicts with keys:
      - question (str)
      - answer (str)
    for 4 questions:
      1) What is the model volume?
      2) What are the overall dimensions? (bounding box)
      3) What are the maximum sizes along X, Y and Z?
      4) Are there chamfers, fillets and holes?
    Supports meta with a single object or a list of objects.
    """
    models = _load_meta(meta_path)
    qa_list: List[Dict[str, str]] = []

    for idx, model in enumerate(models, start=1):
        # 1) Volume
        vol = model.get("mesh_volume", None)
        if vol is None:
            vol = _maybe_volume_from_stl(model)
        vol_str = (
            _trimmed(vol, 6)
            if isinstance(vol, (float, int)) and vol is not None
            else "нет данных"
        )

        # 2) Overall dimensions (bounding box)
        bbox = model.get("bbox_extents", None)
        if bbox and len(bbox) == 3:
            x, y, z = bbox
            dims_str = f"{_trimmed(x)} × {_trimmed(y)} × {_trimmed(z)}"
        else:
            dims_str = "нет данных"

        # 3) Maximum sizes along X, Y, Z
        if bbox and len(bbox) == 3:
            x, y, z = bbox
            axes_str = f"X: {_trimmed(x)}, Y: {_trimmed(y)}, Z: {_trimmed(z)}"
        else:
            axes_str = "нет данных"

        # 4) Presence of chamfers/fillets/holes
        fillets, chamfers = _count_modifying(model)

        def yes_no(count: int, name_single: str, name_plural: str) -> str:
            if count <= 0:
                return "нет"
            if count == 1:
                return f"да (1 {name_single})"
            return f"да ({count} {name_plural})"

        fillets_str = yes_no(fillets, "скругление", "скругления")
        chamfers_str = yes_no(chamfers, "фаска", "фаски")
        if with_holes:
            holes = _count_holes(model)
            holes_str = yes_no(holes, "отверстие", "отверстий")

        # Build the answers (if meta holds several models, prefix with the model number)
        prefix = f"[Модель {idx}] " if len(models) > 1 else ""

        qa_list.append(
            {
                "question": "Какой объем модели?",
                "answer": f"{prefix}{vol_str}",
            }
        )
        qa_list.append(
            {
                "question": "Каковы габаритные размеры изделия? (описывающий параллелепипед)",
                "answer": f"{prefix}{dims_str}",
            }
        )
        qa_list.append(
            {
                "question": "Каковы максимальные размеры модели по осям X, Y и Z?",
                "answer": f"{prefix}{axes_str}",
            }
        )
        qa_list.append(
            {
                "question": (
                    "Есть ли фаски, скругления и отверстия?"
                    if with_holes
                    else "Есть ли фаски, скругления?"
                ),
                "answer": (
                    f"{prefix}Фаски: {chamfers_str}; скругления: {fillets_str}; отверстия: {holes_str}."
                    if with_holes
                    else f"{prefix}Фаски: {chamfers_str}; скругления: {fillets_str}."
                ),
            }
        )

    return qa_list
