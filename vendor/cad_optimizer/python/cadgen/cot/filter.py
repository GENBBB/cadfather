import json
from pathlib import Path

from cadlib.model import CADModel
from cq_parser import CADQuerySyntacticParser

from .formats import make_cot_data, make_questions
from .meta import build_cot_meta


def cot_filter(
    code: str,
    file_path: Path | str | None = None,
    save_questions=True,
    with_holes: bool = True,
    **kwargs,
):
    cs, _ = CADQuerySyntacticParser().parse_code(code)
    cad_model = CADModel.from_dict(cs).create_CAD()
    if file_path is None:
        return True
    file_path = Path(file_path)
    meta_output = file_path.with_name(f"{file_path.stem}_meta.json")
    build_cot_meta(file_path, out_file=meta_output, cad_model=cad_model)
    questions = (
        make_questions(meta_output, with_holes=with_holes) if save_questions else None
    )

    json_path = file_path.with_suffix(".json")
    json_path.write_text(json.dumps(cs, indent=2))
    cad_model.name = file_path.stem
    cad_model.output_folder = file_path.parent
    cad_model.save(
        exclude=["stl", "step", "occ"], render_folder_name=f"{file_path.stem}_render"
    )

    file_path.with_name(f"{file_path.stem}_questions.json").write_text(
        json.dumps(
            {"description": make_cot_data(meta_output), "questions": questions},
            indent=2,
            ensure_ascii=False,
        )
    )

    return True
