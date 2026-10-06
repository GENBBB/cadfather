import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
# from cq_parser.parser import CADQuerySyntacticParser
from tenacity import retry, stop_after_attempt

from .cot.filter import cot_filter


def compound_to_mesh(compound):
    # cq.Compound
    vertices, faces = compound.tessellate(0.001, 0.1)
    return trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)


def sub_command_or_arg(
    code,
    filter=None,
    callable_checks=None,
):
    parser = CADQuerySyntacticParser()
    parts = parser.parse_parts(code)

    def default_filter_eligible_for_sub(part):
        if "segment" in part:
            return "segment"
        elif "arc" in part:
            return "arc"
        elif "chamfer" in part:
            return "chamfer"
        elif "fillet" in part:
            return "fillet"
        elif "box" in part:
            return "box"
        elif "cylinder" in part:
            return "cylinder"
        elif "circle" in part:
            return "circle"
        else:
            return "not_eligible"

    def build_string_from_args(args, remove_spaces=False):
        s = f"{args['fname']}({', '.join([str(x) for x in args['positional']])})"
        if remove_spaces:
            s = s.replace(" ", "")
        return s

    def check_brackets_balance(s):
        return s.count("(") == s.count(")")

    if filter is None:
        filter = default_filter_eligible_for_sub

    eligible_for_sub = defaultdict(lambda: [])
    for part in parts:
        eligible_for_sub[filter(part)].append(part)

    eligible_for_sub.pop("not_eligible")
    try:
        chosen_key = np.random.choice(list(eligible_for_sub.keys()))
    except:
        raise ValueError("Nothing to change.")
    to_sub = np.random.choice(eligible_for_sub[chosen_key])
    while not check_brackets_balance(to_sub):
        to_sub = to_sub[:-1]

    parsed_args = parser._parse_kw_args(to_sub)

    str_before = build_string_from_args(parsed_args)
    if chosen_key == "segment":
        str_before = build_string_from_args(parsed_args, remove_spaces=True)
        if np.random.random() < 0.33 and len(parsed_args["positional"]) > 1:
            parsed_args["fname"] = "arc"
            start = parsed_args["positional"][0]
            end = parsed_args["positional"][1]
            mid = [(start[0] + end[0]) // 2, (start[1] + end[1]) // 2]
            shift_x = np.random.randint(-100, 100)
            shift_y = np.random.randint(-100, 100)
            mid[0] += shift_x
            mid[1] += shift_y
            parsed_args["positional"].insert(1, tuple(mid))
        else:
            i_arg = np.random.choice(range(len(parsed_args["positional"])))
            arg = list(parsed_args["positional"][i_arg])
            shift_x = np.random.randint(-100, 100)
            shift_y = np.random.randint(-100, 100)
            arg[0] += shift_x
            arg[1] += shift_y
            parsed_args["positional"][i_arg] = tuple(arg)
        str_after = build_string_from_args(parsed_args, remove_spaces=True)

    elif chosen_key == "arc":
        str_before = build_string_from_args(parsed_args, remove_spaces=True)
        if np.random.random() < 0.33:
            parsed_args["fname"] = "segment"
            if len(parsed_args["positional"]) == 2:
                parsed_args["positional"] = parsed_args["positional"][1:]
            else:
                parsed_args["positional"] = parsed_args["positional"].pop(1)
        else:
            i_arg = np.random.choice(range(len(parsed_args["positional"])))
            arg = list(parsed_args["positional"][i_arg])
            shift_x = np.random.randint(-100, 100)
            shift_y = np.random.randint(-100, 100)
            arg[0] += shift_x
            arg[1] += shift_y
            parsed_args["positional"][i_arg] = tuple(arg)
        str_after = build_string_from_args(parsed_args, remove_spaces=True)

    elif chosen_key == "chamfer" or chosen_key == "fillet":
        str_before = build_string_from_args(parsed_args, remove_spaces=False)
        for i, part in enumerate(parts):
            if part == str_before:
                if "face" in parts[i - 1]:
                    str_before = "." + parts[i - 1] + "." + part
                    break
        str_after = ""

    elif chosen_key == "cylinder" or chosen_key == "box" or chosen_key == "circle":
        str_before = build_string_from_args(parsed_args, remove_spaces=True)
        i_arg = np.random.choice(range(len(parsed_args["positional"])))
        arg = parsed_args["positional"][i_arg]
        arg = int(arg * np.random.uniform(0.5, 2))
        parsed_args["positional"][i_arg] = arg
        str_after = build_string_from_args(parsed_args, remove_spaces=True)

    else:
        str_after = build_string_from_args(parsed_args)
    code_new = code.replace(str_before, str_after)
    local_vars = {}
    w = exec(code_new, globals(), local_vars)
    w = local_vars["r"].val()
    mesh_new = compound_to_mesh(w)

    if chosen_key != "chamfer" and chosen_key != "fillet":
        w_old = exec(code, globals(), local_vars)
        w_old = local_vars["r"].val()
        mesh_old = compound_to_mesh(w_old)
        volume_ratio = mesh_new.volume / mesh_old.volume
        assert volume_ratio <= 0.9 or volume_ratio >= 1.1

    assert len(mesh_new.faces) > 2
    assert mesh_new.is_watertight
    assert not mesh_new.is_empty

    assert bool(mesh_new.volume > 0)
    assert bool(sum(mesh_new.extents == 0) == 0)

    if callable_checks is not None:
        for callable_check in callable_checks:
            assert callable_check(mesh=mesh_new, code=code_new)

    return code_new, str_before, str_after


def sub_command_filter(
    code: str, file_path: str | Path, with_holes: bool = True, **kwargs
):
    file_path = Path(file_path)
    new_code, str_before, str_after = retry(stop=stop_after_attempt(5))(
        sub_command_or_arg
    )(code)

    changed_file = file_path.with_name(f"{file_path.stem}_changed.py")
    changed_file_step = changed_file.with_suffix(".step")
    changed_file_stl = changed_file.with_suffix(".stl")

    local_vars = {}
    w = exec(new_code, globals(), local_vars)
    w = local_vars["r"].val()
    mesh = compound_to_mesh(w)
    mesh.export(changed_file_stl)
    w.export(str(changed_file_step))

    cot_filter(
        new_code,
        changed_file,
        with_holes=with_holes,
    )
    changed_file.write_text(new_code)
    file_path.with_name("change.json").write_text(
        json.dumps({"before": str_before, "after": str_after}, indent=2)
    )
    return True
