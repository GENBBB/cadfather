"""Face-relative workplane resolution (library home; canonical implementation).

02_cut-style parts select a face with PointOnFaceSelector([x,y,z]) and build a
workplane on it (face_wN), then place cuts via copyWorkplane(face_wN).  The AST
parser cannot resolve point-based face frames statically, so the cut would
collapse to the root frame and the optimiser distorts it.  Executing the code
(cadquery resolves the face correctly), reading each face_wN's real plane, and
rewriting copyWorkplane(face_wN) -> copyWorkplane(cq.Workplane(cq.Plane(...)))
gives the parser a literal frame it already handles.

`needs_face_resolution` is the cheap static gate: only codes that actually
reference copyWorkplane(face_wN) pay the exec cost.  The rewrite is idempotent
(the resolved code contains no face_wN references, so a second pass finds no
frames and returns the input unchanged).
"""
import re


def needs_face_resolution(code: str) -> bool:
    """True iff the code uses a point-selected face workplane the parser cannot
    resolve statically (copyWorkplane(face_wN))."""
    return re.search(r'copyWorkplane\(face_w\d+\)', code) is not None


def resolve_face_frames(code: str) -> str:
    """Execute the code, read each face_wN workplane's real plane, and rewrite
    copyWorkplane(face_wN) to a literal cq.Plane form the parser handles,
    dropping the now-unused selector side-lines.  Returns the code unchanged on
    any failure (missing cadquery/addons, exec error, no face frames)."""
    try:
        import cadquery as cq
    except Exception:
        return code
    ns = {'cq': cq}
    try:
        import cadquery_addons
        ns.update({n: getattr(cadquery_addons, n) for n in dir(cadquery_addons)
                   if not n.startswith('_')})
    except Exception:
        pass
    try:
        exec(code, ns)
    except Exception:
        return code
    frames = {}
    for name, val in list(ns.items()):
        if re.match(r'face_w\d+$', name) and hasattr(val, 'plane'):
            try:
                pl = val.plane
                frames[name] = (tuple(round(c, 5) for c in pl.origin.toTuple()),
                                tuple(round(c, 5) for c in pl.xDir.toTuple()),
                                tuple(round(c, 5) for c in pl.zDir.toTuple()))
            except Exception:
                pass
    if not frames:
        return code
    # Pass 1: rewrite copyWorkplane(face_wN) occurrences; KEEP every line.
    lines = []
    for ln in code.split('\n'):
        for nm, (o, x, n) in frames.items():
            if f'copyWorkplane({nm})' in ln:
                # wrap the Plane in a Workplane: cadquery's copyWorkplane needs a
                # Workplane, and the parser unwraps cq.Workplane(cq.Plane(...)).
                pl = (f'cq.Workplane(cq.Plane(origin=cq.Vector{o}, '
                      f'xDir=cq.Vector{x}, normal=cq.Vector{n}))')
                ln = ln.replace(f'copyWorkplane({nm})', f'copyWorkplane({pl})')
        lines.append(ln)
    # Pass 2: drop selector/frame assignment lines ONLY when the variable is no
    # longer referenced by the remaining code (iterate to a fixpoint so face_sel
    # follows its face_w out).  Dropping unconditionally BREAKS the
    # face_w0-as-chain-root form (face_w0.sketch()... with no copyWorkplane) --
    # NameError at render -- and strips point_selN still needed by attach_at.
    sel_re = re.compile(r'\s*(point_sel\d+|face_sel\d+|face_w\d+)\s*=')
    changed = True
    while changed:
        changed = False
        for i, ln in enumerate(lines):
            m = sel_re.match(ln)
            if not m:
                continue
            name = m.group(1)
            rest = '\n'.join(lines[:i] + lines[i + 1:])
            if not re.search(r'\b' + re.escape(name) + r'\b', rest):
                del lines[i]
                changed = True
                break
    return '\n'.join(lines)
