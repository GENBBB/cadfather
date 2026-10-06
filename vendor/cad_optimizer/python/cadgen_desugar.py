"""Desugar the image2cad/mesh2cad FUNCTIONAL cadgen format into plain
method-chain CadQuery the optimizer's parser understands.

Input format (best.py):
    import cadquery as cq
    from cadgen.extrude import extrude
    ...
    r = None
    r=extrude(r,(20,-11,-2),'ZX',"sketch()....finalize()",86[,False])
    r=hole(r,(-100,-32,39),'YZ',"sketch()....finalize()",101)
    r=orto_cut(...); r=revolve(r,pt,'XY',"...",angle,'Z')

Each cadgen op internally composes `cq.Workplane(WP).workplane(offset=O)`
+ sketch + extrude/revolve and combines via union/cut; the only runtime
work is projecting the anchor point onto the current solid's surface
(offset O + over-pierce direction).  We therefore desugar by executing the
script incrementally WITH cadgen's own functions (projections exact by
construction) while emitting the equivalent chain per call:
    extrude  -> r=<chain>.extrude(H)          (first)  / r=r.union(<chain>)
    revolve  -> r=<chain>.revolve(A, P0, P1)  (first)  / r=r.union(<chain>)
    hole/orto_cut -> r=r.cut(<chain>.extrude(H'))   with cadgen's 10-unit
        over-pierce folded into the cutter span when the pierce direction
        is along the workplane normal (the usual case).
Sketch strings are inlined verbatim -> their coordinate literals stay
TUNABLE parameters; projected offsets/axis points become frozen literals
(same convention as face-frame resolution).  extrude's 0.0005-unit
_surface_translation contact nudge is deliberately omitted.

cadgen only imports in the cad_utils env, so `desugar_cadgen()` runs THIS
FILE under that interpreter as a subprocess (code on stdin, desugared code
on stdout).  Env overrides: CADGEN_PY, CADGEN_PYTHONPATH.  Unsupported ops
(gear/sweep/spring/loft/shell/rib/thread) raise -> callers fall back to
the original code.
"""
import os
import re
import sys

_DEFAULT_PY = ''
_DEFAULT_PP = ''

_WPI = {'XY': 2, 'YZ': 0, 'ZX': 1}
_AXI = {'X': 0, 'Y': 1, 'Z': 2}


def is_cadgen_functional(code: str) -> bool:
    """Cheap static detection of the functional cadgen best.py format."""
    return bool(re.search(r'^from cadgen\.', code, re.M)) and \
        bool(re.search(r'^r\s*=\s*\w+\(\s*r\b', code, re.M))


def _join(wp, off, sk):
    return (f"cq.Workplane('{wp}').workplane(offset={off!r})."
            + sk.strip().lstrip('.'))


def _desugar_impl(code: str, frozen_stl: str = None) -> str:
    """The actual rewrite.  Must run where cadgen imports (cad_utils env).

    Pure path: all ops in {extrude,hole,orto_cut,revolve} -> method-chain code.
    HYBRID path (when an unsupported exotic op is present and frozen_stl given):
    freeze the prefix (everything up to the last non-optimizable statement) by
    executing it via cadgen and exporting `frozen_stl`, then emit
    `r = __MESH__(frozen_stl)` + the optimizable suffix ops as chains.  cq_parser
    turns __MESH__ into a 0-param mesh leaf and only the suffix params are tuned.
    """
    import ast
    from cadgen.extrude import (extrude as cg_extrude,
                                shape_from_cad_object,
                                precompute_surface_compound,
                                closest_surface_point_from_compound)
    from cadgen.hole import hole as cg_hole, _dominant_axis_selector
    from cadgen.orto_cut import orto_cut as cg_orto
    from cadgen.revolve import revolve as cg_revolve, _revolve_axis_points

    import cadquery as cq
    from cadgen.selectors import PointOnEdgeSelector
    try:
        from cadgen.selectors import PointOnFaceSelector
    except Exception:
        PointOnFaceSelector = None

    def _project(cur, point, wp):
        shp = shape_from_cad_object(cur)
        sp = closest_surface_point_from_compound(
            precompute_surface_compound(shp), tuple(map(float, point)))
        return shp, sp, float(sp.toTuple()[_WPI[wp]])

    _OPS = {'extrude': cg_extrude, 'hole': cg_hole, 'orto_cut': cg_orto,
            'revolve': cg_revolve}
    ns = {'cq': cq, 'PointOnEdgeSelector': PointOnEdgeSelector}
    if PointOnFaceSelector is not None:
        ns['PointOnFaceSelector'] = PointOnFaceSelector
    ns.update(_OPS)

    tree = ast.parse(code)
    # exec the original imports so exotic ops (gear/shell/sweep/...) are available
    # in ns for prefix execution in the hybrid path.
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            try:
                exec(ast.get_source_segment(code, stmt), ns)
            except Exception:
                pass

    def _is_opt_opline(stmt):
        return (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == 'r'
                and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Name)
                and stmt.value.func.id in _OPS)

    def _is_exotic_opline(stmt):
        return (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == 'r'
                and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Name)
                and stmt.value.func.id not in _OPS
                and stmt.value.func.id not in ('__MESH__',))

    # meaningful statements (drop imports + `r = None`)
    body = [s for s in tree.body
            if not isinstance(s, (ast.Import, ast.ImportFrom))
            and not (isinstance(s, ast.Assign) and isinstance(s.value, ast.Constant)
                     and s.value.value is None)]
    has_exotic = any(_is_exotic_opline(s) for s in body)

    out = ['import cadquery as cq']
    if 'PointOnEdgeSelector' in code or 'PointOnFaceSelector' in code:
        out.append('try:\n'
                   '    from cadquery_addons.selectors import PointOnEdgeSelector, PointOnFaceSelector\n'
                   'except ImportError:\n'
                   '    from cadgen.selectors import PointOnEdgeSelector, PointOnFaceSelector')

    if has_exotic:
        # HYBRID: freeze everything up to & including the LAST exotic op; the
        # suffix (holes/extrudes/orto_cuts AND interleaved chamfer/fillet edge
        # ops) becomes the optimizable SDF subtree onto the frozen mesh.  Edge
        # ops that can't resolve against the __MESH__ base drop gracefully.
        last_exotic = max(i for i, s in enumerate(body) if _is_exotic_opline(s))
        prefix, suffix = body[:last_exotic + 1], body[last_exotic + 1:]
        if not any(_is_opt_opline(s) for s in suffix):
            raise ValueError('no optimizable op after last exotic')
        if not frozen_stl:
            raise ValueError('hybrid needs frozen_stl output path')
        # execute the whole prefix via cadgen source-exec -> build ns['r']
        ns['r'] = None   # the dropped `r = None` seed
        for stmt in prefix:
            exec(ast.get_source_segment(code, stmt), ns)
        base = ns.get('r')
        if base is None:
            raise ValueError('prefix produced no solid')
        shape = base.val() if hasattr(base, 'val') else base
        cq.exporters.export(shape, frozen_stl)
        out.append(f'r = __MESH__({frozen_stl!r})')
        body = suffix          # only emit + advance the suffix
        first = False          # mesh is the base
    else:
        first = True

    for stmt in body:
        src = ast.get_source_segment(code, stmt)
        is_op_call = (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                      and isinstance(stmt.targets[0], ast.Name)
                      and stmt.targets[0].id == 'r'
                      and isinstance(stmt.value, ast.Call)
                      and isinstance(stmt.value.func, ast.Name))
        if not is_op_call:
            out.append(src)
            exec(src, ns)
            continue
        cur = ns.get('r')
        call = stmt.value
        fn = call.func.id
        if fn not in _OPS:
            raise ValueError(f'unsupported cadgen op: {fn}')
        args = [ast.literal_eval(a) for a in call.args[1:]]  # skip leading r
        kwargs = {k.arg: ast.literal_eval(k.value) for k in call.keywords}

        if fn == 'extrude':
            point, wp, sk, h = args[0], args[1], args[2], args[3]
            pos = args[4] if len(args) > 4 else kwargs.get('point_on_surface', True)
            if cur is None or not pos:
                off = float(point[_WPI[wp]])
            else:
                _, _, off = _project(cur, point, wp)
            chain = _join(wp, off, sk) + f'.extrude({h})'
            out.append(f'r={chain}' if first else f'r=r.union({chain})')
            ns['r'] = cg_extrude(cur, point, wp, sk, h, pos)
            first = False

        elif fn in ('hole', 'orto_cut'):
            point, wp, sk, h = args[0], args[1], args[2], args[3]
            shp, sp, off = _project(cur, point, wp)
            sign, axis = _dominant_axis_selector(shp, sp)
            lo, hi = min(off, off + h), max(off, off + h)
            if _AXI[axis] == _WPI[wp]:
                # over-pierce along the cutter axis: fold into the span
                if sign == '>':
                    hi += 10.0
                else:
                    lo -= 10.0
            # (sideways pierce direction is rare; plain cutter then)
            chain = _join(wp, lo, sk) + f'.extrude({hi - lo!r})'
            out.append(f'r=r.cut({chain})')
            ns['r'] = (cg_hole if fn == 'hole' else cg_orto)(cur, point, wp, sk, h)

        elif fn == 'revolve':
            point, wp, sk, ang, axname = args[0], args[1], args[2], args[3], args[4]
            if cur is None:
                ap = tuple(map(float, point))
                off = ap[_WPI[wp]]
            else:
                _, sp, off = _project(cur, point, wp)
                ap = tuple(float(v) for v in sp.toTuple())
            p0, p1 = _revolve_axis_points(ap, wp, axname)
            chain = _join(wp, off, sk) + f'.revolve({ang}, {tuple(p0)}, {tuple(p1)})'
            out.append(f'r={chain}' if first else f'r=r.union({chain})')
            ns['r'] = cg_revolve(cur, point, wp, sk, ang, axname)
            first = False

    if first:
        raise ValueError('no ops desugared')
    return '\n'.join(out) + '\n'


def desugar_cadgen(code: str, timeout: int = 180, frozen_stl: str = None) -> str:
    """Desugar via a subprocess in the cadgen-capable interpreter.  Falls back
    to an in-process attempt; raises on failure (callers keep the original).
    frozen_stl: enable the HYBRID path -> exotic prefix frozen to this STL, the
    returned code uses `r = __MESH__(frozen_stl)`.  Cache (env
    CADGEN_DESUGAR_CACHE) is bypassed for hybrid (the STL is a per-part side
    effect that must be regenerated)."""
    import subprocess
    cache_dir = os.environ.get('CADGEN_DESUGAR_CACHE')
    cache_f = None
    if cache_dir and not frozen_stl:
        import hashlib
        os.makedirs(cache_dir, exist_ok=True)
        cache_f = os.path.join(
            cache_dir, hashlib.sha1(code.encode()).hexdigest() + '.py')
        if os.path.exists(cache_f):
            with open(cache_f) as fh:
                return fh.read()
    py = os.environ.get('CADGEN_PY', _DEFAULT_PY)
    pp = os.environ.get('CADGEN_PYTHONPATH', _DEFAULT_PP)
    if os.path.exists(py):
        env = dict(os.environ)
        env['PYTHONPATH'] = pp
        if frozen_stl:
            env['CADGEN_FROZEN_STL'] = frozen_stl
        p = subprocess.run([py, os.path.abspath(__file__)], input=code,
                           capture_output=True, text=True, timeout=timeout,
                           env=env)
        if p.returncode == 0 and p.stdout.strip():
            if cache_f:
                with open(cache_f, 'w') as fh:
                    fh.write(p.stdout)
            return p.stdout
        raise RuntimeError(f'desugar subprocess failed: {p.stderr.strip()[:200]}')
    # last resort: maybe cadgen imports right here
    return _desugar_impl(code, frozen_stl)


if __name__ == '__main__':
    # Python prepends THIS script's directory to sys.path, where our
    # cq_parser.py MODULE shadows cadgen's cq_parser PACKAGE -> cadgen
    # imports break.  Drop the script dir before any cadgen import runs.
    _d = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path
                   if os.path.abspath(p or '.') != _d]
    _code = sys.stdin.read()
    try:
        sys.stdout.write(_desugar_impl(_code, os.environ.get('CADGEN_FROZEN_STL')))
    except Exception as e:
        sys.stderr.write(f'{type(e).__name__}: {e}\n')
        sys.exit(3)
