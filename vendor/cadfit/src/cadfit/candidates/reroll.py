"""Postprocess an emitted CadQuery program: detect repeated same-radius sketch
cutouts (`push([(x,y)]).circle(r,mode)`) and re-roll them into a single
GENERATIVE push -- a grid / lattice / polar comprehension -- so the program is
shorter and carries design intent.  Geometrically exact (verified by point-set
match within tolerance); falls back to leaving a group untouched if no pattern.

Public: reroll_program(code) -> (new_code, n_eliminated)
"""
from __future__ import annotations
import re
import numpy as np

_CIRC = re.compile(r"\.push\(\[\(([-\d.]+),\s*([-\d.]+)\)\]\)"
                   r"\.circle\(([-\d.]+)(?:,\s*mode='([a-z])')?\)")
TOL = 0.8


def _cluster(vals, tol):
    vals = sorted(vals); cl, cur = [], [vals[0]]
    for v in vals[1:]:
        (cur if v - cur[-1] <= tol else (cl.append(np.mean(cur)) or cur.clear() or cur)).append(v)
    cl.append(np.mean(cur)); return cl


def _regular(vals, tol):
    if len(vals) < 2:
        return (vals[0], 0.0, 1) if vals else None
    d = np.diff(vals)
    return (vals[0], float(np.mean(d)), len(vals)) if np.max(np.abs(d - np.mean(d))) <= tol else None


def _grid(pts, tol):
    rx = _regular(_cluster([p[0] for p in pts], tol), tol)
    ry = _regular(_cluster([p[1] for p in pts], tol), tol)
    if rx is None or ry is None or rx[2] * ry[2] != len(pts) or rx[2] * ry[2] < 4:
        return None
    gen = [(rx[0] + i * rx[1], ry[0] + j * ry[1]) for i in range(rx[2]) for j in range(ry[2])]
    if not _match(gen, pts, tol):
        return None
    return f"[(({rx[0]:.4f})+i*({rx[1]:.4f}), ({ry[0]:.4f})+j*({ry[1]:.4f})) " \
           f"for i in range({rx[2]}) for j in range({ry[2]})]"


def _lattice(vals, tol):
    """1-D lattice: vals = origin + k*period + unit-cell offsets, repeated."""
    vals = np.sort(np.unique(np.round(vals, 2))); n = len(vals)
    if n < 4:
        return None
    gaps = np.diff(vals)
    cands = sorted({round(float(gaps[i:i + w].sum()), 2)
                    for w in range(1, min(5, n)) for i in range(len(gaps) - w + 1)
                    if tol < gaps[i:i + w].sum() <= (vals[-1] - vals[0])})
    best = None
    for P in cands:
        if sum(1 for v in vals if np.min(np.abs(vals - (v + P))) <= tol) < n * 0.4:
            continue
        offs = sorted((v - vals[0]) % P for v in vals); merged = []
        for o in offs:
            if merged and (abs(o - merged[-1]) <= tol or P - (o - merged[0]) <= tol):
                continue
            merged.append(o)
        k = len(merged)
        if k == 0 or n % k != 0:
            continue
        nper = n // k
        gen = [vals[0] + i * P + o for i in range(nper) for o in merged]
        if len(gen) == n and max(np.min(np.abs(vals - g)) for g in gen) <= tol:
            if best is None or (k, -P) < best[0]:
                best = ((k, -P), P, [round(o, 3) for o in merged], nper, float(vals[0]))
    if best is None:
        return None
    _, P, unit, nper, o0 = best
    return o0, P, unit, nper


def _grid_via_lattice(pts, tol):
    """x lattice (unit cell may be >1) x regular y rows -> generative push."""
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    rows = _cluster(ys, tol); ry = _regular(rows, tol)
    Lx = _lattice(xs, tol)
    if Lx is None or ry is None:
        return None
    o0, P, unit, nper = Lx
    gen = [(o0 + i * P + dx, ry[0] + j * ry[1]) for i in range(nper) for dx in unit for j in range(ry[2])]
    if len(gen) != len(pts) or not _match(gen, pts, tol):
        return None
    return f"[(({o0:.4f})+i*({P:.4f})+dx, ({ry[0]:.4f})+j*({ry[1]:.4f})) " \
           f"for i in range({nper}) for dx in {unit} for j in range({ry[2]})]"


def _polar(pts, tol):
    Pn = np.array(pts); c = Pn.mean(0); rr = np.linalg.norm(Pn - c, axis=1)
    if rr.std() > tol or rr.mean() < 2 * tol or len(pts) < 4:
        return None
    ang = np.sort(np.degrees(np.arctan2(Pn[:, 1] - c[1], Pn[:, 0] - c[0])) % 360)
    d = np.diff(ang)
    if not (len(d) and np.max(np.abs(d - np.mean(d))) <= 2.0):
        return None
    R, a0, da, nn = rr.mean(), ang[0], float(np.mean(d)), len(pts)
    return f"[(({c[0]:.4f})+({R:.4f})*__import__('math').cos(__import__('math').radians(({a0:.3f})+k*({da:.3f}))), " \
           f"({c[1]:.4f})+({R:.4f})*__import__('math').sin(__import__('math').radians(({a0:.3f})+k*({da:.3f})))) " \
           f"for k in range({nn})]"


def _match(gen, pts, tol):
    return len(gen) == len(pts) and all(
        min(np.hypot(gx - px, gy - py) for px, py, *_ in pts) <= tol for gx, gy in gen)


def reroll_program(code: str):
    """Return (rerolled_code, n_ops_eliminated)."""
    hits = list(_CIRC.finditer(code))
    if not hits:
        return code, 0
    groups = {}
    for m in hits:
        x, y, r = float(m.group(1)), float(m.group(2)), float(m.group(3))
        groups.setdefault((round(r, 3), m.group(4) or 'a'), []).append((x, y, m.span(), r))
    eliminated = 0
    removals = []        # spans to delete
    inserts = []         # (pos, text)
    for (r, mode), items in groups.items():
        if len(items) < 4:
            continue
        pts = [(x, y) for x, y, *_ in items]
        gen = (_grid(pts, TOL) or _grid_via_lattice(pts, TOL) or _polar(pts, TOL))
        if not gen:
            continue
        modes = f",mode='{mode}'" if mode != 'a' else ""
        new = f".push({gen}).circle({r}{modes})"
        spans = [it[2] for it in items]
        inserts.append((min(s[0] for s in spans), new))
        removals.extend(spans)
        eliminated += len(items) - 1
    if not removals:
        return code, 0
    # apply edits right-to-left so spans stay valid
    edits = [(s, e, "") for s, e in removals] + [(p, p, t) for p, t in inserts]
    for s, e, t in sorted(edits, key=lambda z: -z[0]):
        code = code[:s] + t + code[e:]
    return code, eliminated
