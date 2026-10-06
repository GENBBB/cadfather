"""Sketch polygons thinned before optimization.

A polygon literal of a CAD model is often a dense approximation of an arc,
with repeated vertices and runs of collinear ones; the optimizer moves every
vertex on its own and folds such a contour into self-intersection (OCC then
builds an invalid body). Only vertices collinear with their neighbours up to
`tol` (the code's 4-digit rounding) go, so the shape does not change; a
thinned contour that crosses itself steps the tolerance down, repeated
vertices alone as the last resort.
"""
import math
import re

POLY = re.compile(r'polygon\(\[(.*?)\]')
PT = re.compile(r'\((-?[\d.e-]+),\s*(-?[\d.e-]+)\)')


def _seg_dist(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = dx * dx + dy * dy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L))
    return math.dist(p, (a[0] + t * dx, a[1] + t * dy))


def _dp(pts, tol):
    """Douglas-Peucker on an open chain, ends kept (iterative)."""
    keep = {0, len(pts) - 1}
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        best, idx = -1.0, None
        for k in range(i + 1, j):
            d = _seg_dist(pts[k], pts[i], pts[j])
            if d > best:
                best, idx = d, k
        if idx is not None and best > tol:
            keep.add(idx)
            stack += [(i, idx), (idx, j)]
    return [pts[k] for k in sorted(keep)]


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def self_intersecting(p):
    """Two non-adjacent edges of the closed polygon cross."""
    n = len(p)
    for j in range(n):
        a, b = p[j], p[(j + 1) % n]
        for k in range(j + 2, n):
            if (k + 1) % n == j:
                continue
            c, d = p[k], p[(k + 1) % n]
            if _cross(a, b, c) * _cross(a, b, d) < 0 and _cross(c, d, a) * _cross(c, d, b) < 0:
                return True
    return False


def thin_polygon(pts, tol):
    """Closed contour: drop repeated vertices (a closing copy of the first one
    included, restored at the end), then DP split at vertex 0 and the vertex
    farthest from it. None if nothing changes or the result crosses itself."""
    closed = len(pts) > 1 and pts[-1] == pts[0]
    q = [p for i, p in enumerate(pts) if i == 0 or p != pts[i - 1]]
    if len(q) > 1 and q[-1] == q[0]:
        q = q[:-1]
    if len(q) < 4:
        return None
    f = max(range(len(q)), key=lambda i: math.dist(q[0], q[i]))
    # a DP step can cut a spike or a hairpin across its neighbour: step the
    # tolerance down, repeated vertices alone as the last resort
    for t_ in (tol, tol / 3, tol / 10, 0.0):
        t = _dp(q[:f + 1], t_)[:-1] + _dp(q[f:] + [q[0]], t_)[:-1] if t_ else q
        if len(t) >= 3 and not self_intersecting(t):
            break
    else:
        return None
    if closed:
        t = t + [t[0]]
    return t if len(t) < len(pts) else None


def thin_code(code, tol=1e-4):
    """Every polygon literal thinned; (code, points before, points after)."""
    n0 = n1 = 0

    def sub(m):
        nonlocal n0, n1
        pts = [(float(x), float(y)) for x, y in PT.findall(m[1])]
        t = thin_polygon(pts, tol)
        n0 += len(pts)
        n1 += len(t or pts)
        if t is None:
            return m[0]
        return 'polygon([' + ','.join(f'({x:.6g},{y:.6g})' for x, y in t) + ']'
    return POLY.sub(sub, code), n0, n1
