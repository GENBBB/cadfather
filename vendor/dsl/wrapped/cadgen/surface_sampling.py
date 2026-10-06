from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import cadquery as cq
import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepGProp import BRepGProp
from OCP.BRepLProp import BRepLProp_SLProps
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GeomAbs import GeomAbs_SurfaceType
from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCP.gp import gp_Pnt
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_REVERSED


MODEL_TOL = 1e-6
AxisName = Literal["X", "-X", "Y", "-Y", "Z", "-Z"]
PositiveAxisName = Literal["X", "Y", "Z"]
PointLike = cq.Vector | Sequence[float]


@dataclass(frozen=True)
class AxisWorkplane:
    point: tuple[float, float, float]
    normal: PositiveAxisName
    xDir: AxisName

    def to_string(self) -> str:
        return (
            "workplane_at_closest_point("
            f"r, ({self.point[0]}, {self.point[1]}, {self.point[2]}), "
            f"{self.normal!r}, {self.xDir!r}"
            ")"
        )

    def transform(self, shift: Sequence[float], scale: float) -> None:
        object.__setattr__(
            self,
            "point",
            tuple((self.point[i] + shift[i]) * scale for i in range(3)),
        )

    def round(self) -> None:
        object.__setattr__(self, "point", tuple(round(v) for v in self.point))

    def to_dict(self) -> dict:
        return {
            "type": "AxisWorkplane",
            "point": self.point,
            "normal": self.normal,
            "xDir": self.xDir,
        }

    @staticmethod
    def from_dict(entity: dict) -> "AxisWorkplane":
        return AxisWorkplane(
            tuple(entity["point"]),
            entity["normal"],
            entity["xDir"],
        )


@dataclass(frozen=True)
class SampledSite:
    face_index: int
    face: cq.Face
    face_area: float
    point: cq.Vector
    outward: cq.Vector
    normal_axis: PositiveAxisName
    normal_sign: float
    face_kind: str
    cylinder_axis: PositiveAxisName | None = None
    cylinder_radius: float | None = None


@dataclass(frozen=True)
class PreparedSurfaceSampler:
    shape: cq.Shape
    surface_compound: cq.Shape
    faces: list[cq.Face]
    areas: list[float]
    triangles: list[list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]]

    @classmethod
    def from_cad_object(
        cls,
        cad_object: cq.Workplane | cq.Shape,
        mesh_deflection: float = 0.005,
    ) -> "PreparedSurfaceSampler":
        shape = shape_from_cad_object(cad_object)
        triangulate_shape(shape, mesh_deflection)
        faces = shape.Faces()
        areas = [face_area(face) for face in faces]
        triangles = [face_triangles(face) for face in faces]
        return cls(
            shape=shape,
            surface_compound=precompute_surface_compound(shape),
            faces=faces,
            areas=areas,
            triangles=triangles,
        )

    def sample_site(self) -> SampledSite:
        if not self.faces:
            raise ValueError("Cannot sample a site from a shape with no faces")

        weights = [
            area**0.42 if area > MODEL_TOL and self.triangles[i] else 0.0
            for i, area in enumerate(self.areas)
        ]
        skipped_faces: list[tuple[int, str]] = []

        while any(weight > 0 for weight in weights):
            face_idx = weighted_choice(list(range(len(self.faces))), weights)
            face = self.faces[face_idx]
            triangles = self.triangles[face_idx]
            if not triangles:
                weights[face_idx] = 0.0
                skipped_faces.append((face_idx, "not triangulated"))
                continue

            tri = weighted_choice(triangles, [item[3] for item in triangles])
            mesh_point = sample_triangle(tri[0], tri[1], tri[2])
            point_np, u, v = closest_point_on_face_surface(face, mesh_point)
            outward_np = normal_at_face_uv(face, u, v)

            face_kind = face_kind_name(face)
            adjusted = adjust_site_to_axis_normal(face, point_np, outward_np, face_kind)
            if adjusted is None:
                weights[face_idx] = 0.0
                skipped_faces.append((face_idx, face_kind))
                continue

            point_np, outward_np, normal_axis, normal_sign, cylinder_axis, cylinder_radius = (
                adjusted
            )
            return SampledSite(
                face_index=face_idx,
                face=face,
                face_area=self.areas[face_idx],
                point=cq.Vector(*point_np),
                outward=cq.Vector(*outward_np),
                normal_axis=normal_axis,
                normal_sign=normal_sign,
                face_kind=face_kind,
                cylinder_axis=cylinder_axis,
                cylinder_radius=cylinder_radius,
            )

        raise ValueError(
            "Surface sampler has no sampleable axis-aligned faces: "
            f"faces={len(self.faces)}, "
            f"triangulated_faces={sum(1 for triangles in self.triangles if triangles)}, "
            f"skipped_faces={skipped_faces[:10]}"
        )


def workplane_at_closest_point(
    cad_object: cq.Workplane | cq.Shape,
    point: PointLike,
    normal: PositiveAxisName,
    xDir: AxisName,
    surface_compound: cq.Shape | None = None,
) -> cq.Workplane:
    normal_vec = axis_vector(normal)
    x_dir_vec = axis_vector(xDir)
    if abs(normal_vec.dot(x_dir_vec)) > 1e-6:
        raise ValueError(f"normal={normal!r} and xDir={xDir!r} must be orthogonal")

    shape = None
    if surface_compound is None or not isinstance(cad_object, cq.Workplane):
        shape = shape_from_cad_object(cad_object)
    if surface_compound is None:
        surface_compound = precompute_surface_compound(shape)

    origin = closest_surface_point_from_compound(surface_compound, point)
    plane = cq.Plane(origin=origin, xDir=x_dir_vec, normal=normal_vec)
    local_workplane = cq.Workplane(plane)

    if isinstance(cad_object, cq.Workplane):
        return cad_object.copyWorkplane(local_workplane)

    base = cq.Workplane("XY").add(shape)
    return base.copyWorkplane(local_workplane)


def workplane_by_point(
    cad_object: cq.Workplane | cq.Shape,
    point: PointLike,
) -> cq.Workplane:
    shape = shape_from_cad_object(cad_object)
    point_vec = point_vector(point)
    point_tuple = point_vec.toTuple()

    candidate_faces = []
    query_vertex = point_vertex(point_tuple)
    for face in shape.Faces():
        extrema = BRepExtrema_DistShapeShape(query_vertex.wrapped, face.wrapped)
        extrema.Perform()
        if extrema.IsDone() and extrema.NbSolution() > 0:
            candidate_faces.append((float(extrema.Value()), face))
    candidate_faces.sort(key=lambda item: item[0])

    for _, face in candidate_faces:
        point_np, u, v = closest_point_on_face_surface(face, point_tuple)
        outward_np = normal_at_face_uv(face, u, v)
        face_kind = face_kind_name(face)
        adjusted = adjust_site_to_axis_normal_near_point(
            face,
            point_np,
            outward_np,
            face_kind,
            np.array(point_tuple, dtype=float),
        )
        if adjusted is None:
            continue

        origin_np, _, normal_axis, normal_sign, _, _ = adjusted
        normal_vec = axis_vector(signed_axis(normal_axis, normal_sign))
        x_dir_vec = axis_vector(default_x_dir_for_normal(normal_axis))
        plane = cq.Plane(
            origin=cq.Vector(*origin_np),
            xDir=x_dir_vec,
            normal=normal_vec,
        )
        return cq.Workplane(plane)

    raise ValueError("No face near point can provide an axis-parallel workplane")


def precompute_surface_compound(cad_object: cq.Workplane | cq.Shape) -> cq.Shape:
    shape = shape_from_cad_object(cad_object)
    faces = shape.Faces()
    if not faces:
        return shape
    return cq.Compound.makeCompound(faces)


def closest_surface_point_from_compound(
    surface_compound: cq.Shape,
    point: PointLike,
) -> cq.Vector:
    query_vertex = point_vertex(point)
    extrema = BRepExtrema_DistShapeShape(query_vertex.wrapped, surface_compound.wrapped)
    extrema.Perform()
    if not extrema.IsDone() or extrema.NbSolution() < 1:
        raise RuntimeError("Could not compute closest point on the CadQuery shape")

    point_on_shape = extrema.PointOnShape2(1)
    return cq.Vector(
        point_on_shape.X(),
        point_on_shape.Y(),
        point_on_shape.Z(),
    )


def shape_from_cad_object(cad_object: cq.Workplane | cq.Shape) -> cq.Shape:
    if isinstance(cad_object, cq.Workplane):
        try:
            return cad_object.findSolid()
        except ValueError:
            value = cad_object.val()
            if isinstance(value, cq.Shape):
                return value
            raise TypeError("cad_object workplane does not contain a CadQuery shape")

    if isinstance(cad_object, cq.Shape):
        return cad_object

    raise TypeError("cad_object must be a cadquery.Workplane or cadquery.Shape")


def point_vector(point: PointLike) -> cq.Vector:
    if isinstance(point, cq.Vector):
        return point
    values = tuple(float(value) for value in point)
    if len(values) != 3:
        raise ValueError("point must contain exactly three coordinates")
    return cq.Vector(*values)


def point_vertex(point: PointLike) -> cq.Vertex:
    vector = point_vector(point)
    return cq.Vertex.makeVertex(vector.x, vector.y, vector.z)


def axis_vector(axis: str) -> cq.Vector:
    axis_vectors = {
        "X": cq.Vector(1, 0, 0),
        "-X": cq.Vector(-1, 0, 0),
        "Y": cq.Vector(0, 1, 0),
        "-Y": cq.Vector(0, -1, 0),
        "Z": cq.Vector(0, 0, 1),
        "-Z": cq.Vector(0, 0, -1),
    }
    try:
        return axis_vectors[axis]
    except KeyError as exc:
        valid = ", ".join(repr(name) for name in axis_vectors)
        raise ValueError(f"axis must be one of: {valid}") from exc


def positive_axis(axis: str) -> PositiveAxisName:
    return axis[-1]  # type: ignore[return-value]


def signed_axis(axis: PositiveAxisName, sign: float) -> AxisName:
    return axis if sign >= 0 else f"-{axis}"  # type: ignore[return-value]


def orthogonal_axes(axis: PositiveAxisName) -> list[PositiveAxisName]:
    return [candidate for candidate in ("X", "Y", "Z") if candidate != axis]


def remaining_axis(axis_a: PositiveAxisName, axis_b: PositiveAxisName) -> PositiveAxisName:
    axes = [axis for axis in ("X", "Y", "Z") if axis not in (axis_a, axis_b)]
    if len(axes) != 1:
        raise ValueError("Expected two distinct axes")
    return axes[0]  # type: ignore[return-value]


def default_x_dir_for_normal(axis: PositiveAxisName) -> PositiveAxisName:
    return {"X": "Y", "Y": "Z", "Z": "X"}[axis]  # type: ignore[return-value]


def axis_name_from_vector(
    vector: Sequence[float] | cq.Vector,
    tol: float = 1e-5,
) -> tuple[PositiveAxisName, float] | None:
    vec = vector if isinstance(vector, cq.Vector) else cq.Vector(*vector)
    if vec.Length <= MODEL_TOL:
        return None
    vec = vec.normalized()
    components = {"X": vec.x, "Y": vec.y, "Z": vec.z}
    axis, value = max(components.items(), key=lambda item: abs(item[1]))
    if abs(abs(value) - 1.0) > tol:
        return None
    return axis, 1.0 if value >= 0 else -1.0  # type: ignore[return-value]


def triangulate_shape(shape: cq.Shape, linear_deflection: float = 0.005) -> None:
    mesher = BRepMesh_IncrementalMesh(shape.wrapped, linear_deflection, False, 0.1, True)
    mesher.Perform()
    if not mesher.IsDone():
        raise RuntimeError("OpenCascade failed to triangulate the shape")


def face_area(face: cq.Face) -> float:
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face.wrapped, props)
    return float(props.Mass())


def face_triangles(face: cq.Face) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    loc = face.wrapped.Location()
    tri = BRep_Tool.Triangulation_s(face.wrapped, loc)
    if tri is None:
        return []

    transform = loc.Transformation()
    triangles = []
    for i in range(1, tri.NbTriangles() + 1):
        triangle = tri.Triangle(i)
        ids = triangle.Get()
        pts = []
        for node_id in ids:
            p = tri.Node(int(node_id)).Transformed(transform)
            pts.append(np.array([p.X(), p.Y(), p.Z()], dtype=float))
        a, b, c = pts
        area = 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))
        if area > MODEL_TOL:
            triangles.append((a, b, c, area))
    return triangles


def weighted_choice(items: Sequence, weights: Sequence[float]):
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Cannot sample from zero total weight")
    target = float(np.random.random()) * total
    acc = 0.0
    for item, weight in zip(items, weights):
        acc += float(weight)
        if acc >= target:
            return item
    return items[-1]


def sample_triangle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    r1 = math.sqrt(float(np.random.random()))
    r2 = float(np.random.random())
    return (1.0 - r1) * a + r1 * (1.0 - r2) * b + r1 * r2 * c


def closest_point_on_face_surface(
    face: cq.Face,
    point: Sequence[float],
) -> tuple[np.ndarray, float, float]:
    surface = BRep_Tool.Surface_s(face.wrapped)
    projected = GeomAPI_ProjectPointOnSurf(gp_Pnt(*point), surface)
    if projected.NbPoints() < 1:
        raise RuntimeError("Could not project sampled mesh point onto exact face surface")
    u, v = projected.LowerDistanceParameters()
    nearest = projected.NearestPoint()
    return np.array([nearest.X(), nearest.Y(), nearest.Z()], dtype=float), float(u), float(v)


def normal_at_face_uv(face: cq.Face, u: float, v: float) -> np.ndarray:
    adaptor = BRepAdaptor_Surface(face.wrapped, True)
    props = BRepLProp_SLProps(adaptor, u, v, 1, MODEL_TOL)
    if not props.IsNormalDefined():
        raise RuntimeError("Normal is not defined at the sampled point")
    normal = np.array(
        [props.Normal().X(), props.Normal().Y(), props.Normal().Z()],
        dtype=float,
    )
    normal /= np.linalg.norm(normal)
    if face.wrapped.Orientation() == TopAbs_REVERSED:
        normal = -normal
    return normal


def face_kind_name(face: cq.Face) -> str:
    surface_type = BRepAdaptor_Surface(face.wrapped, True).GetType()
    if surface_type == GeomAbs_SurfaceType.GeomAbs_Plane:
        return "plane"
    if surface_type == GeomAbs_SurfaceType.GeomAbs_Cylinder:
        return "cylinder"
    if surface_type == GeomAbs_SurfaceType.GeomAbs_Sphere:
        return "sphere"
    return "other"


def adjust_site_to_axis_normal(
    face: cq.Face,
    point: np.ndarray,
    outward: np.ndarray,
    face_kind: str,
) -> tuple[np.ndarray, np.ndarray, PositiveAxisName, float, PositiveAxisName | None, float | None] | None:
    if face_kind == "plane":
        axis = axis_name_from_vector(outward)
        if axis is None:
            return None
        normal_axis, normal_sign = axis
        return point, outward, normal_axis, normal_sign, None, None

    if face_kind == "cylinder":
        return adjust_cylinder_site_to_axis_normal(face, point)

    if face_kind == "sphere":
        return adjust_sphere_site_to_axis_normal(face, point)

    return None


def adjust_site_to_axis_normal_near_point(
    face: cq.Face,
    point: np.ndarray,
    outward: np.ndarray,
    face_kind: str,
    reference_point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, PositiveAxisName, float, PositiveAxisName | None, float | None] | None:
    axis = axis_name_from_vector(outward)
    if axis is not None:
        normal_axis, normal_sign = axis
        return point, outward, normal_axis, normal_sign, None, None

    if face_kind == "plane":
        return None

    candidates = axis_normal_candidates(face, point, face_kind)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: float(np.linalg.norm(item[0] - reference_point)),
    )


def axis_normal_candidates(
    face: cq.Face,
    point: np.ndarray,
    face_kind: str,
) -> list[
    tuple[
        np.ndarray,
        np.ndarray,
        PositiveAxisName,
        float,
        PositiveAxisName | None,
        float | None,
    ]
]:
    if face_kind == "cylinder":
        return cylinder_axis_normal_candidates(face, point)
    if face_kind == "sphere":
        return sphere_axis_normal_candidates(face)
    return []


def adjust_cylinder_site_to_axis_normal(
    face: cq.Face,
    point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, PositiveAxisName, float, PositiveAxisName | None, float | None] | None:
    candidates = cylinder_axis_normal_candidates(face, point)
    np.random.shuffle(candidates)
    return candidates[0] if candidates else None


def cylinder_axis_normal_candidates(
    face: cq.Face,
    point: np.ndarray,
) -> list[
    tuple[
        np.ndarray,
        np.ndarray,
        PositiveAxisName,
        float,
        PositiveAxisName | None,
        float | None,
    ]
]:
    adaptor = BRepAdaptor_Surface(face.wrapped, True)
    cylinder = adaptor.Cylinder()
    axis_dir = cylinder.Axis().Direction()
    axis_vec = np.array([axis_dir.X(), axis_dir.Y(), axis_dir.Z()], dtype=float)
    axis_vec /= np.linalg.norm(axis_vec)
    axis_name = axis_name_from_vector(axis_vec)
    cylinder_axis = axis_name[0] if axis_name is not None else None
    radius = float(cylinder.Radius())
    loc = cylinder.Location()
    axis_origin = np.array([loc.X(), loc.Y(), loc.Z()], dtype=float)
    axial_point = axis_origin + np.dot(point - axis_origin, axis_vec) * axis_vec

    candidates = []
    for axis in ("X", "Y", "Z"):
        base = np.array(axis_vector(axis).toTuple(), dtype=float)
        if abs(float(np.dot(base, axis_vec))) > 1e-5:
            continue
        for sign in (1.0, -1.0):
            normal = sign * base
            candidate = axial_point + radius * normal
            if point_is_on_face(face, candidate):
                candidates.append((candidate, normal, axis, sign, cylinder_axis, radius))

    return candidates  # type: ignore[return-value]


def adjust_sphere_site_to_axis_normal(
    face: cq.Face,
    point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, PositiveAxisName, float, PositiveAxisName | None, float | None] | None:
    candidates = sphere_axis_normal_candidates(face)
    np.random.shuffle(candidates)
    return candidates[0] if candidates else None


def sphere_axis_normal_candidates(
    face: cq.Face,
) -> list[
    tuple[
        np.ndarray,
        np.ndarray,
        PositiveAxisName,
        float,
        PositiveAxisName | None,
        float | None,
    ]
]:
    adaptor = BRepAdaptor_Surface(face.wrapped, True)
    sphere = adaptor.Sphere()
    center_pnt = sphere.Location()
    center = np.array([center_pnt.X(), center_pnt.Y(), center_pnt.Z()], dtype=float)
    radius = float(sphere.Radius())

    candidates = []
    for axis in ("X", "Y", "Z"):
        base = np.array(axis_vector(axis).toTuple(), dtype=float)
        for sign in (1.0, -1.0):
            normal = sign * base
            candidate = center + radius * normal
            if point_is_on_face(face, candidate):
                candidates.append((candidate, normal, axis, sign, None, None))

    return candidates  # type: ignore[return-value]


def point_is_on_face(face: cq.Face, point: Sequence[float], tol: float = 1e-5) -> bool:
    vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(*point)).Vertex()
    extrema = BRepExtrema_DistShapeShape(vertex, face.wrapped)
    extrema.Perform()
    return bool(extrema.IsDone() and extrema.NbSolution() > 0 and extrema.Value() <= tol)
