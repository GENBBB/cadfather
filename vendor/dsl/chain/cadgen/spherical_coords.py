from __future__ import annotations

import math
from typing import Tuple

import cadquery as cq


def normalize_angles_deg(theta: float, phi: float) -> Tuple[float, float]:
    """
    Normalize spherical angles in degrees.

    Conventions:
    - theta is the polar angle, mapped to [0, 180]
    - phi is the azimuth, mapped to [0, 360)

    If theta falls outside [0, 180], it is reflected through the poles and phi
    is shifted by 180 degrees accordingly.

    Examples:
    - (190, 10)   -> (170, 190)
    - (-30, 20)   -> (30, 200)
    - (720+45, -5)-> (45, 355)
    """
    # Bring theta into [0, 360)
    theta = theta % 360.0
    phi = phi % 360.0

    # Reflect angles above 180 through the pole
    if theta > 180.0:
        theta = 360.0 - theta
        phi = (phi + 180.0) % 360.0

    return theta, phi


def angles_to_unit_vector(theta: float, phi: float) -> Tuple[float, float, float]:
    """
    Convert spherical angles (degrees) to a unit 3D vector.

    Convention:
    - theta: polar angle from +z axis, in degrees
    - phi:   azimuth in xy-plane from +x toward +y, in degrees

    Returns:
        (x, y, z), a unit vector
    """
    theta, phi = normalize_angles_deg(theta, phi)

    th = math.radians(theta)
    ph = math.radians(phi)

    x = math.sin(th) * math.cos(ph)
    y = math.sin(th) * math.sin(ph)
    z = math.cos(th)

    return x, y, z


def vector_to_angles(n: Tuple[float, float, float], round_int=False) -> Tuple[float, float]:
    """
    Convert a 3D vector to spherical angles (theta, phi) in degrees.

    The input does not need to be perfectly normalized; it will be normalized internally.
    """
    x, y, z = n
    norm = math.sqrt(x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("Input vector must be non-zero.")

    x /= norm
    y /= norm
    z /= norm

    # Clamp for numerical safety
    z = max(-1.0, min(1.0, z))

    theta = math.degrees(math.acos(z))
    phi = math.degrees(math.atan2(y, x)) % 360.0

    if round:
        theta = round(theta)
        phi = round(phi)

    return theta, phi


def SphericalAnglesDirection(theta, phi):
    """theta and phi to unit cq.Vector"""
    x, y, z = angles_to_unit_vector(theta=theta, phi=phi)
    return cq.Vector(x, y, z)


def Plane(origin, xDir, normal):
    return cq.Workplane(cq.Plane(origin=origin,xDir=xDir,normal=normal))
