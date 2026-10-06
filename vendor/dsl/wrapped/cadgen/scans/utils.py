import numpy as np
import open3d as o3d
import trimesh


def load_and_normalize_mesh(mesh_path: str) -> trimesh.Trimesh:
    """
    Loads a mesh from a file and normalizes it to unit size.
    """
    mesh = trimesh.load(mesh_path)
    mesh.vertices = mesh.vertices / mesh.scale
    return mesh


def subdivide_mesh(mesh, iterations=1):
    """
    Increase the number of triangles in the mesh with Loop subdivision.
    """
    for _ in range(iterations):
        mesh = mesh.subdivide()
    return mesh


def remove_unsupported_triangles(
    mesh: o3d.geometry.TriangleMesh,
    support_pcd: o3d.geometry.PointCloud,
    max_dist: float,
) -> o3d.geometry.TriangleMesh:
    tris = np.asarray(mesh.triangles)
    verts = np.asarray(mesh.vertices)
    centers = verts[tris].mean(axis=1)
    centers_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centers))
    # distance to the nearest point of the source cloud
    d = np.asarray(centers_pcd.compute_point_cloud_distance(support_pcd))
    tri_mask = (d > max_dist).tolist()
    mesh.remove_triangles_by_mask(tri_mask)
    mesh.remove_unreferenced_vertices()
    return mesh


def make_o3d_pc(mesh: trimesh.Trimesh, pc_size: int) -> o3d.geometry.PointCloud:
    """
    Sample points on the surface and interpolate normals from the mesh vertex normals
    (so the normals stay "as in the source mesh").
    """
    # points + indices of the faces they were sampled from
    pts, face_idx = trimesh.sample.sample_surface(mesh, pc_size)  # (N,3), (N,)

    # face vertex triples and the corresponding vertex normals
    faces = mesh.faces[face_idx]  # (N,3) vertex indices
    tri_xyz = mesh.vertices[faces]  # (N,3,3)
    tri_vn = mesh.vertex_normals[faces]  # (N,3,3) normals at the vertices

    # barycentric weights for normal interpolation
    A = tri_xyz[:, 0, :]
    B = tri_xyz[:, 1, :]
    C = tri_xyz[:, 2, :]
    v0 = B - A
    v1 = C - A
    v2 = pts - A

    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    denom = d00 * d11 - d01 * d01 + 1e-18

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    nrm = (
        tri_vn[:, 0, :] * u[:, None]
        + tri_vn[:, 1, :] * v[:, None]
        + tri_vn[:, 2, :] * w[:, None]
    )

    # normalize; do NOT flip the sign (keep it as in the source mesh)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12

    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(pts)
    o3d_pc.normals = o3d.utility.Vector3dVector(nrm)
    return o3d_pc


def single_camera_scan(pcd, diameter=100.0, camera=[1.0, 0.0, 1.0]):
    camera = np.array(camera)
    camera = camera / np.linalg.norm(camera)
    camera = camera * diameter
    HPR_radius = 1e5 * diameter
    _, pt_map = pcd.hidden_point_removal(camera, HPR_radius)
    pcd2 = pcd.select_by_index(pt_map)
    return pcd2, pt_map


def multi_camera_scan(
    original_pcd,
    diameter=100.0,
    min_angle=-np.pi / 2,
    max_angle=0,
    n_cameras=10,
    cos_dimension=0,
    sin_dimension=2,
    keep_scans=False,
    center_camera=[0, 0, 0],
):
    """
        mesh: trimesh.Trimesh
    pc_size: number of points per scan

    About the virtual cameras:
    This algorithm only chooses subsets of the
    original point cloud sampled from the full surface of the mesh.


    The direction from an object towards the camera will be
    In case of rotation around the z-axis:
    r = [cos(angle), sin(angle), 0] + center_camera
    (which means that cos_dimension = 0 and sin_dimension = 1)

    In case of rotation around the y-axis:
    r = [cos(angle), 0, sin(angle)] + center_camera
    (which means that cos_dimension = 0 and sin_dimension = 2)

    In case of rotation around the x-axis:
    r = [0, cos(angle), sin(angle)] + center_camera
    (which means that cos_dimension = 1 and sin_dimension = 2)

    The distance of the camera from the object is monotonically increasing with "diameter",
    however the correspondence is exact due to the usage of the approximate algorithm
    (R. Mehra et.al, Visibility of noisy point cloud data,
    Computers & Graphics, Volume 34, Issue 3, 2010, Pages 219-230)

    angle varies from min_angle to max_angle with n_cameras steps.
    """

    pcds = []
    all_points = set()
    for i, angle in enumerate(np.linspace(min_angle, max_angle, n_cameras)):
        x = diameter * np.cos(angle)
        y = diameter * np.sin(angle)
        color = np.array([i / (n_cameras + 1), i / (n_cameras + 1), 0])

        camera = [0, 0, 0]
        camera[cos_dimension] = x
        camera[sin_dimension] = y
        camera = camera + np.array(center_camera)

        # pcd_cur = copy.deepcopy(original_pcd)
        pcd_cur = original_pcd
        pcd_cur.paint_uniform_color(np.array(color))
        _, pt_map = single_camera_scan(pcd=pcd_cur, diameter=diameter, camera=camera)
        all_points.update(pt_map)

        pcd_cur = pcd_cur.select_by_index(pt_map)
        if keep_scans:
            pcds.append(pcd_cur)

    return pcds, all_points
