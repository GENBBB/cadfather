"""
Visualization of optimization convergence:
- GIF showing target (blue) vs prediction (red) at each step
- Log of intermediate CadQuery scripts with parameter values
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import _cad_grad as cg
from cq_parser import parse_cadquery
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import imageio
import re


def eval_sdf_grid(tree, bounds, resolution=50):
    """Evaluate SDF on a 3D grid. Returns (sdf_grid, grid_axes)."""
    lo, hi = bounds
    x = np.linspace(lo[0], hi[0], resolution)
    y = np.linspace(lo[1], hi[1], resolution)
    z = np.linspace(lo[2], hi[2], resolution)
    grid = np.zeros((resolution, resolution, resolution))
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            for k, zk in enumerate(z):
                grid[i, j, k] = tree.eval_sdf(xi, yj, zk)
    return grid, (x, y, z)


def sdf_to_mesh(grid, axes, level=0.0):
    """Extract mesh from SDF grid using marching cubes."""
    try:
        verts, faces, _, _ = measure.marching_cubes(grid, level=level)
    except (ValueError, RuntimeError):
        return None, None
    # Scale vertices to world coordinates
    x, y, z = axes
    spacing = np.array([x[1] - x[0], y[1] - y[0], z[1] - z[0]])
    origin = np.array([x[0], y[0], z[0]])
    verts = verts * spacing + origin
    return verts, faces


def render_frame(target_verts, target_faces, pred_verts, pred_faces,
                 step, loss, bounds, elev=25, azim=135, figsize=(8, 6)):
    """Render a single frame: target (blue) + prediction (red), transparent."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    lo, hi = bounds
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Draw target mesh (blue, transparent)
    if target_verts is not None and len(target_faces) > 0:
        target_polys = target_verts[target_faces]
        target_coll = Poly3DCollection(target_polys, alpha=0.25,
                                        facecolor='steelblue', edgecolor='steelblue',
                                        linewidth=0.1)
        ax.add_collection3d(target_coll)

    # Draw prediction mesh (red, transparent)
    if pred_verts is not None and len(pred_faces) > 0:
        pred_polys = pred_verts[pred_faces]
        pred_coll = Poly3DCollection(pred_polys, alpha=0.35,
                                      facecolor='tomato', edgecolor='darkred',
                                      linewidth=0.1)
        ax.add_collection3d(pred_coll)

    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f'Step {step}  |  Loss: {loss:.6f}', fontsize=14, fontweight='bold')
    fig.legend(['Target', 'Prediction'], loc='lower right', fontsize=11)

    fig.tight_layout()

    # Convert to image array
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
    plt.close(fig)
    return buf


def inject_params_to_code(code, parse_result, params):
    """Inject optimized parameters back into CadQuery code (approximate)."""
    lines = code.split('\n')
    names = parse_result.param_names
    gt = parse_result.params

    parts = []
    parts.append("# Parameters (name: ground_truth -> optimized):")
    for i, (name, gt_val, opt_val) in enumerate(zip(names, gt, params)):
        delta = opt_val - gt_val
        parts.append(f"#   {name}: {gt_val:.4f} -> {opt_val:.4f}  (delta={delta:+.4f})")
    parts.append("")
    parts.append(code)
    return '\n'.join(parts)


def run_visualization(cq_code, name="optimization", output_dir="viz_output",
                      resolution=48, n_frames=20, steps_per_frame=60,
                      lr=2e-3, perturbation=0.15):
    """Full visualization pipeline for one CadQuery script.

    Creates:
    - {output_dir}/{name}.gif  — animation
    - {output_dir}/{name}_log.txt — intermediate scripts and params
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Parse and build tree
    result = parse_cadquery(cq_code)
    if result.tree_desc is None:
        print(f"  Skipped {name}: no tree")
        return

    tree = cg.create_tree(result.tree_desc)
    tree.set_params(result.params)
    gt_params = list(result.params)

    # 2. Determine bounds from object size
    max_dim = 1.0
    for i, v in enumerate(gt_params):
        n = result.param_names[i] if i < len(result.param_names) else ""
        if 'half' in n or '.r' in n or '.hx' in n or '.hy' in n or '.hz' in n or '.hh' in n or 'depth' in n:
            max_dim = max(max_dim, abs(v))
    pad = max_dim * 1.4
    bounds = ((-pad, -pad, -pad), (pad, pad, pad))

    # 3. Sample SDF points from ground truth
    rng = np.random.RandomState(42)
    sample_range = max_dim * 1.5
    positions = rng.uniform(-sample_range, sample_range, (60000, 3))
    sdf_vals = np.array([tree.eval_sdf(p[0], p[1], p[2]) for p in positions])
    band = max_dim * 0.15
    mask = np.abs(sdf_vals) < band
    positions = positions[mask]
    sdf_vals = sdf_vals[mask]
    if len(positions) > 10000:
        idx = rng.choice(len(positions), 10000, replace=False)
        positions = positions[idx]
        sdf_vals = sdf_vals[idx]
    pos_arr = np.ascontiguousarray(positions, dtype=np.float64)
    dist_arr = np.ascontiguousarray(sdf_vals, dtype=np.float64)

    # 4. Extract target mesh (once)
    print(f"  Extracting target mesh...")
    target_grid, axes = eval_sdf_grid(tree, bounds, resolution)
    target_verts, target_faces = sdf_to_mesh(target_grid, axes)

    # 5. Perturb parameters
    perturbed = list(gt_params)
    for i in range(len(perturbed)):
        n = result.param_names[i] if i < len(result.param_names) else ""
        if n.endswith('.cx') or n.endswith('.cy') or n.endswith('.cz'):
            continue
        if 'hole.hh' in n and perturbed[i] >= 10.0:
            continue
        perturbed[i] *= (1.0 + rng.normal(0, perturbation))
        perturbed[i] = max(perturbed[i], 0.01)

    tree.set_params(perturbed)

    # 6. Run optimization in chunks, capture frames
    frames = []
    log_entries = []
    current_params = list(perturbed)

    # Frame 0: initial perturbed state
    print(f"  Rendering frame 0/{n_frames} (initial)...")
    pred_grid, _ = eval_sdf_grid(tree, bounds, resolution)
    pred_verts, pred_faces = sdf_to_mesh(pred_grid, axes)
    initial_loss = float(np.mean((pred_grid - target_grid) ** 2))
    frame = render_frame(target_verts, target_faces, pred_verts, pred_faces,
                         0, initial_loss, bounds)
    frames.append(frame)
    log_entries.append(f"=== Step 0 (initial, perturbed) ===\n"
                      f"Loss: {initial_loss:.6f}\n"
                      f"{inject_params_to_code(cq_code, result, current_params)}\n")

    total_steps = n_frames * steps_per_frame
    for f_idx in range(1, n_frames + 1):
        print(f"  Optimizing + rendering frame {f_idx}/{n_frames}...")
        opt_result = cg.optimize(
            tree, pos_arr, dist_arr,
            lr=lr, steps=steps_per_frame, batch_size=512,
            early_stop=0.0, verbose=False
        )
        current_params = list(opt_result['params'])
        loss = opt_result['loss_history'][-1]
        step_num = f_idx * steps_per_frame

        # Extract prediction mesh
        pred_grid, _ = eval_sdf_grid(tree, bounds, resolution)
        pred_verts, pred_faces = sdf_to_mesh(pred_grid, axes)
        frame = render_frame(target_verts, target_faces, pred_verts, pred_faces,
                             step_num, loss, bounds)
        frames.append(frame)

        log_entries.append(f"=== Step {step_num} ===\n"
                          f"Loss: {loss:.6f}\n"
                          f"{inject_params_to_code(cq_code, result, current_params)}\n")

    # 7. Hold last frame for a beat
    for _ in range(5):
        frames.append(frames[-1])

    # 8. Save GIF
    gif_path = os.path.join(output_dir, f"{name}.gif")
    imageio.mimsave(gif_path, frames, fps=4, loop=0)
    print(f"  Saved GIF: {gif_path}")

    # 9. Save log
    log_path = os.path.join(output_dir, f"{name}_log.txt")
    with open(log_path, 'w') as f:
        f.write(f"Optimization log for: {name}\n")
        f.write(f"CadQuery code: {cq_code}\n")
        f.write(f"Total steps: {total_steps}\n")
        f.write(f"Ground truth params: {gt_params}\n")
        f.write(f"Initial perturbed params: {perturbed}\n")
        f.write(f"Final params: {current_params}\n\n")
        for entry in log_entries:
            f.write(entry + '\n')
    print(f"  Saved log: {log_path}")


# ============================================================
# Complex test cases for visualization
# ============================================================

DEMO_CASES = {
    "box_with_hole": {
        "code": 'result = cq.Workplane("XY").box(6, 6, 4).hole(2)',
        "desc": "Box with through-hole",
    },
    "union_box_sphere_fillet": {
        "code": 'result = cq.Workplane("XY").box(4, 4, 4).union(cq.Workplane("XY").sphere(3)).fillet(0.5)',
        "desc": "Union of box + sphere with fillet",
    },
    "bottle_profile": {
        "code": 'result = cq.Workplane("XY").moveTo(-2, 0).lineTo(-2, 0.6).threePointArc((0, 1.2), (2, 0.6)).lineTo(2, 0).lineTo(2, -0.6).threePointArc((0, -1.2), (-2, -0.6)).close().extrude(6)',
        "desc": "Bottle profile with arcs + extrude",
    },
    "L_bracket_extruded": {
        "code": 'result = cq.Workplane("XY").moveTo(0,0).lineTo(4,0).lineTo(4,1).lineTo(1,1).lineTo(1,3).lineTo(0,3).close().extrude(2)',
        "desc": "L-bracket profile extruded",
    },
    "box_cut_cyl_chamfer_translate": {
        "code": 'result = cq.Workplane("XY").box(6, 6, 6).cut(cq.Workplane("XY").cylinder(8, 1.5)).chamfer(0.3).translate((0, 0, 1))',
        "desc": "Box with cylindrical cut, chamfer, translated",
    },
    "shell_cylinder": {
        "code": 'result = cq.Workplane("XY").cylinder(6, 3).shell(0.5)',
        "desc": "Hollow cylinder (shell)",
    },
    "loft_rect_to_circle": {
        "code": 'result = cq.Workplane("XY").rect(4, 4).workplane(offset=3).circle(2).loft()',
        "desc": "Loft from rectangle to circle",
    },
    "sweep_circle_L_path": {
        "code": 'result = cq.Workplane("XY").circle(0.5).sweep(cq.Workplane("XZ").moveTo(0, 0).lineTo(0, 3).lineTo(3, 3).wire())',
        "desc": "Circle swept along L-shaped path",
    },
    "rounded_box": {
        "code": 'result = cq.Workplane("XY").box(4, 4, 4).edges().fillet(0.4)',
        "desc": "Box with all edges filleted (rounded box)",
    },

    # ---- Multi-op parts (3-5 3D operations: extrude, revolve, loft, sweep) ----

    "extrude_cut_hole_shell": {
        "code": 'result = cq.Workplane("XY").rect(6, 4).extrude(3).cut(cq.Workplane("XY").cylinder(8, 1)).hole(1.5).shell(0.4)',
        "desc": "Extrude rect -> cut cylinder -> hole -> shell (4 ops)",
    },
    "extrude_union_extrude_fillet": {
        "code": 'result = cq.Workplane("XY").rect(5, 3).extrude(2).union(cq.Workplane("XY").circle(1.5).extrude(4)).fillet(0.3)',
        "desc": "Extrude rect + union extrude circle + fillet (3 ops)",
    },
    "loft_cut_hole_translate": {
        "code": 'result = cq.Workplane("XY").rect(5, 5).workplane(offset=4).circle(2).loft().cut(cq.Workplane("XY").cylinder(8, 0.8)).hole(1).translate((0, 0, 1))',
        "desc": "Loft rect->circle -> cut cyl -> hole -> translate (5 ops)",
    },
    "revolve_cut_translate": {
        "code": 'result = cq.Workplane("XY").moveTo(2, -2).lineTo(4, -2).lineTo(4, 2).lineTo(2, 2).close().revolve().cut(cq.Workplane("XY").box(3, 3, 3)).translate((0, 0, 0.5))',
        "desc": "Revolve rect -> cut box -> translate (3 ops)",
    },
    "sweep_union_extrude_hole": {
        "code": 'result = cq.Workplane("XY").circle(0.6).sweep(cq.Workplane("XZ").moveTo(0, 0).lineTo(0, 4).wire()).union(cq.Workplane("XY").rect(3, 3).extrude(1)).hole(0.5)',
        "desc": "Sweep circle + union extruded rect + hole (4 ops)",
    },
    "extrude_poly_revolve_union": {
        "code": 'result = cq.Workplane("XY").moveTo(0, 0).lineTo(3, 0).lineTo(3, 1).lineTo(1, 1).lineTo(1, 3).lineTo(0, 3).close().extrude(2).union(cq.Workplane("XY").moveTo(3, 0).circle(0.8).revolve())',
        "desc": "Extrude L-bracket + union revolve torus (3 ops)",
    },
    "loft_union_sweep_shell": {
        "code": 'result = cq.Workplane("XY").circle(3).workplane(offset=4).circle(1.5).loft().union(cq.Workplane("XY").circle(0.5).sweep(cq.Workplane("XZ").moveTo(0, 0).lineTo(0, 6).wire())).shell(0.3)',
        "desc": "Loft cone + union swept tube + shell (4 ops)",
    },
}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize CAD optimization convergence")
    parser.add_argument("--case", type=str, default=None,
                        help="Specific case name, or 'all' for all cases")
    parser.add_argument("--output", type=str, default="viz_output",
                        help="Output directory")
    parser.add_argument("--resolution", type=int, default=48,
                        help="Grid resolution for marching cubes")
    parser.add_argument("--frames", type=int, default=20,
                        help="Number of frames in GIF")
    parser.add_argument("--steps-per-frame", type=int, default=60,
                        help="Optimization steps between frames")
    args = parser.parse_args()

    cases = DEMO_CASES
    if args.case and args.case != 'all':
        if args.case in cases:
            cases = {args.case: cases[args.case]}
        else:
            print(f"Unknown case: {args.case}")
            print(f"Available: {', '.join(cases.keys())}")
            sys.exit(1)

    for name, info in cases.items():
        print(f"\n{'='*50}")
        print(f"Case: {name} — {info['desc']}")
        print(f"{'='*50}")
        run_visualization(
            info["code"], name=name, output_dir=args.output,
            resolution=args.resolution, n_frames=args.frames,
            steps_per_frame=args.steps_per_frame,
        )

    print(f"\nDone! GIFs and logs saved to {args.output}/")
