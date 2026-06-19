"""
Simple STL converter module - converts GLB to STL without OpenSCAD dependency.

This module provides a lightweight alternative to the full print pipeline,
converting GLB files to STL format using only trimesh (no OpenSCAD required).

Usage:
    from modules.simple_stl_converter import convert_glb_to_stl

    stl_path = convert_glb_to_stl(glb_path, file_number, output_folder)
"""

import trimesh
import os


def ensure_printable_mesh(mesh):
    """Best-effort improve a mesh for FDM slicing WITHOUT destroying geometry.

    TRELLIS GLB meshes are non-watertight, multi-component shells. An aggressive
    repair (e.g. pymeshfix.repair) discards most of that geometry and collapses
    the mesh to a small fragment -> a "flat"/broken STL. So we only apply gentle,
    geometry-preserving repairs (merge duplicate vertices, fix winding/normals,
    fill small boundary holes) and fall back to the original mesh if the result
    lost substantial geometry or shrank its bounding box. Print-validity only;
    this does not reduce slicer support volume.
    """
    import numpy as np
    import trimesh

    if mesh.is_watertight and mesh.is_winding_consistent:
        return mesh

    repaired = mesh.copy()
    repaired.merge_vertices()
    trimesh.repair.fix_normals(repaired)   # consistent winding + outward normals
    trimesh.repair.fill_holes(repaired)    # gentle small-boundary fill (adds faces, never deletes geometry)

    # Safety net: never return a mesh that lost substantial geometry or whose
    # bounding box collapsed (the symptom of an over-aggressive repair).
    bbox_preserved = np.allclose(mesh.extents, repaired.extents, rtol=0.02, atol=1e-6)
    if len(repaired.faces) < 0.9 * len(mesh.faces) or not bbox_preserved:
        return mesh
    return repaired


def remesh_for_printing(mesh, res=288, smooth_iters=8, max_voxels=30_000_000):
    """Solid voxel remesh + Taubin smoothing -> a single watertight, manifold print solid.

    TRELLIS GLBs are fragmented, non-watertight, multi-component shells (median ~500
    components, 0% watertight) -- not printable as solids. This re-extracts the surface
    from a SOLID voxelization, yielding one closed manifold solid. Validated in
    investigations/mesh_repair: 100% watertight across the corpus, perceptual quality
    impact (DINOv2 vs original) ~0.033 -- at the metric's noise floor.

    Returns the remeshed Trimesh, or None if the result is degenerate (caller falls back).
    The voxel-count guard caps resolution so unusual inputs can't blow up memory.
    """
    import numpy as np
    if mesh is None or len(mesh.faces) == 0:
        return None
    diag = float(np.linalg.norm(mesh.extents))
    if diag <= 0:
        return None
    est_voxels = float(np.prod(mesh.extents)) * (res / diag) ** 3
    if est_voxels > max_voxels:
        res = max(96, int(res * (max_voxels / est_voxels) ** (1.0 / 3.0)))
    vg = mesh.voxelized(pitch=diag / res).fill()
    r = vg.marching_cubes
    r.apply_transform(vg.transform)   # marching_cubes is in voxel-index space; map back to world
    r.merge_vertices()
    trimesh.repair.fix_normals(r)
    if smooth_iters > 0:
        trimesh.smoothing.filter_taubin(r, lamb=0.5, nu=-0.53, iterations=smooth_iters)
    # Validate: non-empty, watertight, and bbox preserved (no collapse).
    if len(r.faces) == 0 or not r.is_watertight:
        return None
    if not np.allclose(mesh.extents, r.extents, rtol=0.05, atol=1e-6):
        return None
    return r


def prepare_printable_mesh(glb_path: str):
    """Load a GLB and return the print-ready mesh (env-gated remesh, else gentle repair).

    Single source of truth: the returned mesh is exactly what gets exported as STL.
    Returns None if the loaded mesh is empty.
    """
    mesh = trimesh.load(glb_path, force='mesh')
    if mesh is None or len(mesh.faces) == 0:
        return None
    mode = os.environ.get('TRELLIS_PRINT_REMESH', 'off').strip().lower()
    remeshed = None
    if mode in ('voxel288_smooth', 'voxel', 'on', 'true', '1'):
        try:
            remeshed = remesh_for_printing(mesh)
        except Exception as e:
            print(f"⚠️  print remesh failed ({type(e).__name__}: {e}); using gentle repair")
    if remeshed is not None:
        print(f"🧱 Applied watertight print remesh ({mode})")
        return remeshed
    return ensure_printable_mesh(mesh)


def export_print_ready(glb_path: str, out_dir: str, basename: str = "print_ready"):
    """Compute the print-ready mesh once; export it as GLB (normals viewer) + STL (download).

    Returns (print_glb_path, stl_path), or (None, None) if the input mesh is empty/degenerate.
    """
    mesh = prepare_printable_mesh(glb_path)
    if mesh is None or len(mesh.faces) == 0:
        return None, None
    os.makedirs(out_dir, exist_ok=True)
    print_glb_path = os.path.join(out_dir, f"{basename}.glb")
    stl_path = os.path.join(out_dir, f"{basename}.stl")
    mesh.export(print_glb_path)
    mesh.export(stl_path)
    return print_glb_path, stl_path


def convert_glb_to_stl(glb_path: str, file_number: str, output_folder: str = "output") -> str:
    """
    Convert GLB file to STL format without base plate or engraving.

    This is a simple converter that doesn't require OpenSCAD. It loads the GLB
    file and exports it directly as STL, ready for 3D printing. Users can add
    base plates, supports, and other modifications in their slicer software.

    Args:
        glb_path: Path to the input GLB file
        file_number: File number for naming (e.g., "0001")
        output_folder: Directory to save the STL file (default: "output")

    Returns:
        str: Path to the generated STL file

    Raises:
        FileNotFoundError: If GLB file doesn't exist
        Exception: If mesh loading or export fails
    """
    # Validate input
    if not os.path.exists(glb_path):
        raise FileNotFoundError(f"GLB file not found: {glb_path}")

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    try:
        # Load GLB file and return print-ready mesh
        print(f"📦 Loading GLB from: {glb_path}")
        mesh = prepare_printable_mesh(glb_path)
        if mesh is None:
            raise Exception("empty mesh — nothing to export")

        # Define output path
        output_filename = f"{file_number}.stl"
        output_path = os.path.join(output_folder, output_filename)

        # Export as STL
        print(f"💾 Exporting STL to: {output_path}")
        mesh.export(output_path)

        print(f"✅ STL conversion complete: {output_filename}")
        return output_path

    except Exception as e:
        error_msg = f"❌ Error converting GLB to STL: {e}"
        print(error_msg)
        raise Exception(error_msg) from e


def batch_convert(glb_files: list[str], output_folder: str = "output") -> list[str]:
    """
    Convert multiple GLB files to STL format.

    Args:
        glb_files: List of GLB file paths
        output_folder: Directory to save STL files

    Returns:
        list[str]: List of generated STL file paths
    """
    stl_paths = []

    for i, glb_path in enumerate(glb_files):
        file_number = f"{i+1:04d}"
        try:
            stl_path = convert_glb_to_stl(glb_path, file_number, output_folder)
            stl_paths.append(stl_path)
        except Exception as e:
            print(f"⚠️ Skipping {glb_path}: {e}")
            continue

    return stl_paths


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 3:
        print("Usage: python simple_stl_converter.py <glb_path> <file_number> [output_folder]")
        sys.exit(1)

    glb_path = sys.argv[1]
    file_number = sys.argv[2]
    output_folder = sys.argv[3] if len(sys.argv) > 3 else "output"

    try:
        stl_path = convert_glb_to_stl(glb_path, file_number, output_folder)
        print(f"\n🎉 Success! STL saved to: {stl_path}")
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        sys.exit(1)
