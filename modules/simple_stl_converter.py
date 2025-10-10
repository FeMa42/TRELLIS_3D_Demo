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
        # Load GLB file
        print(f"📦 Loading GLB from: {glb_path}")
        mesh = trimesh.load(glb_path, force='mesh')

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
