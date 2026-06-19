import os
import numpy as np
import trimesh
from modules.simple_stl_converter import ensure_printable_mesh, remesh_for_printing


def test_repairs_open_mesh_to_watertight():
    box = trimesh.creation.box(extents=(10, 10, 10))
    open_box = trimesh.Trimesh(vertices=box.vertices,
                               faces=box.faces[:-2],   # drop 2 faces -> a hole
                               process=False)
    assert not open_box.is_watertight
    repaired = ensure_printable_mesh(open_box)
    assert repaired.is_watertight
    assert repaired.is_winding_consistent


def test_watertight_mesh_passes_through_valid():
    box = trimesh.creation.box(extents=(10, 10, 10))
    out = ensure_printable_mesh(box)
    assert out.is_watertight


def test_preserves_geometry_of_multicomponent_mesh():
    # Regression: TRELLIS GLBs are non-watertight, multi-component shells.
    # An aggressive repair (e.g. pymeshfix.repair) discards most of that
    # geometry, collapsing the mesh to a small fragment -> a "flat"/broken STL.
    # ensure_printable_mesh must preserve the gross geometry instead.
    b1 = trimesh.creation.box(extents=(10, 10, 10))
    b2 = trimesh.creation.box(extents=(6, 6, 6))
    b2.apply_translation((30, 0, 0))
    b2 = trimesh.Trimesh(vertices=b2.vertices, faces=b2.faces[:-2], process=False)  # hole
    combo = trimesh.util.concatenate([b1, b2])
    assert not combo.is_watertight

    out = ensure_printable_mesh(combo)

    # Both boxes must survive: bounding box preserved, geometry not gutted.
    assert np.allclose(out.extents, combo.extents, rtol=0.05), \
        f"bbox collapsed: {out.extents} vs {combo.extents}"
    assert len(out.faces) >= 0.9 * len(combo.faces), \
        f"geometry gutted: {len(out.faces)} of {len(combo.faces)} faces"


def test_remesh_for_printing_merges_into_one_watertight_solid():
    # Two overlapping boxes = 2 components. The print remesh must fuse them into a
    # single watertight manifold solid while preserving the overall bounding box.
    b1 = trimesh.creation.box(extents=(10, 10, 10))
    b2 = trimesh.creation.box(extents=(10, 10, 10))
    b2.apply_translation((6, 0, 0))                       # overlaps b1 in x
    combo = trimesh.util.concatenate([b1, b2])
    assert len(combo.split(only_watertight=False)) == 2

    out = remesh_for_printing(combo, res=64, smooth_iters=0)  # low res / no smooth = fast & deterministic

    assert out is not None
    assert out.is_watertight
    assert len(out.split(only_watertight=False)) == 1        # fused into ONE solid
    assert np.allclose(out.extents, combo.extents, rtol=0.1), \
        f"bbox not preserved: {out.extents} vs {combo.extents}"


def test_remesh_returns_none_on_empty_mesh():
    empty = trimesh.Trimesh()
    assert remesh_for_printing(empty) is None


def test_prepare_printable_mesh_returns_mesh(tmp_path, monkeypatch):
    box = trimesh.creation.box(extents=(10, 10, 10))
    glb = str(tmp_path / "in.glb"); box.export(glb)
    from modules.simple_stl_converter import prepare_printable_mesh
    monkeypatch.setenv("TRELLIS_PRINT_REMESH", "off")
    m = prepare_printable_mesh(glb)
    assert m is not None and len(m.faces) > 0


def test_export_print_ready_writes_glb_and_stl(tmp_path, monkeypatch):
    box = trimesh.creation.box(extents=(10, 10, 10))
    glb = str(tmp_path / "in.glb"); box.export(glb)
    from modules.simple_stl_converter import export_print_ready
    monkeypatch.setenv("TRELLIS_PRINT_REMESH", "off")
    pg, stl = export_print_ready(glb, str(tmp_path))
    assert pg and stl and os.path.exists(pg) and os.path.exists(stl)
    assert len(trimesh.load(pg, force="mesh").faces) > 0
    assert len(trimesh.load(stl, force="mesh").faces) > 0


def test_export_print_ready_remesh_on_is_watertight(tmp_path, monkeypatch):
    box = trimesh.creation.box(extents=(10, 10, 10))
    open_box = trimesh.Trimesh(vertices=box.vertices, faces=box.faces[:-2], process=False)
    glb = str(tmp_path / "open.glb"); open_box.export(glb)
    from modules.simple_stl_converter import export_print_ready
    monkeypatch.setenv("TRELLIS_PRINT_REMESH", "voxel288_smooth")
    pg, stl = export_print_ready(glb, str(tmp_path), basename="pr")
    assert pg and stl
    assert trimesh.load(stl, force="mesh").is_watertight
