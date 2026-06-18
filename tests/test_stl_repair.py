import numpy as np
import trimesh
from modules.simple_stl_converter import ensure_printable_mesh


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
