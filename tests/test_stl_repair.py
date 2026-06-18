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
