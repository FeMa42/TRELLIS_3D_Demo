import trimesh
from modules.three_d_viewer import create_3d_viewer_html


def _glb(tmp_path):
    p = str(tmp_path / "m.glb")
    trimesh.creation.box(extents=(1, 1, 1)).export(p)
    return p


def test_normals_true_uses_normal_material(tmp_path):
    html = create_3d_viewer_html(_glb(tmp_path), normals=True)
    assert "MeshNormalMaterial" in html


def test_normals_false_keeps_default_material(tmp_path):
    html = create_3d_viewer_html(_glb(tmp_path), normals=False)
    assert "MeshNormalMaterial" not in html
    assert "{{" not in html and "}}" not in html
