import trimesh
from modules.three_d_viewer import create_3d_viewer_html


def _glb(tmp_path):
    p = str(tmp_path / "m.glb")
    trimesh.creation.box(extents=(1, 1, 1)).export(p)
    return p


def test_normals_true_starts_in_normals_mode(tmp_path):
    html = create_3d_viewer_html(_glb(tmp_path), normals=True)
    assert "MeshNormalMaterial" in html
    # MeshNormalMaterial must opt out of ACES tone mapping, or its raw normal
    # colors get crushed to near-black by the renderer.
    assert "toneMapped = false" in html
    assert "viewMode = 'normals'" in html


def test_normals_false_starts_in_solid_mode(tmp_path):
    html = create_3d_viewer_html(_glb(tmp_path), normals=False)
    assert "viewMode = 'solid'" in html
    assert "{{" not in html and "}}" not in html


def test_viewer_has_mode_toggle_and_normal_fallback(tmp_path):
    html = create_3d_viewer_html(_glb(tmp_path))
    # mesh / wireframe / normals toggle present in every viewer
    for mode in ("solid", "wireframe", "normals"):
        assert f"setViewMode('{mode}')" in html
    # decimated print-ready GLBs ship without a NORMAL accessor; the viewer must
    # compute normals so the normals mode (and lighting) work.
    assert "computeVertexNormals" in html
