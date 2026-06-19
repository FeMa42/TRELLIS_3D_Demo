"""Mesh-repair investigation library: repair candidates + quality/printability metrics.

SEPARATE INVESTIGATION — not wired into production. Operates on the GLB meshes in
`output/` (the production pipeline output up to the mesh stage) and evaluates how well
different mesh-repair strategies make them printable (watertight / manifold / single
component) while preserving the original high-quality shape.

Quality/detail metrics are ported verbatim from the printability-eval worktree
(evaluation/dpo/metrics/*): f-score @ tau, mean-curvature Wasserstein, Chamfer-L1 +
normal consistency. Topology metrics follow evaluation/printability_metrics.py.
"""
from __future__ import annotations
import numpy as np
import trimesh
import warnings
warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------------------
# Topology / printability metrics (numpy + trimesh; matches printability_metrics.py defs)
# --------------------------------------------------------------------------------------
def topology(m: trimesh.Trimesh) -> dict:
    if m is None or len(m.faces) == 0:
        return dict(faces=0, verts=0, boundary_edges=-1, nonmanifold_edges=-1,
                    watertight=False, winding=False, components=-1, euler=0,
                    nonmanifold_rate=float("nan"))
    F = np.asarray(m.faces)
    edges = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0), axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    n_boundary = int((counts == 1).sum())
    n_nonman = int((counts > 2).sum())
    n_unique = int(counts.shape[0])
    try:
        comps = len(m.split(only_watertight=False))
    except Exception:
        comps = -1
    return dict(
        faces=int(len(m.faces)), verts=int(len(m.vertices)),
        boundary_edges=n_boundary, nonmanifold_edges=n_nonman,
        watertight=bool(n_boundary == 0 and m.is_watertight),
        winding=bool(m.is_winding_consistent),
        components=comps,
        euler=int(len(m.vertices) - n_unique + len(m.faces)),
        nonmanifold_rate=float(n_nonman) / max(1, n_unique),
    )


# --------------------------------------------------------------------------------------
# Detail / quality-preservation metrics (ported from evaluation/dpo/metrics/*)
# Compare a repaired/candidate mesh against the original baseline mesh (the quality ref).
# --------------------------------------------------------------------------------------
def _sample(mesh, n, rng):
    if len(mesh.faces) == 0:
        return np.zeros((1, 3))
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return pts


def f_score(pred: trimesh.Trimesh, gt: trimesh.Trimesh, tau_fraction=0.01, num_samples=10000):
    """Bidirectional F-score @ tau = tau_fraction * GT bbox diagonal. (precision, recall, f1)"""
    if len(gt.vertices) == 0 or len(pred.vertices) == 0:
        return 0.0, 0.0, 0.0
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(0)
    pp = _sample(pred, num_samples, rng)
    pg = _sample(gt, num_samples, rng)
    diag = float(np.linalg.norm(gt.bounds[1] - gt.bounds[0]))
    if diag == 0:
        return 0.0, 0.0, 0.0
    tau = tau_fraction * diag
    tp, tg = cKDTree(pp), cKDTree(pg)
    d_pg, _ = tg.query(pp, k=1)
    d_gp, _ = tp.query(pg, k=1)
    precision = float((d_pg < tau).mean())
    recall = float((d_gp < tau).mean())
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return precision, recall, f1


def curv_wasserstein(pred, gt, radius_fraction=0.01, num_samples=5000):
    """1-Wasserstein between normalized |mean curvature| sample distributions."""
    if len(gt.vertices) == 0 or len(pred.vertices) == 0:
        return float("inf")
    from scipy.stats import wasserstein_distance
    dg = float(np.linalg.norm(gt.bounds[1] - gt.bounds[0]))
    dp = float(np.linalg.norm(pred.bounds[1] - pred.bounds[0]))
    if dg == 0 or dp == 0:
        return float("inf")
    rg, rp = radius_fraction * dg, radius_fraction * dp
    pp, _ = trimesh.sample.sample_surface(pred, num_samples)
    pg, _ = trimesh.sample.sample_surface(gt, num_samples)
    hp = np.abs(trimesh.curvature.discrete_mean_curvature_measure(pred, pp, rp)) / rp
    hg = np.abs(trimesh.curvature.discrete_mean_curvature_measure(gt, pg, rg)) / rg
    return float(wasserstein_distance(hp, hg))


def chamfer_normals(pred, gt, n_samples=50000):
    """Symmetric Chamfer-L1 and flip-invariant normal consistency."""
    if len(pred.faces) == 0 or len(gt.faces) == 0:
        return float("inf"), 0.0
    from scipy.spatial import cKDTree
    pp, pf = trimesh.sample.sample_surface(pred, n_samples)
    gp, gf = trimesh.sample.sample_surface(gt, n_samples)
    pn, gn = pred.face_normals[pf], gt.face_normals[gf]
    tg, tp = cKDTree(gp), cKDTree(pp)
    d_p, i_p = tg.query(pp)
    d_g, i_g = tp.query(gp)
    chamfer = float(0.5 * (d_p.mean() + d_g.mean()))
    nc = float(0.5 * (np.abs(np.einsum("ij,ij->i", pn, gn[i_p])).mean()
                      + np.abs(np.einsum("ij,ij->i", gn, pn[i_g])).mean()))
    return chamfer, nc


def detail_metrics(pred, gt) -> dict:
    out = {}
    try:
        p, r, f1 = f_score(pred, gt, 0.01)
        out["f1@0.01"], out["p@0.01"], out["r@0.01"] = f1, p, r
    except Exception as e:
        out["f1@0.01"] = float("nan"); out["_f1_err"] = str(e)[:80]
    try:
        _, _, f1b = f_score(pred, gt, 0.005)
        out["f1@0.005"] = f1b
    except Exception:
        out["f1@0.005"] = float("nan")
    try:
        out["curv_wass"] = curv_wasserstein(pred, gt)
    except Exception as e:
        out["curv_wass"] = float("nan"); out["_cw_err"] = str(e)[:80]
    try:
        ch, nc = chamfer_normals(pred, gt)
        # express chamfer as % of bbox diagonal for scale-invariance
        diag = float(np.linalg.norm(gt.bounds[1] - gt.bounds[0]))
        out["chamfer_pct"] = 100.0 * ch / diag if diag else float("nan")
        out["normal_consistency"] = nc
    except Exception as e:
        out["chamfer_pct"] = float("nan"); out["_ch_err"] = str(e)[:80]
    return out


# --------------------------------------------------------------------------------------
# Repair candidates. Each takes a trimesh and returns a repaired trimesh.
# --------------------------------------------------------------------------------------
def cand_baseline(m):
    return m


def cand_trimesh_gentle(m):
    """Current production ensure_printable_mesh: merge + fix_normals + fill_holes, with guard."""
    if m.is_watertight and m.is_winding_consistent:
        return m
    r = m.copy()
    r.merge_vertices()
    trimesh.repair.fix_normals(r)
    trimesh.repair.fill_holes(r)
    if len(r.faces) < 0.9 * len(m.faces) or not np.allclose(m.extents, r.extents, rtol=0.02, atol=1e-6):
        return m
    return r


def cand_largest_component(m):
    comps = m.split(only_watertight=False)
    if len(comps) <= 1:
        return m
    return max(comps, key=lambda c: len(c.faces))


def cand_components_keep(m, area_frac=0.005):
    """Keep connected components whose area >= area_frac of total (drop tiny floaters)."""
    comps = m.split(only_watertight=False)
    if len(comps) <= 1:
        return m
    total = sum(float(c.area) for c in comps)
    keep = [c for c in comps if float(c.area) >= area_frac * total]
    if not keep:
        return m
    return trimesh.util.concatenate(keep)


def cand_open3d_clean(m):
    """open3d topology hygiene: remove duplicated/degenerate/non-manifold, then close small holes."""
    import open3d as o3d
    om = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(m.vertices, np.float64)),
        o3d.utility.Vector3iVector(np.asarray(m.faces, np.int32)))
    om.remove_duplicated_vertices()
    om.remove_duplicated_triangles()
    om.remove_degenerate_triangles()
    om.remove_non_manifold_edges()
    om.remove_unreferenced_vertices()
    r = trimesh.Trimesh(np.asarray(om.vertices), np.asarray(om.triangles), process=True)
    trimesh.repair.fill_holes(r)
    return r


def cand_voxel(m, res=128):
    """Solid-voxelize at pitch = bbox_diag/res, marching-cubes back -> watertight manifold."""
    diag = float(np.linalg.norm(m.extents))
    pitch = diag / res
    vg = m.voxelized(pitch=pitch).fill()
    r = vg.marching_cubes
    r.apply_transform(vg.transform)   # marching_cubes is in voxel-index space; map back to world
    r.merge_vertices()
    trimesh.repair.fix_normals(r)
    return r


def cand_poisson(m, depth=9, n_points=60000):
    """open3d Poisson surface reconstruction, cropped to the original bbox."""
    import open3d as o3d
    om = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(m.vertices, np.float64)),
        o3d.utility.Vector3iVector(np.asarray(m.faces, np.int32)))
    om.compute_vertex_normals()
    pcd = om.sample_points_poisson_disk(n_points)
    pm, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    pm = pm.crop(om.get_axis_aligned_bounding_box())
    r = trimesh.Trimesh(np.asarray(pm.vertices), np.asarray(pm.triangles), process=True)
    r.merge_vertices()
    return r


def cand_voxel_smooth(m, res=224, iterations=10):
    """Voxel remesh then volume-preserving Taubin smoothing (reduces staircase; topology unchanged -> stays watertight)."""
    r = cand_voxel(m, res)
    trimesh.smoothing.filter_taubin(r, lamb=0.5, nu=-0.53, iterations=iterations)
    return r


CANDIDATES = {
    "baseline": cand_baseline,
    "trimesh_gentle": cand_trimesh_gentle,
    "largest_component": cand_largest_component,
    "components_keep": cand_components_keep,
    "open3d_clean": cand_open3d_clean,
    "voxel_96": lambda m: cand_voxel(m, 96),
    "voxel_160": lambda m: cand_voxel(m, 160),
    "voxel_224": lambda m: cand_voxel(m, 224),
    "poisson_8": lambda m: cand_poisson(m, 8),
    "poisson_9": lambda m: cand_poisson(m, 9),
    "voxel_288": lambda m: cand_voxel(m, 288),
    "voxel_224_smooth": lambda m: cand_voxel_smooth(m, 224, 10),
    "voxel_288_smooth": lambda m: cand_voxel_smooth(m, 288, 8),
}
