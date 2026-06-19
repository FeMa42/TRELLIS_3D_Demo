"""Render side-by-side normal-map comparisons (baseline vs repaired) for visual QA.

Uses nvdiffrast (CUDA) — the same renderer the printability-eval worktree used for its
DINOv2 metric. Produces one PNG per shape: rows = candidates, cols = views.

Usage: python render_compare.py 56,57,58,59,60
Outputs: investigations/mesh_repair/renders/compare_model_XXXX.png
"""
import os, sys
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, trimesh, torch
from PIL import Image
import lib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CANDS = ["baseline", "voxel_224", "voxel_288_smooth"]
VIEWS = 3
SIZE = 256
_ctx = None


def _ctx_get():
    global _ctx
    import nvdiffrast.torch as dr
    if _ctx is None:
        _ctx = dr.RasterizeCudaContext(device="cuda")
    return _ctx


def _look_at(eye, tgt, up):
    f = tgt - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def _persp(fovy, aspect, n, f):
    t = 1.0 / np.tan(fovy / 2)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = t / aspect; m[1, 1] = t
    m[2, 2] = (f + n) / (n - f); m[2, 3] = 2 * f * n / (n - f); m[3, 2] = -1
    return m


def render_views(mesh, num=VIEWS, size=SIZE):
    import nvdiffrast.torch as dr
    bg = np.full((size, size, 3), 128, np.uint8)
    if mesh is None or len(mesh.faces) == 0:
        return [bg.copy() for _ in range(num)]
    ctx = _ctx_get(); dev = "cuda"
    c = mesh.bounds.mean(0); ext = float((mesh.bounds[1] - mesh.bounds[0]).max())
    if ext <= 0:
        return [bg.copy() for _ in range(num)]
    v = (np.asarray(mesh.vertices, np.float32) - c) / ext
    f = np.asarray(mesh.faces, np.int32)
    fn = np.nan_to_num(np.asarray(mesh.face_normals, np.float32))
    sv = v[f.reshape(-1)].astype(np.float32)
    sf = np.arange(len(sv), dtype=np.int32).reshape(-1, 3)
    sn = np.repeat(fn, 3, 0).astype(np.float32)
    vt = torch.from_numpy(sv).to(dev); ft = torch.from_numpy(sf).to(dev); nt = torch.from_numpy(sn).to(dev)
    proj = torch.from_numpy(_persp(np.pi / 3, 1.0, 0.1, 10.0)).to(dev)
    outs = []
    for i in range(num):
        a = 2 * np.pi * i / num
        eye = np.array([2 * np.sin(a), 0.3, 2 * np.cos(a)], np.float32)
        view = torch.from_numpy(_look_at(eye, np.zeros(3, np.float32), np.array([0, 1, 0], np.float32))).to(dev)
        vh = torch.cat([vt, torch.ones_like(vt[:, :1])], 1)
        clip = ((vh @ view.T) @ proj.T).unsqueeze(0).contiguous()
        rast, _ = dr.rasterize(ctx, clip, ft, resolution=[size, size])
        ncam = nt @ view[:3, :3].T
        ncam = ncam / (ncam.norm(dim=-1, keepdim=True) + 1e-8)
        nimg, _ = dr.interpolate(ncam.unsqueeze(0).contiguous(), rast, ft)
        nimg = nimg / (nimg.norm(dim=-1, keepdim=True) + 1e-8)
        mask = (rast[..., 3:4] > 0).float()
        rgb = (nimg * 0.5 + 0.5) * mask + torch.full_like(nimg, 0.5) * (1 - mask)
        img = (rgb[0].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()[::-1]
        outs.append(np.ascontiguousarray(img))
    return outs


def main():
    nums = [int(x) for x in sys.argv[1].split(",")]
    os.makedirs(os.path.join(HERE, "renders"), exist_ok=True)
    for n in nums:
        glb = os.path.join(ROOT, "output", f"model_{n:04d}.glb")
        if not os.path.exists(glb):
            continue
        base = trimesh.load(glb, force="mesh")
        rows = []
        for cand in CANDS:
            try:
                m = lib.CANDIDATES[cand](base)
            except Exception as e:
                print(f"{cand} failed on {n}: {e}"); m = None
            row = np.concatenate(render_views(m), axis=1)  # views side by side
            rows.append(row)
        grid = np.concatenate(rows, axis=0)  # candidates stacked
        out = os.path.join(HERE, "renders", f"compare_model_{n:04d}.png")
        Image.fromarray(grid).save(out)
        print(f"wrote {out}  (rows top->bottom: {CANDS})")


if __name__ == "__main__":
    main()
