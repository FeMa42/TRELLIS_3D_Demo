"""DINOv2 perceptual distance between two meshes' normal-map renders.

Ported from the printability-eval worktree (evaluation/dpo/metrics/normal_render_distance.py):
8 nvdiffrast normal-map views (flat per-face normals, 224px), DINOv2 ViT-S/14 features
mean-pooled across views, distance = 1 - cosine_similarity. Lower = more perceptually similar.
Worktree within-shape sampling σ-band ≈ 0.027.
"""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, torch, trimesh
from PIL import Image

_CTX = None
_MODEL = None


def _ctx():
    global _CTX
    import nvdiffrast.torch as dr
    if _CTX is None:
        _CTX = dr.RasterizeCudaContext(device="cuda")
    return _CTX


def _model():
    global _MODEL
    if _MODEL is None:
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
        _MODEL = m.eval().to("cuda")
    return _MODEL


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


def render_normal_views(mesh, num=8, size=224):
    import nvdiffrast.torch as dr
    bg = np.full((size, size, 3), 128, np.uint8)
    if mesh is None or len(mesh.faces) == 0:
        return [bg.copy() for _ in range(num)]
    ctx = _ctx(); dev = "cuda"
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


def _featurize(views):
    import torchvision.transforms as T
    tx = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    batch = torch.stack([tx(Image.fromarray(v)) for v in views]).to("cuda")
    with torch.no_grad():
        out = _model()(batch)
    feats = out if isinstance(out, torch.Tensor) else out.last_hidden_state[:, 0]
    return feats.float().mean(0)


def dinov2_distance(pred, gt):
    fa = _featurize(render_normal_views(pred))
    fb = _featurize(render_normal_views(gt))
    cos = torch.nn.functional.cosine_similarity(fa.unsqueeze(0), fb.unsqueeze(0)).item()
    return 1.0 - cos
