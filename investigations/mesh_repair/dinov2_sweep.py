"""Full-corpus DINOv2 perceptual-distance sweep: repaired mesh vs original baseline.

Loads DINOv2 once; for each shape computes dinov2_distance(candidate, baseline).
Resumable: appends per row, skips (shape,candidate) already in the out CSV, so a
segfault in a voxel remesh only costs the current shape on re-run.

Usage: python dinov2_sweep.py --candidates voxel_224,voxel_288_smooth --out dinov2_results.csv
"""
import os, sys, csv, glob, argparse, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, trimesh
import lib
import normal_dinov2 as nd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def done_set(path):
    if not os.path.exists(path):
        return set()
    return {(r["glb"], r["candidate"]) for r in csv.DictReader(open(path))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="voxel_224,voxel_288_smooth")
    ap.add_argument("--out", default=os.path.join(HERE, "dinov2_results.csv"))
    a = ap.parse_args()
    cands = [c.strip() for c in a.candidates.split(",") if c.strip()]
    glbs = sorted(glob.glob(os.path.join(ROOT, "output", "model_*.glb")))
    done = done_set(a.out)
    new_file = not os.path.exists(a.out)
    f = open(a.out, "a", newline="")
    w = csv.writer(f)
    if new_file:
        w.writerow(["glb", "candidate", "dinov2", "status"]); f.flush()

    base_feats = {}  # cache baseline DINOv2 features per shape
    for glb in glbs:
        name = os.path.basename(glb)
        if all((name, c) in done for c in cands):
            continue
        try:
            base = trimesh.load(glb, force="mesh")
            if len(base.faces) == 0:
                raise RuntimeError("empty baseline")
            if name not in base_feats:
                base_feats[name] = nd._featurize(nd.render_normal_views(base))
            fb = base_feats[name]
        except Exception as e:
            for c in cands:
                if (name, c) not in done:
                    w.writerow([name, c, "", f"baseline_err:{type(e).__name__}"]); f.flush()
            continue
        for c in cands:
            if (name, c) in done:
                continue
            try:
                rep = lib.CANDIDATES[c](base)
                fr = nd._featurize(nd.render_normal_views(rep))
                import torch
                cos = torch.nn.functional.cosine_similarity(fr.unsqueeze(0), fb.unsqueeze(0)).item()
                d = round(1.0 - cos, 5)
                w.writerow([name, c, d, "ok"]); f.flush()
                print(f"  {name} {c:18s} dinov2={d}")
            except Exception as e:
                w.writerow([name, c, "", f"err:{type(e).__name__}:{str(e)[:60]}"]); f.flush()
                print(f"  {name} {c:18s} ERROR {type(e).__name__}")
    f.close()

    # Context: between-shape baseline distances (what a "different object" looks like).
    print("\n--- between-shape baseline distances (context) ---")
    feats = list(base_feats.items())
    import torch
    ds = []
    for i in range(0, min(len(feats), 12) - 1):
        (_, fi), (_, fj) = feats[i], feats[i + 1]
        ds.append(1.0 - torch.nn.functional.cosine_similarity(fi.unsqueeze(0), fj.unsqueeze(0)).item())
    if ds:
        print(f"  between-shape dinov2: median={np.median(ds):.4f} range=[{min(ds):.4f},{max(ds):.4f}] (n={len(ds)} adjacent pairs)")


if __name__ == "__main__":
    main()
