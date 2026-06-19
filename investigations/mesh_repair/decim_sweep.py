"""Decimation sweep: print-ready remesh decimated to target face counts, with quality.

For each shape x target_faces: voxel288_smooth -> decimate_pro(preserve_topology)
-> pymeshfix.repair, then measure faces, watertight, non-manifold edges, components,
f1@0.01 and DINOv2 (vs the ORIGINAL baseline mesh). Resumable per row.

Usage: python decim_sweep.py --shapes 56,57,58,59,60,1,20,40 --targets 50000,25000,12000,6000
"""
import os, sys, csv, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, trimesh
import lib, normal_dinov2 as nd
import warnings; warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def topo(m):
    e = np.sort(m.edges, axis=1); _, c = np.unique(e, axis=0, return_counts=True)
    return dict(faces=len(m.faces), boundary=int((c == 1).sum()), nonman=int((c > 2).sum()),
                watertight=bool(m.is_watertight), comps=len(m.split(only_watertight=False)))


def done_set(path):
    if not os.path.exists(path):
        return set()
    return {(r["glb"], r["target"]) for r in csv.DictReader(open(path))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="56,57,58,59,60,1,20,40")
    ap.add_argument("--targets", default="50000,25000,12000,6000")
    ap.add_argument("--out", default=os.path.join(HERE, "decim_results.csv"))
    a = ap.parse_args()
    nums = [int(x) for x in a.shapes.split(",")]
    targets = [int(x) for x in a.targets.split(",")]
    done = done_set(a.out)
    new = not os.path.exists(a.out)
    f = open(a.out, "a", newline=""); w = csv.writer(f)
    if new:
        w.writerow(["glb", "target", "faces", "watertight", "nonman", "comps", "f1@0.01", "dinov2"]); f.flush()

    base_feat = {}
    for n in nums:
        glb = os.path.join(ROOT, "output", f"model_{n:04d}.glb")
        name = os.path.basename(glb)
        if not os.path.exists(glb) or all((name, str(t)) in done for t in targets):
            continue
        base = trimesh.load(glb, force="mesh")
        if len(base.faces) == 0:
            continue
        if name not in base_feat:
            base_feat[name] = nd._featurize(nd.render_normal_views(base))
        for t in targets:
            if (name, str(t)) in done:
                continue
            try:
                d = lib.cand_voxel_dec(base, target_faces=t)
                tp = topo(d)
                f1 = lib.f_score(d, base, 0.01)[2]
                import torch
                fr = nd._featurize(nd.render_normal_views(d))
                dino = 1.0 - torch.nn.functional.cosine_similarity(fr.unsqueeze(0), base_feat[name].unsqueeze(0)).item()
                row = [name, t, tp["faces"], tp["watertight"], tp["nonman"], tp["comps"], round(f1, 3), round(dino, 4)]
                w.writerow(row); f.flush()
                print(f"  {name} tgt={t:6d}: faces={tp['faces']:6d} wt={tp['watertight']} nonman={tp['nonman']} f1={f1:.3f} dinov2={dino:.4f}")
            except Exception as e:
                w.writerow([name, t, "", "", "", "", "", f"err:{type(e).__name__}"]); f.flush()
                print(f"  {name} tgt={t}: ERROR {type(e).__name__}: {str(e)[:80]}")
    f.close()
    # summary by target
    rows = [r for r in csv.DictReader(open(a.out)) if r["dinov2"] and not r["dinov2"].startswith("err")]
    import statistics as st
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        by[int(r["target"])].append(r)
    print("\n=== summary by target (medians) ===")
    print(f"{'target':>8} {'n':>3} {'faces':>7} {'wt_rate':>7} {'nonman':>6} {'f1@.01':>7} {'dinov2':>7}")
    for t in sorted(by, reverse=True):
        g = by[t]
        wt = sum(1 for r in g if r["watertight"] == "True")
        print(f"{t:>8} {len(g):>3} {int(st.median([float(r['faces']) for r in g])):>7} "
              f"{wt}/{len(g):<4} {st.median([float(r['nonman']) for r in g]):>6.0f} "
              f"{st.median([float(r['f1@0.01']) for r in g]):>7.3f} {st.median([float(r['dinov2']) for r in g]):>7.4f}")


if __name__ == "__main__":
    main()
