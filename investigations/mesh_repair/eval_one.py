"""Worker: evaluate ONE (glb, candidate) pair in an isolated process (crash-safe).

Usage: python eval_one.py <glb_path> <candidate> <out_json> [--dinov2]
Prints/writes a JSON dict of {candidate, status, timing, topology..., detail...}.
Run isolated so a segfault in a repair op (heavy voxelization / open3d) only kills
this worker, not the whole sweep.
"""
import sys, os, json, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trimesh
import lib


def main():
    glb, candidate, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
    want_dinov2 = "--dinov2" in sys.argv[4:]
    rec = {"glb": os.path.basename(glb), "candidate": candidate, "status": "ok"}
    try:
        base = trimesh.load(glb, force="mesh")
        if len(base.faces) == 0:
            raise RuntimeError("empty baseline mesh")
        t0 = time.time()
        out = lib.CANDIDATES[candidate](base)
        rec["repair_s"] = round(time.time() - t0, 3)
        # topology of the repaired mesh
        rec.update({f"topo_{k}": v for k, v in lib.topology(out).items()})
        # detail vs the original baseline mesh (the quality reference)
        t1 = time.time()
        rec.update(lib.detail_metrics(out, base))
        rec["detail_s"] = round(time.time() - t1, 3)
        if want_dinov2:
            try:
                from normal_dinov2 import dinov2_distance
                rec["dinov2"] = dinov2_distance(out, base)
            except Exception as e:
                rec["dinov2"] = float("nan"); rec["_dino_err"] = str(e)[:120]
    except Exception as e:
        rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["trace"] = traceback.format_exc()[-400:]
    with open(out_json, "w") as f:
        json.dump(rec, f)
    print(json.dumps({k: rec[k] for k in rec if k not in ("trace",)}))


if __name__ == "__main__":
    main()
