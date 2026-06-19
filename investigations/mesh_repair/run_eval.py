"""Orchestrator: run every (shape, candidate) in an isolated subprocess; collect a CSV.

Usage:
  python run_eval.py --shapes 56,57,58,59,60,1,10,20,30,40 --out results_stage1.csv
  python run_eval.py --all --out results_all.csv
A crashed/timed-out worker is recorded as status=crashed, never aborts the sweep.
"""
import sys, os, json, csv, argparse, subprocess, tempfile, glob, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
EVAL_ONE = os.path.join(HERE, "eval_one.py")
CANDIDATES = ["baseline", "trimesh_gentle", "largest_component", "components_keep",
              "open3d_clean", "voxel_96", "voxel_160", "voxel_224", "poisson_8", "poisson_9"]


def shapes_arg(a):
    if a.all:
        return sorted(glob.glob(os.path.join(ROOT, "output", "model_*.glb")))
    nums = [int(x) for x in a.shapes.split(",") if x.strip()]
    return [os.path.join(ROOT, "output", f"model_{n:04d}.glb") for n in nums]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="56,57,58,59,60")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--candidates", default=",".join(CANDIDATES))
    ap.add_argument("--out", default=os.path.join(HERE, "results.csv"))
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--dinov2", action="store_true")
    a = ap.parse_args()
    cands = [c.strip() for c in a.candidates.split(",") if c.strip()]
    shapes = [s for s in shapes_arg(a) if os.path.exists(s)]

    rows = []
    for s in shapes:
        for c in cands:
            with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as tf:
                tmp = tf.name
            cmd = [sys.executable, EVAL_ONE, s, c, tmp] + (["--dinov2"] if a.dinov2 else [])
            t0 = time.time()
            try:
                p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=a.timeout)
                if os.path.getsize(tmp) > 0:
                    rec = json.load(open(tmp))
                else:
                    rec = {"glb": os.path.basename(s), "candidate": c, "status": "crashed",
                           "error": f"rc={p.returncode} {p.stderr[-200:]}"}
            except subprocess.TimeoutExpired:
                rec = {"glb": os.path.basename(s), "candidate": c, "status": "timeout"}
            except Exception as e:
                rec = {"glb": os.path.basename(s), "candidate": c, "status": "crashed", "error": str(e)[:200]}
            finally:
                try: os.unlink(tmp)
                except OSError: pass
            rec["wall_s"] = round(time.time() - t0, 2)
            rows.append(rec)
            print(f"  {os.path.basename(s):16s} {c:18s} {rec.get('status'):8s} "
                  f"wt={rec.get('topo_watertight')} comps={rec.get('topo_components')} "
                  f"nonman={rec.get('topo_nonmanifold_edges')} f1@.01={rec.get('f1@0.01')}")

    keys = []
    for r in rows:
        for k in r:
            if k not in keys and k not in ("trace",):
                keys.append(k)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
