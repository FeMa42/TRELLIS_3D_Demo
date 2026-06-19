"""Summarize a results CSV: per-candidate printability vs quality, ranked.

Usage: python analyze.py results_stage1.csv
Quality reference: the 'baseline' candidate is the original mesh; its f1@0.01 vs itself
(~0.89) is the practical ceiling. We want candidates that maximize printability
(watertight, 1 component, 0 non-manifold edges) while keeping detail near that ceiling.
"""
import sys, csv, math
from collections import defaultdict
import statistics as st

def num(x):
    try:
        v = float(x)
        return v if not math.isnan(v) else None
    except (TypeError, ValueError):
        return None

def med(xs):
    xs = [x for x in xs if x is not None]
    return round(st.median(xs), 3) if xs else None

rows = list(csv.DictReader(open(sys.argv[1])))
by = defaultdict(list)
for r in rows:
    by[r["candidate"]].append(r)

order = ["baseline","trimesh_gentle","largest_component","components_keep","open3d_clean",
         "voxel_96","voxel_160","voxel_224","poisson_8","poisson_9"]
cands = [c for c in order if c in by] + [c for c in by if c not in order]

hdr = f"{'candidate':18s} {'n':>3} {'wt_rate':>7} {'comps':>6} {'nonman':>6} {'bound':>6} {'f1@.01':>7} {'f1@.005':>7} {'curv_w':>7} {'chmf%':>6} {'norm_c':>6} {'sec':>5}"
print(hdr); print("-"*len(hdr))
for c in cands:
    g = by[c]
    ok = [r for r in g if r.get("status")=="ok"]
    n = len(ok); bad = len(g)-n
    wt = sum(1 for r in ok if str(r.get("topo_watertight")).lower()=="true")
    line = (f"{c:18s} {n:3d} {wt}/{n:<5} "
            f"{str(med([num(r.get('topo_components')) for r in ok])):>6} "
            f"{str(med([num(r.get('topo_nonmanifold_edges')) for r in ok])):>6} "
            f"{str(med([num(r.get('topo_boundary_edges')) for r in ok])):>6} "
            f"{str(med([num(r.get('f1@0.01')) for r in ok])):>7} "
            f"{str(med([num(r.get('f1@0.005')) for r in ok])):>7} "
            f"{str(med([num(r.get('curv_wass')) for r in ok])):>7} "
            f"{str(med([num(r.get('chamfer_pct')) for r in ok])):>6} "
            f"{str(med([num(r.get('normal_consistency')) for r in ok])):>6} "
            f"{str(med([num(r.get('wall_s')) for r in ok])):>5}")
    if bad: line += f"  ({bad} failed)"
    print(line)

# Pareto pick: among candidates that are watertight on >=80% shapes with 1 component
# and 0 non-manifold edges, the one with highest median f1@0.01.
print("\nPrintable candidates (wt on >=80% shapes, median comps==1, median nonman==0), by detail:")
cand_scores = []
for c in cands:
    ok = [r for r in by[c] if r.get("status")=="ok"]
    if not ok: continue
    wt_rate = sum(1 for r in ok if str(r.get("topo_watertight")).lower()=="true")/len(ok)
    mc = med([num(r.get("topo_components")) for r in ok])
    mn = med([num(r.get("topo_nonmanifold_edges")) for r in ok])
    f1 = med([num(r.get("f1@0.01")) for r in ok])
    if wt_rate>=0.8 and mc==1 and mn==0:
        cand_scores.append((c, f1, wt_rate))
for c,f1,wt in sorted(cand_scores, key=lambda x:-(x[1] or 0)):
    print(f"  {c:18s} f1@0.01={f1}  wt_rate={wt:.0%}")
if not cand_scores:
    print("  (none fully watertight+manifold+single-component at >=80%)")
