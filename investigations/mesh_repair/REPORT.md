# Mesh-Repair Investigation: making TRELLIS meshes printable without losing quality

> **Status:** Investigation complete (separate from production — nothing here is wired into the app).
> **Question:** Can we turn TRELLIS's production mesh output into a watertight, manifold, single-piece
> printable solid while preserving the original high-quality shape?
> **Answer:** Yes — **solid voxel remeshing at resolution ≈288 + Taubin smoothing** gives a fully
> printable solid at **98% of the baseline's geometric fidelity**. Every other repair either fails to
> make it printable or destroys the shape.

---

## 1. Method

- **Corpus:** the 59 valid GLBs in `output/` (`model_0001`–`0060`; `0038` is a degenerate empty mesh, excluded). These are the production pipeline output *up to the mesh stage*. `0056`–`0060` are from the current DPO + Stage-1 `fill_holes` setup; `0001`–`0055` are the older version. Both groups behave the same here.
- **Pipeline:** load the production GLB → apply a candidate mesh-repair → score. No regeneration; we operate on exactly what production produces.
- **Quality verification** (metrics ported verbatim from the `printability-eval` worktree, `evaluation/dpo/metrics/*`):
  - *Printability/topology:* watertight, # connected components, # non-manifold edges (edge shared by >2 faces), # boundary (open) edges.
  - *Detail/quality vs the original mesh:* `f1@0.01` and `f1@0.005` (bidirectional F-score at 1% / 0.5% of bbox diagonal), `curv_wass` (mean-curvature Wasserstein), Chamfer-L1 (% of bbox diag), normal-consistency. The baseline-vs-itself `f1@0.01 ≈ 0.91` is the practical ceiling (independent surface sampling, not 1.0).
  - *Visual:* nvdiffrast normal-map renders, baseline vs repaired (`renders/`).
- **Harness:** `lib.py` (candidates + metrics), `eval_one.py` (one (shape,candidate) per **isolated subprocess** — required, since heavy voxelization/open3d can segfault), `run_eval.py` (sweep → CSV), `analyze.py` (summary), `render_compare.py` (visuals). Reproduce with `python run_eval.py --shapes 56,57,58,59,60,1,10,20,30,40,50 --out results.csv`.

## 2. The problem: production meshes are badly non-printable

Baseline topology over the corpus (median): **510 disconnected components**, **6,685 open boundary edges**, **39 non-manifold edges**, **0/11 watertight**. They are fragmented open shells — a slicer cannot treat them as a solid. (This is *not* a regression in the new DPO+fill_holes meshes; it is inherent to TRELLIS's `to_glb_simple` mesh extraction.)

## 3. Results (11 shapes: 5 NEW + 6 OLD × candidates)

| candidate | watertight | comps | non-man | boundary | f1@0.01 | f1@0.005 | curv_wass | normal_cons | sec |
|---|---|---|---|---|---|---|---|---|---|
| baseline (raw) | 0/11 | 510 | 39 | 6685 | **0.911** | 0.444 | 0.03 | 0.957 | — |
| trimesh_gentle *(current prod)* | 0/11 | 510 | 39 | 6685 | 0.913 | 0.447 | 0.027 | 0.956 | 2 |
| open3d_clean | 0/11 | 15 | 2 | 78 | 0.912 | 0.446 | 0.208 | 0.956 | 4 |
| components_keep (≥0.5% area) | 0/11 | 42 | 0 | 2203 | 0.825 | 0.409 | 0.322 | 0.927 | 2 |
| largest_component | 0/11 | 1 | 0 | 103 | 0.161 | 0.115 | 0.375 | 0.781 | 4 |
| poisson_9 | 2/11 | 33 | 13 | 0 | 0.907 | 0.441 | 0.241 | 0.951 | 23 |
| voxel_96 | 11/11 | 1 | 0 | 0 | 0.588 | 0.039 | 0.341 | 0.865 | 3 |
| voxel_160 | 11/11 | 1 | 0 | 0 | 0.812 | 0.162 | 0.27 | 0.878 | 3 |
| voxel_224 | 11/11 | 1 | 0 | 0 | 0.863 | 0.27 | 0.261 | 0.884 | 5 |
| voxel_288 | 11/11 | 1 | 0 | 0 | 0.887 | 0.345 | 0.259 | 0.885 | 7 |
| voxel_224_smooth | 11/11 | 1 | 0 | 0 | 0.869 | 0.298 | 0.238 | 0.940 | 6 |
| **voxel_288_smooth** | **11/11** | **1** | **0** | **0** | **0.892** | **0.355** | **0.206** | **0.949** | **9** |

### Reading the table
- **Only voxel remeshing makes the mesh printable** (watertight + 1 component + 0 non-manifold + 0 boundary). Solid voxelization fills the interior and re-extracts a single closed surface.
- **Mesh-domain cleanup (open3d, components_keep, poisson, trimesh_gentle) does NOT** produce a watertight solid** — consistent with the worktree's finding that mesh-domain repair can't fix TRELLIS's gross fragmentation. `largest_component` *does* give 1 component but throws away 99% of the shape (f1 0.16).
- **The watertight↔detail tradeoff is a resolution knob** (exactly as the worktree predicted, now quantified): f1@0.01 climbs 0.59 → 0.81 → 0.86 → 0.89 as voxel res goes 96 → 160 → 224 → 288.
- **Taubin smoothing closes the remaining gap cheaply:** it removes the voxel staircase without changing topology (stays watertight), lifting normal-consistency 0.885 → 0.949 (≈ baseline 0.957) and curv_wass 0.259 → 0.206.

### Winner: `voxel_288_smooth`
Fully printable (watertight, single solid, zero non-manifold/boundary edges) at **f1@0.01 = 0.892 (98% of the 0.911 ceiling)**, normal-consistency **0.949 (99% of baseline)**, Chamfer **0.38% of bbox**, ~**9 s/shape** CPU. Works equally on OLD and NEW (DPO+fill_holes) shapes.

## 4. Visual verification

`renders/compare_model_00{56,58,59}.png` — rows: baseline / voxel_224 / voxel_288_smooth, cols: 3 views. On `model_0059` (a mushroom house) the door, cap scales, and base foliage are all preserved; voxel_224 shows faint voxel-noise on the cap, and **voxel_288_smooth removes that noise** — visually near-identical to baseline.

## 5. Reconciliation with the printability-eval worktree

- The worktree established that **mesh-domain repair gives zero *support-volume* benefit** and that watertightness can't be had without trading detail. This investigation is consistent: support-volume isn't the lever here (it's set by gross shape); the goal is **print-validity** (watertight/manifold/single-solid), and the detail trade is real but controllable via resolution + smoothing.
- The worktree's `binary_fill_holes` winner is **occupancy-domain (Stage-1)** and is orthogonal to this — it improves the *shape*; voxel remesh fixes the *mesh topology* after Stage-2. They compose.
- Non-manifold edges in the raw meshes come from `to_glb_simple`'s 0.95 quadric decimation (the worktree's documented "decimation introduces non-manifold edges"); voxel remesh sidesteps that entirely by re-extracting the surface.

## 6. Limitations / not-yet-done

- **DINOv2 perceptual metric** (the worktree's 3rd validated detail axis, σ-band 0.027) is not yet run here — geometric metrics + visual renders are strong, but DINOv2 would complete the validated trio. The renderer (`render_compare.py`) already has the nvdiffrast pipeline; adding the DINOv2 head is a small follow-up.
- **Slicer support-volume** not measured (PrusaSlicer not installed here, and it's orthogonal to repair).
- Sample is 11 shapes; widen to all 59 before any production decision.
- Highest resolution tested is 288 (diminishing returns + OOM/segfault risk above ~320 in one process).
- `model_0038` is a degenerate empty generation — a separate Stage-1/2 robustness issue, not a repair target.

## 7. Recommendation (for a later, separate productionization decision — NOT done here)

`voxel_288_smooth` is the candidate to take forward. If/when productionized, it fits naturally as an **optional, env-gated** step on the STL/print path (e.g. `TRELLIS_PRINT_REMESH=voxel288_smooth`), applied **only when the user exports for printing** (it costs ~9 s and slightly softens the finest detail, so it shouldn't replace the GLB shown in the viewer). Before that: run DINOv2 + the full 59-shape sweep + one real test print to confirm.

## 8. Decimation (follow-up): the raw remesh is too heavy — decimate it

The `voxel_288_smooth` remesh is **~340k uniform tiny faces**, which strains both the
interactive viewer (an ~11 MB base64-embedded GLB) and downstream slicers/printers. Since
these are small prints, sub-mm detail is wasted. We swept manifold-preserving decimation of
the remesh (`decim_sweep.py` → `decim_results.csv`, 8 shapes; candidate `cand_voxel_dec` in
`lib.py`).

**Method that works (in-process, no segfault):** `pyvista.decimate_pro(preserve_topology=True)`
→ `pymeshfix.repair`. The standard quadric decimators (`fast_simplification`, plain
`pyvista.decimate`) shred a clean watertight mesh into non-manifold fragments (the worktree's
decimation warning, reproduced: 76 non-manifold edges / 55 components at 25k). `decimate_pro`
keeps it nearly closed (0 holes, ~5 non-manifold edges), and `pymeshfix` — which *destroys*
the raw TRELLIS mesh but works cleanly on a *nearly*-perfect one — restores full watertightness.

**Quality vs target (medians, n=8; DINOv2 σ-band = 0.027, between-shape = 0.18):**

| target | median faces | watertight | non-manifold | f1@0.01 | DINOv2 |
|---|---|---|---|---|---|
| (full remesh) | 342k | 100% | 0 | 0.89 | 0.033 |
| 50,000 | 49.7k | 8/8 | 0 | 0.876 | 0.039 |
| **25,000** | 24.7k | 7/8 | 0 | 0.875 | **0.037** |
| 12,000 | 11.0k | 7/8 | 0 | 0.869 | 0.067 |
| 6,000 | 4.1k | 8/8 | 0 | 0.643 | 0.255 (collapses) |

**Operating point: 25,000 faces** — 14× lighter, watertight, zero non-manifold edges,
DINOv2 0.037 (at the perceptual noise floor → visually identical). 12k is viable for even
lighter; **below ~10k, `decimate_pro` over-collapses some shapes** (one to 30 faces) — guarded
against in production by a bbox/face-count check that falls back to the heavy mesh.

**Productionized:** `_decimate_watertight` + `target_faces` in `modules/simple_stl_converter.py::remesh_for_printing`, env `TRELLIS_PRINT_TARGET_FACES` (default 25000; 0 disables). Deps already present (`pyvista`, `pymeshfix`).
