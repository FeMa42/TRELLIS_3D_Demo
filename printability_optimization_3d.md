# Printability Optimization for the TRELLIS Text-to-3D Demo

> **For agentic workers:** The implementation plan in Part 4 uses checkbox (`- [ ]`) steps. To execute it task-by-task, use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`.

**Goal:** Make 3D models produced by this demo substantially easier to 3D-print (less support material, cleaner watertight meshes) by applying the validated findings from the `printability-eval` R&D investigation to this repository's TRELLIS pipeline and Streamlit/Gradio apps.

**Architecture:** The printability win lives in **TRELLIS Stage-1 (the sparse-structure occupancy grid)**, not in mesh post-processing. The headline lever is a sub-millisecond `scipy.ndimage.binary_fill_holes` applied to the Stage-1 occupancy before Stage-2, optionally stacked with a small DPO LoRA on the Stage-1 flow model. A secondary fix hardens this repo's GLB→STL export so prints don't fail on non-watertight/non-manifold meshes.

**Tech Stack:** TRELLIS (`TrellisImageTo3DPipeline`), PyTorch, `scipy.ndimage` (already a dependency), PEFT (LoRA, optional), trimesh / pymeshfix (already dependencies), PrusaSlicer CLI (optional, for measurement only).

**Source of findings:** The completed investigation in the worktree at
`/home/damian/Projects/TRELLIS/.claude/worktrees/printability-eval`. Single entry point:
`docs/superpowers/RESULTS.md`; full narrative: `evaluation/dpo/EXPERIMENTS.md` (22 experiments + capstone);
per-experiment diagnostics: `evaluation/results/diagnostics/*.md`. See Part 6 for the full file index.

---

## Global Constraints

- **`scipy` is already pinned** (`requirements.txt:14`, `scipy==1.14.1`); `binary_fill_holes` is importable today. The Stage-1 fix needs **no new dependency**.
- **`peft` is installed (`peft 0.19.1`) but NOT pinned** in `requirements.txt`. The optional LoRA path (Phase B) must add `peft` to `requirements.txt`.
- **Stage-1 occupancy grid is 64³ boolean.** The sparse-structure decoder emits **raw logits** (no sigmoid); occupancy is `logits > 0` (i.e. `sigmoid > 0.5`). Do not add a sigmoid.
- **All slicer numbers below are profile-specific:** PrusaSlicer, **Prusa MK3S, 0.4 mm nozzle, 0.2 mm layer, 15 % gyroid infill, 45° support-overhang threshold**, mesh rescaled so the longest bbox edge = **50 mm**. Rankings can shift under other FDM/SLA profiles.
- **The metric of "printability" is slicer support-material volume (mm³, lower = better)**, not topology counts. Detail preservation is tracked separately (`f1@0.01`, `curv_wass`, `dinov2`; DINOv2 σ-band = 0.027).
- **Stage-2 normalizes away small Stage-1 edits.** Only *large* occupancy changes (fill_holes, dilation) survive to the mesh. Threshold/CFG tweaks are no-ops — do not pursue them.
- This repo runs TRELLIS in conda env **`trellis_blackwell`** with `ATTN_BACKEND=xformers`, `SPCONV_ALGO=native` (set in `streamlit-app.py:19-20` / `gradio-app.py:17-18`). Default run config has `ENABLE_TRELLIS_CPU_OFFLOAD=false`.

---

# Part 1 — Executive Summary (TL;DR)

1. **The first sparse-structure stage is the root cause.** Of the holes/unprintable geometry in final meshes, **100 % of the confidently-classified ones originate in Stage-1** (the 64³ occupancy grid); **0 % are introduced by Stage-2**. Stage-2 (FlexiCubes) can *heal* small Stage-1 gaps but not large ones — so the fix must happen at Stage-1.

2. **The single best lever is free and tiny:** apply `scipy.ndimage.binary_fill_holes` to the Stage-1 occupancy grid before Stage-2. It cuts slicer support volume by **~15 % (≈ −1300 mm³ median on cars, ≈ −964 mm³ median on general objects)** while preserving detail (`f1@0.01 ≈ 0.80`). Cost: microseconds of NumPy, **no new dependency, no training, no GPU cost.** This is the recommended Phase A and gives most of the benefit.

3. **Stacking a small DPO LoRA on the Stage-1 flow model adds more, super-additively.** The final validated operating point is **r=16 LoRA (slicer-weighted reward `w_sv=0.25`) + `binary_fill_holes`**:
   - General objects (Thingi10k, the closest analogue to demo content): **−3091 ± 189 mm³ mean, 78 % of shapes improved**.
   - Cars (MeshFleet, the training distribution): **−1332 ± 126 mm³ mean, 91 % improved**.
   - The car-trained LoRA **transfers to general objects with an even larger effect** — important because the demo generates arbitrary objects. (Phase B, optional.)

4. **"Post-processing made a lot of things worse" — confirmed, with nuance.** *Mesh-domain* cleanup (PyMeshFix hole-filling, largest-component) had **literally zero effect on support volume**. Aggressive morphology (erode variants) bought big support reductions **only by destroying fine detail** (`f1@0.01` 0.80 → 0.07). Probabilistic dilation and multi-seed logit averaging made support **worse**. The lesson: do printability work in the **occupancy domain (upstream of Stage-2)**, and treat mesh post-processing as *print-validity hygiene only* (watertight/manifold), never as a support-volume lever. This repo's current export (aggressive 0.95 decimation + visibility-mincut hole removal + bare STL with no watertight guarantee) is a print-validity hazard worth auditing (Phase C).

5. **Dead ends — do not repeat:** Stage-1 logit-threshold sweeps, CFG sweeps, multi-seed logit averaging, probabilistic dilation, fill_holes+erode, LoRA rank ≥ 64, reward re-weighting beyond `w_sv=0.25`, mesh-domain PyMeshFix for support reduction. All refuted with data (Part 5).

---

# Part 2 — Research Synthesis (the gathered findings)

## 2.1 How TRELLIS produces a mesh, and where printability dies

TRELLIS image-to-3D is two stages:

- **Stage-1 — Sparse structure.** A flow model (`sparse_structure_flow_model`) denoises a latent; a decoder (`sparse_structure_decoder`) expands it to a dense **64³ occupancy logit field**. The baseline binarizes with `decoder(z_s) > 0` (logit > 0 ⇔ prob > 0.5) and keeps the occupied voxel coordinates. *This grid defines the gross shape — and therefore the overhangs that require print supports.*
- **Stage-2 — Structured latent (SLAT) + mesh.** A second flow model fills those voxels with latent features; a FlexiCubes mesh decoder turns them into the final mesh, then GLB/STL.

**Attribution evidence (the load-bearing result).** On 10 hole-prone shapes, of the 6 that had holes in the Stage-2 mesh, **4 were confidently Stage-1-origin and 0 were Stage-2-introduced** (2 ambiguous due to render resolution). Restricting to confident cases: **100 % of Stage-2 holes had a Stage-1 occupancy gap upstream.** Stage-2 FlexiCubes closed small Stage-1 gaps in 4/10 cases (it *can* heal) but "can't heal arbitrarily large Stage-1 gaps."
*Source: `evaluation/results/diagnostics/stage1_attribution_summary.md`; EXPERIMENTS §7.*

**Why small Stage-1 tweaks don't work.** Stage-2 FlexiCubes **normalizes away small occupancy perturbations**. Empirically: binarization-threshold sweeps (−0.5…+0.5) move support < 0.2 %; CFG sweeps (1.5…7.0) move it < 4 %; multi-seed logit averaging makes it **+4.5–5 % worse** (averaging pushes occupancy toward 0.5 → Stage-2 over-decodes). Only **large** occupancy edits survive: `binary_fill_holes` and hard dilation.
*Source: `intervention_sweep_summary.md`; EXPERIMENTS §16.*

**Mechanistic root cause.** Stage-1 is a Dice-trained binary classifier that is under-confident at boundaries; at 64³ the surface is dominated by axis-aligned stair-stepping, and sub-1/256 features are collapsed by the `BLOCKS@octree_depth=8 + voxelize@1/64` preprocessing floor — independent of how clean the input mesh was.
*Source: `docs/superpowers/specs/2026-06-01-printability-research-synthesis.md`.*

## 2.2 What was tried — the full intervention catalog and verdicts

| Family | Intervention | Verdict | Numbers |
|---|---|---|---|
| **Occupancy morphology** | **`binary_fill_holes`** | ✅ **WINNER (detail-safe)** | −15 % support (median 8460 → 7160 mm³), `f1@0.01`=0.80, `dinov2`=0.013. Microsecond cost. |
| | `fill_holes`+`erode1`/`erode2` | ❌ refuted | Support −30 % / −58 % **but** `f1@0.01` collapses 0.80 → 0.28 → 0.07; verts 196k → 37k. Detail destroyed. |
| | `hard_dilate_1/2`, `morph_reconstruction` | ⚠️ support win, off detail-band | Support 6860–7600 mm³ but `f1@0.01`≈0.25 (blobby). |
| | `prob_dilate_{0.3,0.5,0.7}` | ❌ refuted | **+25–35 % WORSE** (spurious fringe protrusions). |
| | `gaussian_1.0` dilation | ❌ refuted | +4 % worse. |
| **Logit-space** | threshold sweep, CFG sweep | ❌ no-op | within ±0.2–4 % of baseline (Stage-2 smooths them). |
| | multi-seed logit averaging (`avg_n4/n8`) | ❌ refuted | **+4.5–5 % WORSE**. |
| **DPO / RLHF** | DPO LoRA r=16, slicer reward `w_sv=0.25` | ✅ stacks with fill_holes | −215 mm³ alone (multi-seed); see §2.3 for combined. |
| | LoRA rank sweep | r=16 ✅ / r=32 ✅ / **r=64 ❌ broken** | r=64 overfits (20 M params on 128 pairs) → support **worse**. |
| | reward re-weighting (`w_sv≥0.40`, composite, `w_wt`) | ❌ all regress | `w_sv=0.25` is the Pareto-optimal weight; everything else dilutes it. |
| | data scale-up 138 → 580 pairs | ❌ no help | variance doubled, signal narrowed. |
| **Inference selection** | best-of-N (slice N, keep lowest support) | ✅ works, training-free | best-of-32 ≈ **−1480 mm³ median (~23 %)**; mode-collapse refuted (SNR 0.34/0.49/0.54 @ N=4/16/32). Cost: N slices/shape. |
| **Mesh-domain cleanup** | **PyMeshFix `fill_small_boundaries`** (n=10…300) | ❌ **ZERO support effect** | byte-identical 8460 mm³; only nudges watertight frac 0.77 → 0.83–0.90. |
| | `largest_component_only` | ❌ negligible | −0.4 %. |

*Sources: `intervention_sweep_summary.md`, `fill_holes_pareto_summary.md`, `fill_holes_erode_detail_summary.md`, `interventions.py`; EXPERIMENTS §13–§21.*

## 2.3 The validated operating point (latest)

> **Note:** `RESULTS.md` (2026-06-03) headlines **r=32**+fill_holes. The later Phase-2 multi-seed work (2026-06-06, `phase2_C_r16_multiseed_summary.md`) **supersedes** that: **r=16 + fill_holes is the final recommendation** — it beats r=32 on every per-shape statistic with tighter cross-seed variance and half the parameters.

| Configuration | Dataset | Mean Δ support volume | Improved | Notes |
|---|---|---|---|---|
| **`fill_holes` only** (no training) | cars | **≈ −1300 mm³ median (−15 %)** | — | Zero deps, microseconds. **Start here.** |
| `fill_holes` only (transfer) | general (Thingi10k) | −2446 mean / −964 median | 76 % | |
| **r=16 LoRA + `fill_holes`** | **general (Thingi10k)** | **−3091 ± 189 mm³** | **78 %** | **Final recommendation.** |
| r=16 LoRA + `fill_holes` | cars (MeshFleet) | −1332 ± 126 mm³ | 91 % | |
| r=32 LoRA + `fill_holes` | cars (MeshFleet) | −1287 ± 168 mm³ | 93 % | superseded by r=16 |
| DPO LoRA alone (r=16) | cars | −215 mm³ | — | small alone; magic is stacking |

- **Super-additivity:** combined effect exceeds the sum of parts (r=16: up to 1.76× single-seed, ~1.0–1.3× multi-seed). The LoRA shifts Stage-1 occupancy toward a "more fillable" topology that `fill_holes` then closes more effectively.
- **Generalization (key for the demo):** the LoRA was trained on cars but **transfers to general objects with a *larger* absolute effect** (transfer ratio 1.6× at r=16). General objects are also far more printable to begin with (Thingi10k median support 255 mm³ vs cars' 8493 mm³; 43 % of general objects need *zero* support). The demo's text→image→3D content is much closer to the general distribution than to cars.
- **Detail cost:** `f1@0.01 ≈ 0.80` (≈ 20 % geometric detail compromise vs reference), `dinov2 = 0.013` (well inside the 0.027 σ-band). Acceptable for printed objects.

*Sources: `phase2_C_r16_multiseed_summary.md`, `phase2_A_transfer_summary.md`, `r32_plus_fill_holes_multiseed_summary.md`, `dpo_plus_fill_holes_summary.md`; EXPERIMENTS §17–§22.*

## 2.4 The exact reference implementation (from the worktree)

**The fill_holes operation** (`evaluation/dpo/diagnostics/interventions.py`):
```python
import scipy.ndimage as ndi
def fill_holes(occ: np.ndarray) -> np.ndarray:
    """Fill interior voids; input/output are (64,64,64) bool."""
    return ndi.binary_fill_holes(occ.astype(bool)).astype(bool)
```
Wired between Stage-1 and Stage-2 (`evaluation/dpo/diagnostics/_run_dpo_plus_fill_holes.py`):
```python
_z, occ = sample_stage1(pipe, cond, seed=SEED, steps=25, cfg=5.0)  # occ = (decoder(z_s) > 0)[0,0].cpu().numpy()
occ = binary_fill_holes(occ.astype(bool))                          # the fix
verts, faces = stage2_and_mesh(pipe, cond, occ)                    # coords = np.argwhere(occ) → sample_slat → decode_slat
```

**The LoRA wrap** (`evaluation/dpo/diagnostics/post_hoc_slicer.py`):
```python
from peft import PeftModel
def _apply_lora(pipeline, lora_dir):
    flow = pipeline.models["sparse_structure_flow_model"]
    pipeline.models["sparse_structure_flow_model"] = PeftModel.from_pretrained(flow, str(lora_dir), is_trainable=False)
def _unwrap_lora(pipeline):
    w = pipeline.models["sparse_structure_flow_model"]
    pipeline.models["sparse_structure_flow_model"] = w.unload() if hasattr(w, "unload") else w.merge_and_unload()
```
Trained checkpoints on disk in the worktree:
- **r=16 (recommended):** `evaluation/results/sweep_r16_slicer_w025/r16_slicer_w025/final/` (`adapter_config.json`, `adapter_model.safetensors`)
- r=32: `evaluation/results/sweep_r32_slicer_w025/r32_slicer_w025/final/`
- reward config: `evaluation/dpo/configs/reward_varianta_slicer_w025.yaml`; base model `microsoft/TRELLIS-image-large`.
- LoRA hparams: rank 16, α 32, target modules `[to_qkv, to_q, to_kv, to_out]`, β=500, lr=1e-4, KL λ=0, 3 epochs, uniform timestep sampling, ~3 min/train.

---

# Part 3 — Integration Map (this repository)

| What | Where (this repo) | Action |
|---|---|---|
| **Stage-1 occupancy → coords** | `trellis/pipelines/trellis_image_to_3d.py:204-205` in `sample_sparse_structure()` — `coords = torch.argwhere(decoder(z_s)>0)[:, [0,2,3,4]].int()` | **Insert `binary_fill_holes` here** (Phase A). Covers both the normal `run()` path and `_run_with_cpu_offload()` (both call this method). |
| **Stage-1 flow model (LoRA target)** | `pipeline.models["sparse_structure_flow_model"]` — key confirmed at `trellis_image_to_3d.py:191` | **Wrap with PEFT** (Phase B). |
| **TRELLIS pipeline load** | `modules/model_manager.py:_load_trellis_pipeline()` (~`:244` `from_pretrained`, `:245` `.to(device)`, `:249` optional offload) | **Attach LoRA after line 244** (Phase B). |
| **App 3D-gen call** | `modules/generation_pipeline.py:generate_3d_model()` → `self.trellis_pipeline.run(...)` (`:318-329`); exposes `sparse_structure_steps/cfg`, `slat_steps/cfg`, `seed`, `texture_size`, `use_simple_glb` | Optionally surface a "printability mode" flag. |
| **GLB export (live path)** | `trellis/utils/postprocessing_utils.py::to_glb_simple` (default `use_simple_glb=True`), called with `simplify=0.95, fill_holes=True, remove_floating=True` (`generation_pipeline.py:352-361`) | **Audit** (Phase C): 0.95 decimation keeps ~5 % of faces; `_fill_holes` is a *visibility-mincut* that can carve interior geometry; final `mesh.fill_holes()` is weak. |
| **STL export** | `modules/simple_stl_converter.py::convert_glb_to_stl()` — bare `trimesh.load(...).export(.stl)`, **no watertight/manifold check or repair** | **Harden** (Phase C): add manifold/watertight repair before export so prints don't fail. Wired in `streamlit-app.py:294`, `gradio-app.py:379`. |
| **Env / run config** | `run_streamlit_app.sh` (env `trellis_blackwell`, `ENABLE_TRELLIS_CPU_OFFLOAD=false`, `IMAGE_MODEL=qwen`) | New toggles added as env vars to match repo style. |
| **Deps** | `requirements.txt` — `scipy==1.14.1` ✅, `pymeshfix==0.17.0` ✅, `trimesh==4.5.3` ✅; `peft` **missing** | Add `peft` for Phase B. |

**Caveat on LoRA + CPU offload:** PEFT wraps the `nn.Module`; this repo's custom offload hooks operate on `pipeline.models[...]`. The default run config has `ENABLE_TRELLIS_CPU_OFFLOAD=false`, so Phase B is safe by default. If offload is later enabled, verify the offload manager re-resolves the wrapped module (test before shipping).

**Caveat on inference config:** the worktree validated the LoRA at `steps=25, cfg=5.0`; this repo's `generate_3d_model` defaults to `sparse_structure_steps=24, sparse_structure_cfg=7.5`. `fill_holes` is robust to this. For the LoRA, prefer matching `cfg≈5.0` (a small change in `generation_pipeline.py`) or treat current cfg as a tuning variable to verify.

---

# Part 4 — Implementation Plan

Recommended order: **Phase A (fill_holes) → Phase C (watertight STL) → Phase B (LoRA, optional) → Phase D (measurement/UI, optional)**. Phase A alone delivers most of the printability benefit with zero new dependencies.

## Phase A — Stage-1 `binary_fill_holes` (the headline fix)

### Task A1: Add a testable occupancy hole-fill helper

**Files:**
- Create: `trellis/utils/printability_utils.py`
- Test: `tests/test_printability_utils.py`

**Interfaces:**
- Produces: `fill_occupancy_holes(occ: np.ndarray) -> np.ndarray` — accepts a 3-D `(D,H,W)` or batched `(N,...,D,H,W)` boolean/numeric array, returns a `bool` array of the same shape with interior voids filled per 3-D volume.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_printability_utils.py
import numpy as np
from trellis.utils.printability_utils import fill_occupancy_holes

def test_fills_interior_cavity_3d():
    occ = np.ones((9, 9, 9), dtype=bool)
    occ[4, 4, 4] = False                      # one interior void
    out = fill_occupancy_holes(occ)
    assert out.dtype == bool
    assert out[4, 4, 4] == True               # void filled
    assert out.sum() == 9 * 9 * 9

def test_leaves_open_surface_untouched():
    occ = np.zeros((9, 9, 9), dtype=bool)
    occ[1:8, 1:8, 0:4] = True                 # solid block, open to a face
    out = fill_occupancy_holes(occ)
    assert out.sum() == occ.sum()             # nothing enclosed → no change

def test_batched_volume():
    occ = np.ones((2, 1, 9, 9, 9), dtype=bool)
    occ[0, 0, 4, 4, 4] = False
    out = fill_occupancy_holes(occ)
    assert out.shape == occ.shape
    assert out[0, 0, 4, 4, 4] == True
```

- [ ] **Step 2: Run the test, verify it fails**
Run: `pytest tests/test_printability_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: trellis.utils.printability_utils`.

- [ ] **Step 3: Implement the helper**
```python
# trellis/utils/printability_utils.py
"""Occupancy-domain printability helpers (TRELLIS Stage-1).

Based on the printability-eval investigation: applying scipy.ndimage.binary_fill_holes
to the Stage-1 64^3 occupancy grid (before Stage-2) reduces slicer support volume by
~15% (≈ -1300 mm^3 median) while preserving fine detail (f1@0.01 ≈ 0.80).
"""
import numpy as np
import scipy.ndimage as ndi


def fill_occupancy_holes(occ: np.ndarray) -> np.ndarray:
    """Fill interior voids in a boolean occupancy grid.

    Applies ``scipy.ndimage.binary_fill_holes`` to each trailing (D, H, W) volume.
    Accepts a 3-D array or any leading batch/channel dims, e.g. (N, 1, D, H, W).
    Returns a boolean array of the same shape. Open (surface-connected) cavities
    are left untouched — only topologically enclosed voids are filled.
    """
    occ = np.asarray(occ, dtype=bool)
    if occ.ndim < 3:
        raise ValueError(f"expected >=3 dims (..., D, H, W), got shape {occ.shape}")
    if occ.ndim == 3:
        return ndi.binary_fill_holes(occ)
    spatial = occ.shape[-3:]
    flat = occ.reshape(-1, *spatial)
    filled = np.empty_like(flat)
    for i in range(flat.shape[0]):
        filled[i] = ndi.binary_fill_holes(flat[i])
    return filled.reshape(occ.shape)
```

- [ ] **Step 4: Run the test, verify it passes**
Run: `pytest tests/test_printability_utils.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**
```bash
git add trellis/utils/printability_utils.py tests/test_printability_utils.py
git commit -m "feat(printability): add Stage-1 occupancy fill_holes helper"
```

### Task A2: Wire fill_holes into Stage-1 sampling behind an env-var toggle

**Files:**
- Modify: `trellis/pipelines/trellis_image_to_3d.py` (`sample_sparse_structure`, around `:204-205`)

**Interfaces:**
- Consumes: `fill_occupancy_holes` from Task A1.
- Behaviour: when env var `TRELLIS_STAGE1_FILL_HOLES` is truthy (`1/true/yes`), the Stage-1 occupancy is hole-filled before coordinate extraction. Default off → byte-identical to current behaviour. Covers both `run()` and `_run_with_cpu_offload()` (both call this method).

- [ ] **Step 1: Read the current method to confirm the exact lines**
Run: `sed -n '176,207p' trellis/pipelines/trellis_image_to_3d.py`
Expected: ends with `coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()` then `return coords`.

- [ ] **Step 2: Replace the occupancy→coords lines**
Replace:
```python
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()

        return coords
```
with:
```python
        decoder = self.models['sparse_structure_decoder']
        occ = decoder(z_s) > 0   # (num_samples, 1, D, H, W) bool; raw logits, no sigmoid

        # Printability fix (printability-eval): fill enclosed voids in the Stage-1
        # occupancy grid before Stage-2. ~15% less slicer support volume, detail-safe.
        if os.environ.get('TRELLIS_STAGE1_FILL_HOLES', 'false').lower() in ('1', 'true', 'yes'):
            from trellis.utils.printability_utils import fill_occupancy_holes
            occ_np = fill_occupancy_holes(occ[:, 0].detach().cpu().numpy())
            occ[:, 0] = torch.from_numpy(occ_np).to(occ.device)

        coords = torch.argwhere(occ)[:, [0, 2, 3, 4]].int()

        return coords
```

- [ ] **Step 3: Confirm `os` is imported at module top**
Run: `grep -n '^import os' trellis/pipelines/trellis_image_to_3d.py`
Expected: a match. If absent, add `import os` to the imports block (the only change needed).

- [ ] **Step 4: Smoke-test that the toggle off changes nothing and on imports cleanly**
Run:
```bash
python -c "import ast; ast.parse(open('trellis/pipelines/trellis_image_to_3d.py').read()); print('parse-ok')"
TRELLIS_STAGE1_FILL_HOLES=true python -c "from trellis.utils.printability_utils import fill_occupancy_holes; print('helper-ok')"
```
Expected: `parse-ok` then `helper-ok`.

- [ ] **Step 5: Commit**
```bash
git add trellis/pipelines/trellis_image_to_3d.py
git commit -m "feat(printability): fill Stage-1 occupancy holes when TRELLIS_STAGE1_FILL_HOLES=1"
```

### Task A3: End-to-end verification on a generated object

**Files:** none (manual verification). Run in the `trellis_blackwell` env with this repo's env vars.

- [ ] **Step 1: Generate one object twice (off vs on) from the same seed**
Use `modules.generation_pipeline.GenerationPipeline.generate_3d_model(image, base_seed=42, ...)` (or a short script calling `trellis_pipeline.run(image, seed=42)`), once with `TRELLIS_STAGE1_FILL_HOLES` unset and once `=true`. Save both GLBs.

- [ ] **Step 2: Confirm the meshes differ and detail is preserved**
Expected: the filled mesh has equal-or-fewer interior boundary loops and visually closed cavities; silhouette/detail substantially unchanged (this matches `f1@0.01 ≈ 0.80`). If the filled mesh looks blobby or loses surface features, stop — that indicates the wrong morphology (do **not** add erode/dilate; Part 5).

- [ ] **Step 3 (optional, quantitative): measure support volume with PrusaSlicer**
If PrusaSlicer CLI is available, scale each mesh to 50 mm bbox-max and slice with the locked profile (Part 2 / Global Constraints); compare support-material volume. Expected: filled mesh ~15 % lower. (Or port `evaluation/dpo/slicer.py` + `slicer_profile_fdm_default.ini` from the worktree — see Phase D.)

- [ ] **Step 4: Decide default**
Recommended: set `TRELLIS_STAGE1_FILL_HOLES=true` in `run_streamlit_app.sh` (and the Gradio launch) so the demo ships printable-by-default.
```bash
git add run_streamlit_app.sh
git commit -m "chore(printability): enable Stage-1 fill_holes by default in run script"
```

## Phase C — Harden GLB→STL export for print validity

> **Rationale:** The worktree proved mesh-domain cleanup does **not** reduce support volume — so this phase is **not** about support volume (that's Phase A/B, upstream). It is about ensuring the STL the user prints is **watertight and manifold** so the slice doesn't fail. This repo's current path (0.95 decimation + visibility-mincut `_fill_holes` + bare STL export, no repair) is a print-validity risk.

### Task C1: Add a watertight/manifold repair before STL export

**Files:**
- Modify: `modules/simple_stl_converter.py::convert_glb_to_stl()`
- Test: `tests/test_stl_repair.py`

**Interfaces:**
- Produces: `ensure_printable_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh` — returns a mesh that is manifold and watertight where feasible (uses `pymeshfix`, already a dependency).

- [ ] **Step 1: Write the failing test** (synthetic open box → repaired watertight)
```python
# tests/test_stl_repair.py
import numpy as np, trimesh
from modules.simple_stl_converter import ensure_printable_mesh

def test_repairs_open_mesh_to_watertight():
    box = trimesh.creation.box(extents=(10, 10, 10))
    open_box = trimesh.Trimesh(vertices=box.vertices,
                               faces=box.faces[:-2],   # drop 2 faces → a hole
                               process=False)
    assert not open_box.is_watertight
    repaired = ensure_printable_mesh(open_box)
    assert repaired.is_watertight
    assert repaired.is_winding_consistent
```

- [ ] **Step 2: Run, verify it fails**
Run: `pytest tests/test_stl_repair.py -v`
Expected: FAIL — `ensure_printable_mesh` undefined.

- [ ] **Step 3: Implement and call it in the converter**

> **⚠️ Lesson learned (do NOT use pymeshfix `repair()` here).** The original draft used `pymeshfix.MeshFix(...).repair()`. On real TRELLIS GLBs — which are non-watertight, multi-component shells — `repair()` discards most of the geometry and collapses the mesh to a small fragment, producing **flat/broken STLs** (observed: 14,147 faces → 1,808, volume 0.268 → 0.013). The unit test (a clean box missing 2 faces) repaired fine and never exposed this. Use **gentle, geometry-preserving** trimesh repair with a **fallback to the original mesh** instead. This is consistent with the worktree finding that aggressive mesh-domain ops destroy TRELLIS geometry.

Add to `modules/simple_stl_converter.py`:
```python
def ensure_printable_mesh(mesh):
    """Best-effort improve a mesh for FDM slicing WITHOUT destroying geometry.

    TRELLIS GLBs are non-watertight, multi-component shells; an aggressive repair
    collapses them to a flat fragment. Apply only gentle, geometry-preserving
    repairs and fall back to the original if geometry/bbox is lost.
    """
    import numpy as np, trimesh
    if mesh.is_watertight and mesh.is_winding_consistent:
        return mesh
    repaired = mesh.copy()
    repaired.merge_vertices()
    trimesh.repair.fix_normals(repaired)   # consistent winding + outward normals
    trimesh.repair.fill_holes(repaired)    # gentle small-boundary fill (never deletes geometry)
    bbox_preserved = np.allclose(mesh.extents, repaired.extents, rtol=0.02, atol=1e-6)
    if len(repaired.faces) < 0.9 * len(mesh.faces) or not bbox_preserved:
        return mesh                        # over-aggressive result -> keep original
    return repaired
```
Then in `convert_glb_to_stl`, after `mesh = trimesh.load(glb_path, force='mesh')` and before `mesh.export(...)`, insert `mesh = ensure_printable_mesh(mesh)`. Regression test: a two-box multi-component non-watertight mesh must keep its bounding box (see `tests/test_stl_repair.py::test_preserves_geometry_of_multicomponent_mesh`).

- [ ] **Step 4: Run, verify it passes**
Run: `pytest tests/test_stl_repair.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**
```bash
git add modules/simple_stl_converter.py tests/test_stl_repair.py
git commit -m "feat(printability): repair meshes to watertight/manifold before STL export"
```

### Task C2: Audit decimation aggressiveness (investigation, then decide)

**Files:** `trellis/utils/postprocessing_utils.py` (`to_glb_simple`), `modules/generation_pipeline.py:352-361`.

- [ ] **Step 1: Measure detail/print-validity at `simplify=0.95` vs `0.9` vs `0.8`**
Generate the same object at three `simplify` values (thread the value through `generate_3d_model` / `to_glb_simple`), and compare: face count, `is_watertight`, thin-wall survival, and (if available) slicer support volume. Aggressive 0.95 keeps only ~5 % of faces and can thin walls below printable thickness.

- [ ] **Step 2: Decide and document**
If 0.95 degrades printable detail, lower the default for the print path (e.g. 0.9) or expose it as a UI/env setting. Record the chosen value and the evidence in this file. Do **not** expect any decimation change to reduce *support volume* — that lever is Phase A/B.

- [ ] **Step 3: Commit any change**
```bash
git add -A && git commit -m "tune(printability): adjust mesh decimation for print detail (see printability_optimization_3d.md)"
```

## Phase B — Optional: DPO LoRA on the Stage-1 flow model

> Stacks on top of Phase A for the full **−3091 mm³ mean (general objects)** effect. Adds a `peft` dependency and a checkpoint file. Skip if Phase A's improvement is sufficient for the demo.

### Task B1: Vendor the trained LoRA checkpoint and pin `peft`

**Files:**
- Create: `checkpoints/printability_lora_r16/` (copied from the worktree)
- Modify: `requirements.txt`, `.gitignore`

- [ ] **Step 1: Copy the r=16 checkpoint into this repo**
```bash
mkdir -p checkpoints/printability_lora_r16
cp /home/damian/Projects/TRELLIS/.claude/worktrees/printability-eval/evaluation/results/sweep_r16_slicer_w025/r16_slicer_w025/final/* checkpoints/printability_lora_r16/
ls checkpoints/printability_lora_r16/   # expect adapter_config.json, adapter_model.safetensors
```

- [ ] **Step 2: Pin `peft`**
Add `peft==0.19.1` to `requirements.txt`.

- [ ] **Step 3: Decide checkpoint tracking**
The r=16 adapter is small (LoRA weights only). Track it in git, or add to `.gitignore` + document the copy step here. Recommended: track it so the demo is self-contained.

- [ ] **Step 4: Commit**
```bash
git add requirements.txt checkpoints/printability_lora_r16 .gitignore
git commit -m "feat(printability): vendor r16 slicer-w025 Stage-1 LoRA + pin peft"
```

### Task B2: Load the LoRA onto `sparse_structure_flow_model` behind an env var

**Files:**
- Modify: `modules/model_manager.py::_load_trellis_pipeline()` (after `from_pretrained`/`.to`, ~`:244-248`)

**Interfaces:**
- Behaviour: when `TRELLIS_STAGE1_LORA` is set to a checkpoint dir, wrap the flow model via PEFT. Unset → unchanged.

- [ ] **Step 1: Insert the LoRA wrap after the pipeline is on device**
After `pipeline.to(self.trellis_device)` and before offload setup, add:
```python
lora_dir = os.environ.get('TRELLIS_STAGE1_LORA')
if lora_dir:
    from peft import PeftModel
    flow = pipeline.models['sparse_structure_flow_model']
    pipeline.models['sparse_structure_flow_model'] = PeftModel.from_pretrained(
        flow, lora_dir, is_trainable=False)
    print(f"[printability] Loaded Stage-1 LoRA from {lora_dir}")
```

- [ ] **Step 2: Guard the CPU-offload interaction**
Confirm the default run config keeps `ENABLE_TRELLIS_CPU_OFFLOAD=false` (it does). If offload is enabled, verify the offload manager resolves the *wrapped* module before shipping; otherwise wrap before offload-hook registration.

- [ ] **Step 3: Verify load + a single generation**
Run with `TRELLIS_STAGE1_LORA=checkpoints/printability_lora_r16 TRELLIS_STAGE1_FILL_HOLES=true` and confirm the log line prints and one object generates without error.

- [ ] **Step 4 (recommended): match the LoRA's inference cfg**
The LoRA was trained/validated at `cfg≈5.0`. Either set `sparse_structure_cfg=5.0` for the print path in `generate_3d_model`, or verify quality at the current 7.5. Document the choice.

- [ ] **Step 5: Commit**
```bash
git add modules/model_manager.py modules/generation_pipeline.py
git commit -m "feat(printability): optional Stage-1 DPO LoRA via TRELLIS_STAGE1_LORA"
```

## Phase D — Optional: measurement harness, best-of-N, and UI

- [ ] **D1 — Slicer measurement harness.** Port `evaluation/dpo/slicer.py` + `evaluation/configs/slicer_profile_fdm_default.ini` from the worktree into a small `tools/measure_printability.py` that scales a GLB/STL to 50 mm bbox-max and returns support volume. Use it to quantify Phase A/B gains on demo-style objects. (Needs PrusaSlicer CLI; throughput ~2000 meshes/hr.)
- [ ] **D2 — Best-of-N (training-free alternative/complement).** For a "max printability" mode, generate N Stage-1 candidates, slice each, keep the lowest-support mesh. Best-of-32 ≈ −23 % support at zero training cost but N× generation+slicing time. Best as an opt-in for users who will physically print.
- [ ] **D3 — UI surfacing.** Expose a "Printability mode" toggle in the Streamlit/Gradio apps that sets `TRELLIS_STAGE1_FILL_HOLES` (and optionally the LoRA / best-of-N) for that generation, with a one-line tooltip ("Optimizes the model to need less support material when 3D-printed").

---

# Part 5 — Risks, Caveats, and Dead Ends (do NOT repeat)

**Refuted interventions (data-backed):**
- **Stage-1 logit-threshold sweeps, CFG sweeps, multi-seed logit averaging** → no-ops or worse; Stage-2 smooths small occupancy edits away.
- **Probabilistic dilation (`prob_dilate`)** → +25–35 % more support.
- **`fill_holes` + erode (`erode1/erode2`)** → big support wins but detail destroyed (`f1@0.01` 0.80 → 0.07). Tempting but wrong; the demo must keep recognizable detail.
- **LoRA rank ≥ 64** → overfits (20 M params on 128 pairs), support gets worse.
- **Reward re-weighting beyond `w_sv=0.25`** (composite, `w_wt`, `w_sv≥0.40`) → all regress; the operating point sits on a fragile Pareto edge.
- **Scaling the DPO dataset** (138 → 580 pairs) → variance doubles, no gain.
- **Mesh-domain PyMeshFix / largest-component for support reduction** → literally zero support effect (it only raises watertight fraction; that's why Phase C is framed as print-validity, not support reduction).

**Methodological cautions carried over:**
- **Single-seed scouts overestimate by ~30 %** and pick lucky seeds. Any A/B you run here should use ≥2–3 Stage-1 seeds. (Stage-2 `sample_slat` is effectively unseeded in TRELLIS, so per-shape metrics drift run-to-run.)
- **Voxel-IoU is a dead metric** for topology-improving changes (GT artifacts). Judge with slicer support volume + the detail trio (`f1@0.01`, `curv_wass`, `dinov2`).
- **All numbers are tied to one Prusa MK3S FDM profile.** If the demo targets resin/SLA or a different FDM printer, re-measure — rankings can shift. (Open pending item in the worktree.)
- **Largest meshes OOM'd** in the worktree's Thingi10k runs (10–17 per batch excluded) — very large/dense demo meshes may need the CPU-offload path or downsampling.

**Integration risks specific to this repo:**
- PEFT-wrapped flow model + custom CPU-offload hooks are untested together (Phase B caveat). Safe at the default `ENABLE_TRELLIS_CPU_OFFLOAD=false`.
- The demo's Stage-1 cfg (7.5) differs from the LoRA's validation cfg (5.0). `fill_holes` is robust; the LoRA should be verified at the demo's cfg or the cfg lowered.

---

# Part 6 — Where Everything Lives (worktree reference index)

Root: `/home/damian/Projects/TRELLIS/.claude/worktrees/printability-eval/`

| What | Path |
|---|---|
| Single entry point / results overview | `docs/superpowers/RESULTS.md` |
| Full narrative (22 experiments + capstone) | `evaluation/dpo/EXPERIMENTS.md` |
| Per-experiment diagnostics (19 files) | `evaluation/results/diagnostics/*.md` |
| Original investigation plan | `docs/superpowers/specs/2026-06-01-printability-investigation-plan.md` |
| Prior-art research synthesis | `docs/superpowers/specs/2026-06-01-printability-research-synthesis.md` |
| Phase 2 (Thingi10k generalization) design | `docs/superpowers/specs/2026-06-03-phase2-thingi10k-generalization-design.md` |
| **fill_holes & morphology implementations** | `evaluation/dpo/diagnostics/interventions.py` |
| **LoRA apply/unwrap + post-hoc driver** | `evaluation/dpo/diagnostics/post_hoc_slicer.py`, `_run_dpo_plus_fill_holes.py` |
| Stage-1/Stage-2 sampling helpers | `evaluation/dpo/sampling.py` |
| Mesh-domain postprocess (the ZERO-effect one) | `evaluation/dpo/diagnostics/mesh_postprocess.py` |
| Stage-1 vs Stage-2 attribution | `evaluation/dpo/diagnostics/stage1_attribution.py` |
| Slicer wrapper + locked FDM profile | `evaluation/dpo/slicer.py`, `evaluation/configs/slicer_profile_fdm_default.ini` |
| DPO trainer (LoRA via `peft.get_peft_model`) | `evaluation/dpo/trainer.py`; toolkit docs `evaluation/dpo/README.md` |
| **r=16 LoRA checkpoint (recommended)** | `evaluation/results/sweep_r16_slicer_w025/r16_slicer_w025/final/` |
| r=32 LoRA checkpoint | `evaluation/results/sweep_r32_slicer_w025/r32_slicer_w025/final/` |
| Reward config | `evaluation/dpo/configs/reward_varianta_slicer_w025.yaml` |

**Key diagnostics for the headline claims:** `stage1_attribution_summary.md` (Stage-1 is the cause), `intervention_sweep_summary.md` (fill_holes wins, logit tricks lose), `fill_holes_erode_detail_summary.md` (why not erode), `phase2_C_r16_multiseed_summary.md` (final r=16 operating point), `phase2_A_transfer_summary.md` (generalizes to non-car objects).
