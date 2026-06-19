# Print-Ready Mesh Viewer — Design Spec

**Date:** 2026-06-19
**Status:** Approved (design); pending implementation plan.

## 1. Goal

After a 3D generation, run the print-ready watertight remesh **immediately** (not lazily at
download time) and show the user **two side-by-side interactive 3D viewers**: the existing colored
GLB on the left, and the **print-ready mesh rendered as normals** on the right. **Remove the
Gaussian splat** (and its preview video). The right viewer shows *exactly* the mesh the STL-download
button produces — "what you see is what you print."

This applies to **both** the Streamlit (`streamlit-app.py`) and Gradio (`gradio-app.py`) apps.

## 2. Current state (relevant flow)

- **`modules/generation_pipeline.py::generate_3d_model`** (≈266–387) returns **`(video_path, glb_path)`**.
  - Calls `trellis_pipeline.run(...)` (≈318–329) **without** `formats=`, so it falls through to the
    default `['mesh','gaussian','radiance_field']` (`trellis/pipelines/trellis_image_to_3d.py:302`) —
    all three are always decoded.
  - **Preview video** (≈337–344) is rendered **only** from the gaussian splat:
    `render_utils.render_video(outputs['gaussian'][0])` → `gaussian_splat.mp4`. Gated by `sample_video`.
  - **GLB** (≈349–381): `use_simple_glb=True` → `to_glb_simple(outputs['mesh'][0], simplify=0.95,
    color=(180,180,220), ...)` (mesh-only); `use_simple_glb=False` → `to_glb_new(outputs['gaussian'][0],
    outputs['mesh'][0], ...)` (textured, **needs the gaussian**).
- The apps set `sample_video = use_gaussian_rendering` and `use_simple_glb = not use_gaussian_rendering`
  from `USE_GAUSSIAN_RENDERING` (streamlit ≈226–227, gradio ≈315–316). So the env flag overloads BOTH
  the video and the GLB-export path.
- **STL** is produced **lazily** at download time via `convert_glb_to_stl(glb_path, ...)`
  (streamlit `prepare_3d_model_for_printing` ≈305; gradio `prepare_for_download` ≈390), which already
  honors `TRELLIS_PRINT_REMESH` (→ `remesh_for_printing`, else `ensure_printable_mesh`) in
  `modules/simple_stl_converter.py`.
- **Viewer** `modules/three_d_viewer.py::create_3d_viewer_html(glb_path, **opts)` → `ThreeDViewer`:
  base64-embeds the GLB, loads it via **GLTFLoader** (≈248–256), and in a `model.traverse` block
  (≈271–280) keeps the GLB's own material (sets `side=DoubleSide`, metalness/roughness). Hardcoded to GLB.
- **Display:**
  - Streamlit (≈765–785): if `video_path==""` → viewer only; else `st.video(video_path)` +
    `components.html(create_3d_viewer_html(glb_path), height=550)`.
  - Gradio: `iFrame(height=550)` as `model_output` (≈553/555), `gr.Video` (≈551) when gaussian;
    `image_gallery.select(...)` has two output branches (≈700–719) for gaussian vs non-gaussian.

**Coupling:** the preview video is the **only** consumer of the gaussian; both apps already handle an
empty `video_path` gracefully. Removing the gaussian is safe.

## 3. Design

### 3.1 Data flow (new)

```
generate_3d_model(image, ...)
  └─ trellis_pipeline.run(..., formats=['mesh'])      # gaussian/RF no longer decoded
  └─ to_glb_simple(mesh) -> glb_path                  # colored GLB (left viewer + GLB download)
  └─ export_print_ready(glb_path, temp_dir)           # remesh ONCE
        -> print_ready_glb_path  (right viewer, normals)
        -> stl_path              (STL download)
  return (glb_path, print_ready_glb_path, stl_path)   # video dropped
```

App display: `st.columns(2)` / `gr.Row` →
left `create_3d_viewer_html(glb_path)`, right `create_3d_viewer_html(print_ready_glb_path, normals=True)`.

### 3.2 Components & interfaces

**A. `modules/simple_stl_converter.py`**
- Extract the load + remesh-or-repair logic currently inline in `convert_glb_to_stl` into:
  - `prepare_printable_mesh(glb_path: str) -> trimesh.Trimesh` — loads the GLB, applies the env-gated
    remesh (`remesh_for_printing` when `TRELLIS_PRINT_REMESH` enabled, else `ensure_printable_mesh`),
    returns the resulting mesh. This is the single source of truth for "the printable mesh."
  - `export_print_ready(glb_path: str, out_dir: str, basename: str = "print_ready") -> tuple[str, str]`
    — calls `prepare_printable_mesh` once, exports `<basename>.glb` and `<basename>.stl`, returns
    `(print_glb_path, stl_path)`. Returns `(None, None)` if the input mesh is empty/degenerate.
- `convert_glb_to_stl` is refactored to call `prepare_printable_mesh` then export (behavior unchanged;
  existing tests still pass).

**B. `modules/generation_pipeline.py::generate_3d_model`**
- Add `formats=['mesh']` to the `trellis_pipeline.run(...)` call.
- Delete the gaussian video block; remove the `sample_video`/`use_simple_glb=False` (`to_glb_new`)
  branches — always export the colored GLB via `to_glb_simple`.
- After `glb.export(glb_path)`, call `export_print_ready(glb_path, temp_dir)`.
- **New return:** `(glb_path, print_ready_glb_path, stl_path)`. (Drop `video_path`.) **Decision:**
  remove the now-unused `sample_video` and `use_simple_glb` parameters from the signature and fix the
  two app call sites (no backward-compat shim — these are internal callers only).

**C. `modules/three_d_viewer.py`**
- Add `normals: bool = False` to `create_3d_viewer_html` / `ThreeDViewer`. When `True`, in the
  `model.traverse` material block (≈271–280) replace each mesh material with
  `new THREE.MeshNormalMaterial({ side: THREE.DoubleSide })` instead of keeping the GLB material.
  Loader and the rest are unchanged.

**D. `streamlit-app.py`**
- Update the local `generate_3d_model` wrapper (≈213–236) and its call site (≈751–754) to the new
  3-tuple return; stop passing `sample_video`/`use_simple_glb`.
- Replace the results display (≈765–785): `col_left, col_right = st.columns(2)`;
  left `components.html(create_3d_viewer_html(glb_path), height=550)`,
  right `components.html(create_3d_viewer_html(print_ready_glb_path, normals=True), height=550)`,
  with captions ("Colored model" / "Print-ready (normals)"). Remove `st.video`.
- STL download uses the `stl_path` from generation (no lazy `convert_glb_to_stl`); keep GLB download
  = `glb_path`. Gallery save unchanged.

**E. `gradio-app.py`**
- Update `select_and_generate_3d_for_index` (≈270–355) to the new 3-tuple; build two viewer HTMLs
  (colored + normals); remove the gaussian/video branch.
- UI: replace the single `model_output` iFrame + `gr.Video` with a `gr.Row` of two `iFrame`s
  (`model_output_colored`, `model_output_normals`). Update `image_gallery.select(...)` outputs
  (≈700–719) to a single branch returning both viewer HTMLs. STL/GLB downloads use the generated paths.

### 3.3 Gaussian removal scope
Remove: the gaussian video render, the `to_glb_new` (textured) branch, `sample_video` usage, the
apps' `gr.Video`/`st.video` and `use_gaussian_rendering` branching. **Decision:** remove
`get_gaussian_rendering_setting()` and all `USE_GAUSSIAN_RENDERING` reads / session flags from both
apps (obsolete). Also remove `USE_GAUSSIAN_RENDERING` from `run_streamlit_app.sh`. Do NOT remove the
`slat_decoder_gs` model class or `trellis/utils/render_utils.py` (library code; notebooks/examples may
use them).

## 4. Edge cases
- **Remesh off / degenerate:** `prepare_printable_mesh` returns the `ensure_printable_mesh` result
  (== the STL download); the normals viewer shows that. Consistent semantics: normals viewer ≡ STL.
- **Empty generation** (e.g., the known degenerate case): `export_print_ready` returns `(None, None)`;
  the app shows the colored viewer plus a "no print-ready mesh available" placeholder on the right and
  disables the STL download.
- **Large print-ready GLB** (~340k faces base64-embedded in the viewer HTML): acceptable; note as a
  possible future optimization (decimation), out of scope here.

## 5. Testing
- Unit (`tests/`): `prepare_printable_mesh` returns a mesh (watertight when `TRELLIS_PRINT_REMESH` on);
  `export_print_ready` writes a valid GLB + STL and returns paths (and `(None, None)` on an empty mesh);
  `create_3d_viewer_html(path, normals=True)` output contains `MeshNormalMaterial` and `normals=False`
  does not. Existing STL/printability tests must still pass.
- Manual end-to-end: run each app, generate, confirm two side-by-side viewers (colored + normals), no
  video, STL/GLB downloads work, right viewer matches the downloaded STL.

## 6. Global constraints
- Conda env `trellis_blackwell`; `TRELLIS_PRINT_REMESH=voxel288_smooth` default-on (already wired).
- `generate_3d_model`'s new return tuple is `(glb_path, print_ready_glb_path, stl_path)` — both apps'
  call sites must match exactly.
- The normals viewer is fed a **GLB** (reuse GLTFLoader); the remeshed trimesh is exported to GLB.
- No production behavior change to `convert_glb_to_stl`'s STL output (shared helper).
