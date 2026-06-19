# Print-Ready Mesh Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After generation, run the print-ready remesh immediately and show two side-by-side interactive viewers (colored GLB + print-ready mesh as normals) in both apps, with the Gaussian splat removed.

**Architecture:** `generate_3d_model` decodes mesh only, exports the colored GLB, then computes the print-ready mesh once and exports it as both a GLB (for the normals viewer) and an STL (download). The existing Three.js viewer gains a `normals` flag (→ `MeshNormalMaterial`). Both apps render two viewers side-by-side and drop the gaussian video.

**Tech Stack:** Python, trimesh, TRELLIS pipeline, Three.js (r128, GLTFLoader), Streamlit (`components.html`), Gradio (`gradio_iframe.iFrame`), conda env `trellis_blackwell`.

## Global Constraints

- Conda env `trellis_blackwell`. Activate for any test run:
  `export PATH=/home/damian/miniconda3/bin:$PATH; eval "$(conda shell.bash hook)"; conda activate trellis_blackwell`
- `TRELLIS_PRINT_REMESH` (default `voxel288_smooth`) already gates the remesh in `modules/simple_stl_converter.py`; do NOT change that contract.
- `generate_3d_model`'s new return is exactly `(glb_path, print_ready_glb_path, stl_path)` (video dropped). Both app call sites must match.
- The normals viewer is fed a **GLB** (reuse GLTFLoader); the remeshed trimesh is exported to GLB via `mesh.export('*.glb')`.
- Semantics: the normals viewer shows exactly the mesh `export_print_ready` writes as STL.
- Remove `get_gaussian_rendering_setting()` and all `USE_GAUSSIAN_RENDERING` reads from both apps and from `run_streamlit_app.sh`. Do NOT remove `slat_decoder_gs` or `trellis/utils/render_utils.py` (library code).
- Line numbers below are guides; each task says to grep/read first, then edit, so they survive drift.
- Apps can't run headless in CI — app tasks verify via `ast.parse` + import smoke checks; full UX is a manual end-to-end step.

---

### Task 1: Print-ready mesh helpers in `simple_stl_converter.py`

**Files:**
- Modify: `modules/simple_stl_converter.py` (add two functions; refactor `convert_glb_to_stl` to reuse one)
- Test: `tests/test_stl_repair.py`

**Interfaces:**
- Consumes: existing `remesh_for_printing(mesh)`, `ensure_printable_mesh(mesh)` in the same file.
- Produces:
  - `prepare_printable_mesh(glb_path: str) -> trimesh.Trimesh | None` — loads the GLB, returns the env-gated print-ready mesh (remesh when `TRELLIS_PRINT_REMESH` enabled, else `ensure_printable_mesh`); `None` if the loaded mesh is empty.
  - `export_print_ready(glb_path: str, out_dir: str, basename: str = "print_ready") -> tuple[str | None, str | None]` — computes the mesh once, writes `<basename>.glb` and `<basename>.stl`, returns `(print_glb_path, stl_path)` or `(None, None)` on empty.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_stl_repair.py`)

```python
def test_prepare_printable_mesh_returns_mesh(tmp_path):
    import os
    box = trimesh.creation.box(extents=(10, 10, 10))
    glb = str(tmp_path / "in.glb"); box.export(glb)
    from modules.simple_stl_converter import prepare_printable_mesh
    os.environ["TRELLIS_PRINT_REMESH"] = "off"
    m = prepare_printable_mesh(glb)
    assert m is not None and len(m.faces) > 0


def test_export_print_ready_writes_glb_and_stl(tmp_path):
    import os
    box = trimesh.creation.box(extents=(10, 10, 10))
    glb = str(tmp_path / "in.glb"); box.export(glb)
    from modules.simple_stl_converter import export_print_ready
    os.environ["TRELLIS_PRINT_REMESH"] = "off"
    pg, stl = export_print_ready(glb, str(tmp_path))
    assert pg and stl and os.path.exists(pg) and os.path.exists(stl)
    assert len(trimesh.load(pg, force="mesh").faces) > 0
    assert len(trimesh.load(stl, force="mesh").faces) > 0


def test_export_print_ready_remesh_on_is_watertight(tmp_path):
    import os
    box = trimesh.creation.box(extents=(10, 10, 10))
    open_box = trimesh.Trimesh(vertices=box.vertices, faces=box.faces[:-2], process=False)
    glb = str(tmp_path / "open.glb"); open_box.export(glb)
    from modules.simple_stl_converter import export_print_ready
    os.environ["TRELLIS_PRINT_REMESH"] = "voxel288_smooth"
    pg, stl = export_print_ready(glb, str(tmp_path), basename="pr")
    assert trimesh.load(stl, force="mesh").is_watertight
    os.environ["TRELLIS_PRINT_REMESH"] = "off"   # restore default for other tests
```

- [ ] **Step 2: Run the tests, verify they fail**

Run: `python -m pytest tests/test_stl_repair.py -q -k "prepare_printable or export_print_ready"`
Expected: FAIL — `ImportError: cannot import name 'prepare_printable_mesh'`.

- [ ] **Step 3: Add the two functions** (place them right BEFORE `def convert_glb_to_stl`)

```python
def prepare_printable_mesh(glb_path: str):
    """Load a GLB and return the print-ready mesh (env-gated remesh, else gentle repair).

    Single source of truth: the returned mesh is exactly what gets exported as STL.
    Returns None if the loaded mesh is empty.
    """
    mesh = trimesh.load(glb_path, force='mesh')
    if mesh is None or len(mesh.faces) == 0:
        return None
    mode = os.environ.get('TRELLIS_PRINT_REMESH', 'off').strip().lower()
    remeshed = None
    if mode in ('voxel288_smooth', 'voxel', 'on', 'true', '1'):
        try:
            remeshed = remesh_for_printing(mesh)
        except Exception as e:
            print(f"⚠️  print remesh failed ({type(e).__name__}: {e}); using gentle repair")
    if remeshed is not None:
        print("🧱 Applied watertight print remesh (voxel288_smooth)")
        return remeshed
    return ensure_printable_mesh(mesh)


def export_print_ready(glb_path: str, out_dir: str, basename: str = "print_ready"):
    """Compute the print-ready mesh once; export it as GLB (normals viewer) + STL (download).

    Returns (print_glb_path, stl_path), or (None, None) if the input mesh is empty/degenerate.
    """
    mesh = prepare_printable_mesh(glb_path)
    if mesh is None or len(mesh.faces) == 0:
        return None, None
    os.makedirs(out_dir, exist_ok=True)
    print_glb_path = os.path.join(out_dir, f"{basename}.glb")
    stl_path = os.path.join(out_dir, f"{basename}.stl")
    mesh.export(print_glb_path)
    mesh.export(stl_path)
    return print_glb_path, stl_path
```

- [ ] **Step 4: Refactor `convert_glb_to_stl` to reuse `prepare_printable_mesh`**

In `convert_glb_to_stl`, find the block that loads the GLB and computes `mesh` (the `mesh = trimesh.load(...)` through the `mode = os.environ.get('TRELLIS_PRINT_REMESH'...)` / `ensure_printable_mesh` logic, ending before `mesh.export(output_path)`). Replace that whole block with:

```python
        mesh = prepare_printable_mesh(glb_path)
        if mesh is None:
            raise Exception("empty mesh — nothing to export")
```

Leave the surrounding `print(...)` logs, `output_path` computation, and `mesh.export(output_path)` intact.

- [ ] **Step 5: Run all converter tests, verify PASS**

Run: `python -m pytest tests/test_stl_repair.py -q`
Expected: all pass (the 8 existing + 3 new).

- [ ] **Step 6: Commit**

```bash
git add modules/simple_stl_converter.py tests/test_stl_repair.py
git commit -m "feat(print-ready): prepare_printable_mesh + export_print_ready helpers" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `normals` flag in the 3D viewer

**Files:**
- Modify: `modules/three_d_viewer.py` (`create_3d_viewer_html`, `ThreeDViewer.__init__`/`generate_html`, and the material `traverse` JS block ≈271–280)
- Test: `tests/test_three_d_viewer.py` (create)

**Interfaces:**
- Produces: `create_3d_viewer_html(glb_path, normals: bool = False, **opts) -> str` — when `normals=True`, the generated HTML applies `THREE.MeshNormalMaterial` to every mesh instead of the GLB's material.

- [ ] **Step 1: Read the viewer to find the wiring**

Run: `grep -n "def create_3d_viewer_html\|class ThreeDViewer\|def __init__\|def generate_html\|def _generate_javascript\|traverse\|child.material" modules/three_d_viewer.py`
Note how `create_3d_viewer_html` passes options into `ThreeDViewer`, and the `model.traverse(function (child) { ... })` block that sets `child.material.side` / metalness / roughness.

- [ ] **Step 2: Write the failing test** (`tests/test_three_d_viewer.py`)

```python
import trimesh
from modules.three_d_viewer import create_3d_viewer_html


def _glb(tmp_path):
    p = str(tmp_path / "m.glb")
    trimesh.creation.box(extents=(1, 1, 1)).export(p)
    return p


def test_normals_true_uses_normal_material(tmp_path):
    html = create_3d_viewer_html(_glb(tmp_path), normals=True)
    assert "MeshNormalMaterial" in html


def test_normals_false_keeps_default_material(tmp_path):
    html = create_3d_viewer_html(_glb(tmp_path), normals=False)
    assert "MeshNormalMaterial" not in html
```

- [ ] **Step 3: Run it, verify it fails**

Run: `python -m pytest tests/test_three_d_viewer.py -q`
Expected: FAIL — `test_normals_true_uses_normal_material` (no `MeshNormalMaterial` in output); the `normals` kwarg may also raise `TypeError` if not yet accepted.

- [ ] **Step 4: Thread the `normals` flag through**

(a) In `create_3d_viewer_html`, add `normals: bool = False` to the signature and pass it into `ThreeDViewer` (e.g. store as `self.normals = normals` in `__init__`, accepting `normals=False`).

(b) In the material `traverse` JS block (≈271–280), make the per-mesh material conditional. Replace the body that sets `child.material.side = THREE.DoubleSide` (and metalness/roughness) so that when `self.normals` is true it emits:

```python
        # inside the f-string building the traverse block:
        material_js = (
            "child.material = new THREE.MeshNormalMaterial({ side: THREE.DoubleSide });"
            if self.normals else
            "child.material.side = THREE.DoubleSide;\n"
            "                    if (child.material.metalness !== undefined) child.material.metalness = 0.3;\n"
            "                    if (child.material.roughness !== undefined) child.material.roughness = 0.7;"
        )
```

and interpolate `{material_js}` where the old per-child material lines were. (Match the file's actual variable names and indentation — read Step 1 output. The exact metalness/roughness values must match what the file currently uses.)

- [ ] **Step 5: Run the tests, verify PASS**

Run: `python -m pytest tests/test_three_d_viewer.py -q`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add modules/three_d_viewer.py tests/test_three_d_viewer.py
git commit -m "feat(print-ready): add normals (MeshNormalMaterial) mode to 3D viewer" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `generate_3d_model` — mesh-only, no gaussian, export print-ready

**Files:**
- Modify: `modules/generation_pipeline.py` (`generate_3d_model`, ≈266–387)

**Interfaces:**
- Consumes: `export_print_ready(glb_path, out_dir)` from Task 1.
- Produces: `generate_3d_model(...) -> tuple[str, str | None, str | None]` returning `(glb_path, print_ready_glb_path, stl_path)`. Removes `sample_video` and `use_simple_glb` parameters.

- [ ] **Step 1: Read the function**

Run: `sed -n '260,390p' modules/generation_pipeline.py` and `grep -n "def generate_3d_model\|trellis_pipeline.run\|render_video\|gaussian_splat\|to_glb_simple\|to_glb_new\|sample_video\|use_simple_glb\|return video_path" modules/generation_pipeline.py`

- [ ] **Step 2: Edit `trellis_pipeline.run(...)` to decode mesh only**

Add `formats=['mesh']` to the `self.trellis_pipeline.run(...)` call (alongside the existing `seed=...`, `sparse_structure_sampler_params=...`, `slat_sampler_params=...`).

- [ ] **Step 3: Remove the gaussian video block and force the simple GLB**

Delete the `if sample_video: ... render_utils.render_video(outputs['gaussian'][0]) ... else: video_path = ""` block. Remove the `if use_simple_glb:` / `else:` branching so only the `to_glb_simple(outputs['mesh'][0], simplify=0.95, color=(180, 180, 220), fill_holes=True, remove_floating=True)` path remains (delete the `to_glb_new(outputs['gaussian'][0], ...)` branch). Keep `glb.export(glb_path)`.

- [ ] **Step 4: Export the print-ready mesh and change the return**

Immediately after `glb.export(glb_path)`, add:

```python
        # Print-ready watertight remesh (env-gated) -> GLB (normals viewer) + STL (download).
        from modules.simple_stl_converter import export_print_ready
        print_ready_glb_path, stl_path = export_print_ready(glb_path, temp_dir)
```

(`temp_dir` is the directory already used for `glb_path` — confirm its variable name in Step 1 and reuse it.)

Change the function's `return` to:

```python
        return glb_path, print_ready_glb_path, stl_path
```

- [ ] **Step 5: Remove `sample_video` and `use_simple_glb` from the signature**

Delete those two parameters from `def generate_3d_model(self, ...)`. (Confirm via Step 1 they are no longer referenced in the body.)

- [ ] **Step 6: Verify (no GPU run)**

```bash
python -c "import ast; ast.parse(open('modules/generation_pipeline.py').read()); print('parse-ok')"
grep -n "outputs\['gaussian'\]\|render_video\|to_glb_new\|sample_video\|use_simple_glb" modules/generation_pipeline.py || echo "gaussian/video refs removed"
```
Expected: `parse-ok`, and no remaining `gaussian`/`render_video`/`to_glb_new`/`sample_video`/`use_simple_glb` references in `generate_3d_model`.

- [ ] **Step 7: Commit**

```bash
git add modules/generation_pipeline.py
git commit -m "feat(print-ready): mesh-only generation + export print-ready GLB/STL; drop gaussian video" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Streamlit app — side-by-side viewers, drop gaussian

**Files:**
- Modify: `streamlit-app.py` (local `generate_3d_model` wrapper ≈213–236; results display ≈765–785; gaussian setting ≈87–89; STL prep ≈305)

**Interfaces:**
- Consumes: `generate_3d_model(...) -> (glb_path, print_ready_glb_path, stl_path)` (Task 3); `create_3d_viewer_html(path, normals=True)` (Task 2); `export_print_ready` already ran in generation.

- [ ] **Step 1: Read the call/display sites**

Run: `grep -n "generate_3d_model\|video_path\|st.video\|create_3d_viewer_html\|components.html\|get_gaussian_rendering_setting\|use_gaussian_rendering\|convert_glb_to_stl\|sample_video\|use_simple_glb\|download_button\|st.columns" streamlit-app.py`

- [ ] **Step 2: Update the generation wrapper + call site**

In the local `generate_3d_model` wrapper (≈213–236): drop the `sample_video`/`use_simple_glb` args passed to `generation_pipeline.generate_3d_model(...)`; return the new 3-tuple `(glb_path, print_ready_glb_path, stl_path)`. At the call site (≈751–754) change
`st.session_state.video_path, glb_path = generate_3d_model(...)`
to
`glb_path, print_ready_glb_path, stl_path = generate_3d_model(...)`
and store `print_ready_glb_path` and `stl_path` in `st.session_state`.

- [ ] **Step 3: Replace the results display with two columns**

Replace the results block (≈765–785, the `if len(video_path)==0:` / else with `st.video`) with:

```python
        col_left, col_right = st.columns(2)
        with col_left:
            st.caption("🎨 Colored model")
            components.html(create_3d_viewer_html(glb_path), height=550)
        with col_right:
            if print_ready_glb_path:
                st.caption("🧱 Print-ready (normals)")
                components.html(create_3d_viewer_html(print_ready_glb_path, normals=True), height=550)
            else:
                st.caption("🧱 Print-ready")
                st.info("No printable mesh could be generated for this model.")
```

(Use the session-state names from Step 2 if the display reads from session state rather than locals.)

- [ ] **Step 4: Use the generated STL; remove lazy conversion + gaussian setting**

In `prepare_3d_model_for_printing` (≈305), use the `stl_path` produced during generation instead of calling `convert_glb_to_stl(glb_output, ...)` (the STL already exists; copy/move it to the output dir / gallery as the existing code does for the GLB). Remove `get_gaussian_rendering_setting()` (≈87–89) and any `use_gaussian_rendering` reads/branches.

- [ ] **Step 5: Verify**

```bash
python -c "import ast; ast.parse(open('streamlit-app.py').read()); print('parse-ok')"
grep -n "st.video\|get_gaussian_rendering_setting\|use_gaussian_rendering\|sample_video\|use_simple_glb" streamlit-app.py || echo "gaussian/video refs removed"
```
Expected: `parse-ok`; no `st.video`/gaussian refs remain.

- [ ] **Step 6: Commit**

```bash
git add streamlit-app.py
git commit -m "feat(print-ready): streamlit side-by-side colored+normals viewers; drop gaussian video" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Gradio app — side-by-side viewers, drop gaussian

**Files:**
- Modify: `gradio-app.py` (`select_and_generate_3d_for_index` ≈270–355; UI components ≈551–555; `image_gallery.select(...)` outputs ≈700–719; `prepare_for_download` ≈390; gaussian setting ≈55–57, ≈116, ≈315–316)

**Interfaces:**
- Consumes: `generate_3d_model(...) -> (glb_path, print_ready_glb_path, stl_path)` (Task 3); `create_3d_viewer_html(path, normals=True)` (Task 2).

- [ ] **Step 1: Read the sites**

Run: `grep -n "generate_3d_model\|select_and_generate_3d_for_index\|create_3d_viewer_html\|iFrame\|gr.Video\|video_output\|model_output\|get_gaussian_rendering_setting\|use_gaussian_rendering\|image_gallery.select\|prepare_for_download\|convert_glb_to_stl\|sample_video\|use_simple_glb" gradio-app.py`

- [ ] **Step 2: Update generation handler**

In `select_and_generate_3d_for_index` (≈270–355): drop `sample_video`/`use_simple_glb` args; unpack `(glb_path, print_ready_glb_path, stl_path)`; build two HTMLs:
```python
            colored_html = create_3d_viewer_html(glb_path, container_height="550px")
            normals_html = (create_3d_viewer_html(print_ready_glb_path, normals=True, container_height="550px")
                            if print_ready_glb_path else
                            "<div style='height:550px;display:flex;align-items:center;justify-content:center;"
                            "color:#888'>No printable mesh available</div>")
```
Return a single (non-gaussian) tuple that includes both `colored_html` and `normals_html` (+ status + the paths the download step needs). Remove the gaussian/`gr.Video` return branch.

- [ ] **Step 3: Replace UI components with two iFrames in a Row**

Where `model_output = iFrame(height=550)` and `video_output = gr.Video(...)` are defined (≈551–555), replace with:
```python
        with gr.Row():
            model_output_colored = iFrame(height=550)
            model_output_normals = iFrame(height=550)
```
Delete `video_output`.

- [ ] **Step 4: Update `image_gallery.select(...)` wiring**

In the `.select(handle_gallery_select, ...)` outputs (≈700–719), collapse the two gaussian/non-gaussian branches into one that maps the handler's returns to `[model_output_colored, model_output_normals, ...status..., ...]`. Keep the `.then(prepare_for_download, ...)` chain.

- [ ] **Step 5: Use the generated STL; drop gaussian setting**

In `prepare_for_download` (≈390), use the `stl_path` from generation instead of `convert_glb_to_stl(glb_output, ...)`. Remove `get_gaussian_rendering_setting()` (≈55–57), the session `'use_gaussian_rendering'` flag (≈116), and the `sample_video`/`use_simple_glb` derivation (≈315–316).

- [ ] **Step 6: Verify**

```bash
python -c "import ast; ast.parse(open('gradio-app.py').read()); print('parse-ok')"
grep -n "gr.Video\|video_output\|get_gaussian_rendering_setting\|use_gaussian_rendering\|sample_video\|use_simple_glb" gradio-app.py || echo "gaussian/video refs removed"
```
Expected: `parse-ok`; no gaussian/video refs remain.

- [ ] **Step 7: Commit**

```bash
git add gradio-app.py
git commit -m "feat(print-ready): gradio side-by-side colored+normals viewers; drop gaussian video" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Remove `USE_GAUSSIAN_RENDERING` from the run script + manual end-to-end

**Files:**
- Modify: `run_streamlit_app.sh`

- [ ] **Step 1: Remove the env line**

Run: `grep -n "USE_GAUSSIAN_RENDERING" run_streamlit_app.sh`
Delete the `export USE_GAUSSIAN_RENDERING=...` line (and its comment).

- [ ] **Step 2: Verify the script still parses**

Run: `bash -n run_streamlit_app.sh && echo "bash-syntax-ok"`
Expected: `bash-syntax-ok`.

- [ ] **Step 3: Commit**

```bash
git add run_streamlit_app.sh
git commit -m "chore(print-ready): drop obsolete USE_GAUSSIAN_RENDERING from run script" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 4: Manual end-to-end (human-run, both apps)**

On the GPU box: launch each app, generate a model, and confirm: (a) two side-by-side viewers appear — colored GLB (left) and print-ready normals (right); (b) no video; (c) the right viewer matches the downloaded STL; (d) STL and GLB downloads work; (e) generation logs `🧱 Applied watertight print remesh`. Record the result in the PR/commit notes. (Not automatable here — apps need a display + GPU + a conditioning image.)

---

## Notes
- The print-ready GLB is large (~340k faces, base64-embedded in the viewer HTML). Acceptable; a manifold-preserving decimation is a possible future optimization (out of scope; would need re-validation per `investigations/mesh_repair/REPORT.md`).
- `generate_3d_model` now always pays the ~9 s remesh; it no longer renders the 300-frame gaussian video, so net time is roughly a wash with less VRAM.
