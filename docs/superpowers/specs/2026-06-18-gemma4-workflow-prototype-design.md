# Gemma 4 Workflow Prototype — Design

**Date:** 2026-06-18
**Status:** Approved (design)
**Repo:** `/home/damian/gamescom/TRELLIS_3D_Demo`

## Overview

Prototype, in a Jupyter notebook, a Gemma 4–driven workflow that (1) turns a user
concept into an optimized 3D-generation prompt and (2) uses Gemma 4's vision to
pick the best of several candidate images to feed into TRELLIS. The notebook is a
proving ground: validate Gemma 4's prompt quality and image judgment before
porting the logic into the Streamlit/Gradio apps.

## Goals

- Replace the current trivial prompt enhancement (fixed-suffix concatenation in
  `modules/generation_pipeline.py:_create_enhanced_prompt`) with a Gemma 4 prompt
  writer.
- Replace ImageReward (`reward_model.score`) for image selection with a Gemma 4
  multimodal "pick the best image for 3D" judgment.
- Run the full loop end-to-end (concept → prompt → images → pick → TRELLIS → GLB)
  in a notebook, with every intermediate result visible for inspection.
- Write the two Gemma 4 functions so they lift cleanly into a future
  `modules/gemma4_pipeline.py` without rework.

## Non-goals

- Integrating into the Streamlit/Gradio apps (separate follow-up; tracked).
- Upgrading the apps' production env to transformers 5.x (prototype stays isolated
  in `trellis_tf5_test`).
- Fine-tuning Gemma 4 or comparing multiple variants (E4B-it only for now; model id
  is a single variable so a 31B A/B is a later one-line change).
- Replacing the image generator or TRELLIS (reused as-is).

## Workflow

```
concept (str)
  → gemma4_generate_prompt(concept)     # Gemma 4 (text)   → optimized 3D prompt (str)
  → qwen_generate(prompt, n=4)          # Qwen-Image       → List[PIL.Image] (512x512)
  → gemma4_rank_images(prompt, images)  # Gemma 4 (vision) → {best_idx, scores, rationale}
  → trellis.run(best_image)             # TRELLIS          → {mesh, gaussian, ...}
  → to_glb_simple(...)                  #                  → GLB file
```

Each step is its own notebook cell; intermediate artifacts (enhanced prompt, the
4-image grid, the chosen image + scores + rationale) are displayed inline before
the (slower) TRELLIS step runs.

## Components

### `gemma4_generate_prompt(concept, system_prompt=DEFAULT_3D_SYSTEM) -> str`
- Chat-template call to Gemma 4 E4B-it (text only).
- `DEFAULT_3D_SYSTEM` seeds from the existing `gemma_qwen_image.ipynb` system prompt
  (3D-render keywords, neutral background, "only output the prompt", SFW guard),
  adapted for Gemma 4.
- Returns the enhanced prompt string (stripped of any meta-commentary).

### `gemma4_rank_images(prompt, images) -> {"best_idx": int, "scores": List[float], "rationale": str}`
- Single multimodal call: the `prompt` plus all N images.
- Primary path: pass the N images as separate image inputs.
- Fallback path: if E4B-it does not accept multiple images per turn, composite the N
  images into one labeled 2x2 grid image and pass that single image. The function
  supports both and selects based on a capability check / config flag.
- Instructed to return JSON: `{"best_index": int, "scores": [float x N] (0-10, 3D-print
  suitability), "reason": str}`. Parsed with a strict-then-tolerant parser (regex
  fallback for non-JSON drift); on parse failure, fall back to highest-scored or
  index 0 with a logged warning.

### Driver: `run_concept(concept) -> dict`
Chains the four steps; returns prompt, images, ranking result, and GLB path for
batch experimentation over several concepts.

## Notebook structure

`gemma4_workflow_prototype.ipynb` (in repo root):
1. Setup + load Gemma 4 (E4B-it), Qwen-Image, TRELLIS once; `MODEL_ID`,
   `IMAGE_MODEL` as top-level variables.
2. `gemma4_generate_prompt` + example concepts (show before/after vs. fixed suffix).
3. Image generation (4 candidates, shown as a grid).
4. `gemma4_rank_images` + display (winner highlighted, scores + rationale).
5. TRELLIS on the winner → GLB.
6. `run_concept` driver for quick multi-concept runs.

## Environment

- Conda env: `trellis_tf5_test` (transformers 5.12.1, hf_hub 1.19.0 — required for
  Gemma 4; already validated to keep Qwen/FLUX/TRELLIS working, TRELLIS e2e passes).
- No CPU offload, full precision (ample VRAM on the RTX PRO 6000 Blackwell cards).
- Runtime env vars (as for the apps): `ATTN_BACKEND=xformers`, `SPCONV_ALGO=native`,
  `CUDA_HOME=/usr/local/cuda-12.9` on PATH (TRELLIS/nvdiffrast).
- Register a Jupyter kernel for `trellis_tf5_test`, or launch jupyter from it.

## Success criteria

- Gemma 4 prompts produce visibly better candidate images than the fixed-suffix method.
- Gemma 4's pick matches a human's choice, with a sensible rationale, across several
  test concepts.
- The loop runs end-to-end with no manual fiddling — i.e., the two functions are
  ready to port into `modules/`.

## Risks & mitigations

- **Multi-image input**: E4B-it may not accept N separate images per call → 2x2 grid
  fallback built into `gemma4_rank_images`.
- **Structured-output drift**: VLM may not emit clean JSON → strict instruction +
  tolerant parser + safe fallback.
- **First-run downloads**: Gemma 4 E4B (~few GB) + Qwen-Image weights.
- **Env isolation**: prototype stays in `trellis_tf5_test`; production apps remain on
  transformers 4.57.6 until integration is separately approved.

## Follow-up (out of scope here)

Port `gemma4_generate_prompt` and `gemma4_rank_images` into `modules/gemma4_pipeline.py`,
wire into `generation_pipeline.py` (prompt step + replace the `reward_model.score`
selection), and decide the apps' transformers-5.x migration.
