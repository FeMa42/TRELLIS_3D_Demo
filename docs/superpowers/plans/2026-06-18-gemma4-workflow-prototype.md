# Gemma 4 Workflow Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prototype, in a Jupyter notebook, a Gemma 4–driven loop that writes an optimized 3D prompt and uses Gemma 4 vision to pick the best candidate image for TRELLIS.

**Architecture:** Pure, testable helpers (JSON parse, image grid, prompt clean) plus two model-driven functions (`gemma4_generate_prompt`, `gemma4_rank_images`) live in `modules/gemma4_pipeline.py` (the eventual app-port module). A notebook imports them and demonstrates the end-to-end loop (concept → prompt → 4 images → pick → TRELLIS → GLB), displaying every intermediate result.

**Tech Stack:** Python 3.10, transformers 5.12.1 (Gemma 4), diffusers (Qwen-Image), TRELLIS, PIL, pytest. Conda env `trellis_tf5_test`.

## Global Constraints

- Conda env: `trellis_tf5_test` (transformers 5.12.1, huggingface_hub 1.19.0). Python at `~/miniconda3/envs/trellis_tf5_test/bin/python`.
- Gemma 4 model id: `google/gemma-4-E4B-it` (multimodal, instruction-tuned). Exposed as a single top-level `MODEL_ID` variable.
- Image generator: `Qwen/Qwen-Image`, 4 candidates, 512x512. Exposed as `IMAGE_MODEL`.
- Runtime env vars (set in notebook cell 1 before importing trellis): `ATTN_BACKEND=xformers`, `SPCONV_ALGO=native`, `CUDA_HOME=/usr/local/cuda-12.9` (prepend `$CUDA_HOME/bin` to `PATH`).
- No CPU offload, `torch.bfloat16`, full precision (ample VRAM).
- Ranking output contract: `{"best_idx": int, "scores": List[float] (len N, 0-10), "rationale": str}` with `0 <= best_idx < N`.
- Repo root: `/home/damian/gamescom/TRELLIS_3D_Demo`. Run all commands from there.
- Do not modify the apps or the production env. New files only (plus the notebook).

---

## File Structure

- Create: `modules/gemma4_pipeline.py` — pure helpers + `load_gemma4`, `gemma4_generate_prompt`, `gemma4_rank_images`. One responsibility: the Gemma 4 prompt-writer + image-judge, with no notebook/UI code so it ports directly into the app.
- Create: `tests/test_gemma4_pipeline.py` — pytest for the pure helpers (no model load).
- Create: `gemma4_workflow_prototype.ipynb` — the demo/eval harness.

---

### Task 1: Pure helpers (`_clean_prompt`, `_parse_ranking_json`, `_make_grid`)

These have no model dependency and are fully unit-tested first.

**Files:**
- Create: `modules/gemma4_pipeline.py`
- Test: `tests/test_gemma4_pipeline.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `_clean_prompt(text: str) -> str` — strips code fences, leading "Prompt:"/quotes, surrounding whitespace.
  - `_parse_ranking_json(text: str, n: int) -> dict` — returns `{"best_idx": int, "scores": List[float], "rationale": str}`; tolerant of fenced/`json`-prefixed output; clamps `best_idx` into `[0, n)`; pads/truncates `scores` to length `n`; on total failure returns `{"best_idx": 0, "scores": [0.0]*n, "rationale": "parse-failed"}`.
  - `_make_grid(images: List["PIL.Image.Image"], cols: int = 2) -> "PIL.Image.Image"` — composites images into a labeled grid (index drawn top-left of each cell).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gemma4_pipeline.py`:

```python
from PIL import Image
from modules.gemma4_pipeline import _clean_prompt, _parse_ranking_json, _make_grid


def test_clean_prompt_strips_fences_and_labels():
    assert _clean_prompt('```\nPrompt: a red dragon\n```') == "a red dragon"
    assert _clean_prompt('"a blue car"') == "a blue car"
    assert _clean_prompt("  plain text  ") == "plain text"


def test_parse_ranking_json_plain():
    out = _parse_ranking_json('{"best_index": 2, "scores": [1,2,9,3], "reason": "c"}', 4)
    assert out["best_idx"] == 2
    assert out["scores"] == [1.0, 2.0, 9.0, 3.0]
    assert out["rationale"] == "c"


def test_parse_ranking_json_fenced_and_clamped():
    out = _parse_ranking_json('```json\n{"best_index": 9, "scores": [1,2], "reason": "x"}\n```', 4)
    assert out["best_idx"] == 3            # clamped into [0, 4)
    assert len(out["scores"]) == 4         # padded to n


def test_parse_ranking_json_garbage_falls_back():
    out = _parse_ranking_json("the model rambled with no json", 3)
    assert out["best_idx"] == 0
    assert out["scores"] == [0.0, 0.0, 0.0]
    assert out["rationale"] == "parse-failed"


def test_make_grid_dimensions():
    imgs = [Image.new("RGB", (64, 64), c) for c in ("red", "green", "blue", "white")]
    grid = _make_grid(imgs, cols=2)
    assert grid.size == (128, 128)         # 2x2 of 64px cells
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/damian/gamescom/TRELLIS_3D_Demo && ~/miniconda3/envs/trellis_tf5_test/bin/python -m pytest tests/test_gemma4_pipeline.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'modules.gemma4_pipeline'` (or import error).

- [ ] **Step 3: Write minimal implementation**

Create `modules/gemma4_pipeline.py`:

```python
"""Gemma 4 prompt-writer + multimodal image-judge for the 3D pipeline.

Pure helpers are unit-tested; the model-driven functions are smoke-validated in
the prototype notebook. No notebook/UI code here so this ports into the app.
"""
import json
import re
from typing import List, Dict

from PIL import Image, ImageDraw


def _clean_prompt(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
    t = re.sub(r"\n?```$", "", t)
    t = re.sub(r"^\s*(prompt|enhanced prompt)\s*:\s*", "", t, flags=re.IGNORECASE)
    t = t.strip().strip('"').strip("'").strip()
    return t


def _parse_ranking_json(text: str, n: int) -> Dict:
    fallback = {"best_idx": 0, "scores": [0.0] * n, "rationale": "parse-failed"}
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return fallback
    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError:
        return fallback
    try:
        best = int(raw.get("best_index", raw.get("best_idx", 0)))
    except (TypeError, ValueError):
        best = 0
    best = max(0, min(best, n - 1))
    scores = raw.get("scores", []) or []
    scores = [float(s) for s in scores][:n]
    scores += [0.0] * (n - len(scores))
    rationale = str(raw.get("reason", raw.get("rationale", ""))).strip()
    return {"best_idx": best, "scores": scores, "rationale": rationale}


def _make_grid(images: List[Image.Image], cols: int = 2) -> Image.Image:
    if not images:
        raise ValueError("no images to grid")
    cells = [im.convert("RGB") for im in images]
    cw = max(im.width for im in cells)
    ch = max(im.height for im in cells)
    rows = (len(cells) + cols - 1) // cols
    grid = Image.new("RGB", (cols * cw, rows * ch), "black")
    draw = ImageDraw.Draw(grid)
    for i, im in enumerate(cells):
        x, y = (i % cols) * cw, (i // cols) * ch
        grid.paste(im, (x, y))
        draw.text((x + 4, y + 4), str(i), fill="yellow")
    return grid
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/damian/gamescom/TRELLIS_3D_Demo && ~/miniconda3/envs/trellis_tf5_test/bin/python -m pytest tests/test_gemma4_pipeline.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
cd /home/damian/gamescom/TRELLIS_3D_Demo
git add modules/gemma4_pipeline.py tests/test_gemma4_pipeline.py
git commit -m "feat(gemma4): add pure helpers for prompt cleaning, ranking JSON parse, image grid"
```

---

### Task 2: Model loader + `gemma4_generate_prompt`

Model-driven; validated by a smoke check (the model is too heavy for CI, so this is a real-load sanity assert, not a mocked unit test).

**Files:**
- Modify: `modules/gemma4_pipeline.py` (append)

**Interfaces:**
- Consumes: `_clean_prompt` (Task 1).
- Produces:
  - `load_gemma4(model_id: str = "google/gemma-4-E4B-it") -> tuple(model, processor)` — loads `Gemma4ForConditionalGeneration` + `AutoProcessor`, `torch.bfloat16`, `device_map="cuda"`.
  - `DEFAULT_3D_SYSTEM: str` — the 3D prompt-writer system prompt.
  - `gemma4_generate_prompt(model, processor, concept: str, system_prompt: str = DEFAULT_3D_SYSTEM, max_new_tokens: int = 200) -> str` — returns the cleaned enhanced prompt.

- [ ] **Step 1: Append loader, system prompt, and generator**

Append to `modules/gemma4_pipeline.py`:

```python
import torch
from transformers import AutoProcessor, Gemma4ForConditionalGeneration

DEFAULT_3D_SYSTEM = (
    "You write image-generation prompts for single 3D objects that will be "
    "converted to 3D models. Output ONLY the prompt, no preamble or commentary. "
    "Describe one object on a neutral plain background, centered, 3/4 view, soft "
    "studio lighting, no harsh shadows. Add render keywords: 'high quality 3D "
    "render', 'stylized 3D model', 'optimized for 3D printing'. Keep it safe-for-work."
)


def load_gemma4(model_id: str = "google/gemma-4-E4B-it"):
    processor = AutoProcessor.from_pretrained(model_id)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    return model, processor


def _generate(model, processor, messages, max_new_tokens):
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    in_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(out[0][in_len:], skip_special_tokens=True)


def gemma4_generate_prompt(model, processor, concept, system_prompt=DEFAULT_3D_SYSTEM,
                           max_new_tokens=200):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": concept}]},
    ]
    return _clean_prompt(_generate(model, processor, messages, max_new_tokens))
```

> Note: `apply_chat_template(..., return_dict=True)` is the Gemma 3/4 multimodal
> interface. If the installed model card documents a different message schema,
> adjust `messages`/`_generate` here — confirm against the model card on first load
> (Task 4, Step 1). The signatures above stay the same.

- [ ] **Step 2: Smoke-validate the import path (no model download)**

Run: `cd /home/damian/gamescom/TRELLIS_3D_Demo && ~/miniconda3/envs/trellis_tf5_test/bin/python -c "from modules.gemma4_pipeline import load_gemma4, gemma4_generate_prompt, DEFAULT_3D_SYSTEM; print('imports OK')"`
Expected: `imports OK` (verifies `Gemma4ForConditionalGeneration` is importable in this env; does NOT download weights).

- [ ] **Step 3: Commit**

```bash
cd /home/damian/gamescom/TRELLIS_3D_Demo
git add modules/gemma4_pipeline.py
git commit -m "feat(gemma4): add model loader and 3D prompt generator"
```

---

### Task 3: `gemma4_rank_images` (multimodal pick-the-best)

Uses the tested helpers; primary path = N separate images, fallback = single 2x2 grid.

**Files:**
- Modify: `modules/gemma4_pipeline.py` (append)

**Interfaces:**
- Consumes: `_parse_ranking_json`, `_make_grid` (Task 1); `_generate` (Task 2).
- Produces:
  - `RANK_QUESTION: str` — the judge instruction (asks for the JSON contract).
  - `gemma4_rank_images(model, processor, prompt: str, images: List[Image.Image], use_grid: bool = False, max_new_tokens: int = 250) -> dict` — returns `{"best_idx", "scores", "rationale"}` (the Global Constraints contract).

- [ ] **Step 1: Append the ranking function**

Append to `modules/gemma4_pipeline.py`:

```python
RANK_QUESTION = (
    "You are selecting the best image to convert into a single 3D model. "
    "Given the target prompt and {n} candidate images (indexed from 0), pick the "
    "one with: a single clearly-separated object, neutral/plain background, full "
    "object visible, clean silhouette, minimal occlusion, even lighting. "
    'Respond ONLY with JSON: {{"best_index": <int>, "scores": [<float 0-10> per '
    'image in order], "reason": "<one short sentence>"}}. Target prompt: "{prompt}".'
)


def gemma4_rank_images(model, processor, prompt, images, use_grid=False,
                       max_new_tokens=250):
    n = len(images)
    question = RANK_QUESTION.format(n=n, prompt=prompt)
    if use_grid:
        grid = _make_grid(images, cols=2)
        content = [
            {"type": "image", "image": grid},
            {"type": "text",
             "text": question + " The image is a labeled grid; indices are drawn "
                     "in each cell (row-major)."},
        ]
    else:
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": content}]
    text = _generate(model, processor, messages, max_new_tokens)
    return _parse_ranking_json(text, n)
```

- [ ] **Step 2: Smoke-validate import**

Run: `cd /home/damian/gamescom/TRELLIS_3D_Demo && ~/miniconda3/envs/trellis_tf5_test/bin/python -c "from modules.gemma4_pipeline import gemma4_rank_images, RANK_QUESTION; print('rank import OK')"`
Expected: `rank import OK`.

- [ ] **Step 3: Re-run the helper tests (ensure no regression)**

Run: `cd /home/damian/gamescom/TRELLIS_3D_Demo && ~/miniconda3/envs/trellis_tf5_test/bin/python -m pytest tests/test_gemma4_pipeline.py -q`
Expected: PASS (5 passed).

- [ ] **Step 4: Commit**

```bash
cd /home/damian/gamescom/TRELLIS_3D_Demo
git add modules/gemma4_pipeline.py
git commit -m "feat(gemma4): add multimodal image-ranking with grid fallback"
```

---

### Task 4: Prototype notebook (`gemma4_workflow_prototype.ipynb`)

The eval harness: loads models once, demonstrates each step, runs the full loop.
Build it by writing a `.py` cell-script and converting, or author cells directly.
Each cell below is one notebook cell.

**Files:**
- Create: `gemma4_workflow_prototype.ipynb`

**Interfaces:**
- Consumes: `load_gemma4`, `gemma4_generate_prompt`, `gemma4_rank_images`, `_make_grid` from `modules.gemma4_pipeline`.
- Produces: a `run_concept(concept)` driver (notebook-local).

- [ ] **Step 1: Cell 1 — setup + load models (and confirm the Gemma 4 message schema)**

```python
import os
os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPCONV_ALGO"] = "native"
os.environ["CUDA_HOME"] = "/usr/local/cuda-12.9"
os.environ["PATH"] = "/usr/local/cuda-12.9/bin:" + os.environ["PATH"]
import torch
from PIL import Image
from modules.gemma4_pipeline import (
    load_gemma4, gemma4_generate_prompt, gemma4_rank_images, _make_grid,
)

MODEL_ID = "google/gemma-4-E4B-it"
IMAGE_MODEL = "Qwen/Qwen-Image"

gemma, processor = load_gemma4(MODEL_ID)
# Sanity: confirm the multimodal chat-template path works end-to-end on a tiny call.
print(gemma4_generate_prompt(gemma, processor, "a small toy car"))
```

Expected: prints a single enhanced prompt line (no preamble). If this errors on the
message schema, fix `_generate`/`messages` in `modules/gemma4_pipeline.py` per the
model card, re-run.

- [ ] **Step 2: Cell 2 — load the Qwen-Image generator**

```python
from diffusers import DiffusionPipeline

qwen = DiffusionPipeline.from_pretrained(IMAGE_MODEL, torch_dtype=torch.bfloat16).to("cuda")

def qwen_generate(prompt, n=4, steps=40, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = qwen([prompt] * n, height=512, width=512, num_inference_steps=steps,
               true_cfg_scale=5.5, generator=g)
    return list(out.images)
```

Expected: defines `qwen_generate`; downloads Qwen-Image weights on first run.

- [ ] **Step 3: Cell 3 — prompt-gen demo (before/after vs. the fixed suffix)**

```python
concept = "a pretzel-copter: a helicopter whose cockpit is shaped like a pretzel"
suffix_prompt = concept + " Render of high quality 3D model on neutral background, optimized for 3D printing."
gemma_prompt = gemma4_generate_prompt(gemma, processor, concept)
print("SUFFIX:", suffix_prompt)
print("GEMMA :", gemma_prompt)
```

Expected: the Gemma prompt is a richer, single-object 3D description.

- [ ] **Step 4: Cell 4 — generate candidates and show the grid**

```python
images = qwen_generate(gemma_prompt, n=4, seed=1337)
_make_grid(images, cols=2)   # displays the 2x2 grid inline
```

Expected: 4 candidate images shown as a labeled 2x2 grid.

- [ ] **Step 5: Cell 5 — rank with Gemma 4 and show the pick**

```python
rank = gemma4_rank_images(gemma, processor, gemma_prompt, images, use_grid=False)
print("scores:", rank["scores"], "| best:", rank["best_idx"])
print("why:", rank["rationale"])
best_image = images[rank["best_idx"]]
best_image
```

Expected: per-image scores, the chosen index + one-line rationale, and the winning
image displayed. (If multi-image input errors, set `use_grid=True` and re-run.)

- [ ] **Step 6: Cell 6 — TRELLIS on the winner → GLB**

```python
from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline
from trellis.utils.postprocessing_utils import to_glb_simple

trellis = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
trellis.to("cuda:0")
out = trellis.run(best_image, seed=1)
glb = to_glb_simple(out["mesh"][0], simplify=0.95, color=(180, 180, 220), remove_floating=True, verbose=False)
glb.export("gemma4_prototype_out.glb")
print("GLB written:", out.keys())
```

Expected: `GLB written: dict_keys([...])` and `gemma4_prototype_out.glb` on disk.

- [ ] **Step 7: Cell 7 — `run_concept` driver for batch experimentation**

```python
def run_concept(concept, n=4, seed=1337):
    prompt = gemma4_generate_prompt(gemma, processor, concept)
    imgs = qwen_generate(prompt, n=n, seed=seed)
    rank = gemma4_rank_images(gemma, processor, prompt, imgs)
    display(_make_grid(imgs, cols=2))
    print(concept, "->", prompt)
    print("pick", rank["best_idx"], "scores", rank["scores"], "|", rank["rationale"])
    return prompt, imgs, rank

for c in ["a red dragon curled up", "a vintage camera", "a mushroom house"]:
    run_concept(c)
```

Expected: each concept runs the prompt→images→pick loop with inline display.

- [ ] **Step 8: Commit the notebook (outputs cleared)**

```bash
cd /home/damian/gamescom/TRELLIS_3D_Demo
~/miniconda3/envs/trellis_tf5_test/bin/jupyter nbconvert --clear-output --inplace gemma4_workflow_prototype.ipynb
git add gemma4_workflow_prototype.ipynb
git commit -m "feat(gemma4): add end-to-end workflow prototype notebook"
```

---

## Self-Review

**1. Spec coverage:**
- Prompt-gen replacement → Task 2 (`gemma4_generate_prompt`) + Cell 3. ✓
- Multimodal image-selection replacing ImageReward → Task 3 (`gemma4_rank_images`) + Cell 5. ✓
- End-to-end loop in a notebook → Task 4 (Cells 1-7). ✓
- Functions liftable into `modules/gemma4_pipeline.py` → created in Tasks 1-3 (already in `modules/`). ✓
- Single multi-image comparison call + grid fallback → Task 3 (`use_grid`). ✓
- E4B-it, Qwen-Image, `trellis_tf5_test`, no offload, env vars → Global Constraints + Cell 1-2. ✓
- Success criteria (before/after prompt, sensible pick, no-fiddle loop) → Cells 3, 5, 7. ✓
- Risks (multi-image, JSON drift, downloads) → grid fallback (Task 3), tolerant parser (Task 1), download notes (Cells 1-2). ✓

**2. Placeholder scan:** No TBD/TODO; every code step shows full code; the one "adjust per model card" note is a concrete contingency with the surrounding signatures fixed, not a placeholder. ✓

**3. Type consistency:** `_parse_ranking_json` returns `best_idx`/`scores`/`rationale`; `gemma4_rank_images` returns the same dict; notebook reads `rank["best_idx"]`/`["scores"]`/`["rationale"]`. `load_gemma4` returns `(model, processor)`; all callers pass `(gemma, processor)`. `_make_grid(images, cols)` consistent across module and notebook. ✓
