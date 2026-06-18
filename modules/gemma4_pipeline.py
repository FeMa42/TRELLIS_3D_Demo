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


# --- model-driven (Gemma 4) ------------------------------------------------
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
