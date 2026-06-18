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
