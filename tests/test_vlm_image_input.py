"""
Tests for the image-handling side of VLMPlanner.
No model weights loaded — only PIL loading and message construction are tested.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from pathlib import Path
from PIL import Image

from vlm.planner import VLMPlanner


def _planner() -> VLMPlanner:
    """Return a VLMPlanner without loading weights."""
    p = VLMPlanner.__new__(VLMPlanner)
    p.model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    p._model = None
    p._processor = None
    return p


# ── _to_pil ────────────────────────────────────────────────────────────────────

def test_to_pil_from_path(synthetic_scene_image):
    """Loading an image from a file path returns an RGB PIL Image."""
    planner = _planner()
    img = planner._to_pil(synthetic_scene_image)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size == (224, 224)


def test_to_pil_from_string_path(synthetic_scene_image):
    """Accepts a string path as well as a pathlib.Path."""
    planner = _planner()
    img = planner._to_pil(str(synthetic_scene_image))
    assert isinstance(img, Image.Image)


def test_to_pil_passthrough_pil_image(synthetic_pil_image):
    """A PIL Image passed directly is returned unchanged (no re-loading)."""
    planner = _planner()
    result = planner._to_pil(synthetic_pil_image)
    assert result is synthetic_pil_image


def test_to_pil_converts_to_rgb(tmp_path):
    """RGBA or grayscale images are converted to RGB."""
    rgba_img = Image.new("RGBA", (64, 64), color=(255, 0, 0, 128))
    path = tmp_path / "rgba.png"
    rgba_img.save(path)

    planner = _planner()
    result = planner._to_pil(path)
    assert result.mode == "RGB"


def test_to_pil_missing_file_raises():
    planner = _planner()
    with pytest.raises(FileNotFoundError):
        planner._to_pil("/nonexistent/path/image.jpg")


# ── _build_messages ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = "You are a robot task planner."


def test_build_messages_structure(synthetic_pil_image):
    """Message list has exactly system + user roles."""
    planner = _planner()
    msgs = planner._build_messages(SYSTEM_PROMPT, "pick the cup", [synthetic_pil_image])

    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


def test_build_messages_system_content(synthetic_pil_image):
    planner = _planner()
    msgs = planner._build_messages(SYSTEM_PROMPT, "pick the cup", [synthetic_pil_image])
    assert msgs[0]["content"] == SYSTEM_PROMPT


def test_build_messages_user_has_image_and_text(synthetic_pil_image):
    """User message contains one image block followed by one text block."""
    planner = _planner()
    msgs = planner._build_messages(SYSTEM_PROMPT, "pick the cup", [synthetic_pil_image])

    user_content = msgs[1]["content"]
    types = [block["type"] for block in user_content]
    assert types == ["image", "text"]


def test_build_messages_text_matches_command(synthetic_pil_image):
    planner = _planner()
    command = "place the red cup on the shelf"
    msgs = planner._build_messages(SYSTEM_PROMPT, command, [synthetic_pil_image])

    text_blocks = [b for b in msgs[1]["content"] if b["type"] == "text"]
    assert len(text_blocks) == 1
    assert text_blocks[0]["text"] == command


def test_build_messages_multiple_images(synthetic_pil_image):
    """Multiple images each appear as a separate image block before the text."""
    planner = _planner()
    imgs = [synthetic_pil_image, synthetic_pil_image]  # two images
    msgs = planner._build_messages(SYSTEM_PROMPT, "pick the cup", imgs)

    user_content = msgs[1]["content"]
    types = [block["type"] for block in user_content]
    assert types == ["image", "image", "text"]


def test_build_messages_image_block_contains_pil(synthetic_pil_image):
    """Each image block stores the PIL Image object under the 'image' key."""
    planner = _planner()
    msgs = planner._build_messages(SYSTEM_PROMPT, "cmd", [synthetic_pil_image])

    image_blocks = [b for b in msgs[1]["content"] if b["type"] == "image"]
    assert len(image_blocks) == 1
    assert isinstance(image_blocks[0]["image"], Image.Image)
