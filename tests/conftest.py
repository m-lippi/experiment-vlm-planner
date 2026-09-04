"""
Shared pytest fixtures.
"""

import pytest
from PIL import Image


@pytest.fixture
def synthetic_scene_image(tmp_path):
    """
    Minimal 224x224 RGB image for testing PIL loading and message construction.
    Generated programmatically — no external file dependency.
    """
    img = Image.new("RGB", (224, 224), color=(120, 80, 60))
    path = tmp_path / "scene.jpg"
    img.save(path)
    return path


@pytest.fixture
def synthetic_pil_image():
    """A PIL Image directly (no file I/O)."""
    return Image.new("RGB", (224, 224), color=(200, 150, 100))
