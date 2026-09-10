import numpy as np
import pytest
import torch
from PIL import Image

from vlm.perception import PerceptionModule


class _FakeInputs(dict):
    @property
    def input_ids(self):
        return None

    def to(self, _device):
        return self


class _FakeProcessor:
    def __call__(self, **_kwargs):
        return _FakeInputs()

    def post_process_grounded_object_detection(self, *_args, **_kwargs):
        return [{
            "boxes": torch.tensor([
                [0.0, 0.0, 8.0, 8.0],
                [10.0, 0.0, 18.0, 8.0],
            ]),
            "scores": torch.tensor([0.9, 0.8]),
        }]


class _FakeModel:
    def __call__(self, **_kwargs):
        return object()


def _perception_with_two_detections():
    perception = PerceptionModule()
    perception._processor = _FakeProcessor()
    perception._model = _FakeModel()
    return perception


def _get_pose(perception, excluded_positions):
    return perception.get_pose(
        "red_cup",
        Image.new("RGB", (20, 10)),
        np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]]),
        np.eye(4),
        depth_image=np.full((10, 20), 1000, dtype=np.uint16),
        excluded_positions=excluded_positions,
        exclusion_radius=0.1,
    )


def test_get_pose_skips_detection_at_handled_position():
    perception = _perception_with_two_detections()

    pose = _get_pose(perception, [(0.4, 0.4, 0.2)])

    assert pose is not None
    assert pose["x"] == pytest.approx(1.4)
    assert pose["y"] == pytest.approx(0.4)
    assert perception._last_detection["box"] == [10.0, 0.0, 18.0, 8.0]


def test_get_pose_returns_none_when_every_detection_is_handled():
    perception = _perception_with_two_detections()

    pose = _get_pose(perception, [(0.4, 0.4), (1.4, 0.4)])

    assert pose is None
    assert perception._last_detection is None
