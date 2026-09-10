import numpy as np
from PIL import Image

from scripts.run_loop_host import (
    _annotate_handled_objects,
    _format_ros_double,
    _placed_object_position,
    _update_iteration_debug,
    _world_label,
)


def test_format_ros_double_keeps_ros_parameter_type():
    assert _format_ros_double(30.0) == "30.000000"
    assert _format_ros_double(20) == "20.000000"


def test_annotate_handled_objects_draws_cross_with_direct_calibration():
    image = Image.new("RGB", (100, 100), "black")
    K = np.array([[50.0, 0.0, 50.0], [0.0, 50.0, 50.0], [0.0, 0.0, 1.0]])
    cam_to_base = np.eye(4)

    result = _annotate_handled_objects(
        image,
        {"cube": (0.0, 0.0, 1.0)},
        ".",
        camera_matrix=K,
        cam_to_base=cam_to_base,
    )

    assert result.getpixel((50, 50)) == (0, 220, 0)
    assert result.getpixel((45, 50)) == (0, 220, 0)


def test_placed_object_position_uses_original_name_and_gazebo_fallback():
    result = _placed_object_position(
        original_location="table",
        resolved_location="work_surface",
        estimates={},
        poses={},
        gazebo_poses={"table": {"x": 0.65, "y": -0.10}},
    )

    assert result == (0.45, -0.10)


def test_placed_object_position_prefers_resolved_perception_pose_with_height():
    result = _placed_object_position(
        original_location="green tray",
        resolved_location="target_tray",
        estimates={"target_tray": (0.31, 0.22)},
        poses={
            "target_tray": {
                "position": {"x": 0.31, "y": 0.22, "z": 0.35},
            },
        },
        gazebo_poses={"green tray": {"x": 0.9, "y": 0.9}},
    )

    assert result == (0.31, 0.22, 0.35)


def test_world_label_does_not_report_gazebo_world_for_real_robot():
    class Args:
        real_ros2 = True
        world = "office"

    assert _world_label(Args()) == "real_ros2"


def test_update_iteration_debug_preserves_pre_step_state(tmp_path):
    iteration = tmp_path / "iter_01"
    iteration.mkdir()
    path = iteration / "debug.json"
    path.write_text('{"world":"real_ros2","robot_state_before_step":{"available":true}}')

    _update_iteration_debug(iteration, {"execution": {"attempted": True}})

    import json

    result = json.loads(path.read_text())
    assert result["world"] == "real_ros2"
    assert result["robot_state_before_step"]["available"] is True
    assert result["execution"]["attempted"] is True
