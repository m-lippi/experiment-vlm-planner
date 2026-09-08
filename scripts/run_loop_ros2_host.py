#!/usr/bin/env python3
"""Closed-loop real-robot execution using calibrated ROS 2 RGB-D topics."""

import sys
from pathlib import Path

from run_loop_host import main


if __name__ == "__main__":
    data = Path(__file__).resolve().parent.parent / "data"
    required = ("overview_camera_info.json", "overview_camera_pose.json")
    missing = [name for name in required if not (data / name).exists()]
    if missing and not any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        print(
            "[ERROR] Missing overview calibration: " + ", ".join(missing),
            file=sys.stderr,
        )
        print(
            "Run calibrate_overview_camera_apriltag.py before the real loop.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    main(real_ros2_default=True)
