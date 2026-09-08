#!/usr/bin/env python3
"""Shared geometry and image helpers for the AprilTag calibration scripts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import cv2
except ImportError:  # Reported with an actionable message when detection starts.
    cv2 = None


_ENCODINGS = {
    "mono8": (1, False),
    "rgb8": (3, False),
    "r8g8b8": (3, False),
    "bgr8": (3, True),
    "rgba8": (4, False),
    "bgra8": (4, True),
}


def ros_image_to_gray(msg) -> np.ndarray:
    """Decode a ROS Image without cv_bridge, including padded row strides."""
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed")
    encoding = msg.encoding.lower()
    if encoding not in _ENCODINGS:
        raise ValueError(
            f"unsupported image encoding {msg.encoding!r}; expected one of "
            f"{', '.join(sorted(_ENCODINGS))}"
        )
    channels, bgr = _ENCODINGS[encoding]
    row = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)
    pixels = row[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
    if channels == 1:
        return pixels[:, :, 0].copy()
    if channels == 4:
        pixels = pixels[:, :, :3]
    code = cv2.COLOR_BGR2GRAY if bgr else cv2.COLOR_RGB2GRAY
    return cv2.cvtColor(pixels, code)


def camera_matrix(msg) -> tuple[np.ndarray, np.ndarray]:
    """Return K and distortion coefficients from sensor_msgs/CameraInfo."""
    K = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(msg.d, dtype=np.float64)
    if not np.isfinite(K).all() or K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
        raise ValueError("CameraInfo contains invalid focal lengths")
    return K, distortion


class AprilTagDetector:
    """Small compatibility wrapper around OpenCV's AprilTag dictionaries."""

    _FAMILIES = {
        "tag16h5": "DICT_APRILTAG_16h5",
        "tag25h9": "DICT_APRILTAG_25h9",
        "tag36h10": "DICT_APRILTAG_36h10",
        "tag36h11": "DICT_APRILTAG_36h11",
    }

    def __init__(self, family: str = "tag36h11") -> None:
        if cv2 is None or not hasattr(cv2, "aruco"):
            raise RuntimeError(
                "OpenCV was built without the aruco module. Install "
                "opencv-contrib-python-headless (and remove opencv-python-headless)."
            )
        try:
            dictionary_id = getattr(cv2.aruco, self._FAMILIES[family])
        except KeyError as exc:
            raise ValueError(
                f"unsupported family {family!r}; choose from {', '.join(self._FAMILIES)}"
            ) from exc
        except AttributeError as exc:
            raise RuntimeError("this OpenCV version has no AprilTag dictionaries") from exc

        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        if hasattr(cv2.aruco, "ArucoDetector"):
            params = cv2.aruco.DetectorParameters()
            self._detect = cv2.aruco.ArucoDetector(dictionary, params).detectMarkers
        else:
            params = cv2.aruco.DetectorParameters_create()
            self._detect = lambda image: cv2.aruco.detectMarkers(
                image, dictionary, parameters=params
            )

    def corners(self, gray: np.ndarray, tag_id: int) -> np.ndarray | None:
        corners, ids, _ = self._detect(gray)
        if ids is None:
            return None
        for marker_corners, marker_id in zip(corners, ids.reshape(-1)):
            if int(marker_id) == tag_id:
                return np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
        return None


def estimate_camera_from_tag(
    corners: np.ndarray,
    tag_size: float,
    K: np.ndarray,
    distortion: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Estimate T_camera_tag and its RMS corner reprojection error.

    The tag frame has X right, Y up and Z out of the printed tag. ``tag_size``
    is the black square's outer edge length in metres.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed")
    half = tag_size / 2.0
    object_points = np.array(
        [[-half, half, 0.0], [half, half, 0.0],
         [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float64,
    )
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        np.asarray(corners, dtype=np.float64),
        K,
        distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        raise RuntimeError("solvePnP failed")
    rotation, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = tvec.reshape(3)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, distortion)
    error = float(
        np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - corners) ** 2, axis=1)))
    )
    return transform, error


def transform_from_ros(msg) -> np.ndarray:
    """Convert geometry_msgs/Transform to a homogeneous matrix."""
    q = msg.rotation
    norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if norm < 1e-12:
        raise ValueError("zero-length quaternion")
    x, y, z, w = q.x / norm, q.y / norm, q.z / norm, q.w / norm
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = [msg.translation.x, msg.translation.y, msg.translation.z]
    return result


def average_transforms(transforms: Iterable[np.ndarray]) -> np.ndarray:
    """Average translations robustly and rotations on SO(3)."""
    values = list(transforms)
    if not values:
        raise ValueError("cannot average an empty transform list")
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = np.median([value[:3, 3] for value in values], axis=0)
    rotation_sum = np.sum([value[:3, :3] for value in values], axis=0)
    u, _, vt = np.linalg.svd(rotation_sum)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    result[:3, :3] = rotation
    return result


def matrix_to_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    """Convert R = Rz(yaw) Ry(pitch) Rx(roll) to ROS fixed-axis RPY."""
    pitch = math.atan2(-rotation[2, 0], math.hypot(rotation[0, 0], rotation[1, 0]))
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return roll, pitch, yaw


def validate_transform(transform: np.ndarray, name: str) -> None:
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3):
        raise ValueError(f"{name} rotation determinant is not +1")


def write_json(path: Path, data: dict) -> None:
    """Write JSON atomically, so interrupted calibration cannot corrupt a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
    temporary.replace(path)
