"""Spot robot waypoint following demo utilities."""

from .waypoint_follower import WaypointTrajectoryFollower
from .terrain_configs import FLAT_TERRAIN_CFG
from .camera_utils import ThirdPersonCamera
from .physics_utils import attach_payload_to_robot
from .keyboard_controller import KeyboardController
from .progress_based_waypointFollower import ProgressBasedWaypointFollower
from .waypoint_library import (
    create_basic_square,
    create_square_with_midpoints,
    create_fine_grid_square,
    create_figure_eight,
    create_circle_approximation,
    create_spiral,
    create_slalom
)

__version__ = "0.1.0"

__all__ = [
    "WaypointTrajectoryFollower",
    "FLAT_TERRAIN_CFG",
    "ThirdPersonCamera",
    "attach_payload_to_robot",
    "KeyboardController",
    "ProgressBasedWaypointFollower",
]