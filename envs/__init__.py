"""Spot robot waypoint following demo utilities."""

from .spot_demo import SpotRoughDemo
from .spot_env_with_stepfield import SpotStepfieldEnv
from .spot_rough_env_cfg import SpotRoughEnvTestCfg_PLAY
from .spot_rough_env_with_multimesh_raycaster_cfg import SpotRoughEnvMultimeshTestCfg_PLAY
from .spot_rough_env_with_multimesh_raycaster_cfg_old import SpotRoughEnvMultiMeshRayCasterTestCfg_PLAY

__version__ = "0.1.0"

__all__ = [
    "SpotRoughDemo",
    "SpotStepfieldEnv",
    "SpotRoughEnvTestCfg_PLAY",
    "SpotRoughEnvMultimeshTestCfg_PLAY"
]