# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from ..terrain_generator_cfg import TerrainGeneratorCfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.15),
            step_width=0.3,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.15),
            step_width=0.3,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.3, grid_width=0.45, grid_height_range=(0.05, 0.25), platform_width=1
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),

        #NOTE: new change ----------
       "incline_ramp": terrain_gen.HfPyramidSlopedTerrainCfg(
           proportion=0.15,  # 20% of terrain tiles
           slope_range=(0.1, 0.35), 
           platform_width=1,  # Small platform at center
           border_width=0.1,
           inverted=False,  # Slopes go upward from center
       ),

       "incline_ramp_inverted": terrain_gen.HfPyramidSlopedTerrainCfg(
           proportion=0.2,  # 20% of terrain tiles
           slope_range=(0.1, 0.35), 
           platform_width=1,  # Small platform at center
           border_width=0.1,
           inverted=True,  # Slopes go upward from center
       ),
    },
)
"""Rough terrains configuration."""

