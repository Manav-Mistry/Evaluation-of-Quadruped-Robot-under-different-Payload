from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen

FLAT_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1.0,  # 100% flat terrain
            slope_range=(0.0, 0.0),  # Zero slope = flat
            platform_width=2.0,
            border_width=0.25
        ),
    },
)

from isaaclab.terrains import TerrainImporterCfg

CUSTOM_USD_PATH = "/home/manav/Desktop/Test course 3D models/continous_ramps/continous_ramps_with_only_colliders.usd"

# Option 1: Use USD terrain directly
RAMP_TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/ground",  # This is where it will be spawned
    terrain_type="usd",
    usd_path=CUSTOM_USD_PATH,
    num_envs=1,  # Number of environments
    env_spacing=8.0,  # Spacing if you have multiple environments
    visual_material=None,  # Material is in the USD file
    physics_material=None,  # Physics material is in the USD file
    debug_vis=False,
)