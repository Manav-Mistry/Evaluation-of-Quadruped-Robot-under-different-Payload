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
