# pyright: reportAttributeAccessIssue=false
# pyright: reportOptionalMemberAccess = false

"""
Test Spot policy on individual training terrain types.

Select the active terrain by uncommenting one ACTIVE_TERRAIN line near the
bottom of the TERRAIN SELECTION section. All others must remain commented.

The robot receives a constant 1 m/s forward command for 15 seconds.
Watch the viewer to see if it walks successfully or falls.

Usage:
    python terrain_test.py --headless        # no viewer
    python terrain_test.py                   # with viewer (default)
"""

import argparse
import os
import sys

sys.path.append("/home/manav/IsaacLab/")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import scripts.reinforcement_learning.rsl_rl.cli_args as cli_args  # type: ignore
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test Spot policy on individual terrain types.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── imports after sim launch ──────────────────────────────────────────────────
import torch
import isaaclab.terrains as terrain_gen
import isaaclab.sim as sim_utils
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

from envs import SpotRoughEnvTestCfg_PLAY

# =============================================================================
# CONFIGURATION
# =============================================================================
CHECKPOINT_PATH = "/home/manav/IsaacLab/logs/rsl_rl/spot_rough/2026-01-05_11-55-39/exported/policy.pt"
TEST_DURATION_S  = 15.0   # seconds
FORWARD_VEL      = 1.0    # m/s  [lin_vel_x]

# =============================================================================
# TERRAIN CONFIGS  (one for each training sub-terrain, proportion=1.0)
# =============================================================================

_BASE = dict(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
)

TERRAIN_pyramid_stairs = TerrainGeneratorCfg(
    **_BASE,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.05, 0.15),
            step_width=0.3,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
    },
)

TERRAIN_pyramid_stairs_inv = TerrainGeneratorCfg(
    **_BASE,
    sub_terrains={
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.05, 0.15),
            step_width=0.3,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
    },
)

TERRAIN_boxes = TerrainGeneratorCfg(
    **_BASE,
    sub_terrains={
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0,
            grid_width=0.45,
            grid_height_range=(0.05, 0.25),
            platform_width=1,
        ),
    },
)

TERRAIN_random_rough = TerrainGeneratorCfg(
    **_BASE,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(0.02, 0.10),
            noise_step=0.02,
            border_width=0.25,
        ),
    },
)

TERRAIN_incline_ramp = TerrainGeneratorCfg(
    **_BASE,
    sub_terrains={
        "incline_ramp": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1.0,
            slope_range=(0.1, 0.35),
            platform_width=1,
            border_width=0.1,
            inverted=False,
        ),
    },
)

TERRAIN_incline_ramp_inverted = TerrainGeneratorCfg(
    **_BASE,
    sub_terrains={
        "incline_ramp_inverted": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1.0,
            slope_range=(0.1, 0.35),
            platform_width=1,
            border_width=0.1,
            inverted=True,
        ),
    },
)

# =============================================================================
# SELECT ACTIVE TERRAIN — uncomment exactly ONE line
# =============================================================================
ACTIVE_TERRAIN = TERRAIN_pyramid_stairs
# ACTIVE_TERRAIN = TERRAIN_pyramid_stairs_inv
# ACTIVE_TERRAIN = TERRAIN_boxes
# ACTIVE_TERRAIN = TERRAIN_random_rough
# ACTIVE_TERRAIN = TERRAIN_incline_ramp
# ACTIVE_TERRAIN = TERRAIN_incline_ramp_inverted


# =============================================================================
# HELPERS
# =============================================================================

def build_env_cfg(terrain_cfg: TerrainGeneratorCfg):
    """Configure SpotRoughEnvTestCfg_PLAY for a single-terrain test run."""
    env_cfg = SpotRoughEnvTestCfg_PLAY()

    # Single environment, long enough episode
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = TEST_DURATION_S + 5.0
    env_cfg.curriculum = None

    # Replace terrain with the selected single-terrain config
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_cfg,
        max_init_terrain_level=0,   # always start on the first (only) tile
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # Clean test: no noise, no randomisation (already set by PLAY cfg, but be explicit)
    env_cfg.observations.policy.enable_corruption = False

    return env_cfg


def main():
    terrain_name = list(ACTIVE_TERRAIN.sub_terrains.keys())[0]

    print(f"\n{'='*60}")
    print(f"  TERRAIN TEST: {terrain_name}")
    print(f"  Duration : {TEST_DURATION_S} s")
    print(f"  Command  : {FORWARD_VEL} m/s forward (no turning)")
    print(f"  Policy   : {CHECKPOINT_PATH}")
    print(f"{'='*60}\n")

    # Build env
    env_cfg  = build_env_cfg(ACTIVE_TERRAIN)
    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    env      = RslRlVecEnvWrapper(base_env)
    device   = env.unwrapped.device

    # Load policy
    policy = torch.jit.load(CHECKPOINT_PATH, map_location=device)
    policy.eval()
    print("Policy loaded.\n")

    # Constant forward command:  [lin_vel_x, lin_vel_y, ang_vel_z]
    forward_cmd = torch.tensor([[FORWARD_VEL, 0.0, 0.0]], device=device)

    # Reset
    obs, _   = env.reset()
    step_dt  = env.unwrapped.step_dt          # should be ~0.02 s at 50 Hz
    max_steps = int(TEST_DURATION_S / step_dt)

    print(f"step_dt={step_dt:.4f}s  →  running for {max_steps} steps\n")

    fell  = False
    count = 0

    while simulation_app.is_running() and count < max_steps:
        with torch.inference_mode():
            # Inject forward command into observation before policy call
            # Velocity command occupies indices 9-11 in the policy obs vector
            # (after base_lin_vel[0:3], base_ang_vel[3:6], projected_gravity[6:9])
            obs["policy"][:, 9:12] = forward_cmd

            action = policy(obs["policy"])
            obs, _, terminated, _ = env.step(action)

        # Detect fall (terminated before time-out)
        if terminated.any() and count < max_steps - 5:
            fell = True
            sim_t = count * step_dt
            print(f"  [FELL]  Robot terminated at step {count}  (sim_time = {sim_t:.2f} s)")
            break

        if count % 100 == 0:
            robot = env.unwrapped.scene["robot"]
            p = robot.data.root_pos_w[0]
            sim_t = count * step_dt
            print(f"  t={sim_t:5.1f}s | pos = ({p[0]:6.2f}, {p[1]:6.2f}, {p[2]:5.3f})")

        count += 1

    # ── Final result ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if fell:
        print(f"  RESULT : FAILED — robot fell on '{terrain_name}'")
    else:
        print(f"  RESULT : PASSED — robot completed {TEST_DURATION_S}s on '{terrain_name}'")
    print(f"{'='*60}\n")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
