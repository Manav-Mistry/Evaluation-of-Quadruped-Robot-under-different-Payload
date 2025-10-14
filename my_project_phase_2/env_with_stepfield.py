# pyright: reportAttributeAccessIssue=false
# pyright: reportOptionalMemberAccess = false


import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import scripts.reinforcement_learning.rsl_rl.cli_args as cli_args  # isort: skip


from isaaclab.app import AppLauncher
import random

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates an interactive demo with the H1 rough terrain environment."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import torch
import numpy as np

import carb
import omni
import omni.usd
from isaacsim.core.utils.stage import get_current_stage
from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, Sdf, UsdPhysics
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab.assets import RigidObject, RigidObjectCfg
import isaaclab.sim as sim_utils
from pxr import UsdGeom, Gf, UsdPhysics

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.spot_rough_env_cfg import SpotRoughEnvCfg_PLAY
from scripts.demos.spot_rough_env.spot_rough_env_test import SpotRoughEnvTestCfg_PLAY
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

# Use VisualizationMarkers for drawing in Isaac Lab
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import VisualizationMarkersCfg
import isaaclab.markers.visualization_markers as marker_utils

from isaaclab.assets import AssetBaseCfg

TASK = "Isaac-Velocity-Rough-Spot-v0"
RL_LIBRARY = "rsl_rl"


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


class WaypointTrajectoryFollower:
    def __init__(self, waypoints, segment_time=2.0):
        """
        waypoints: list of [x, y, yaw] in world frame
        segment_time: duration (s) to move between consecutive waypoints
        """
        self.waypoints = np.array(waypoints, dtype=np.float32)
        self.segment_time = segment_time

        # total trajectory duration
        self.total_time = segment_time * (len(waypoints) - 1)

        self.markers = None

    
    def setup_markers(self):
        """Setup visualization markers for drawing the path"""
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/WaypointPath",
            markers={
                "waypoint_sphere": sim_utils.SphereCfg(
                    radius=0.1,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),  # Cyan
                ),
            },
        )
        self.markers = VisualizationMarkers(marker_cfg)
        
    def draw_path(self):
        """Draw the waypoint trajectory as spheres"""
        if self.markers is None:
            print("Markers not initialized. Call setup_markers() first.")
            return
            
        # Create positions from waypoints (x, y, z=0.15 to keep above ground)
        positions = np.array([[wp[0], wp[1], 0.15] for wp in self.waypoints], dtype=np.float32)
        
        # Create marker indices (all use the same marker type - index 0)
        marker_indices = np.zeros(len(self.waypoints), dtype=np.int32)
        
        # Visualize the waypoints
        self.markers.visualize(translations=positions, marker_indices=marker_indices)



    def get_reference(self, t):
        """
        Return desired [x, y, yaw] at time t using linear interpolation.
        """
        if hasattr(t, "item"):
            t = t.item()
        

        if t >= self.total_time:
            return self.waypoints[-1]

        # which segment are we in?
        seg_idx = int(t // self.segment_time)
        tau = (t % self.segment_time) / self.segment_time  # normalized [0,1]

        p0 = self.waypoints[seg_idx]
        p1 = self.waypoints[seg_idx + 1]

        # simple linear interpolation
        interp = (1 - tau) * p0 + tau * p1
        return interp
    

    def get_command(self, t, dt, current_yaw):
        """
        Compute velocity command [vx, vy, yaw_rate] at time t
        from finite differences of reference trajectory.
        """

        # Convert tensor values to float if needed
        # if hasattr(t, 'item'):
        #     t = t.item()
        # if hasattr(dt, 'item'):
        #     dt = dt.item()

        if t >= self.total_time:
            return (None, None), np.zeros(3, dtype=np.float32)

        ref_now = self.get_reference(t)
        ref_next = self.get_reference(t + dt)

        error_x = (ref_next[0] - ref_now[0])
        error_y = (ref_next[1] - ref_now[1])
        
        error = (error_x, error_y)


        dx_world = (ref_next[0] - ref_now[0]) / dt
        dy_world = (ref_next[1] - ref_now[1]) / dt
        
        # Transform from world frame to base frame
        dx_base = dx_world * np.cos(current_yaw) + dy_world * np.sin(current_yaw)
        dy_base = -dx_world * np.sin(current_yaw) + dy_world * np.cos(current_yaw)

        # handle yaw properly (wrap-around)
        yaw_now, yaw_next = ref_now[2], ref_next[2]
        yaw_err = np.arctan2(np.sin(yaw_next - yaw_now), np.cos(yaw_next - yaw_now))
        yaw_rate = yaw_err / dt

        return error, np.array([dx_base, dy_base, yaw_rate], dtype=np.float32)
    

    def get_command_with_feedback(self, t, dt, current_pos, current_yaw):
        # Get CURRENT waypoint goal (not next timestep reference)
        ref_now = self.get_reference(t)
        
        # Error to the GOAL (not to next interpolation point)
        error_x_world = ref_now[0] - current_pos[0]
        error_y_world = ref_now[1] - current_pos[1]
        
        # Use proportional control
        kp = 3  # Tune this
        dx_world = kp * error_x_world
        dy_world = kp * error_y_world
        
        # Clip to training limits
        dx_world = np.clip(dx_world, -2.0, 3.0)
        dy_world = np.clip(dy_world, -1.5, 1.5)
        
        # Transform to base frame
        dx_base = dx_world * np.cos(current_yaw) + dy_world * np.sin(current_yaw)
        dy_base = -dx_world * np.sin(current_yaw) + dy_world * np.cos(current_yaw)
        
        # Yaw control (same as before)
        yaw_err = np.arctan2(np.sin(ref_now[2] - current_yaw), 
                            np.cos(ref_now[2] - current_yaw))
        yaw_rate = 2.0 * yaw_err  # Proportional yaw control
        yaw_rate = np.clip(yaw_rate, -2.0, 2.0)
        
        error = (error_x_world, error_y_world)
        return error, np.array([dx_base, dy_base, yaw_rate], dtype=np.float32)
    


class SpotRoughDemo:
    

    def __init__(self):
        """Initializes environment config designed for the interactive model and sets up the environment,
        loads pre-trained checkpoints, and registers keyboard events."""
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(TASK, args_cli)

        # create a payload ----------------------------
        cube_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_0/Cube",
            spawn=sim_utils.CuboidCfg(
                size=(0.1, 0.1, 0.1),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(),
        )

        # create environment ---------------------------------
        env_cfg = SpotRoughEnvTestCfg_PLAY()
        
        # The user requested to keep only one environment
        env_cfg.scene.num_envs = 1
        env_cfg.episode_length_s = 1000000
        env_cfg.curriculum = None
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 5.2)
        env_cfg.commands.base_velocity.ranges.heading = (-1.0, 1.0)
        env_cfg.scene.payload = cube_cfg
        
        env_cfg.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator", 
            terrain_generator=FLAT_TERRAIN_CFG,  # Changed from ROUGH_TERRAINS_CFG
            max_init_terrain_level=0,  # Changed from 5 (0 = easiest/flattest level)
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

        env_cfg.scene.usd_object = AssetBaseCfg(
            prim_path="/World/envs/env_0/stepfield",  # Use wildcard for multiple envs
            spawn=sim_utils.UsdFileCfg(
                usd_path="/home/manav/Desktop/Step_Fields/1/ex-12_test_courses_continuous_ramps.usd",  # Replace with your USD file path
                scale=(1.0, 1.0, 1.0),  # Adjust scale as needed
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(4.0, 0.0, 0.5),  # Initial position (x, y, z)
                rot=(1.0, 0.0, 0.0, 0.0),  # Initial rotation (w, x, y, z quaternion)
            ),
        )

        env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.5)
        env_cfg.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        self._attach_payload()
        self.env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device
        
        checkpoint = "/home/manav/IsaacLab/logs/rsl_rl/spot_flat/2025-08-27_11-21-29/exported/policy.pt"
        self.policy = torch.jit.load(checkpoint, map_location=self.device)

        self.create_camera()
        self.commands = torch.zeros(env_cfg.scene.num_envs, 3, device=self.device)
        
        self._camera_local_transform = torch.tensor([-2.5, 0.0, 0.8], device=self.device)
        self.viewport.set_active_camera(self.camera_path)



    def _attach_payload(self):
        spot_body_prim_path = "/World/envs/env_0/Robot/body"
        # stage = get_current_stage()
        # spot = stage.GetPrimAtPath(spot_body_prim_path)
        # print(spot)
        cube_prim_path = "/World/envs/env_0/Cube"
        
        stage = get_current_stage()
        fixed_joint = UsdPhysics.FixedJoint.Define(stage, "/World/FixedJoint")

        fixed_joint.CreateBody0Rel().SetTargets([spot_body_prim_path])
        fixed_joint.CreateBody1Rel().SetTargets([cube_prim_path])

        # set the local transform
        fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.14343))


    def create_camera(self):
        """Creates a camera to be used for third-person view."""
        stage = get_current_stage()
        self.viewport = get_viewport_from_window_name("Viewport")
        self.camera_path = "/World/Camera"
        self.perspective_path = "/OmniverseKit_Persp"
        camera_prim = stage.DefinePrim(self.camera_path, "Camera")
        camera_prim.GetAttribute("focalLength").Set(8.5)
        coi_prop = camera_prim.GetProperty("omni:kit:centerOfInterest")
        if not coi_prop or not coi_prop.IsValid():
            camera_prim.CreateAttribute(
                "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
            ).Set(Gf.Vec3d(0, 0, -10))
        self.viewport.set_active_camera(self.perspective_path)


    def update_camera(self):
        """Updates the per-frame transform of the third-person view camera to follow
        the single robot's torso transform."""

        base_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[0, :]
        base_quat = self.env.unwrapped.scene["robot"].data.root_quat_w[0, :]

        camera_pos = quat_apply(base_quat, self._camera_local_transform) + base_pos

        camera_state = ViewportCameraState(self.camera_path, self.viewport)
        eye = Gf.Vec3d(camera_pos[0].item(), camera_pos[1].item(), camera_pos[2].item())
        target = Gf.Vec3d(base_pos[0].item(), base_pos[1].item(), base_pos[2].item() + 0.6)
        camera_state.set_position_world(eye, True)
        camera_state.set_target_world(target, True)


def main():
    """Main function."""
    demo_spot = SpotRoughDemo()

    waypoints = [
        [0, 0, 0],       
        [2, 0, 0], 
        [2, 0, np.pi/2],      # Move to (1,0), stay facing 0°
        [2, 2, np.pi/2], 
        [2, 2, np.pi],            # Move to (1,1), stay facing 0°
        [0, 2, np.pi],       # Move to (2,1), stay facing 0°
        [0, 2, 3*np.pi/2],
        [0, 0, 3*np.pi/2]               # Move to (2,2), stay facing 0°
    ]

    follower = WaypointTrajectoryFollower(waypoints)

    follower.setup_markers()

    obs, _ = demo_spot.env.reset()

    follower.draw_path()

    # Debug: Check terrain and spawn info
    terrain_origins = demo_spot.env.unwrapped.scene.terrain.terrain_origins
    print(f"Terrain origins: {terrain_origins}")

    count = 0

    while simulation_app.is_running():
        demo_spot.update_camera()
        # Print robot position every physics step
        robot_pos = demo_spot.env.unwrapped.scene["robot"].data.root_pos_w[0, :3]

        with torch.inference_mode():
            action = demo_spot.policy(obs)
            obs, _, _, _ = demo_spot.env.step(action)

            sim_time = demo_spot.env.unwrapped.episode_length_buf[0] * demo_spot.env.unwrapped.step_dt
            dt = demo_spot.env.unwrapped.step_dt

            position = robot_pos.cpu().numpy()
            robot_quat = demo_spot.env.unwrapped.scene["robot"].data.root_quat_w[0, :]
            w, x, y, z = robot_quat[0].item(), robot_quat[1].item(), robot_quat[2].item(), robot_quat[3].item()
            yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            error, base_command = follower.get_command_with_feedback(sim_time, dt, position, yaw)
            # error, base_command = follower.get_command(sim_time, dt, yaw)
            # print("Base command: ", base_command, type(base_command))
            demo_spot.commands = torch.from_numpy(base_command).unsqueeze(0).to(demo_spot.device)

            obs[:, 196:199] = demo_spot.commands
            count+=1

        if count % 50 == 0:
            print("-------------------------------------------------------")
            print(f"Robot position: x={robot_pos[0]:.3f}, y={robot_pos[1]:.3f}, z={robot_pos[2]:.3f}")
            print(f"Error: {error}")
            print(f"Command : {base_command}")
            print("-------------------------------------------------------")


if __name__ == "__main__":
    main()
    simulation_app.close()
