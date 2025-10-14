# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates an interactive demo with the spot rough terrain environment.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/spot_locomotion.py

"""

"""Launch Isaac Sim Simulator first."""
# pyright: reportAttributeAccessIssue=false
# pyright: reportOptionalMemberAccess = false


import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import scripts.reinforcement_learning.rsl_rl.cli_args as cli_args  # isort: skip


from isaaclab.app import AppLauncher

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

from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.spot_rough_env_cfg import SpotRoughEnvCfg_PLAY

TASK = "Isaac-Velocity-Rough-Spot-v0"
RL_LIBRARY = "rsl_rl"


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
        env_cfg = SpotRoughEnvCfg_PLAY()
        
        # The user requested to keep only one environment
        env_cfg.scene.num_envs = 1
        env_cfg.episode_length_s = 1000000
        env_cfg.curriculum = None
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 5.2)
        env_cfg.commands.base_velocity.ranges.heading = (-1.0, 1.0)
        env_cfg.scene.payload = cube_cfg
        
        self.env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device

        self._attach_payload()
        
        checkpoint = "/home/manav/IsaacLab/logs/rsl_rl/spot_flat/2025-08-27_11-21-29/exported/policy.pt"
        self.policy = torch.jit.load(checkpoint, map_location=self.device)

        self.create_camera()
        self.commands = torch.zeros(env_cfg.scene.num_envs, 3, device=self.device)
        self.commands[:,:]= self.env.unwrapped.command_manager.get_command("base_velocity") # type: ignore
        self.set_up_keyboard()
        
        self._camera_local_transform = torch.tensor([-2.5, 0.0, 0.8], device=self.device)


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


    def set_up_keyboard(self):
        """Sets up interface for keyboard input and registers the desired keys for control."""
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        T = 1
        R = 0.5
        self._key_to_control = {
            "UP": torch.tensor([2.0, 0.0, 0.0], device=self.device),
            "DOWN": torch.tensor([-2.0, 0.0, 0.0], device=self.device),
            "LEFT": torch.tensor([0.0, 2.0, 0.0], device=self.device),
            "RIGHT": torch.tensor([0.0, -2.0, 0.0], device=self.device),
            "N": torch.tensor([0.0, 0.0, 2.0], device=self.device),
            "M": torch.tensor([0.0, 0.0, -2.0], device=self.device),
            "ZEROS": torch.tensor([0.0, 0.0, 0.0], device=self.device)
        }

    def _on_keyboard_event(self, event):
        """Checks for a keyboard event and assign the corresponding command control depending on key pressed.
        
        Since there is only one environment, the commands are always applied to the robot at index 0.
        """
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._key_to_control:
                self.commands[0] = self._key_to_control[event.input.name]
                self.viewport.set_active_camera(self.camera_path)


            elif event.input.name == "ESCAPE":
                pass

            elif event.input.name == "C":
                if self.viewport.get_active_camera() == self.camera_path:
                    self.viewport.set_active_camera(self.perspective_path)
                else:
                    self.viewport.set_active_camera(self.camera_path)

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.commands[0] = self._key_to_control["ZEROS"]

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
    obs, _ = demo_spot.env.reset()
    while simulation_app.is_running():
        demo_spot.update_camera()
        with torch.inference_mode():
            action = demo_spot.policy(obs)
            obs, _, _, _ = demo_spot.env.step(action)
            obs[:, 196:199] = demo_spot.commands


if __name__ == "__main__":
    main()
    simulation_app.close()
