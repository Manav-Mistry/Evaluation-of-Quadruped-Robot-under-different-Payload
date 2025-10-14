# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportAttributeAccessIssue=false
# pyright: reportOptionalMemberAccess = false

"""
This script demonstrates an interactive demo with the spot rough terrain environment.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/spot_locomotion.py

"""

"""Launch Isaac Sim Simulator first."""

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
from isaacsim.core.utils.stage import get_current_stage
from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, Sdf
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.spot_rough_env_cfg import SpotRoughEnvCfg_PLAY

TASK = "Isaac-Velocity-Rough-Spot-v0"
RL_LIBRARY = "rsl_rl"


class SpotRoughDemo:
    """This class provides an interactive demo for the Spot rough terrain environment.
    It loads a pre-trained checkpoint for the Isaac-Velocity-Rough-Spot-v0 task, trained with RSL RL
    and defines a set of keyboard commands for directing motion of selected robots.
    
    This version of the demo only contains one robot and the keyboard commands
    are automatically mapped to it.
    
    The following keyboard controls can be used to control the robot:
    * UP: go forward
    * LEFT: turn left
    * RIGHT: turn right
    * DOWN: stop
    * C: switch between third-person and perspective views
    * ESC: exit current third-person view"""

    def __init__(self):
        """Initializes environment config designed for the interactive model and sets up the environment,
        loads pre-trained checkpoints, and registers keyboard events."""
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(TASK, args_cli)
        # load the trained jit policy
        checkpoint = "/home/manav/IsaacLab/logs/rsl_rl/spot_flat/2025-08-27_11-21-29/exported/policy.pt"
        # create envionrment
        env_cfg = SpotRoughEnvCfg_PLAY()
        # The user requested to keep only one environment
        env_cfg.scene.num_envs = 1
        env_cfg.episode_length_s = 1000000
        env_cfg.curriculum = None
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 5.2)
        env_cfg.commands.base_velocity.ranges.heading = (-1.0, 1.0)
        # wrap around environment for rsl-rl
        self.env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device
        # load previously trained model
        # ppo_runner = OnPolicyRunner(self.env, agent_cfg.to_dict(), log_dir=None, device=self.device)
        # ppo_runner.load(checkpoint)
        # obtain the trained policy for inference
        # self.policy = ppo_runner.get_inference_policy(device=self.device)
        self.policy = torch.jit.load(checkpoint, map_location=self.device)

        self.create_camera()
        self.commands = torch.zeros(env_cfg.scene.num_envs, 3, device=self.device)
        self.commands[:,:]= self.env.unwrapped.command_manager.get_command("base_velocity") # type: ignore
        self.set_up_keyboard()
        # Set the selected ID to 0 since there is only one environment
        self._selected_id = 0
        self._camera_local_transform = torch.tensor([-2.5, 0.0, 0.8], device=self.device)

    def create_camera(self):
        """Creates a camera to be used for third-person view."""
        stage = get_current_stage()
        self.viewport = get_viewport_from_window_name("Viewport")
        # Create camera
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
            # Arrow keys map to pre-defined command vectors to control navigation of robot
            if event.input.name in self._key_to_control:
                # Apply commands to the single robot at index 0
                self.commands[0] = self._key_to_control[event.input.name]
            # Escape key exits out of the current selected robot view
            elif event.input.name == "ESCAPE":
                # With only one robot and no selection logic, this can clear the selection.
                # However, for a single robot, we can ignore this.
                pass
            # C key swaps between third-person and perspective views
            elif event.input.name == "C":
                if self.viewport.get_active_camera() == self.camera_path:
                    self.viewport.set_active_camera(self.perspective_path)
                else:
                    self.viewport.set_active_camera(self.camera_path)
        # On key release, the robot stops moving
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            # Apply zero command on key release
            self.commands[0] = self._key_to_control["ZEROS"]

    def _update_camera(self):
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
        # Update camera to follow the robot
        demo_spot._update_camera() # type: ignore
        with torch.inference_mode():
            action = demo_spot.policy(obs)
            obs, _, _, _ = demo_spot.env.step(action)
            # overwrite command based on keyboard input
            obs[:, 196:199] = demo_spot.commands


if __name__ == "__main__":
    main()
    simulation_app.close()
