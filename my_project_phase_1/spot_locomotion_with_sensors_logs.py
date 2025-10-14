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

import argparse
import os
import sys
import csv
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
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
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.spot_rough_env_cfg import SpotRoughEnvCfg_PLAY
from datetime import datetime

TASK = "Isaac-Velocity-Rough-Spot-v0"
RL_LIBRARY = "rsl_rl"


class SpotRoughDemo:
    """This class provides an interactive demo for the Spot rough terrain environment.
    It loads a pre-trained checkpoint for the Isaac-Velocity-Rough-Spot-v0 task, trained with RSL RL
    and defines a set of keyboard commands for directing motion of selected robots.

    A robot can be selected from the scene through a mouse click. Once selected, the following
    keyboard controls can be used to control the robot:

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
        env_cfg.scene.num_envs = 100
        env_cfg.episode_length_s = 1000000
        env_cfg.curriculum = None
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 2.0)
        env_cfg.commands.base_velocity.ranges.heading = (-1.0, 1.0)
        # wrap around environment for rsl-rl
        self.env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device
        
        self.policy = torch.jit.load(checkpoint, map_location=self.device)

        self.create_camera()
        self.commands = torch.zeros(env_cfg.scene.num_envs, 3, device=self.device)
        self.commands[:,:]= self.env.unwrapped.command_manager.get_command("base_velocity") # type: ignore
        self.set_up_keyboard()
        self._prim_selection = omni.usd.get_context().get_selection()
        self._selected_id = None
        self._previous_selected_id = None
        self._camera_local_transform = torch.tensor([-2.5, 0.0, 0.8], device=self.device)

        print("--- Setting up contact sensor logging ---")
        self.contact_sensor: ContactSensor = self.env.unwrapped.scene.sensors["contact_forces"]

        robot = self.env.unwrapped.scene["robot"]

        body_names = robot.body_names

        body_names_to_idx_map = {name: i for i, name in enumerate(body_names)}

        self.foot_indices: Dict[str, int] = {
            name: idx for name, idx in body_names_to_idx_map.items() if "foot" in name
        }

        print(f"All body names from articulation: {body_names}")
        print(f"Found foot indices: {self.foot_indices}")
        print("-----------------------------------------")
        

        ### ADDED FOR CSV LOGGING ###
        print("--- Setting up CSV logging ---")
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_filename = f"contact_log_{timestamp}.csv"

        self.logfile = open(self.log_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.logfile)

        # Create and write the header row. This is crucial for knowing what each column is.
        # We'll create a sorted list of feet to ensure the column order is always the same.
        self.sorted_foot_names = sorted(self.foot_indices.keys())
        
        header = ['rl_step']
        for foot_name in self.sorted_foot_names:
            header.extend([
                f'{foot_name}_Fx', f'{foot_name}_Fy', f'{foot_name}_Fz',
                f'{foot_name}_contact'
            ])
        
        self.csv_writer.writerow(header)
        print(f"Logging contact forces to: {self.log_filename}")
        print("----------------------------")


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
        """Checks for a keyboard event and assign the corresponding command control depending on key pressed."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # Arrow keys map to pre-defined command vectors to control navigation of robot
            if event.input.name in self._key_to_control:
                if self._selected_id:
                    self.commands[self._selected_id] = self._key_to_control[event.input.name]
            # Escape key exits out of the current selected robot view
            elif event.input.name == "ESCAPE":
                self._prim_selection.clear_selected_prim_paths()
            # C key swaps between third-person and perspective views
            elif event.input.name == "C":
                if self._selected_id is not None:
                    if self.viewport.get_active_camera() == self.camera_path:
                        self.viewport.set_active_camera(self.perspective_path)
                    else:
                        self.viewport.set_active_camera(self.camera_path)
        # On key release, the robot stops moving
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if self._selected_id:
                self.commands[self._selected_id] = self._key_to_control["ZEROS"]

    def update_selected_object(self):
        """Determines which robot is currently selected and whether it is a valid H1 robot.
        For valid robots, we enter the third-person view for that robot.
        When a new robot is selected, we reset the command of the previously selected
        to continue random commands."""

        self._previous_selected_id = self._selected_id
        selected_prim_paths = self._prim_selection.get_selected_prim_paths()
        if len(selected_prim_paths) == 0:
            self._selected_id = None
            self.viewport.set_active_camera(self.perspective_path)
        elif len(selected_prim_paths) > 1:
            print("Multiple prims are selected. Please only select one!")
        else:
            prim_splitted_path = selected_prim_paths[0].split("/")
            # a valid robot was selected, update the camera to go into third-person view
            if len(prim_splitted_path) >= 4 and prim_splitted_path[3][0:4] == "env_":
                self._selected_id = int(prim_splitted_path[3][4:])
                if self._previous_selected_id != self._selected_id:
                    self.viewport.set_active_camera(self.camera_path)
                self._update_camera()
            else:
                print("The selected prim was not a H1 robot")

        # Reset commands for previously selected robot if a new one is selected
        if self._previous_selected_id is not None and self._previous_selected_id != self._selected_id:
            self.env.unwrapped.command_manager.reset([self._previous_selected_id])
            self.commands[:, 0:3] = self.env.unwrapped.command_manager.get_command("base_velocity")

    def _update_camera(self):
        """Updates the per-frame transform of the third-person view camera to follow
        the selected robot's torso transform."""

        base_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[self._selected_id, :]  # - env.scene.env_origins
        base_quat = self.env.unwrapped.scene["robot"].data.root_quat_w[self._selected_id, :]

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
    rl_step_counter = 0

    while simulation_app.is_running():
        demo_spot.update_selected_object()
        with torch.inference_mode():
            action = demo_spot.policy(obs)
            obs, _, _, _ = demo_spot.env.step(action)
            obs[:, 196:199] = demo_spot.commands

            rl_step_counter += 1

            # Log contact sensor data every n RL steps
            if rl_step_counter % 5 == 0:
                # A threshold to decide if a force magnitude constitutes a "contact"
                CONTACT_THRESHOLD = 1.0  # in Newtons

                # Get the latest data from the sensor's history buffer
                force_history = demo_spot.contact_sensor.data.net_forces_w_history

                # Create a list to hold all the data for the current row
                data_row = [rl_step_counter]
                # We will log data for the first environment (env_0)
                env_idx = 0
                
                # Loop through the sorted list of foot names to ensure consistent column order
                for foot_name in demo_spot.sorted_foot_names:
                    body_idx = demo_spot.foot_indices[foot_name]
                    
                    # Get force and contact state (same logic as before)
                    foot_force_history = force_history[env_idx, :, body_idx, :]
                    current_force_vec = foot_force_history[0, :]
                    force_magnitudes = torch.norm(foot_force_history, dim=-1)
                    is_in_contact = torch.max(force_magnitudes) > CONTACT_THRESHOLD

                    # Append the data to our row list
                    force_cpu = current_force_vec.cpu().numpy()
                    data_row.extend([
                        force_cpu[0],  # Fx
                        force_cpu[1],  # Fy
                        force_cpu[2],  # Fz
                        1 if is_in_contact.item() else 0  # Contact state as 1 or 0
                    ])
                    
                demo_spot.csv_writer.writerow(data_row)

                print(f"\n[RL Step {rl_step_counter}] Contact Forces for Env {env_idx}: (Data saved to CSV)")

    print("Simulation ended. Closing log file.")
    demo_spot.logfile.close()

if __name__ == "__main__":
    try:
        main()
        simulation_app.close()
    finally:
        simulation_app.close()
