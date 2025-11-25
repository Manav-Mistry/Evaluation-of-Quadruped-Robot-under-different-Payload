# pyright: reportAttributeAccessIssue=false
# pyright: reportOptionalMemberAccess = false

"""
This script has two controls 1)keyboard and 2) waypoint follower.

.. code-block:: bash

    # Usage
    python main_with_control_toggle.py --control keyboard
    python main_with_control_toggle.py --control waypoint

    Default is keyboard control.

"""

import argparse
import os
import sys
import numpy as np
import torch

import h5py
from datetime import datetime

sys.path.append( "/home/manav/IsaacLab/")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



import scripts.reinforcement_learning.rsl_rl.cli_args as cli_args # type: ignore
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(
    description="Interactive demo with the Spot robot for rough terrain navigation."
)
# Add control mode argument
parser.add_argument(
    "--control",
    type=str,
    default="keyboard",
    choices=["keyboard", "waypoint"],
    help="Control mode: keyboard for manual control, waypoint for autonomous following"
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import modules after app launch
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

# Import custom modules
from my_utils import WaypointTrajectoryFollower
from my_utils import FLAT_TERRAIN_CFG
from my_utils import KeyboardController

from hdf5_logger_utils import HDF5Logger, LoggingSchema

from envs import SpotStepfieldEnv, SpotRoughDemo
from envs import SpotRoughEnvTestCfg_PLAY, SpotRoughEnvMultimeshTestCfg_PLAY, SpotRoughEnvMultiMeshRayCasterTestCfg_PLAY # 3rd one with old policy

# Constants
TASK = "Isaac-Velocity-Rough-Spot-v0"
RL_LIBRARY = "rsl_rl"
CHECKPOINT_PATH = "/home/manav/IsaacLab/logs/rsl_rl/spot_rough/2025-10-19_12-57-22/exported/policy.pt"
# /home/manav/IsaacLab/logs/rsl_rl/spot_flat/2025-08-27_11-21-29/exported
# "/home/manav/IsaacLab/logs/rsl_rl/spot_rough/2025-10-19_12-57-22/exported/policy.pt"

# HDF5 Logging Configuration
HDF5_CONFIG = {
    'enabled': True,
    'buffer_size': 100,
    'save_dir': '/home/manav/my_isaaclab_project/logged_data',
    'experiment_prefix': 'imu_test'
}

# Example usage:
def create_waypoints():
    """
    Generates waypoints with 'In-Place Turns' using Continuous Yaw (Unwrapping).
    This ensures the robot always takes the shortest turn (Right or Left).
    """
    # Define your path
    path_xy = [
        [-1, 0],
        [0.34, 0],
        [1.52, 0],
        [2.90, 0],
        [4, 1.2],
        [6.57, 1.2],
        [6.57, 0],      # <--- The Problematic Point
        [4.07, 0],
        [3.16, 1.26],
        [1.87, 1.26],
    ]

    waypoints = []
    
    # Initialize with robot's starting yaw
    # (Assuming start is 0.0, change if robot starts facing elsewhere)
    current_yaw = 0.0 
    waypoints.append([path_xy[0][0], path_xy[0][1], current_yaw])

    YAW_THRESHOLD = 0.05  # Threshold to trigger an in-place turn

    for i in range(len(path_xy) - 1):
        curr_pos = path_xy[i]
        next_pos = path_xy[i+1]

        # 1. Calculate raw geometric angle (-pi to +pi)
        dx = next_pos[0] - curr_pos[0]
        dy = next_pos[1] - curr_pos[1]
        raw_yaw = np.arctan2(dy, dx)

        # 2. "Smart Unwrap": Find the version of this angle closest to current_yaw
        # We check raw_yaw, raw_yaw + 2pi, and raw_yaw - 2pi
        # and pick the one with the smallest distance to current_yaw.
        
        candidates = [
            raw_yaw,
            raw_yaw + 2 * np.pi,
            raw_yaw - 2 * np.pi
        ]
        
        # Select the candidate that minimizes abs(candidate - current_yaw)
        desired_yaw = min(candidates, key=lambda x: abs(x - current_yaw))

        # 3. Check for Turn
        # If the orientation changes significantly, insert a turn waypoint
        if abs(desired_yaw - current_yaw) > YAW_THRESHOLD:
            waypoints.append([curr_pos[0], curr_pos[1], desired_yaw])
        
        # 4. Add Move Waypoint
        waypoints.append([next_pos[0], next_pos[1], desired_yaw])
        
        # Update current_yaw for the next iteration's comparison
        current_yaw = desired_yaw

    return np.array(waypoints)


def run_keyboard_control(demo):
    """Run demo with keyboard control."""
    print("\n" + "="*60)
    print("KEYBOARD CONTROL MODE")
    print("="*60)
    
    # Initialize keyboard controller with camera support
    keyboard = KeyboardController(
        num_envs=1, 
        device=demo.device,
        camera_path=demo.camera.camera_path,
        perspective_path=demo.camera.perspective_path,
        viewport=demo.camera.viewport
    )
    keyboard.print_controls()

    # define schema for data collection
    schema = LoggingSchema.payload_experiment()
    
    # Reset environment
    obs, _ = demo.env.reset()
    
    print("Starting keyboard control... Press CTRL+C to exit.\n")
    
    logger = HDF5Logger(
        config=HDF5_CONFIG,
        control_mode='keyboard',
        schema=schema,
        num_envs=1,
        max_timesteps=1000
    )


    try:
        count=0
        while simulation_app.is_running() and count < 110: # 
            # robot_pos = demo.env.unwrapped.scene["robot"].data.root_pos_w[0, :3]
            payload_imu = demo.env.unwrapped.scene["payload_imu"].data
            

            demo.update_camera()
            
            with torch.inference_mode():
                action = demo.policy(obs)
                obs, _, _, _ = demo.env.step(action)

                if count % 10 == 0: # 
                    logger.log({
                        'payload_pos_w': payload_imu.pos_w[0], # torch tensor OK - auto-converted
                        'payload_quat_w': payload_imu.quat_w[0],
                        'payload_lin_acc_b': payload_imu.lin_acc_b[0],
                        'payload_ang_acc_b': payload_imu.ang_acc_b[0],
                    })
                    print(count)
                    
                # Get keyboard command
                keyboard_command = keyboard.get_command()
                obs[:, 9:12] = keyboard_command
            
            count+=1

        logger.close()

                
    finally:
        keyboard.cleanup()


def run_waypoint_control(demo):
    """Run demo with waypoint following control."""
    print("\n" + "="*60)
    print("WAYPOINT CONTROL MODE")
    print("="*60)
    
    # Setup waypoint following
    waypoints = create_waypoints()

    # --------------
    #   one segment dist / seg_time = expected velocity 
    #   here
    #   seg_dist(2 meter) / seg_time (2 sec) = 1 m/s velocity
    # --------------
    follower = WaypointTrajectoryFollower(waypoints, segment_time=2.0, kp=3, kd=1)

    follower.setup_markers()
    
    # Reset environment
    obs, _ = demo.reset()
    
    # Draw waypoint path
    follower.draw_path()
    
    print(f"Following {len(waypoints)} waypoints...")
    print("Press CTRL+C to exit.\n")
    
    # Debug: Print terrain info
    terrain_origins = demo.env.unwrapped.scene.terrain.terrain_origins
    print(f"Terrain origins: {terrain_origins}\n")
    
    # Main simulation loop
    count = 0
    while simulation_app.is_running(): # 50Hz 
        demo.update_camera()
        
        # Get robot position BEFORE stepping
        robot_pos = demo.env.unwrapped.scene["robot"].data.root_pos_w[0, :3]

        with torch.inference_mode():
            action = demo.policy(obs)
            obs, _, _, _ = demo.env.step(action)

            # Get timing info
            sim_time = demo.env.unwrapped.episode_length_buf[0] * demo.env.unwrapped.step_dt
            dt = demo.env.unwrapped.step_dt

            # Convert position and get yaw
            position = robot_pos.cpu().numpy()
            robot_quat = demo.env.unwrapped.scene["robot"].data.root_quat_w[0, :]
            w, x, y, z = robot_quat[0].item(), robot_quat[1].item(), robot_quat[2].item(), robot_quat[3].item()
            yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            
            # Get waypoint command
            # error, base_command = follower.get_command_with_feedback(sim_time, dt, position, yaw)
            error, base_command = follower.get_command_with_feedback_PD(sim_time, dt, position, yaw)

            # Update commands
            demo.commands = torch.from_numpy(base_command).unsqueeze(0).to(demo.device)
            obs[:, 9:12] = demo.commands
            count += 1

        # Print status periodically
        if count % 10 == 0:
            print("-------------------------------------------------------")
            print(demo.get_dt)
            print("-------------------------------------------------------")


def main():
    """Main execution function."""
    # Parse RSL-RL configuration
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(TASK, args_cli)
    
    # Initialize demo
    print("Initializing Spot environment...")
    
    demo = SpotStepfieldEnv(
        env_cfg_class=SpotRoughEnvMultimeshTestCfg_PLAY,
        checkpoint_path=CHECKPOINT_PATH,
        terrain_cfg=FLAT_TERRAIN_CFG,
    )
    
    print("Environment initialized successfully!\n")
    
    # Run based on control mode
    if args_cli.control == "keyboard":
        run_keyboard_control(demo)
    elif args_cli.control == "waypoint":
        run_waypoint_control(demo)
    


if __name__ == "__main__":
    try:
        main()
    finally:
        print("before sim close")
        simulation_app.close()
        print("simulation closed")