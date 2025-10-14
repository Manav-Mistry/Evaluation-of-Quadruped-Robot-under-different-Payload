# pyright: reportAttributeAccessIssue=false
# pyright: reportOptionalMemberAccess = false

import argparse
import os
import sys
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "/home/manav/IsaacLab/"))

import scripts.reinforcement_learning.rsl_rl.cli_args as cli_args # type: ignore

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(
    description="Interactive demo with the Spot robot for rough terrain navigation."
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
from my_utils import ProgressBasedWaypointFollower
from my_utils import FLAT_TERRAIN_CFG
from my_utils import create_fine_grid_square, create_basic_square

from envs import SpotRoughDemo
from envs import SpotRoughEnvTestCfg_PLAY

# Constants
TASK = "Isaac-Velocity-Rough-Spot-v0"
RL_LIBRARY = "rsl_rl"
CHECKPOINT_PATH = "/home/manav/IsaacLab/logs/rsl_rl/spot_flat/2025-08-27_11-21-29/exported/policy.pt"

def main():
    """Main execution function."""
    # Parse RSL-RL configuration
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(TASK, args_cli)

    # Initialize demo
    demo = SpotRoughDemo(
        env_cfg_class=SpotRoughEnvTestCfg_PLAY,
        checkpoint_path=CHECKPOINT_PATH,
        terrain_cfg=FLAT_TERRAIN_CFG
    )
    
    # Setup waypoint following
    waypoints = create_basic_square() * 1  # loop square N times

    # --------------
    # Progress-based follower - no segment_time needed!
    # Robot will move at its own pace through waypoints
    # --------------
    follower = ProgressBasedWaypointFollower(
        waypoints, 
        kp=2,
        kd=0.3,
        threshold_normal=0.15,
        threshold_final=0.15,
        velocity_threshold_final=0.1,
        lookahead_distance=0.25
    )
    follower.setup_markers()

    
    # Reset environment
    obs, _ = demo.reset()
    
    # Draw waypoint path
    follower.draw_path()
    
    # Debug: Print terrain info
    terrain_origins = demo.env.unwrapped.scene.terrain.terrain_origins
    print(f"Terrain origins: {terrain_origins}")
    
    # Main simulation loop
    count = 0
    while simulation_app.is_running():

        demo.update_camera()
        
        with torch.inference_mode():
            # Get policy action
            action = demo.policy(obs)
            obs, _, _, _ = demo.step(action)
            
            # Get robot state
            robot_state = demo.get_robot_state()
            position = robot_state['position']
            yaw = robot_state['yaw']
            velocity = robot_state.get('velocity', np.zeros(3))  # Need velocity for progress check

            # Get dt (no sim_time needed for progress-based!)
            dt = demo.get_dt()
            
            # Progress-based PD controller
            error, base_command = follower.get_command_PD(dt, position, yaw, velocity)
            
            # Check if trajectory is complete
            if follower.is_complete():
                print("Trajectory complete! Stopping robot.")
                base_command = np.zeros(3, dtype=np.float32)  # Stop
                # Optional: break or reset
                # follower.reset()  # To restart trajectory
            
            # Update commands in observation
            demo.commands = torch.from_numpy(base_command).unsqueeze(0).to(demo.device)
            obs[:, 196:199] = demo.commands

            count += 1

        
        # Print status periodically
        if count % 50 == 0:
            print("-------------------------------------------------------")
            print(f"Waypoint: {follower.current_waypoint_index}/{len(waypoints)-1} | State: {follower.state.name}")
            print(f"Robot position: x={position[0]:.3f}, y={position[1]:.3f}, z={position[2]:.3f}")
            print(f"Error: ({error[0]:.2f}, {error[1]:.2f}, {error[2]:.2f})")
            print(f"Command: [{base_command[0]:.2f}, {base_command[1]:.2f}, {base_command[2]:.2f}]")
            print("-------------------------------------------------------")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()