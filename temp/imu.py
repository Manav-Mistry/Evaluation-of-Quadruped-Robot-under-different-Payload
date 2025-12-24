# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Example on using the IMU sensor.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ImuCfg
from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip


@configclass
class ImuSensorSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    payload = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube",
                spawn=sim_utils.CuboidCfg(
                    size=(0.1, 0.1, 0.1),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 1.0), 
                        metallic=0.2
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0, 0, 10)
                )
            )

    imu_cube = ImuCfg(prim_path="{ENV_REGEX_NS}/Cube", debug_vis=True)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Simulate physics
    while simulation_app.is_running():

        # if count % 400 == 0:
        #     # reset counter
        #     count = 0
        #     current_vel = scene["payload"].data.root_vel_w
            
        #     # Create velocity change: [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]
        #     vel_change = torch.tensor([[0.0, 5.0, 0.0, 0.0, 0.0, 0.0]], device=current_vel.device)
        #     new_vel = current_vel + vel_change
            
        #     scene["payload"].write_root_velocity_to_sim(new_vel)

        #     # # Don't forget to apply changes!
        #     scene.write_data_to_sim()
           

        # -- write data to sim
        # scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)

        # print information from the sensors
        if count % 1 == 0:
        # print("-------------------------------")
            print(scene["payload"])
            print("Received linear velocity: ", scene["imu_cube"].data.lin_vel_b)
            print("Received angular velocity: ", scene["imu_cube"].data.ang_vel_b)
            
            print("Received linear acceleration: ", scene["imu_cube"].data.lin_acc_b)
            # print("Received angular acceleration: ", scene["imu_cube"].data.ang_acc_b)
        # print("-------------------------------")

def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.02, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    # design scene
    scene_cfg = ImuSensorSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()