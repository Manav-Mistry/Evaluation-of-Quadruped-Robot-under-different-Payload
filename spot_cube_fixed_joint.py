# Simple example: Spawn Spot robot with a cube attached via fixed joint

import argparse

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Spawn Spot robot with a cube attached via fixed joint.")
parser.add_argument("--num_envs", type=int, default=5, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows after app launch."""

import torch
from pxr import Gf, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaacsim.core.utils.stage import get_current_stage

# Import Spot robot config (includes actuators and default joint positions)
from isaaclab_assets.robots.spot import SPOT_CFG


def create_fixed_joint(robot_body_path: str, payload_path: str, joint_path: str, local_offset: tuple = (0.0, 0.0, 0.15)):
    """
    Create a fixed joint between the robot body and the payload cube.

    Args:
        robot_body_path: USD path to the robot's body link
        payload_path: USD path to the payload object
        joint_path: USD path where the fixed joint will be created
        local_offset: Local position offset as (x, y, z) tuple - position of cube relative to robot body
    """
    stage = get_current_stage()

    # Create fixed joint
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, joint_path)

    # Set body relationships
    fixed_joint.CreateBody0Rel().SetTargets([robot_body_path])
    fixed_joint.CreateBody1Rel().SetTargets([payload_path])

    # Set local transform on body0 (robot body) - where the joint attaches
    # This defines the offset from robot body origin to where the cube will be
    local_pos0 = Gf.Vec3f(float(local_offset[0]), float(local_offset[1]), float(local_offset[2]))
    fixed_joint.CreateLocalPos0Attr().Set(local_pos0)

    # Set local transform on body1 (cube) - cube's origin attaches at this point
    # Setting to (0,0,0) means the cube's center attaches to the offset point on robot
    fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    # Set default rotations (identity quaternion)
    fixed_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    fixed_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # Make it rigid (high break force/torque)
    fixed_joint.CreateBreakForceAttr().Set(1e10)
    fixed_joint.CreateBreakTorqueAttr().Set(1e10)

    # Debug: verify the attribute was set
    actual_pos = fixed_joint.GetLocalPos0Attr().Get()
    print(f"Created fixed joint at {joint_path} with LocalPos0={actual_pos}")


@configclass
class SpotCubeSceneCfg(InteractiveSceneCfg):
    """Scene configuration with Spot robot and a cube."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg()
    )

    # Dome light
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Spot robot - use SPOT_CFG which includes actuators (PD controllers) and default joint positions
    # Default joint positions from SPOT_CFG:
    #   - [fh]l_hx: 0.1 (left hip_x), [fh]r_hx: -0.1 (right hip_x)
    #   - f[rl]_hy: 0.9 (front hip_y), h[rl]_hy: 1.1 (hind hip_y)
    #   - .*_kn: -1.5 (all knees)
    # Actuators: DelayedPDActuatorCfg for hips, RemotizedPDActuatorCfg for knees
    #   - stiffness=60.0, damping=1.5
    robot: ArticulationCfg = SPOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            joint_pos={
                # Use default standing pose from SPOT_CFG
                "[fh]l_hx": 0.1,   # left hip_x
                "[fh]r_hx": -0.1,  # right hip_x
                "f[rl]_hy": 0.9,  # front hip_y
                "h[rl]_hy": 1.1,  # hind hip_y
                ".*_kn": -1.5,    # all knees
            },
            joint_vel={".*": 0.0},
        ),
    )

    # Cube (payload) to attach to the robot
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0),  # Green cube
                metallic=0.2
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),  # Initial position (will be constrained by joint)
        ),
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Get robot articulation
    robot = scene["robot"]

    # Get the default joint positions (standing pose) - this is what the PD controller will target
    default_joint_pos = robot.data.default_joint_pos.clone()

    # Simulate physics
    while simulation_app.is_running():
        # Write default joint positions as targets to keep the robot standing
        # The actuators (PD controllers) will generate torques to track these positions
        robot.set_joint_position_target(default_joint_pos)
        scene.write_data_to_sim()

        # Perform step
        sim.step()

        # Update sim-time
        sim_time += sim_dt
        count += 1

        # Update scene buffers
        scene.update(sim_dt)

        # Print robot info periodically
        if count % 500 == 0:
            num_robots = scene["robot"].data.root_pos_w.shape[0]
            robot_pos = scene["robot"].data.root_pos_w[0]
            print(f"[t={sim_time:.2f}s] {num_robots} robots active | Robot 0 pos: {robot_pos.cpu().numpy()}")


def main():
    """Main function."""
    # Initialize simulation context (500Hz physics like in Spot training configs)
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.002,
        device=args_cli.device,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)

    # Set camera view - positioned to see all 5 robots in a row
    sim.set_camera_view(eye=[6.0, -12.0, 4.0], target=[6.0, 0.0, 0.0])

    # Create scene with 5 envs arranged in a row (spacing along Y-axis)
    # env_spacing=3.0 means 3m between each robot
    scene_cfg = SpotCubeSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)

    # Define different payload offsets (x, y) for each environment
    # z offset (height above body) is constant at 0.15
    payload_offsets = [
        (0.0, 0.0, 0.15),    # env 0: centered
        (0.15, 0.0, 0.15),   # env 1: forward
        (0.25, 0.0, 0.15),   # env 2: more forward
        (-0.15, 0.0, 0.15),  # env 3: backward
        (-0.25, 0.0, 0.15),  # env 4: more backward
    ]

    # Create fixed joints between robot body and cube for each environment
    for env_idx in range(args_cli.num_envs):
        # Get offset for this environment (cycle through if more envs than offsets)
        offset = payload_offsets[env_idx % len(payload_offsets)]
        create_fixed_joint(
            robot_body_path=f"/World/envs/env_{env_idx}/Robot/body",
            payload_path=f"/World/envs/env_{env_idx}/Cube",
            joint_path=f"/World/envs/env_{env_idx}/FixedJoint",
            local_offset=offset,
        )
        print(f"  Env {env_idx}: payload offset = {offset}")

    # Reset simulation
    sim.reset()

    # Update scene to get initial positions
    scene.update(sim_cfg.dt)

    # Debug: Print relative positions of cubes to their robot bodies
    print("\n[DEBUG] Checking cube positions relative to robot bodies:")
    robot_positions = scene["robot"].data.root_pos_w
    cube_positions = scene["cube"].data.root_pos_w
    for i in range(min(args_cli.num_envs, len(payload_offsets))):
        robot_pos = robot_positions[i].cpu().numpy()
        cube_pos = cube_positions[i].cpu().numpy()
        relative_pos = cube_pos - robot_pos
        print(f"  Env {i}: Robot={robot_pos}, Cube={cube_pos}, Relative={relative_pos}")

    print(f"\n[INFO]: Setup complete. {args_cli.num_envs} Spot robots with attached cubes are ready!")
    print("[INFO]: Each green cube is attached to its Spot's body via a fixed joint.")

    # Run simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
