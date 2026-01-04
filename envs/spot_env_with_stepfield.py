# pyright: reportAttributeAccessIssue=false
# pyright: reportOptionalMemberAccess = false

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from isaaclab.assets import RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.terrains import TerrainImporterCfg, TerrainImporter
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from my_utils import ThirdPersonCamera
from my_utils import attach_payload_to_robot

import numpy as np
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns, ImuCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg


# USD_PATH = "/home/manav/Desktop/Test course 3D models/continous_ramps/continous_ramps_with_only_colliders.usd"
USD_PATH_BASELINE_FLAT ="/home/manav/Desktop/Test course 3D models/base_line_flat/base_line_flat_with_only_colliders.usd"

class SpotStepfieldEnv:
    
    def __init__(self, env_cfg_class, checkpoint_path, terrain_cfg, camera_mode):
        self.robot_init_position = (-1, 0.7, 0.5) #(-1, 0.7, 0.5)
         
        cube_cfg = self._create_payload_config()
        imu_cfg = self._attach_imu()
        
        # attach IMU to Spot body
        imu_spot_cfg = self._attach_imu_spot()
        
        # Setup environment configuration
        env_cfg = self._setup_environment_config(
            env_cfg_class, 
            cube_cfg, 
            terrain_cfg,
            imu_cfg,
            imu_spot_cfg
        )

        # Attach payload to robot
        for env_idx in range(env_cfg.scene.num_envs):
            attach_payload_to_robot(
                robot_body_path=f"/World/envs/env_{env_idx}/Robot/body",
                payload_path=f"/World/envs/env_{env_idx}/Cube",
                env_idx= env_idx,
                local_offset=(0.0, 0.0, 0.14343),
            )
        
        # Create environment
        try:
            base_env = ManagerBasedRLEnv(cfg=env_cfg)
            self.env = RslRlVecEnvWrapper(base_env)
        except Exception as e:
            print(f"ERROR during environment creation: {e}")
            import traceback
            traceback.print_exc()
            raise

        self.device = self.env.unwrapped.device
        
        # Add ramps after env is initialized
        

        # terrain_importer: TerrainImporter = self.env.unwrapped.scene.terrain
        # terrain_importer.import_usd(name="Ramp", usd_path=CUSTOM_USD_PATH)

        # Load trained policy
        self.policy = torch.jit.load(checkpoint_path, map_location=self.device)
        
        # Setup camera
        self.camera = ThirdPersonCamera(mode=camera_mode)

        if camera_mode == "static":
            self.camera.set_local_transform(
                torch.tensor([-2.5, 0.0, 2.5], device=self.device)
            )
        else :
            self.camera.set_local_transform(
                torch.tensor([-2.5, 0.0, 0.8], device=self.device)
            )
        self.camera.activate()
        
        # Initialize commands
        self.commands = torch.zeros(env_cfg.scene.num_envs, 3, device=self.device)


    def _create_payload_config(self):
        """Create configuration for the payload cube."""
        return RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube", #{ENV_REGEX_NS}/Robot/body /World/envs/env_0/Cube
            spawn=sim_utils.CuboidCfg(
                size=(0.1, 0.1, 0.1),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0), 
                    metallic=0.2
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=self.robot_init_position,
            ),
        )
    
    def _attach_imu(self):
        return ImuCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            debug_vis=True,
        )
    
    def _attach_imu_spot(self):
        return ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/body",
            debug_vis=True,
        )
    
    def _setup_environment_config(self, env_cfg_class, cube_cfg, terrain_cfg, imu_cfg, imu_spot_cfg):
        """Setup and configure the environment."""
        env_cfg = env_cfg_class()
        env_cfg.scene.num_envs = 1
        env_cfg.episode_length_s = 1000000
        env_cfg.curriculum = None

        env_cfg.commands.base_velocity.ranges.lin_vel_x = (-2, 5.2)
        env_cfg.commands.base_velocity.ranges.heading = (-1.0, 1.0)

        env_cfg.scene.payload = cube_cfg

        env_cfg.scene.payload_imu = imu_cfg
        env_cfg.scene.robot_imu = imu_spot_cfg

        # Spawn Test Ramps without Rigidbody: Working
        # env_cfg.scene.custom_ramp = AssetBaseCfg(
        #     prim_path="{ENV_REGEX_NS}/CustomRamp",
        #     spawn=sim_utils.UsdFileCfg(
        #         usd_path=USD_PATH_BASELINE_FLAT,
        #         scale=(1.0, 1.0, 1.0),

        #     ),
        # )

        # Spawn Test Ramps with Rigidbody: Working but collision property is incorrectly set
        # env_cfg.scene.custom_ramp = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/CustomRamp",
        #     spawn=sim_utils.UsdFileCfg(
        #         usd_path="/home/manav/Desktop/Test course 3D models/continous_ramps/continuous_ramps_new_test.usd",
        #         scale=(1.0, 1.0, 1.0),
        #     ),
        #     init_state=RigidObjectCfg.InitialStateCfg(
        #         pos=(0, -0.7, 0.5),
        #     ),
        # )
        
        env_cfg.scene.robot.init_state.pos = self.robot_init_position # (-1, 0.7, 0.5)
        env_cfg.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        
        return env_cfg

    def update_camera(self):
        """Update camera to follow the robot."""
        robot = self.env.unwrapped.scene["robot"]
        base_pos = robot.data.root_pos_w[0, :]
        base_quat = robot.data.root_quat_w[0, :]
        self.camera.update(base_pos, base_quat)
    
    def get_robot_state(self):
        """
        Get current robot state.
        
        Returns:
            Dictionary with 'position' (numpy array) and 'yaw' (float)
        """
        robot = self.env.unwrapped.scene["robot"]
        robot_pos = robot.data.root_pos_w[0, :3]
        robot_quat = robot.data.root_quat_w[0, :]
        
        # Extract yaw from quaternion
        w, x, y, z = (
            robot_quat[0].item(), 
            robot_quat[1].item(), 
            robot_quat[2].item(), 
            robot_quat[3].item()
        )
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        
        return {
            'position': robot_pos.cpu().numpy(),
            'yaw': yaw
        }
    
    def reset(self):
        """Reset the environment and return initial observation."""
        return self.env.reset()
    
    def step(self, action):
 
        return self.env.step(action)
    
    def get_sim_time(self):
        """Get current simulation time."""
        return (self.env.unwrapped.episode_length_buf[0] * 
                self.env.unwrapped.step_dt)
    
    def get_dt(self): # internal calculation: self.cfg.sim.dt * self.cfg.decimation
        """Get simulation time step."""
        return self.env.unwrapped.step_dt
