# pyright: reportAttributeAccessIssue=false
# pyright: reportOptionalMemberAccess = false

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from isaaclab.assets import RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from my_utils import ThirdPersonCamera
from my_utils import attach_payload_to_robot

import numpy as np


class SpotRoughDemo:
    """Interactive demo for Spot robot with terrain navigation."""
    
    def __init__(self, env_cfg_class, checkpoint_path, terrain_cfg):
      
        cube_cfg = self._create_payload_config()
        
        # Setup environment configuration
        env_cfg = self._setup_environment_config(
            env_cfg_class, 
            cube_cfg, 
            terrain_cfg
        )
        
        # Attach payload to robot
        for env_idx in range(env_cfg.scene.num_envs):
            attach_payload_to_robot(
                robot_body_path=f"/World/envs/env_{env_idx}/Robot/body",
                payload_path=f"/World/envs/env_{env_idx}/Cube",
                env_idx= env_idx,
                local_offset=(0.0, 0.4, 0.14343),
            )
        
        # Create environment
        self.env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device
        
        # Load trained policy
        self.policy = torch.jit.load(checkpoint_path, map_location=self.device)
        
        # Setup camera
        self.camera = ThirdPersonCamera()
        self.camera.set_local_transform(
            torch.tensor([-2.5, 0.0, 0.8], device=self.device)
        )
        self.camera.activate()
        
        # Initialize commands
        self.commands = torch.zeros(env_cfg.scene.num_envs, 3, device=self.device)
    
    def _create_payload_config(self):
        """Create configuration for the payload cube."""
        return RigidObjectCfg(
            prim_path="/World/envs/env_0/Cube",
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
            init_state=RigidObjectCfg.InitialStateCfg(),
        )
    
    def _setup_environment_config(self, env_cfg_class, cube_cfg, terrain_cfg):
        """Setup and configure the environment."""
        env_cfg = env_cfg_class()
        
        # Basic environment settings
        env_cfg.scene.num_envs = 1
        env_cfg.episode_length_s = 1000000
        env_cfg.curriculum = None
        
        # Command ranges
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (-2, 5.2)
        env_cfg.commands.base_velocity.ranges.heading = (-1.0, 1.0)
        
        # Add payload
        env_cfg.scene.payload = cube_cfg
        
        # Setup terrain
        env_cfg.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=terrain_cfg,
            max_init_terrain_level=0,
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
        
        # Robot initial state
        env_cfg.scene.robot.init_state.pos = (-2, 0.0, 0.5)
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
        """
        Step the environment with given action.
        
        Args:
            action: Action tensor from policy
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        return self.env.step(action)
    
    def get_sim_time(self):
        """Get current simulation time."""
        return (self.env.unwrapped.episode_length_buf[0] * 
                self.env.unwrapped.step_dt)
    
    def get_dt(self):
        """Get simulation time step."""
        return self.env.unwrapped.step_dt
