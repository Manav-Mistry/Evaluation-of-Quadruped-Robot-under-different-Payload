# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math
import isaaclab.sim as sim_utils
# import isaaclab.terrains as terrain_gen
from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns

from isaaclab.markers.config import VisualizationMarkersCfg
# from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns
from isaaclab.assets import AssetBaseCfg
##
# Pre-defined configs
##
from isaaclab_assets.robots.spot import SPOT_CFG  # isort: skip

USD_PATH_BASELINE_FLAT_NOWALL = "/home/manav/Desktop/NoWalls/ex-12_test_courses_baseline_flat_NOWALL/baseline_flat_with_only_colliders.usd"
USD_PATH_BASELINE_FLAT ="/home/manav/Desktop/Test course 3D models/base_line_flat/base_line_flat_with_only_colliders.usd"
USD_PATH_CONTINUOUS_RAMPS = "/home/manav/Desktop/Test course 3D models/continous_ramps/continous_ramps_with_only_colliders.usd"
USD_PATH_CROSSING_RAMPS = "/home/manav/Desktop/Test course 3D models/crossing_ramps/crossing_ramps_with_only_colliders.usd"
USD_PATH_INCLINE_RAMP = "/home/manav/Desktop/Test course 3D models/incline_crossover/baseline_incline_crossover_with_only_colliders.usd"
USD_PATH_CROSSING_RAMPS_INCLINE_CROSSOVER = "/home/manav/Desktop/Test course 3D models/crossing_ramps_incline_crossover/crossing_ramps_incline_crossover_with_only_colliders.usd"
USD_PATH_CROSSING_RAMPS_INCLINE_CROSSOVER_LARGER = "/home/manav/Desktop/extra_padding/crossing_ramps_incline_crossover_larger_with_only_colliders.usd"
USD_PATH_CONTINUOUS_INCLINE_RAMPS_LARGER = "/home/manav/Desktop/extra_padding/continuous_incline/continuous_incline_larger_with_only_colliders.usd"

RAY_CASTER_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "hit": sim_utils.SphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    },
)

@configclass
class SpotActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class SpotCommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.6), lin_vel_y=(-1.5, 1.5), ang_vel_z=(-2.0, 2.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class SpotObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # `` observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class SpotEventCfg:
    """Configuration for randomization."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # add_base_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="body"),
    #         "mass_distribution_params": (-5.0, 5.0),
    #         "operation": "add",
    #     },
    # )
    # NEW
    # base_com = EventTerm(
    #     func=mdp.randomize_rigid_body_com,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="body"),
    #         "com_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1)},
    #     },
    # )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (-0.0, 0.0)},
            "velocity_range": {
                "x": (0, 0),
                "y": (0, 0),
                "z": (0, 0),
                "roll": (0, 0),
                "pitch": (0, 0),
                "yaw": (0, 0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=spot_mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # interval
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0), 
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
    #     },
    # )


@configclass
class SpotRewardsCfg:
    # -- task 
    base_linear_velocity = RewardTermCfg(
        func=spot_mdp.base_linear_velocity_reward,
        weight=1.0,
        params={"std": 1.0, "ramp_rate": 1, "ramp_at_vel": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )

    base_angular_velocity = RewardTermCfg(
        func=spot_mdp.base_angular_velocity_reward,
        weight=0.5,
        params={"std": 2.0, "asset_cfg": SceneEntityCfg("robot")},
    )

    air_time = RewardTermCfg(
        func=spot_mdp.air_time_reward,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "mode_time": 0.3,
            "velocity_threshold": 0.5,
        },
    )

    gait = RewardTermCfg(
        func=spot_mdp.GaitReward,
        weight=2.5,
        params={
            "std": 0.1,
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "synced_feet_pair_names": (("fl_foot", "hr_foot"), ("fr_foot", "hl_foot")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )


    # -- penalties
    # NEW
    # base_height = RewardTermCfg(
    #     func=mdp.base_height_l2,
    #     weight=-1.0,
    #     params={
    #         "target_height": 0.5,
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "sensor_cfg": SceneEntityCfg("height_scanner"),
    #     },
    # )
    action_smoothness = RewardTermCfg(func=spot_mdp.action_smoothness_penalty, weight=-0.1)
   
    lin_vel_z_l2 = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05)
    
    undesired_contacts = RewardTermCfg(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*uleg"), "threshold": 1.0},
    )
    foot_slip = RewardTermCfg(
        func=spot_mdp.foot_slip_penalty,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )
    joint_acc = RewardTermCfg(
        func=spot_mdp.joint_acceleration_penalty,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_pos = RewardTermCfg(
        func=spot_mdp.joint_position_penalty,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
        },
    )
    joint_torques = RewardTermCfg(
        func=spot_mdp.joint_torques_penalty,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_vel = RewardTermCfg(
        func=spot_mdp.joint_velocity_penalty,
        weight=-1.0e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )


@configclass
class SpotTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # body_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="body"), "threshold": 1.0},
    # )
    # terrain_out_of_bounds = DoneTerm(
    #     func=mdp.terrain_out_of_bounds,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
    #     time_out=True,
    # )


@configclass
class SpotRoughEnvCfg(LocomotionVelocityRoughEnvCfg):

    # Basic settings
    observations: SpotObservationsCfg = SpotObservationsCfg()
    actions: SpotActionsCfg = SpotActionsCfg()
    commands: SpotCommandsCfg = SpotCommandsCfg()

    # MDP setting
    rewards: SpotRewardsCfg = SpotRewardsCfg()
    terminations: SpotTerminationsCfg = SpotTerminationsCfg()
    events: SpotEventCfg = SpotEventCfg()

    # Viewer
    viewer = ViewerCfg(eye=(10.5, 10.5, 0.3), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # general settings
        self.decimation = 10  # 50 Hz
        self.episode_length_s = 20.0  
        # simulation settings
        self.sim.dt = 0.002  # 500 Hz
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt

        # switch robot to Spot-d
        self.scene.robot = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # terrain
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            debug_vis=False,
            visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0), roughness=0.5), #diffuse_color=(1, 1, 1)
        )

        self.scene.custom_ramp = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/CustomRamp",
            spawn=sim_utils.UsdFileCfg(
                usd_path=USD_PATH_BASELINE_FLAT_NOWALL ,
                scale=(1.0, 1.0, 1.0),

            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, 0.0, -0.06),
            ),
        )


        # no height scan
        # num_rays = (size_x / resolution + 1) * (size_y / resolution + 1)
        #          = (1.6 / 0.1 + 1) * (1.0 / 0.1 + 1)
        #          = (16 + 1) * (10 + 1)
        #          = 17 * 11 = 187

        self.scene.height_scanner = MultiMeshRayCasterCfg (
            prim_path="{ENV_REGEX_NS}/Robot/body",
            update_period= self.decimation * self.sim.dt,
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0, 0.0, 20.0)),
            mesh_prim_paths=[
                "/World/ground",
                MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/CustomRamp", merge_prim_meshes=True),
            ],
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            visualizer_cfg=RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/RayCaster"),

        )

        self.scene.height_scanner.update_period = self.decimation * self.sim.dt


class SpotRoughEnvCorrectedCfg_PLAY(SpotRoughEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None

        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
