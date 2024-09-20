# Copyright (c) 2022-2024, The lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg, EventTermCfg, ObservationGroupCfg,ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns, CameraCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

##
# Pre-defined configs
##
# from omni.isaac.lab.terrains.config.rough import CUSTOM_TERRAIN_CFG, NOISE_TERRAIN_CFG, TERRAIN_SZ
from omni.isaac.lab_assets.unitree import UNITREE_GO2_Z1_CFG
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR


DEBUG_VIS=False

_EPISODE_LENGTH=10.0
CURRICULUM_EPISODE_LENGTH=40.0

UNWANTED_CONTACT_BODIES_H=[".*_hip", ".*_thigh", "base", "Head_.*", "gripper.*"]
UNWANTED_CONTACT_BODIES_L=[".*_calf"]
ILLEGAL_CONTACT_BODIES=["base", "Head_.*", "gripper.*"]

legs_joints = UNITREE_GO2_Z1_CFG.actuators["base_legs"].joint_names_expr
arm_joints = UNITREE_GO2_Z1_CFG.actuators["base_arm"].joint_names_expr


#################################
#TABLE 2 in unified policy article 
#Velocity
MIN_CMD_LINSPEED = 0
MAX_CMD_LINSPEED = 0.9 #1.5
MAX_CMD_YAWSPEED = 1.0 #math.pi/2
#Radius
L_MIN = 0.2
L_MAX= 0.7
#Pitch
P_MAX = 2*math.pi/5
#Yaw
Y_MAX = 3*math.pi/5
T_TRAJ_MIN = 1 # seconds
T_TRAJ_MAX = 3 # seconds

#################################


MAX_PUSHSPEED = 0.5
MAX_EE_Z = 1.0
#TERRAIN_SZ = 4.0 # overwrite

USE_HEIGHT_SCAN = False
USE_CONTACT_SEN = False


##
# Scene definition
##

# flat_terrain = TerrainImporterCfg(
#     prim_path="/World/ground",
#     terrain_type="plane",
#     collision_group=-1,
#     physics_material=sim_utils.RigidBodyMaterialCfg(
#         friction_combine_mode="multiply",
#         restitution_combine_mode="multiply",
#         static_friction=1.0,
#         dynamic_friction=1.0,
#     ),
#     visual_material=sim_utils.MdlFileCfg(
#         mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
#         project_uvw=True,
#     ),
#     debug_vis=DEBUG_VIS,
# )

# terrain = TerrainImporterCfg(
# 	prim_path="/World/ground",
# 	terrain_type="usd",
# 	usd_path="omniverse://localhost/Projects/random_terrain.usd",
# 	collision_group=-1,
# 	physics_material=sim_utils.RigidBodyMaterialCfg(
# 		friction_combine_mode="multiply",
# 		restitution_combine_mode="multiply",
# 		static_friction=1.0,
# 		dynamic_friction=1.0,
# 	),
# 	visual_material=sim_utils.MdlFileCfg(
# 		mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
# 		project_uvw=True,
# 	),
# 	debug_vis=DEBUG_VIS,
# )

# curriculum_terrain = TerrainImporterCfg(
# 	prim_path="/World/ground",
# 	terrain_type="generator",
# 	terrain_generator=CUSTOM_TERRAIN_CFG.replace(size=(TERRAIN_SZ,TERRAIN_SZ)),
# 	max_init_terrain_level=0,
# 	collision_group=-1,
# 	physics_material=sim_utils.RigidBodyMaterialCfg(
# 		friction_combine_mode="multiply",
# 		restitution_combine_mode="multiply",
# 		static_friction=1.0,
# 		dynamic_friction=1.0,
# 	),
# 	visual_material=sim_utils.MdlFileCfg(
# 		mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
# 		project_uvw=True,
# 	),
# 	debug_vis=DEBUG_VIS,
# )

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=DEBUG_VIS,
    )
    
    # robots
    robot: ArticulationCfg = UNITREE_GO2_Z1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=DEBUG_VIS,
        mesh_prim_paths=["/World/ground"],
    ) if USE_HEIGHT_SCAN else None

    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", 
        history_length=3, 
        track_air_time=True, 
        track_pose=True
    ) if USE_CONTACT_SEN else None


    # frame
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gripperStator",
                name="end_effector",
            ),
        ],
    )
    feet_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path=f"{{ENV_REGEX_NS}}/Robot/{foot_name}", name=foot_name) for foot_name in ["FL_foot","FR_foot","RL_foot","RR_foot"]
        ],
    )
    
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

##
# MDP settings
##

@configclass
class CommandsCfg:

    # Modify ee_pos_cmd
    ee_pos_cmd = mdp.UniformPoseSphereCommandCfg( #suppose to return quaternion
    	asset_name="robot",
    	body_name="gripperStator",
    	resampling_time_range=(T_TRAJ_MIN, T_TRAJ_MAX),
    	debug_vis=True,
    	ranges=mdp.UniformPoseSphereCommandCfg.Ranges(
    		l_radius=(L_MIN, L_MAX),
    		s_pitch=(-P_MAX, P_MAX), 
    		s_yaw=(-Y_MAX, Y_MAX), 
    		roll=(-math.pi, math.pi), 
    		pitch=(-math.pi/2, math.pi/2), 
    		yaw=(-math.pi, math.pi)
    	),
    )

    # ee_pos_cmd
    # ee_pos_cmd = mdp.UniformPoseCommandCfg( #suppose to return quaternion
    #     asset_name="robot",
    #     body_name="gripperStator",
    #     resampling_time_range=(T_TRAJ_MIN, T_TRAJ_MAX),
    #     debug_vis=DEBUG_VIS,
    #     ranges=mdp.UniformPoseCommandCfg.Ranges(
    #         pos_x=(-0.5, 0.5),
    #         pos_y=(-0.5, 0.5), 
    #         pos_z=(0.2, 1), 
    #         roll=(-math.pi, math.pi), 
    #         pitch=(-math.pi, math.pi), 
    #         yaw=(-math.pi, math.pi)
    #     ),
    # )


    base_velocity_cmd = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(_EPISODE_LENGTH, _EPISODE_LENGTH),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=DEBUG_VIS,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-MAX_CMD_LINSPEED, MAX_CMD_LINSPEED),
            #lin_vel_y=(-MAX_CMD_LINSPEED, MAX_CMD_LINSPEED),
            lin_vel_y=(-0.0, 0.0),
            ang_vel_z=(-MAX_CMD_YAWSPEED, MAX_CMD_YAWSPEED),
            heading=(-math.pi, math.pi)
            #lin_vel_x=(0.5, 0.5),
            #lin_vel_y=(-0.0, 0.0),
            #ang_vel_z=(-MAX_CMD_ANGSPEED, MAX_CMD_ANGSPEED),
            #heading=(-0.0, 0.0)
        )
    )

# @configclass
# class CurriculumCommandsCfg(CommandsCfg):
# 	def __post_init__(self):
# 		super().__post_init__()
# 		self.ee_pos_cmd.resampling_time_range = (CURRICULUM_EPISODE_LENGTH, CURRICULUM_EPISODE_LENGTH)
# 		self.base_velocity_cmd.resampling_time_range = (CURRICULUM_EPISODE_LENGTH, CURRICULUM_EPISODE_LENGTH)


@configclass
class ActionsCfg:
    #legs_pos = mdp.JointEffortActionCfg(asset_name="robot", joint_names=legs_joints, scale=1.0)
    #arm_pos = mdp.JointEffortActionCfg(asset_name="robot", joint_names=arm_joints, scale=1.0)
    legs_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=legs_joints, scale=1.0, use_default_offset=True)
    arm_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=arm_joints, scale=1.0, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    # object_pos = ObservationTermCfg(
    #         func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")}
    #     )

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy group."""

        #Base state R5(roll, pitch and ang velocities)
        base_ori_roll_pitch = ObservationTermCfg(func=mdp.base_ori_roll_pitch, noise=Unoise(n_min=-0.2, n_max=0.2))
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))

        #base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        # projected_gravity = ObservationTermCfg(
        #     func=mdp.projected_gravity,
        #     noise=Unoise(n_min=-0.05, n_max=0.05),
        # )
        
        #Arm State R12 (joint pose and velocity) (maybe R14)
        #Leg State R28 (joint pos, joint vel, foot contact)
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        # TODO :add foot contact info


        #Last action R18 (ee pose cmd, ee orientation cmd, base vel cmd)
        actions = ObservationTermCfg(func=mdp.last_action)

        #end-effector position and orientation command
        eepos_commands = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "ee_pos_cmd"})

        #base velocity command [v_cmd, w_yaw_cmd]
        vel_commands = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity_cmd"})

        #Environment extrinsics R20 (...) for sim-to-real transfer



        #OTHER
        #
        # binary_contact = ObservationTermCfg(func=mdp.binary_contact, params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_foot")})
        # height_scan = ObservationTermCfg(
        # 	func=mdp.height_scan,
        # 	params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        # 	noise=Unoise(n_min=-0.1, n_max=0.1),
        # 	clip=(-1.0, 1.0),
        # ) if USE_HEIGHT_SCAN else None

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"), 
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add"
        },
    )
    # reset
    reset_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }
        },
    )
    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    # interval
    # push_robot = EventTermCfg(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(4.0, _EPISODE_LENGTH),
    #     params={"velocity_range": {"x": (-MAX_PUSHSPEED, MAX_PUSHSPEED), "y": (-MAX_PUSHSPEED, MAX_PUSHSPEED)}}
    # )


@configclass
class RewardsCfg:


    #################################### LOCOMOTION REWARD ##############################################
    #r_loco_following
    r_cmd_linvel_x = RewardTermCfg(func=mdp.track_lin_vel_x_yaw_frame, params={"command_name": "base_velocity_cmd"}, weight=0.5)
    r_cmd_angvel_yaw = RewardTermCfg(func=mdp.track_ang_vel_z_world_exp, params={"command_name": "base_velocity_cmd"}, weight=0.15)
    
    #r_loco_energy
    r_loco_energy = RewardTermCfg(func=mdp.r_joint_leg_power, weight=0.00005)

    #r_loco_alive``
    r_loco_alive = RewardTermCfg(func=mdp.is_alive, params={"command_name": "base_velocity_cmd"}, weight=1.0)
    #####################################################################################################

    ################################### MANIPULATION REWARD #############################################

    r_manip_following = RewardTermCfg(func=mdp.track_pose_orientation, params={"command_name": "ee_pos_cmd",
                                                                               "asset_cfg": SceneEntityCfg("robot", body_names="gripperStator")}, weight=0.5)
    r_manip_energy = RewardTermCfg(func=mdp.r_joint_arm_power, weight=0.004)
    # r_manip_alive = 0


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)

    # Minimum height termination only working on flat terrain
    t_min_base_heigh = TerminationTermCfg(func=mdp.root_height_below_minimum, params={"minimum_height": 0.10})

    #termination according to base position and ee pose cmd 
    #t_wrong_ee_cmd = TerminationTermCfg(func=mdp.wrong_ee_cmd_for_base_oritation, params={"command_name": "ee_pos_cmd"})


@configclass
class CurriculumCfg:
    pass



##
# Environment configuration
##


@configclass
class FlatEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = _EPISODE_LENGTH
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True


        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.sim.physics_material = self.scene.terrain.physics_material

        # articulation settings
        #self.scene.robot.spawn.articulation_props.fix_root_link=True
        self.scene.robot.spawn.articulation_props.solver_position_iteration_count = 20 # a high number prevents objects from going through the ground
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True


        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_sensor is not None:
            self.scene.contact_sensor.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False