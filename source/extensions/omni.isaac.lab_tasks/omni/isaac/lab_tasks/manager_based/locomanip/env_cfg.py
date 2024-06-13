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
from omni.isaac.lab.terrains.config.rough import CUSTOM_TERRAIN_CFG, NOISE_TERRAIN_CFG, TERRAIN_SZ
from omni.isaac.lab_assets.unitree import UNITREE_GO2_Z1_CFG
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR


DEBUG_VIS=False

FLAT_EPISODE_LENGTH=10.0
CURRICULUM_EPISODE_LENGTH=40.0

UNWANTED_CONTACT_BODIES_H=[".*_hip", ".*_thigh", "base", "Head_.*", "gripper.*"]
UNWANTED_CONTACT_BODIES_L=[".*_calf"]
ILLEGAL_CONTACT_BODIES=["base", "Head_.*", "gripper.*"]

legs_joints = UNITREE_GO2_Z1_CFG.actuators["base_legs"].joint_names_expr
arm_joints = UNITREE_GO2_Z1_CFG.actuators["base_arm"].joint_names_expr

MAX_COM_LINSPEED = 1.5
MAX_COM_ANGSPEED = math.pi/2
MAX_PUSHSPEED = 0.5
MAX_EE_Z = 1.0
#TERRAIN_SZ = 4.0 # overwrite

USE_HEIGHT_SCAN = False


##
# Scene definition
##

#flat_terrain = TerrainImporterCfg(
#	prim_path="/World/ground",
#	terrain_type="plane",
#	collision_group=-1,
#	physics_material=sim_utils.RigidBodyMaterialCfg(
#		friction_combine_mode="multiply",
#		restitution_combine_mode="multiply",
#		static_friction=1.0,
#		dynamic_friction=1.0,
#	),
#	visual_material=sim_utils.MdlFileCfg(
#		mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
#		project_uvw=True,
#	),
#	debug_vis=DEBUG_VIS,
#)
flat_terrain = TerrainImporterCfg(
	prim_path="/World/ground",
	terrain_type="usd",
	usd_path="omniverse://localhost/Projects/random_terrain.usd",
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

curriculum_terrain = TerrainImporterCfg(
	prim_path="/World/ground",
	terrain_type="generator",
	terrain_generator=CUSTOM_TERRAIN_CFG.replace(size=(TERRAIN_SZ,TERRAIN_SZ)),
	max_init_terrain_level=0,
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

@configclass
class VelSceneCfg(InteractiveSceneCfg):
	"""Configuration for the terrain scene with a legged robot."""

	# ground terrain
	terrain: TerrainImporterCfg = MISSING
	
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
	contact_sensor = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, track_pose=True)
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
class FlatCommandsCfg:
	ee_pos = mdp.UniformPoseCommandCfg(
		asset_name="robot",
		body_name="gripperStator",
		resampling_time_range=(FLAT_EPISODE_LENGTH, FLAT_EPISODE_LENGTH),
		debug_vis=DEBUG_VIS,
		ranges=mdp.UniformPoseCommandCfg.Ranges(
			pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
		),
	)
	base_velocity = mdp.UniformVelocityCommandCfg(
		asset_name="robot",
		resampling_time_range=(FLAT_EPISODE_LENGTH, FLAT_EPISODE_LENGTH),
		rel_standing_envs=0.02,
		rel_heading_envs=1.0,
		heading_command=True,
		vel_world = True, # prevents drift from accumulating (makes the robot turn in circles and messes up the curriculum level up detection)
		heading_control_stiffness=0.5,
		debug_vis=DEBUG_VIS,
		ranges=mdp.UniformVelocityCommandCfg.Ranges(
			lin_vel_x=(-MAX_COM_LINSPEED, MAX_COM_LINSPEED),
			lin_vel_y=(-MAX_COM_LINSPEED, MAX_COM_LINSPEED),
			ang_vel_z=(-MAX_COM_ANGSPEED, MAX_COM_ANGSPEED),
			heading=(-math.pi, math.pi)
			#lin_vel_x=(0.5, 0.5),
			#lin_vel_y=(-0.0, 0.0),
			#ang_vel_z=(-MAX_COM_ANGSPEED, MAX_COM_ANGSPEED),
			#heading=(-0.0, 0.0)
		)
	)
@configclass
class CurriculumCommandsCfg(FlatCommandsCfg):
	def __post_init__(self):
		super().__post_init__()
		self.ee_pos.resampling_time_range = (CURRICULUM_EPISODE_LENGTH, CURRICULUM_EPISODE_LENGTH)
		self.base_velocity.resampling_time_range = (CURRICULUM_EPISODE_LENGTH, CURRICULUM_EPISODE_LENGTH)


@configclass
class ActionsCfg:
	#legs_pos = mdp.JointEffortActionCfg(asset_name="robot", joint_names=legs_joints, scale=1.0)
	#arm_pos = mdp.JointEffortActionCfg(asset_name="robot", joint_names=arm_joints, scale=1.0)
	legs_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=legs_joints, scale=1.0, use_default_offset=True)
	arm_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=arm_joints, scale=1.0, use_default_offset=True)


@configclass
class ObservationsCfg:
	"""Observation specifications for the MDP."""

	@configclass
	class PolicyCfg(ObservationGroupCfg):
		"""Observations for policy group."""

		# observation terms (order preserved)
		joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
		joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
		binary_contact = ObservationTermCfg(func=mdp.binary_contact, params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_foot")})
		base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
		projected_gravity = ObservationTermCfg(
			func=mdp.projected_gravity,
			noise=Unoise(n_min=-0.05, n_max=0.05),
		)
		base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
		eepos_commands = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "ee_pos"})
		vel_commands = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"})
		actions = ObservationTermCfg(func=mdp.last_action)
		height_scan = ObservationTermCfg(
			func=mdp.height_scan,
			params={"sensor_cfg": SceneEntityCfg("height_scanner")},
			noise=Unoise(n_min=-0.1, n_max=0.1),
			clip=(-1.0, 1.0),
		) if USE_HEIGHT_SCAN else None

		def __post_init__(self):
			self.enable_corruption = True
			self.concatenate_terms = True

	# observation groups
	policy: PolicyCfg = PolicyCfg()


@configclass
class FlatEventCfg:
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
	push_robot = EventTermCfg(
		func=mdp.push_by_setting_velocity,
		mode="interval",
		interval_range_s=(4.0, FLAT_EPISODE_LENGTH),
		params={"velocity_range": {"x": (-MAX_PUSHSPEED, MAX_PUSHSPEED), "y": (-MAX_PUSHSPEED, MAX_PUSHSPEED)}}
	)
@configclass
class CurriculumEventCfg:
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
	push_robot = EventTermCfg(
		func=mdp.push_by_setting_velocity,
		mode="interval",
		interval_range_s=(4.0, FLAT_EPISODE_LENGTH),
		params={"velocity_range": {"x": (-MAX_PUSHSPEED, MAX_PUSHSPEED), "y": (-MAX_PUSHSPEED, MAX_PUSHSPEED)}},
	)


@configclass
class FlatRewardsCfg:
	r_com_linvel_lin = RewardTermCfg(func=mdp.r_com_linvel_lin, params={"command_name": "base_velocity",               "maxerr": 1.3*MAX_COM_LINSPEED}, weight=3.0)
	r_com_angvel_lin = RewardTermCfg(func=mdp.r_com_angvel_lin, params={"command_name": "base_velocity",               "maxerr": MAX_COM_ANGSPEED    }, weight=3.0)
	
	r_joint_acc_lin = RewardTermCfg(func=mdp.r_joint_acc_lin, params={                                                 "maxerr": 4500                }, weight=1.0)
	r_action_rate_lin = RewardTermCfg(func=mdp.r_action_rate_lin, params={                                             "maxerr": 2.0*math.pi         }, weight=1.0)
	r_joint_power_lin = RewardTermCfg(func=mdp.r_joint_power_lin, params={                                             "maxerr": 3000                }, weight=1.0)
	
	r_gait = RewardTermCfg(
		func=mdp.GaitReward,
		weight=0.0,#weight=10.0,
		params={
			"std": 0.1,
			"max_err": 0.2,
			"velocity_threshold": 0.5,
			"synced_feet_pair_names": (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot")),
			"asset_cfg": SceneEntityCfg("robot"),
			"sensor_cfg": SceneEntityCfg("contact_sensor"),
			"command_name": "base_velocity",
		},
	)
	
	#r_flat_orientation_lin = RewardTermCfg(func=mdp.r_flat_orientation_lin, params={                                   "maxerr": 2.0                 }, weight=1.0)
	r_joint_pose_lin = RewardTermCfg(func=mdp.r_joint_pose_lin, params={                                               "maxerr": 2.0*math.pi         }, weight=2.0)
	r_acc_lin = RewardTermCfg(func=mdp.r_acc_lin, params={                                                             "maxerr": 5000                }, weight=1.0)
	#r_height_linmax = RewardTermCfg(func=mdp.r_height_linmax, params={                                                 "maxerr": 0.5                 }, weight=1.0)
	r_nbcontact_lin = RewardTermCfg(
		func=mdp.r_nbcontact_lin, params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=UNWANTED_CONTACT_BODIES_L+UNWANTED_CONTACT_BODIES_H)}, weight=2.0)
	
@configclass
class CurriculumRewardsCfg:
	r_com_linvel_lin = RewardTermCfg(func=mdp.r_com_linvel_lin, params={"command_name": "base_velocity",               "maxerr": 1.3*MAX_COM_LINSPEED}, weight=3.0)
	r_com_angvel_lin = RewardTermCfg(func=mdp.r_com_angvel_lin, params={"command_name": "base_velocity",               "maxerr": MAX_COM_ANGSPEED    }, weight=3.0)
	
	r_joint_acc_lin = RewardTermCfg(func=mdp.r_joint_acc_lin, params={                                                 "maxerr": 4500                }, weight=1.0)
	r_action_rate_lin = RewardTermCfg(func=mdp.r_action_rate_lin, params={                                             "maxerr": 2.0*math.pi         }, weight=1.0)
	r_joint_power_lin = RewardTermCfg(func=mdp.r_joint_power_lin, params={                                             "maxerr": 3000                }, weight=1.0)
	
	r_gait = RewardTermCfg(
		func=mdp.GaitReward,
		weight=0.0,#weight=10.0,
		params={
			"std": 0.1,
			"max_err": 0.2,
			"velocity_threshold": 0.5,
			"synced_feet_pair_names": (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot")),
			"asset_cfg": SceneEntityCfg("robot"),
			"sensor_cfg": SceneEntityCfg("contact_sensor"),
			"command_name": "base_velocity",
		},
	)
	
	#r_flat_orientation_lin = RewardTermCfg(func=mdp.r_flat_orientation_lin, params={                                   "maxerr": 2.0                 }, weight=1.0)
	r_joint_pose_lin = RewardTermCfg(func=mdp.r_joint_pose_lin, params={                                               "maxerr": 2.0*math.pi         }, weight=2.0)
	r_acc_lin = RewardTermCfg(func=mdp.r_acc_lin, params={                                                             "maxerr": 5000                }, weight=1.0)
	#r_height_linmax = RewardTermCfg(func=mdp.r_height_linmax, params={                                                 "maxerr": 0.5                 }, weight=1.0)
	r_nbcontact_lin = RewardTermCfg(
		func=mdp.r_nbcontact_lin, params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=UNWANTED_CONTACT_BODIES_L+UNWANTED_CONTACT_BODIES_H)}, weight=2.0)


@configclass
class FlatTerminationsCfg:
	time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
	#base_contact = TerminationTermCfg(
	#	func=mdp.illegal_contact,
	#	params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=UNWANTED_CONTACT_BODIES_H), "threshold": 1.0},
	#)
@configclass
class CurriculumTerminationsCfg:
	out_of_bounds = TerminationTermCfg(func=mdp.root_out_of_curriculum, time_out=True)
	time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
	#base_contact = TerminationTermCfg(
	#	func=mdp.illegal_contact,
	#	params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=ILLEGAL_CONTACT_BODIES), "threshold": 1.0},
	#)


@configclass
class FlatCurriculumCfg:
	pass
@configclass
class CurriculumCurriculumCfg:
	terrain_levels = CurriculumTermCfg(func=mdp.terrain_levels)


##
# Environment configuration
##


@configclass
class EnvCfg(ManagerBasedRLEnvCfg):
	def __post_init__(self):
		"""Post initialization."""
		
		# general settings
		self.decimation = 4
		
		# simulation settings
		self.sim.dt = 0.005
		self.sim.disable_contact_processing = True
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
@configclass
class FlatEnvCfg(EnvCfg):
	"""Configuration for the locomotion velocity-tracking environment."""

	# Scene settings
	scene = VelSceneCfg(num_envs=4096, env_spacing=2.5, terrain=flat_terrain)
	# Basic settings
	observations = ObservationsCfg()
	actions = ActionsCfg()
	commands = FlatCommandsCfg()
	# MDP settings
	rewards = FlatRewardsCfg()
	terminations = FlatTerminationsCfg()
	events = FlatEventCfg()
	curriculum = FlatCurriculumCfg()
	
	episode_length_s = FLAT_EPISODE_LENGTH
	
	def __post_init__(self):
		super().__post_init__()
		self.scene.robot.spawn.articulation_props.solver_position_iteration_count = 5
@configclass
class CurriculumEnvCfg(EnvCfg):
	"""Configuration for the locomotion velocity-tracking environment."""

	# Scene settings
	scene: VelSceneCfg = VelSceneCfg(num_envs=4096, env_spacing=2.5, terrain=curriculum_terrain)
	# Basic settings
	observations = ObservationsCfg()
	actions = ActionsCfg()
	commands = CurriculumCommandsCfg()
	# MDP settings
	rewards = CurriculumRewardsCfg()
	terminations = CurriculumTerminationsCfg()
	events = CurriculumEventCfg()
	curriculum = CurriculumCurriculumCfg()
	
	episode_length_s = CURRICULUM_EPISODE_LENGTH
	
	def __post_init__(self):
		super().__post_init__()
		self.scene.terrain.terrain_generator.size = (TERRAIN_SZ,TERRAIN_SZ)


@configclass
class FlatEnvCfg_PlayControl(FlatEnvCfg):
	def __post_init__(self):
		super().__post_init__()
		
		self.episode_length_s = 3600
		
		# disable randomization for play
		self.observations.policy.enable_corruption = False
		# remove random pushing event
		self.events.base_external_force_torque = None
		self.events.push_robot = None
		# control velocities directly
		self.commands.base_velocity.heading_command = True
		self.commands.base_velocity.vel_world = False
		# add contact termination
		self.terminations.play_contact = TerminationTermCfg(
			func=mdp.illegal_contact,
			params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=ILLEGAL_CONTACT_BODIES), "threshold": 1.0},
		)
		
		self.commands.base_velocity.resampling_time_range = (self.episode_length_s+1, self.episode_length_s+1)
@configclass
class CurriculumEnvCfg_PlayControl(FlatEnvCfg_PlayControl):
	def __post_init__(self):
		super().__post_init__()

		self.scene.terrain.terrain_generator.num_rows = 5
		#self.scene.terrain.terrain_generator.size = (8.0,8.0)
		self.scene.terrain.min_init_terrain_level=0
		self.scene.terrain.max_init_terrain_level=0
		self.curriculum.terrain_levels = None
		self.terminations.out_of_bounds = None