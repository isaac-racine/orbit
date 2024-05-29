# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from math import pi, sqrt
from dataclasses import MISSING

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.orbit.envs import RLTaskEnvCfg
from omni.isaac.orbit.managers import CurriculumTermCfg
from omni.isaac.orbit.managers import EventTermCfg
from omni.isaac.orbit.managers import ObservationGroupCfg
from omni.isaac.orbit.managers import ObservationTermCfg
from omni.isaac.orbit.managers import RewardTermCfg
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.managers import TerminationTermCfg
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.sensors import ContactSensorCfg, RayCasterCfg, patterns, CameraCfg
from omni.isaac.orbit.terrains import TerrainImporterCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.noise import AdditiveUniformNoiseCfg as Unoise

import omni.isaac.orbit_tasks.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from omni.isaac.orbit.terrains.config.rough import VEL_CUSTOM_TERRAIN_CFG, ROUGH_TERRAINS_CFG
from omni.isaac.orbit_assets.unitree import UNITREE_GO2_CFG


DEBUG_VIS=False
EPISODE_LENGTH=20.0

ILLEGAL_CONTACT_BODIES=["base", "Head_.*", ".*_hip", ".*_thigh", ".*_calf"]
UNWANTED_CONTACT_BODIES=["base", "Head_.*", ".*_hip", ".*_thigh", ".*_calf"]
WANTED_CONTACT_BODIES=[".*_foot"]
MAX_COMSPEED = 2.0
MAX_PUSHSPEED = 1.0


##
# Scene definition
##


@configclass
class VelSceneCfg(InteractiveSceneCfg):
	"""Configuration for the terrain scene with a legged robot."""

	# ground terrain
	#terrain = TerrainImporterCfg(
	#	prim_path="/World/ground",
	#	terrain_type="generator",
	#	terrain_generator=VEL_CUSTOM_TERRAIN_CFG,
	#	min_init_terrain_level=0,
	#	max_init_terrain_level=0,
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
	# plane terrain
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
	robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
	# sensors
	height_scanner = RayCasterCfg(
		prim_path="{ENV_REGEX_NS}/Robot/base",
		offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
		attach_yaw_only=True,
		pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
		debug_vis=DEBUG_VIS,
		mesh_prim_paths=["/World/ground"],
	)
	contact_sensor = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, track_pose=True, debug_vis=DEBUG_VIS)

##
# MDP settings
##


@configclass
class CommandsCfg:
	"""Command specifications for the MDP."""

	base_velocity = mdp.UniformVelocityCommand2Cfg(
		asset_name="robot",
		resampling_time_range=(math.inf, math.inf),
		debug_vis=DEBUG_VIS,
		ranges=mdp.UniformVelocityCommandCfg.Ranges(
			lin_vel_x=(-MAX_COMSPEED, MAX_COMSPEED), lin_vel_y=(-MAX_COMSPEED, MAX_COMSPEED), heading=(-math.pi, math.pi)
		),
	)

@configclass
class ActionsCfg:
	"""Action specifications for the MDP."""
	
	#joint_act = mdp.JointEffortActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25)
	#joint_act = mdp.JointEffortActionCfg(asset_name="robot", joint_names=[".*"])
	joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=False)
	#joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=False)

@configclass
class ObservationsCfg:
	"""Observation specifications for the MDP."""

	@configclass
	class PolicyCfg(ObservationGroupCfg):
		"""Observations for policy group."""

		# observation terms (order preserved)
		velocity_commands = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"})
		actions = ObservationTermCfg(func=mdp.last_action)
		
		base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
		base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
		projected_gravity = ObservationTermCfg(
			func=mdp.projected_gravity,
			noise=Unoise(n_min=-0.05, n_max=0.05),
		)
		joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
		joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
		feet_forces = ObservationTermCfg(
			func=mdp.received_forces,
			params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_foot")},
			noise=Unoise(n_min=-1, n_max=1)
		)
		
		height_scan = ObservationTermCfg(
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
		params={"asset_cfg": SceneEntityCfg("robot", body_names="base"), "mass_range": (-1.0, 3.0), "operation": "add"},
	)

	# reset
	base_external_force_torque = EventTermCfg(
		func=mdp.apply_external_force_torque,
		mode="reset",
		params={
			"asset_cfg": SceneEntityCfg("robot", body_names="base"),
			"force_range": (0.0, 0.0),
			"torque_range": (-0.0, 0.0),
		},
	)
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
	#push_robot = EventTermCfg(
	#	func=mdp.push_by_setting_velocity,
	#	mode="interval",
	#	interval_range_s=(3.0, min(5.0, EPISODE_LENGTH)),
	#	params={"velocity_range": {"x": (-MAX_PUSHSPEED, MAX_PUSHSPEED), "y": (-MAX_PUSHSPEED, MAX_PUSHSPEED)}},
	#)

@configclass
class RewardsCfg:
	"""Reward terms for the MDP."""
	
	"""
	# -- task
	#track_lin_vel_xy_exp = RewardTermCfg(func=mdp.track_lin_vel_xy_exp, weight=10.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
	#track_dir_xy_exp = RewardTermCfg(func=mdp.track_dir_xy_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
	#track_heading_exp = RewardTermCfg(func=mdp.track_heading_exp, weight=0.75, params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
	#track_speed_xy_exp = RewardTermCfg(func=mdp.track_speed_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
	## -- penalties
	#lin_vel_z_l2 = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-2.0)
	#ang_vel_xy_l2 = RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05)
	#dof_torques_l2 = RewardTermCfg(func=mdp.joint_torques_l2, weight=-0.0002)
	#dof_acc_l2 = RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7)
	#action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)
	#feet_air_time = RewardTermCfg(
	#	func=mdp.feet_air_time,
	#	weight=0.01,
	#	params={
	#		"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_foot"),
	#		"command_name": "base_velocity",
	#		"threshold": 0.5,
	#	},
	#)
	#flat_orientation_l2 = RewardTermCfg(func=mdp.flat_orientation_l2, weight=-0.1)
	"""
	
	"""
	# command tracking rewards
	track_dir_xy_unit = RewardTermCfg(func=mdp.track_dir_xy_unit, weight=5.0, params={"command_name": "base_velocity"})
	track_heading_unit = RewardTermCfg(func=mdp.track_heading_unit, weight=5.0, params={"command_name": "base_velocity"})
	track_speed_xy_unit = RewardTermCfg(func=mdp.track_speed_xy_unit, weight=2.0, params={"command_name": "base_velocity",
		"max_err": 0.5
	})
	
	# task shaping penalties
	#undesired_contacts_unit = RewardTermCfg(func=mdp.undesired_contacts_unit, weight=-10.0, params={ # 1 if contact 0 otherwise
	#	"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=UNWANTED_CONTACT_BODIES),
	#	"treshold": 0.1 # N
	#})
	#unflat_orientation_unit = RewardTermCfg(func=mdp.unflat_orientation_unit, weight=-1.0, params={
	#	"min_dot": 0.1 # -1 is perfectly flat, 1 is flipped over
	#})
	joint_acc_unit = RewardTermCfg(func=mdp.joint_acc_unit, weight=-0.5, params={ # proportional to max acceleration for all joints
		"max_acc": 2.0e3
	})
	joint_vel_unit = RewardTermCfg(func=mdp.joint_vel_unit, weight=-0.5, params={ # proportional to max velocity for all joints
		"max_vel": 0.25e2
	})
	joint_torques_unit = RewardTermCfg(func=mdp.joint_torques_unit, weight=-0.1, params={ # proportional to max torque for all joints
		"max_torque": 40 # Nm (45 Nm is peak for GO2)
	})
	action_rate_unit = RewardTermCfg(func=mdp.action_rate_unit, weight=-0.5, params={
		"max_act": 1.0e1
	})
	#prolonged_contacts_unit = RewardTermCfg(func=mdp.prolonged_contacts_unit, weight=-0.1, params={ # proportional to max time for all contacts
	#	"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=UNWANTED_CONTACT_BODIES),
	#	"max_time": 4 # s
	#})
	#contact_forces_unit = RewardTermCfg(func=mdp.contact_forces_unit, weight=-0.001, params={ # proportional to max force for all contacts
	#	"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=UNWANTED_CONTACT_BODIES),
	#	"max_force": 50 # N
	#})
	
	# task shaping rewards
	#joint_goodpose_unit = RewardTermCfg(func=mdp.joint_goodpose_unit, weight=1.0)
	#flat_orientation_unit = RewardTermCfg(func=mdp.flat_orientation_unit, weight=0.1, params={
	#	"max_dot": -0.75 # -1 is perfectly flat, 1 is flipped over
	#})
	## reward only ground / foot contact
	#desired_contacts_unit = RewardTermCfg(func=mdp.desired_contacts_unit, weight=0.1, params={ # 1 if contact 0 otherwise
	#	"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=WANTED_CONTACT_BODIES),
	#	"treshold": 0.1 # N
	#})
	"""
	
	
	#p_contact_sparse = RewardTermCfg(func=mdp.p_contact_sparse, params={                                               
	#	"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=UNWANTED_CONTACT_BODIES)},                                                                    weight=0.3         )
	#rp_joint_acc_bilin = RewardTermCfg(func=mdp.rp_joint_acc_bilin, has_two=True,               params={               "minerr": 20,   "maxerr": 500         }, weight=1, weight2=1)
	#rp_flat_orientation_bilin = RewardTermCfg(func=mdp.rp_flat_orientation_bilin, has_two=True, params={               "minerr": 0.3,  "maxerr": 2           }, weight=1, weight2=1)
	#																									               
	#rp_vel_xy_bilin = RewardTermCfg(func=mdp.rp_vel_xy_bilin, has_two=True, params={"command_name": "base_velocity",   "minerr": 0.25, "maxerr": MAX_COMSPEED}, weight=1, weight2=1)
	#rp_heading_bilin = RewardTermCfg(func=mdp.rp_heading_bilin, has_two=True, params={"command_name": "base_velocity", "minerr": pi/2, "maxerr": pi          }, weight=1, weight2=1)
	
	
	#r_joint_acc_lin = RewardTermCfg(func=mdp.r_joint_acc_lin, params={                             "maxerr": 500             }, weight=1)
	#r_flat_orientation_lin = RewardTermCfg(func=mdp.r_flat_orientation_lin, params={               "maxerr": 2               }, weight=1)
	#r_joint_pose_lin = RewardTermCfg(func=mdp.r_joint_pose_lin, params={                           "maxerr": pi*1.5          }, weight=1)
	#
	#r_vel_xy_lin = RewardTermCfg(func=mdp.r_vel_xy_lin, params={"command_name": "base_velocity",   "maxerr": MAX_COMSPEED*1.5}, weight=1)
	#r_heading_lin = RewardTermCfg(func=mdp.r_heading_lin, params={"command_name": "base_velocity", "maxerr": pi              }, weight=1)
	

@configclass
class TerminationsCfg:
	"""Termination terms for the MDP."""
	
	#out_of_bounds = TerminationTermCfg(func=mdp.root_out_of_curriculum, time_out=True)
	time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
	illegal_contact = TerminationTermCfg(
		func=mdp.illegal_contact,
		params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=ILLEGAL_CONTACT_BODIES), "threshold": 0.1},
	)
	
	#c_contact_err = TerminationTermCfg(func=mdp.c_contact_err, params={
	#	"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=ILLEGAL_CONTACT_BODIES), "maxprob": 0.01})

@configclass
class CurriculumCfg:
	"""Curriculum terms for the MDP."""

	#terrain_levels = CurriculumTermCfg(func=mdp.terrain_levels_vel2)


##
# Environment configuration
##


@configclass
class UnitreeGo2VelCustomEnvCfg(RLTaskEnvCfg):
	"""Configuration for the locomotion velocity-tracking environment."""

	# Scene settings
	scene: VelSceneCfg = VelSceneCfg(num_envs=4096, env_spacing=2.5)
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
		self.episode_length_s = EPISODE_LENGTH
		
		# simulation settings
		self.sim.dt = 0.005
		self.sim.disable_contact_processing = True
		self.sim.physics_material = self.scene.terrain.physics_material
		
		# articulation settings
		self.scene.robot.spawn.articulation_props.solver_position_iteration_count = 20 # a high number prevents objects from going through the ground
		self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
		
		# update sensor update periods
		# we tick all the sensors based on the smallest update period (physics update period)
		if self.scene.height_scanner is not None:
			self.scene.height_scanner.update_period = self.decimation * self.sim.dt
		if self.scene.contact_sensor is not None:
			self.scene.contact_sensor.update_period = self.sim.dt

@configclass
class UnitreeGo2VelCustomEnvCfg_PLAYCONTROL(UnitreeGo2VelCustomEnvCfg):
	def __post_init__(self):
		# post init of parent
		super().__post_init__()
		
		self.episode_length_s = 3600
		
		self.scene.terrain.terrain_generator.num_rows = 5
		#self.scene.terrain.terrain_generator.size = (8.0,8.0)
		self.scene.terrain.min_init_terrain_level=0
		self.scene.terrain.max_init_terrain_level=0
		self.curriculum.terrain_levels = None
		self.terminations.out_of_bounds = None
		
		# disable randomization for play
		self.observations.policy.enable_corruption = False
		# remove random pushing event
		self.events.base_external_force_torque = None
		self.events.push_robot = None
		
		self.episode_length_s = 3600
		
		self.commands.base_velocity = mdp.UserVelocityCommandCfg(
			asset_name="robot",
			debug_vis=DEBUG_VIS,
			resampling_time_range=(math.inf, math.inf), # not used 
		)