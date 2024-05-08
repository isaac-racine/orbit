# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
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
from omni.isaac.orbit_assets.unitree import UNITREE_GO2_CFG
from omni.isaac.orbit.terrains.config.rough import CUSTOM_TERRAINS_CFG  # isort: skip


EPISODE_LENGTH=20.0


##
# Scene definition
##


@configclass
class VelSceneCfg(InteractiveSceneCfg):
	"""Configuration for the terrain scene with a legged robot."""

	# ground terrain
	terrain = TerrainImporterCfg(
		prim_path="/World/ground",
		terrain_type="generator",
		collision_group=-1,
		max_init_terrain_level=0,
		init_terrain_row=0,
		linear_progression=True,
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
		debug_vis=True,
		terrain_generator=CUSTOM_TERRAINS_CFG
	)
	
	# robots
	robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
	
	# sensors
	height_scanner = RayCasterCfg(
		prim_path="{ENV_REGEX_NS}/Robot/base",
		offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
		attach_yaw_only=True,
		pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
		debug_vis=False,
		mesh_prim_paths=["/World/ground"],
	)
	contact_sensor = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=0, track_air_time=True)
	
	# lights
	light = AssetBaseCfg(
		prim_path="/World/light",
		spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
	)
	sky_light = AssetBaseCfg(
		prim_path="/World/skyLight",
		spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
	)

	# Camera (takes alot of processing power --> might need to reduce the numof envs for training) 

	#Pinhole camera type
	# camera_front = CameraCfg(
	# 	prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
	# 	update_period=0.1,
	# 	height=480,
	# 	width=640,
	# 	data_types=["rgb", "distance_to_image_plane"],
	# 	spawn=sim_utils.PinholeCameraCfg(
	# 		focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
	# 	),
	# 	offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),	#To modify for real offset on Go2
	# )

	# The fisheye camera type (is not recognised and the sim is defaulting to a pinhole camera type)
	# camera_360 = CameraCfg(
	# 	prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
	# 	update_period=0.1,
	# 	height=480,
	# 	width=640,
	# 	data_types=["rgb", "distance_to_image_plane"],
	# 	spawn=sim_utils.FisheyeCameraCfg(),
	# 	offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),	#To modify for real offset on Go2
	# )


##
# MDP settings
##


@configclass
class CommandsCfg:
	"""Command specifications for the MDP."""

	base_velocity = mdp.UniformVelocityCommandCfg(
		asset_name="robot",
		resampling_time_range=(4.0,EPISODE_LENGTH-4.0),
		rel_standing_envs=0.02,
		rel_heading_envs=1.0,
		heading_command=True,
		heading_control_stiffness=0.5,
		debug_vis=True,
		ranges=mdp.UniformVelocityCommandCfg.Ranges(
			lin_vel_x=(-2.0, 2.0), lin_vel_y=(-2.0, 2.0), ang_vel_z=(-math.pi, math.pi), heading=(-math.pi, math.pi)
		),
	)


@configclass
class ActionsCfg:
	"""Action specifications for the MDP."""

	joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


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
		root_quat_w = ObservationTermCfg(func=mdp.root_quat_w, noise=Unoise(n_min=-0.05, n_max=0.05))
		joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
		joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
		feet_forces = ObservationTermCfg(
			func=mdp.received_forces,
			params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_foot")},
			noise=Unoise(n_min=-0.1, n_max=0.1)
		)
		
		#height_scan = ObservationTermCfg(
		#	func=mdp.height_scan,
		#	params={"sensor_cfg": SceneEntityCfg("height_scanner")},
		#	noise=Unoise(n_min=-0.1, n_max=0.1),
		#	clip=(-1.0, 1.0),
		#)

		def __post_init__(self):
			self.enable_corruption = True
			self.concatenate_terms = True

	# observation groups
	policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
	"""Configuration for events."""

	physics_material = EventTermCfg(
		func=mdp.randomize_rigid_body_material,
		mode="startup",
		params={
			"asset_cfg": SceneEntityCfg("robot", body_names=".*"),
			"static_friction_range": (0.5, 1.0),
			"dynamic_friction_range": (0.3, 0.8),
			"restitution_range": (0.0, 0.5),
			"num_buckets": 64,
		},
	)
	add_base_mass = EventTermCfg(
		func=mdp.randomize_rigid_body_mass,
		mode="startup",
		params={"asset_cfg": SceneEntityCfg("robot", body_names="base"), "mass_range": (-5.0, 5.0), "operation": "add"},
	)
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
				"x": (-0.5, 0.5),
				"y": (-0.5, 0.5),
				"z": (-0.5, 0.5),
				"roll": (-0.5, 0.5),
				"pitch": (-0.5, 0.5),
				"yaw": (-0.5, 0.5),
			},
		},
	)
	reset_robot_joints = EventTermCfg(
		func=mdp.reset_joints_by_scale,
		mode="reset",
		params={
			"position_range": (0.5, 1.5),
			"velocity_range": (0.0, 0.0),
		},
	)
	push_robot = EventTermCfg(
		func=mdp.push_by_setting_velocity,
		mode="interval",
		interval_range_s=(0.0,EPISODE_LENGTH),
		params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
	)


@configclass
class RewardsCfg:
	"""Reward terms for the MDP."""

	# -- task
	track_lin_vel_xy_exp = RewardTermCfg(
		func=mdp.track_lin_vel_xy_exp, weight=2.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
	)
	track_ang_vel_z_exp = RewardTermCfg(
		func=mdp.track_ang_vel_z_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
	)
	# -- steering
	lin_vel_z_l2 = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-2.0)
	ang_vel_xy_l2 = RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05)
	dof_torques_l2 = RewardTermCfg(func=mdp.joint_torques_l2, weight=-1.0e-5)
	dof_acc_l2 = RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7)
	action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)
	feet_air_time = RewardTermCfg(
		func=mdp.feet_air_time,
		weight=0.125,
		params={
			"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_foot"),
			"command_name": "base_velocity",
			"threshold": 0.5,
		},
	)
	undesired_contacts = RewardTermCfg(
		func=mdp.undesired_contacts,
		weight=-1.0,
		params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_thigh"), "threshold": 1.0},
	)
	flat_orientation_l2 = RewardTermCfg(func=mdp.flat_orientation_l2, weight=-2.0)
	dof_pos_limits = RewardTermCfg(func=mdp.joint_pos_limits, weight=-2.0)


@configclass
class TerminationsCfg:
	"""Termination terms for the MDP."""
	out_of_bounds = TerminationTermCfg(func=mdp.root_out_of_curriculum, time_out=True)
	time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
	base_contact = TerminationTermCfg(
		func=mdp.illegal_contact,
		params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="base"), "threshold": 1.0},
	)


@configclass
class CurriculumCfg:
	"""Curriculum terms for the MDP."""

	terrain_levels = CurriculumTermCfg(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class LocomotionVelocityCustomEnvCfg(RLTaskEnvCfg):
	"""Configuration for the locomotion velocity-tracking environment."""

	# Scene settings
	scene: VelSceneCfg = VelSceneCfg(num_envs=4096, env_spacing=0.0)
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
		
		# update sensor update periods
		# we tick all the sensors based on the smallest update period (physics update period)
		if self.scene.height_scanner is not None:
			self.scene.height_scanner.update_period = self.decimation * self.sim.dt
		if self.scene.contact_sensor is not None:
			self.scene.contact_sensor.update_period = self.sim.dt
