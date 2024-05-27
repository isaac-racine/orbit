# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`omni.isaac.orbit.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import ContactSensor

if TYPE_CHECKING:
	from omni.isaac.orbit.envs import RLTaskEnv
	from omni.isaac.orbit.managers.command_manager import CommandTerm

"""
MDP terminations.
"""


def time_out(env: RLTaskEnv) -> torch.Tensor:
	"""Terminate the episode when the episode length exceeds the maximum episode length."""
	return env.episode_length_buf >= env.max_episode_length


def command_resample(env: RLTaskEnv, command_name: str, num_resamples: int = 1) -> torch.Tensor:
	"""Terminate the episode based on the total number of times commands have been re-sampled.

	This makes the maximum episode length fluid in nature as it depends on how the commands are
	sampled. It is useful in situations where delayed rewards are used :cite:`rudin2022advanced`.
	"""
	command: CommandTerm = env.command_manager.get_term(command_name)
	return torch.logical_and((command.time_left <= env.step_dt), (command.command_counter == num_resamples))


"""
Root terminations.
"""


def bad_orientation(
	env: RLTaskEnv, limit_angle: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
	"""Terminate when the asset's orientation is too far from the desired orientation limits.

	This is computed by checking the angle between the projected gravity vector and the z-axis.
	"""
	# extract the used quantities (to enable type-hinting)
	asset: RigidObject = env.scene[asset_cfg.name]
	return torch.acos(-asset.data.projected_gravity_b[:, 2]).abs() > limit_angle


def root_height_below_minimum(
	env: RLTaskEnv, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
	"""Terminate when the asset's root height is below the minimum height.

	Note:
		This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
	"""
	# extract the used quantities (to enable type-hinting)
	asset: RigidObject = env.scene[asset_cfg.name]
	return asset.data.root_pos_w[:, 2] < minimum_height


def root_out_of_curriculum(
	env: RLTaskEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
	"""Terminate when the asset's root position is outside the terrain.
		Only usable with generator terrain.
	"""
	
	# extract the used quantities (to enable type-hinting)
	asset: RigidObject = env.scene[asset_cfg.name]
	terrain: TerrainImporter = env.scene.terrain
	dist = torch.abs(asset.data.root_pos_w[:, :2] - env.scene.env_origins[:,:2])
	return torch.logical_or(dist[:,0] > terrain.cfg.terrain_generator.size[0]/2, dist[:,1] > terrain.cfg.terrain_generator.size[1]/2)

"""
Joint terminations.
"""


def joint_pos_out_of_limit(env: RLTaskEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
	"""Terminate when the asset's joint positions are outside of the soft joint limits."""
	# extract the used quantities (to enable type-hinting)
	asset: Articulation = env.scene[asset_cfg.name]
	# compute any violations
	out_of_upper_limits = torch.any(asset.data.joint_pos > asset.data.soft_joint_pos_limits[..., 1], dim=1)
	out_of_lower_limits = torch.any(asset.data.joint_pos < asset.data.soft_joint_pos_limits[..., 0], dim=1)
	return torch.logical_or(out_of_upper_limits[:, asset_cfg.joint_ids], out_of_lower_limits[:, asset_cfg.joint_ids])


def joint_pos_out_of_manual_limit(
	env: RLTaskEnv, bounds: tuple[float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
	"""Terminate when the asset's joint positions are outside of the configured bounds.

	Note:
		This function is similar to :func:`joint_pos_out_of_limit` but allows the user to specify the bounds manually.
	"""
	# extract the used quantities (to enable type-hinting)
	asset: Articulation = env.scene[asset_cfg.name]
	if asset_cfg.joint_ids is None:
		asset_cfg.joint_ids = slice(None)
	# compute any violations
	out_of_upper_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] > bounds[1], dim=1)
	out_of_lower_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] < bounds[0], dim=1)
	return torch.logical_or(out_of_upper_limits, out_of_lower_limits)


def joint_vel_out_of_limit(env: RLTaskEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
	"""Terminate when the asset's joint velocities are outside of the soft joint limits."""
	# extract the used quantities (to enable type-hinting)
	asset: Articulation = env.scene[asset_cfg.name]
	# compute any violations
	limits = asset.data.soft_joint_vel_limits
	return torch.any(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) > limits[:, asset_cfg.joint_ids], dim=1)


def joint_vel_out_of_manual_limit(
	env: RLTaskEnv, max_velocity: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
	"""Terminate when the asset's joint velocities are outside the provided limits."""
	# extract the used quantities (to enable type-hinting)
	asset: Articulation = env.scene[asset_cfg.name]
	# compute any violations
	return torch.any(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) > max_velocity, dim=1)


def joint_effort_out_of_limit(env: RLTaskEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
	"""Terminate when effort applied on the asset's joints are outside of the soft joint limits.

	In the actuators, the applied torque are the efforts applied on the joints. These are computed by clipping
	the computed torques to the joint limits. Hence, we check if the computed torques are equal to the applied
	torques.
	"""
	# extract the used quantities (to enable type-hinting)
	asset: Articulation = env.scene[asset_cfg.name]
	# check if any joint effort is out of limit
	out_of_limits = torch.isclose(
		asset.data.computed_torque[:, asset_cfg.joint_ids], asset.data.applied_torque[:, asset_cfg.joint_ids]
	)
	return torch.any(out_of_limits, dim=1)


"""
Contact sensor.
"""


def illegal_contact(env: RLTaskEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
	"""Terminate when the contact force on the sensor exceeds the force threshold."""
	# extract the used quantities (to enable type-hinting)
	contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
	net_contact_forces = contact_sensor.data.net_forces_w_history
	# check if any contact force exceeds the threshold
	return torch.any(
		torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
	)


"""
Constraint terminations
"""


def constraint(err: float, errmax: float, pmax: float):
	prob = torch.clip((err-errmax)/errmax, min=0,max=1) * pmax # there is a chance the termination will not happen
	return torch.rand_like(prob) <= prob

def c_joint_val_err(env: RLTaskEnv, maxerr: float, maxprob: float, val_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
	asset: Articulation = env.scene[asset_cfg.name]
	
	data = getattr(asset.data, val_name)[:, asset_cfg.joint_ids]
	err = torch.linalg.norm(data, dim=-1)
	return constraint(err, maxerr, maxprob)
def c_joint_acc_err(env: RLTaskEnv, maxerr: float, maxprob: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
	return c_joint_val_err(env, maxerr, maxprob, "joint_acc", asset_cfg)
def c_joint_vel_err(env: RLTaskEnv, maxerr: float, maxprob: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
	return c_joint_val_err(env, maxerr, maxprob, "joint_vel", asset_cfg)
def c_joint_torque_err(env: RLTaskEnv, maxerr: float, maxprob: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
	return c_joint_val_err(env, maxerr, maxprob, "applied_torque", asset_cfg)

def c_base_orientation_err(env: RLTaskEnv, maxerr: float, maxprob: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
	asset: RigidObject = env.scene[asset_cfg.name]
	
	dir_g = torch.nn.functional.normalize(asset.data.projected_gravity_b, dim=-1)
	err = torch.linalg.norm(env.nZ - dir_g, dim=-1) # 2 at max
	return constraint(err, maxerr, maxprob)

def c_contact_err(env: RLTaskEnv, maxprob: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
	contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
	
	isbad = torch.amax(contact_sensor.data.current_contact_time[:,sensor_cfg.body_ids], dim=-1) > 0.01
	return constraint(isbad.float(), 0.1, maxprob)