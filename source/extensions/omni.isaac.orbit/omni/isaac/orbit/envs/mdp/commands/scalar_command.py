# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.isaac.orbit.utils.math as math_utils
from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.managers import CommandTerm
from omni.isaac.orbit.markers import VisualizationMarkers
from omni.isaac.orbit.markers.config import BLUE_ARROW_Z_MARKER_CFG, GREEN_ARROW_Z_MARKER_CFG

if TYPE_CHECKING:
	from omni.isaac.orbit.envs import BaseEnv

	from .commands_cfg import NormalVelocityCommandCfg, UniformVelocityCommandCfg, UserVelocityCommandCfg


class UniformSpeedCommand(CommandTerm):
	r"""Command generator that generates a speed (x,y) command from uniform distribution.

	The command comprises of a linear velocity in x and y direction and an angular velocity around
	the z-axis. It is given in the robot's base frame.

	If the :attr:`cfg.heading_command` flag is set to True, the angular velocity is computed from the heading
	error similar to doing a proportional control on the heading error. The target heading is sampled uniformly
	from the provided range. Otherwise, the angular velocity is sampled uniformly from the provided range.

	Mathematically, the angular velocity is computed as follows from the heading command:

	.. math::

		\omega_z = \frac{1}{2} \text{wrap_to_pi}(\theta_{\text{target}} - \theta_{\text{current}})

	"""

	cfg: UniformSpeedCommandCfg
	"""The configuration of the command generator."""

	def __init__(self, cfg: UniformSpeedCommandCfg, env: BaseEnv):
		"""Initialize the command generator.

		Args:
			cfg: The configuration of the command generator.
			env: The environment.
		"""
		# initialize the base class
		super().__init__(cfg, env)

		# obtain the robot asset
		# -- robot
		self.robot: Articulation = env.scene[cfg.asset_name]

		# crete buffers to store the command
		# -- command: value
		self.speed_command_b = torch.zeros(self.num_envs, device=self.device)
		# -- metrics
		self.metrics["error_speed"] = torch.zeros(self.num_envs, device=self.device)

	def __str__(self) -> str:
		"""Return a string representation of the command generator."""
		msg = "UniformSpeedCommand:\n"
		msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
		msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
		return msg

	"""
	Properties
	"""

	@property
	def command(self) -> torch.Tensor:
		"""The desired base speed command in the base frame. Shape is (num_envs)."""
		return self.speed_command_b

	"""
	Implementation specific functions.
	"""

	def _update_metrics(self):
		# time for which the command was executed
		max_command_time = self.cfg.resampling_time_range[1]
		max_command_step = max_command_time / self._env.step_dt
		# logs data
		self.metrics["error_speed"] += (
			torch.abs(self.speed_command_b - torch.norm(self.robot.data.root_lin_vel_b[:, :2], dim=-1)) / max_command_step
		)

	def _resample_command(self, env_ids: Sequence[int]):
		# sample velocity commands
		r = torch.empty(len(env_ids), device=self.device)
		# -- speed
		self.speed_command_b[env_ids] = r.uniform_(*self.cfg.range_speed)

	def _update_command(self):
		pass

	def _set_debug_vis_impl(self, debug_vis: bool):
		# set visibility of markers
		# note: parent only deals with callbacks. not their visibility
		if debug_vis:
			# create markers if necessary for the first tome
			if not hasattr(self, "base_speed_goal_visualizer"):
				# -- goal
				marker_cfg = GREEN_ARROW_Z_MARKER_CFG.copy()
				marker_cfg.prim_path = "/Visuals/Command/speed_goal"
				marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
				self.base_speed_goal_visualizer = VisualizationMarkers(marker_cfg)
				# -- current
				marker_cfg = BLUE_ARROW_Z_MARKER_CFG.copy()
				marker_cfg.prim_path = "/Visuals/Command/speed_current"
				marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
				self.base_speed_visualizer = VisualizationMarkers(marker_cfg)
			# set their visibility to true
			self.base_speed_goal_visualizer.set_visibility(True)
			self.base_speed_visualizer.set_visibility(True)
		else:
			if hasattr(self, "base_speed_goal_visualizer"):
				self.base_speed_goal_visualizer.set_visibility(False)
				self.base_speed_visualizer.set_visibility(False)

	def _debug_vis_callback(self, event):
		# get marker location
		# -- base state
		speed_des_arrow_pos = self.robot.data.root_pos_w.clone()
		speed_arrow_pos = self.robot.data.root_pos_w.clone()
		speed_des_arrow_pos[:,2] += 0.5
		speed_des_arrow_pos[:,0] += 0.5
		speed_arrow_pos[:,2] += 0.5
		speed_arrow_pos[:,0] -= 0.5
		# -- resolve the scales
		base_scale = self.base_speed_goal_visualizer.cfg.markers["arrow"].scale
		speed_des_arrow_scale = base_scale.clone()
		speed_des_arrow_scale[:,0] *= self.command
		speed_arrow_scale = base_scale.clone()
		speed_arrow_scale[:,0] *= torch.norm(self.robot.data.root_lin_vel_b[:, :2], dim=-1)
		# display markers
		self.base_vel_goal_visualizer.visualize(speed_des_arrow_pos, None, speed_des_arrow_scale)
		self.base_vel_visualizer.visualize(speed_arrow_pos, None, speed_arrow_scale)
