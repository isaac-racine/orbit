# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward manager for computing reward signals for a given world."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import ConstraintTermCfg

if TYPE_CHECKING:
	from omni.isaac.orbit.envs import RLTaskEnv


class ConstraintManager(ManagerBase):
	"""Manager for computing constraint signals for a given world.

	The constraint manager computes the total delta as the max across all constraint probabilities. The constraint
	terms are parsed from a nested config class containing the constraint manger's settings and constraint
	terms configuration.

	The constraint terms are parsed from a config class containing the manager's settings and each term's
	parameters. Each constraint term should instantiate the :class:`ConstraintTermCfg` class.

	.. note::

		NOT weighted with time step.

	"""

	_env: RLTaskEnv
	"""The environment instance."""

	def __init__(self, cfg: object, env: RLTaskEnv):
		"""Initialize the constraint manager.

		Args:
			cfg: The configuration object or dictionary (``dict[str, ConstraintTermCfg]``).
			env: The environment instance.
		"""
		super().__init__(cfg, env)
		# prepare extra info to store individual constraint term information
		self._episode_c_max = dict()
		for term_name in self._term_names:
			self._episode_c_max[term_name] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
		# create buffer for managing reward per environment
		self._delta_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

	def __str__(self) -> str:
		"""Returns: A string representation for reward manager."""
		msg = f"<ConstrinatManager> contains {len(self._term_names)} active terms.\n"

		# create table for term information
		table = PrettyTable()
		table.title = "Active Constraint Terms"
		table.field_names = ["Index", "Name", "Max. prob.", "Terrain lvs", "Terrain types"]
		# set alignment of table columns
		table.align["Name"] = "l"
		table.align["Max. prob."] = "r"
		table.align["Terrain lvs"] = "r"
		table.align["Terrain types"] = "r"
		# add info on each term
		for index, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
			table.add_row([
				index, name, term_cfg.pmax,
				term_cfg.curriculum_row_range if term_cfg.curriculum_dependency else "N/A",
				term_cfg.curriculum_col_range if term_cfg.curriculum_dependency else "N/A",
			])
		# convert table to string
		msg += table.get_string()
		msg += "\n"

		return msg

	"""
	Properties.
	"""

	@property
	def active_terms(self) -> list[str]:
		"""Name of active reward terms."""
		return self._term_names

	"""
	Operations.
	"""

	def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
		"""Returns the episodic max constraint.

		Args:
			env_ids: The environment ids for which the episodic max constraint terms is to be returned. Defaults to all the environment ids.

		Returns:
			Dictionary of episodic max constraint terms.
		"""
		# resolve environment ids
		if env_ids is None:
			env_ids = slice(None)
		# store information
		extras = {}
		for term_name in self._term_names:
			# store information
			# r_1 + r_2 + ... + r_n
			episodic_sum_avg = torch.mean(self._episode_c_max[term_name][env_ids])
			extras["Episode Constraint/" + term_name] = episodic_sum_avg / self._env.max_episode_length_s
			# reset episodic sum
			self._episode_c_max[term_name][env_ids] = 0.0
		# reset all the constraint terms
		for term_cfg in self._class_term_cfgs:
			term_cfg.func.reset(env_ids=env_ids)
		# return logged information
		return extras

	def compute(self, dt: float) -> torch.Tensor:
		"""Computes the reward signal as a weighted sum of individual terms.

		This function calls each reward term managed by the class and adds them to compute the net
		reward signal. It also updates the episodic sums corresponding to individual reward terms.

		Args:
			dt: The time-step interval of the environment.

		Returns:
			The net reward signal of shape (num_envs,).
		"""
		# reset computation
		self._delta_buf[:] = 0.0
		# iterate over all the reward terms
		for term_name, term_cfg in zip(self._term_names, self._term_cfgs):
			# compute term's value
			value = torch.clip(term_cfg.func(self._env, **term_cfg.params), min=0)
			
			# check curriculum
			if term_cfg.curriculum_dependency:
				terrain = self._env.scene.terrain				
				row_r = term_cfg.curriculum_row_range ; col_r = term_cfg.curriculum_col_range
				lower_row, upper_row = (row_r[0], row_r[1] if row_r[1] != -1 else terrain.max_terrain_level-1)
				lower_col, upper_col = (col_r[0], col_r[1] if col_r[1] != -1 else terrain.max_terrain_type-1)
				curriculum_ok = torch.logical_and( 
					torch.logical_and(lower_row <= terrain.terrain_levels, terrain.terrain_levels <= upper_row),
					torch.logical_and(lower_col <= terrain.terrain_types, terrain.terrain_types <= upper_col)
				)
				
				value = torch.where(curriculum_ok, value, 0.0)
			
			# update max (no moving average)
			valuemax = self._episode_c_max[term_name]
			valuemax[:] = torch.where(valuemax > value, valuemax, value)
			
			# compute delta prob
			delta = term_cfg.pmax * torch.clip(value / valuemax, min=0,max=1)
			self._delta_buf[:] = torch.where(self._delta_buf > delta, self._delta_buf, delta)
		
		return self._delta_buf

	"""
	Operations - Term settings.
	"""

	def set_term_cfg(self, term_name: str, cfg: ConstraintTermCfg):
		"""Sets the configuration of the specified term into the manager.

		Args:
			term_name: The name of the constraint term.
			cfg: The configuration for the constraint term.

		Raises:
			ValueError: If the term name is not found.
		"""
		if term_name not in self._term_names:
			raise ValueError(f"Constraint term '{term_name}' not found.")
		# set the configuration
		self._term_cfgs[self._term_names.index(term_name)] = cfg

	def get_term_cfg(self, term_name: str) -> ConstraintTermCfg:
		"""Gets the configuration for the specified term.

		Args:
			term_name: The name of the constraint term.

		Returns:
			The configuration of the constraint term.

		Raises:
			ValueError: If the term name is not found.
		"""
		if term_name not in self._term_names:
			raise ValueError(f"Constraint term '{term_name}' not found.")
		# return the configuration
		return self._term_cfgs[self._term_names.index(term_name)]

	"""
	Helper functions.
	"""

	def _prepare_terms(self):
		"""Prepares a list of constraint functions."""
		# parse remaining constraint terms and decimate their information
		self._term_names: list[str] = list()
		self._term_cfgs: list[ConstraintTermCfg] = list()
		self._class_term_cfgs: list[ConstraintTermCfg] = list()

		# check if config is dict already
		if isinstance(self.cfg, dict):
			cfg_items = self.cfg.items()
		else:
			cfg_items = self.cfg.__dict__.items()
		# iterate over all the terms
		for term_name, term_cfg in cfg_items:
			# check for non config
			if term_cfg is None:
				continue
			# check for valid config type
			if not isinstance(term_cfg, ConstraintTermCfg):
				raise TypeError(
					f"Configuration for the term '{term_name}' is not of type ConstraintTermCfg."
					f" Received: '{type(term_cfg)}'."
				)
			# resolve common parameters
			self._resolve_common_term_cfg(term_name, term_cfg, min_argc=1)
			# add function to list
			self._term_names.append(term_name)
			self._term_cfgs.append(term_cfg)
			# check if the term is a class
			if isinstance(term_cfg.func, ManagerTermBase):
				self._class_term_cfgs.append(term_cfg)
