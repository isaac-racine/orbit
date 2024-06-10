# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.terrains import TerrainImporter

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`omni.isaac.lab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def terrain_levels_vel2(
	env: ManagerBasedRLEnv, env_ids: Sequence[int], command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
	asset: Articulation = env.scene[asset_cfg.name]
	terrain: TerrainImporter = env.scene.terrain
	command = env.command_manager.get_command(command_name)
	
	ep_time = env.episode_length_buf[env_ids]*env.step_dt
	
	# straight line distance the robot should have walked if it respected the command
	required_distance = torch.norm(command[env_ids, :2], dim=1) * ep_time
	# compute the distance the robot walked
	distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
	# compute the distance difference
	distdiff = torch.abs(required_distance - distance)
	reldiff = distdiff / required_distance
	
	# exception if required distance is too small
	ignore = required_distance < 0.2
	
	# robots that walked far enough progress to harder terrains
	move_up = torch.logical_and(distance > 0.95*terrain.cfg.terrain_generator.size[0]/2, reldiff < 0.05, ~ignore)
	# robots that walked too little
	move_down = torch.logical_and(reldiff > 0.5, ~ignore)
	move_down *= ~move_up
	# update terrain levels
	terrain.update_env_origins(env_ids, move_up, move_down)
	# return the mean terrain level
	return torch.mean(terrain.terrain_levels.float())