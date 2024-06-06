# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`omni.isaac.orbit.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.terrains import TerrainImporter

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv


def terrain_levels_pos(
    env: RLTaskEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:

    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    term = env.command_manager.get_term("base_position")
    command = term.command

    dist_goal = torch.norm(command[env_ids, :3], dim=1)
    dist_totgoal = torch.norm(term.pos_command_w[env_ids, :3] - env.scene.env_origins[env_ids, :3], dim=1)
    dist_walked = torch.norm(asset.data.root_pos_w[env_ids, :3] - env.scene.env_origins[env_ids, :3], dim=1)

    move_up = dist_goal < 0.2
    move_down = dist_walked < dist_totgoal * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())
