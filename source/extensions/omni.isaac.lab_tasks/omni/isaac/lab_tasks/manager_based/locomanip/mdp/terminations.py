# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`omni.isaac.lab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
import omni.isaac.lab.utils.math as math_utils

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ModifiedManagerBasedRLEnv
    from omni.isaac.lab.managers.command_manager import CommandTerm

"""
MDP terminations.
"""
def time_out(env: ModifiedManagerBasedRLEnv) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    return env.episode_length_buf >= env.max_episode_length


def root_height_below_minimum(
    env: ModifiedManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height

def wrong_ee_cmd_for_base_oritation(
    env: ModifiedManagerBasedRLEnv, command_name: str = "ee_pos_cmd", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the end effector position command is considered not ok for the current base orientation."""


    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # [w,x,y,z]
    quat = asset.data.root_quat_w
    
    euler_angles = math_utils.euler_xyz_from_quat(quat)

    roll = euler_angles[0]
    pitch = euler_angles[1]

    # Initialize the condition tensor
    condition_tensor = torch.zeros(command.shape[0], dtype=torch.bool, device=command.device)
    
    # Check conditions
    roll_condition = torch.logical_and(command[:, 1] > 0, roll > 0.2)
    roll_condition = torch.logical_or(roll_condition, torch.logical_and(command[:, 1] < 0, roll < -0.2))
    
    pitch_condition = torch.logical_and(command[:, 2] > 0, pitch > 0.2)
    pitch_condition = torch.logical_or(pitch_condition, torch.logical_and(command[:, 2] < 0, pitch < -0.2))
    
    # Combine the conditions
    condition_tensor = torch.logical_or(roll_condition, pitch_condition)
    
    return condition_tensor
