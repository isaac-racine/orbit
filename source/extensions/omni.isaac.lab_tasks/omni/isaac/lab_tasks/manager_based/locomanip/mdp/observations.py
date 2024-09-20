# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`omni.isaac.lab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING


import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import RayCaster

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv

"""
Root state.
"""

def base_ori_roll_pitch (env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """base orientation in the simulation world frame."""

    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    quat = asset.data.root_quat_w

    #Euler angle following the XYZ convention (roll, pitch, yaw)
    euler_angles = math_utils.euler_xyz_from_quat(quat)
    
    combined = torch.stack((euler_angles[0], euler_angles[1]), dim=0).transpose(0, 1)
    # print(torch.stack((euler_angles[0], euler_angles[1]), dim=0))
    # print(combined)

    return combined


def base_ori_roll (env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """base orientation in the simulation world frame."""

    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    quat = asset.data.root_quat_w

    #Euler angle following the XYZ convention (roll, pitch, yaw)
    euler_angles = math_utils.euler_xyz_from_quat(quat)
    roll = euler_angles[0]
    # print(euler_angles)  # Should output: ()
    # print(roll.shape)  # Should output: ()

    return roll

def base_ori_pitch (env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """base orientation in the simulation world frame."""

    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    quat = asset.data.root_quat_w

    #Euler angle following the XYZ convention (roll, pitch, yaw)
    euler_angles = math_utils.euler_xyz_from_quat(quat)
    pitch = euler_angles[1]

    return pitch