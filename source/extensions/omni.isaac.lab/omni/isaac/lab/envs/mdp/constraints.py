# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable constraint functions.

The functions can be passed to the :class:`omni.isaac.lab.managers.ConstraintTermCfg` object to include
the constraint introduced by the function.
"""

from __future__ import annotations

import torch
from math import pi
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.managers.manager_term_cfg import ConstraintTermCfg
from omni.isaac.lab.sensors import ContactSensor

if TYPE_CHECKING:
	from omni.isaac.lab.envs import ManagerBasedRLEnv

"""
General.
"""

def c_joint_val(env: ManagerBasedRLEnv, limval: float, val_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
	asset: Articulation = env.scene[asset_cfg.name]
	
	data = getattr(asset.data, val_name)[:, asset_cfg.joint_ids]
	val = torch.amax(torch.abs(data), dim=-1)
	return val - limval
def c_joint_acc(env: ManagerBasedRLEnv, limval: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
	return c_joint_val(env, limval, "joint_acc", asset_cfg)
def c_joint_vel(env: ManagerBasedRLEnv, limval: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
	return c_joint_val(env, limval, "joint_vel", asset_cfg)
def c_joint_torque(env: ManagerBasedRLEnv, limval: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
	return c_joint_val(env, limval, "applied_torque", asset_cfg)