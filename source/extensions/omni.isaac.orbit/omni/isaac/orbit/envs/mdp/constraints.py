# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable constraint functions.

The functions can be passed to the :class:`omni.isaac.orbit.managers.ConstraintTermCfg` object to include
the contraint introduced by the function.
"""

from __future__ import annotations

import torch
from math import pi
from typing import TYPE_CHECKING

import omni.isaac.orbit.utils.math as math_utils
from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.managers.manager_base import ManagerTermBase
from omni.isaac.orbit.managers.manager_term_cfg import ConstraintTermCfg
from omni.isaac.orbit.sensors import ContactSensor

if TYPE_CHECKING:
	from omni.isaac.orbit.envs import RLTaskEnv

"""
General.
"""

def c_joint_val(env: RLTaskEnv, limval: float, val_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
	asset: Articulation = env.scene[asset_cfg.name]
	
	data = getattr(asset.data, val_name)[:, asset_cfg.joint_ids]
	val = torch.amax(torch.abs(data), dim=-1)
	return val - limval
def c_joint_acc(env: RLTaskEnv, limval: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
	return c_joint_val(env, limval, "joint_acc", asset_cfg)
def c_joint_vel(env: RLTaskEnv, limval: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
	return c_joint_val(env, limval, "joint_vel", asset_cfg)
def c_joint_torque(env: RLTaskEnv, limval: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
	return c_joint_val(env, limval, "applied_torque", asset_cfg)
