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
    #print(c_joint_val(env, limval, "joint_acc", asset_cfg))
    return c_joint_val(env, limval, "joint_acc", asset_cfg)

def c_joint_vel(env: ManagerBasedRLEnv, limval: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    #print(c_joint_val(env, limval, "joint_vel", asset_cfg))
    return c_joint_val(env, limval, "joint_vel", asset_cfg)

def c_joint_torque(env: ManagerBasedRLEnv, limval: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    #print(c_joint_val(env, limval, "applied_torque", asset_cfg))
    return c_joint_val(env, limval, "applied_torque", asset_cfg)


#Action rate CaT (based on paper)
def c_action_rate(env: ManagerBasedRLEnv, limval: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    #print(torch.amax(torch.abs(env.action_manager.action - env.action_manager.prev_action)/env.step_dt, dim=-1) - limval)
    return torch.amax(torch.abs(env.action_manager.action - env.action_manager.prev_action)/env.step_dt, dim=-1) - limval

#Action rate IsaacLab (default)
def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2-kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def c_base_ori_xy(env: ManagerBasedRLEnv, limval: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:	
    
    asset: RigidObject = env.scene[asset_cfg.name]
    xy_ori = torch.norm(asset.data.projected_gravity_b[:, :2], dim=1)

    return torch.amax(torch.abs(xy_ori)) - limval

def c_hip_ori(env: ManagerBasedRLEnv, limval: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:	
    
    asset: Articulation = env.scene[asset_cfg.name]
    hip_ori = asset.data.joint_pos[:, [0,3,6,9]] - asset.data.default_joint_pos[:, [0,3,6,9]]
    print(asset_cfg.joint_names)
    print(asset_cfg.joint_ids)
    
    return torch.amax(torch.abs(hip_ori)) - limval

# def c_air_time(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, limval: float) -> torch.Tensor:

#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     # compute the reward
#     air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]

#     first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
#     last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
#     reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
#     # no reward for zero command
#     reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1