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
from omni.isaac.lab.sensors import RayCaster, ContactSensor





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

def ee_goal_pos_w (env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = math_utils.combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    # curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    return des_pos_w

def ee_goal_ori_w(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:

    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = math_utils.quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    # curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    return des_quat_w

def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = math_utils.quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    return math_utils.quat_error_magnitude(curr_quat_w, des_quat_w)

def foot_contact (env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    
    asset: RigidObject = env.scene[asset_cfg.name]
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold

    print(is_contact)
    
    return is_contact


def get_masses (env: ManagerBasedRLEnv, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    # if env_ids is None:
    #     env_ids = torch.arange(env.scene.num_envs, device="cpu")
    # else:
    #     env_ids = env_ids.cpu()

    # # resolve body indices
    # if asset_cfg.body_ids == slice(None):
    #     body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    # else:
    #     body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current masses of the bodies (num_assets, num_bodies)
    masses = asset.root_physx_view.get_masses()

    print(masses)

    return masses
