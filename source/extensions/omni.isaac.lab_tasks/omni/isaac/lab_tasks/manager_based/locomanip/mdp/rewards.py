# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject, Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor

from omni.isaac.lab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul, quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ModifiedManagerBasedRLEnv


############################# LOCOMOTION #################################

def track_lin_vel_x_yaw_frame(
    env: ModifiedManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_x_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_x_error = torch.sum(
        torch.abs(vel_x_yaw[:, 0] - env.command_manager.get_command(command_name)[:, 0]), dim=-1
    )
    return -lin_vel_x_error

def track_ang_vel_z_world_exp( #yaw velocity
    env: ModifiedManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.abs( asset.data.root_ang_vel_w[:, 2] - env.command_manager.get_command(command_name)[:, 2])


    return torch.exp(-ang_vel_error)

def r_joint_leg_power(env: ModifiedManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
	asset: Articulation = env.scene[asset_cfg.name]
	
	torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
	speed = asset.data.joint_vel[:, asset_cfg.joint_ids]
	err = torch.sum(torch.square(torch.abs(torque * speed)), dim=1)
	return -err


def is_alive(env: ModifiedManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for being alive."""

    return (~env.termination_manager.terminated).float() * (0.2 + 0.5 * env.command_manager.get_command(command_name)[:, 0])


############################ MANIPULATION #################################

def position_command_error(env: ModifiedManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)

def track_pose_orientation(env: ModifiedManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot",body_names='gripperStator')) -> torch.Tensor:
    """Penalize tracking pose and orientation error using shortest path.

    The function computes the pose and orientation error between the desired pose and orientation (from the command) and the
    current pose and orientation of the asset's body (in world frame). The pose and orientation error is computed as the shortest
    path between the desired and current pose and orientations.
    """

    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    #POSE
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b) #verify if ok
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3] # type: ignore


    # print(curr_pos_w)
    # print('-------------')
    # print(des_pos_w)
    err_pos = torch.square(torch.norm(curr_pos_w - des_pos_w, dim=1))

    #ORIENTATION
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b) #verify if ok
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore

    err_ori = torch.square(quat_error_magnitude(curr_quat_w, des_quat_w))
    
    return  torch.exp(-torch.sqrt(err_pos+err_ori))  # exp(−∥[p,o]−[p_cmd​,o_cmd​]∥) = exp(−∥difference p​∥^2+∥difference o​∥^2​)

def r_joint_arm_power(env: ModifiedManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
	asset: Articulation = env.scene[asset_cfg.name]
	
	torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
	speed = asset.data.joint_vel[:, asset_cfg.joint_ids]
	err = torch.sum(torch.abs(torque * speed), dim=1)
	return -err