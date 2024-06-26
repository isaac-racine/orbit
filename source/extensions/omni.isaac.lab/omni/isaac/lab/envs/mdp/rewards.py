# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable reward functions.

The functions can be passed to the :class:`omni.isaac.lab.managers.RewardTermCfg` object to include
the reward introduced by the function.
"""

from __future__ import annotations

import torch
from math import pi
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.managers.manager_term_cfg import RewardTermCfg
from omni.isaac.lab.sensors import ContactSensor, FrameTransformer
from omni.isaac.lab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

"""
General.
"""


def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for being alive."""
    return (~env.termination_manager.terminated).float()


def is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.termination_manager.terminated.float()


class is_terminated_term(ManagerTermBase):
    """Penalize termination for specific terms that don't correspond to episodic timeouts.

    The parameters are as follows:

    * attr:`term_keys`: The termination terms to penalize. This can be a string, a list of strings
      or regular expressions. Default is ".*" which penalizes all terminations.

    The reward is computed as the sum of the termination terms that are not episodic timeouts.
    This means that the reward is 0 if the episode is terminated due to an episodic timeout. Otherwise,
    if two termination terms are active, the reward is 2.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)
        # find and store the termination terms
        term_keys = cfg.params.get("term_keys", ".*")
        self._term_names = env.termination_manager.find_terms(term_keys)

    def __call__(self, env: ManagerBasedRLEnv, term_keys: str | list[str] = ".*") -> torch.Tensor:
        # Return the unweighted reward for the termination terms
        reset_buf = torch.zeros(env.num_envs, device=env.device)
        for term in self._term_names:
            # Sums over terminations term values to account for multiple terminations in the same step
            reset_buf += env.termination_manager.get_term(term)

        return (reset_buf * (~env.termination_manager.time_outs)).float()


"""
Root penalties.
"""


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2-kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2-kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2-kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def base_height_l2(
    env: ManagerBasedRLEnv, target_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize asset height from its target using L2-kernel.

    Note:
        Currently, it assumes a flat terrain, i.e. the target height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # TODO: Fix this for rough-terrain.
    return torch.square(asset.data.root_pos_w[:, 2] - target_height)


def body_lin_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the linear acceleration of bodies using L2-kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.norm(asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :], dim=-1), dim=1)


"""
Joint penalties.
"""


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2-kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the L2 norm.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def joint_vel_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation using an L1-kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L1-kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the L1 norm.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2-kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the L2 norm.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_deviation_l1(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def joint_vel_limits(
    env: ManagerBasedRLEnv, soft_ratio: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint velocities if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint velocity and the soft limits.

    Args:
        soft_ratio: The ratio of the soft limits to be used.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = (
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
        - asset.data.soft_joint_vel_limits[:, asset_cfg.joint_ids] * soft_ratio
    )
    # clip to max error = 1 rad/s per joint to avoid huge penalties
    out_of_limits = out_of_limits.clip_(min=0.0, max=1.0)
    return torch.sum(out_of_limits, dim=1)


"""
Action penalties.
"""


def applied_torque_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize applied torques if they cross the limits.

    This is computed as a sum of the absolute value of the difference between the applied torques and the limits.

    .. caution::
        Currently, this only works for explicit actuators since we manually compute the applied torques.
        For implicit actuators, we currently cannot retrieve the applied torques from the physics engine.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    # TODO: We need to fix this to support implicit joints.
    out_of_limits = torch.abs(
        asset.data.applied_torque[:, asset_cfg.joint_ids] - asset.data.computed_torque[:, asset_cfg.joint_ids]
    )
    return torch.sum(out_of_limits, dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2-kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2-kernel."""
    return torch.sum(torch.square(env.action_manager.action), dim=1)


"""
Contact sensor.
"""


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)


def contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # compute the violation
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
    # compute the penalty
    return torch.sum(violation.clip(min=0.0), dim=1)


"""
Velocity-tracking rewards.
"""


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2)


# unit task rewards : [0,1] range ---------------------------------------------------------------------------------------------------------------


# linear task rewards (rewards err closer to 0): ---------------------------------------------------------------------------------------------------------
# [0,1] range : 0 is maxerr, 1 is err=0
# form :
# 	err = clip(|err|)
# 	rew = 1 - err / maxerr


def r_joint_val_lin(
    env: ManagerBasedRLEnv, maxerr: float, val_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    data = getattr(asset.data, val_name)[:, asset_cfg.joint_ids]
    err = torch.linalg.norm(data, dim=-1)
    return lin(err, maxerr)


# linear max task rewards (rewards err closer to errmax): ---------------------------------------------------------------------------------------------------------
# [0,1] range : 0 is err=0, 1 is err=errmax
# form :
# 	err = clip(|err|)
# 	rew = err / maxerr


def track_dir_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of direction commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command: torch.Tensor = env.command_manager.get_command(command_name)
    # compute the error
    dir_error = torch.sum(
        torch.square(
            torch.nn.functional.normalize(command[:, :2], dim=1)
            - torch.nn.functional.normalize(asset.data.root_lin_vel_b[:, :2], dim=1)
        ),
        dim=1,
    )
    return torch.exp(-dir_error / std**2)


def track_speed_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of speed commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command: torch.Tensor = env.command_manager.get_command(command_name)
    # compute the error
    speed_error = torch.square(
        torch.linalg.norm(command[:, :2], dim=1) - torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    )
    return torch.exp(-speed_error / std**2)


def track_heading_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of heading commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    # compute the error
    heading_error = torch.square(env.command_manager.get_command(command_name)[:, 2])
    return torch.exp(-heading_error / std**2)


def over_speed_xy(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize going over speed commands (xy axes)."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command: torch.Tensor = env.command_manager.get_command(command_name)
    # compute the error
    over_speed = (
        torch.linalg.norm(command[:, :2], dim=1) < torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    ).float()

    return over_speed


# command pos is relative to base pos, so when the agent is at the command pos, the command pos is 0,0
def track_pos_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    # compute the error
    pos_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :3]),
        dim=1,
    )
    return torch.exp(-pos_error / std**2)


# unit task rewards : [0,1] range ---------------------------------------------------------------------------------------------------------------


# command tracking rewards
def track_dir_xy_unit(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of direction commands (xy axes) using normalized value."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command: torch.Tensor = env.command_manager.get_command(command_name)

    dir_b = torch.nn.functional.normalize(asset.data.root_lin_vel_b[:, :2])
    dir_c = torch.nn.functional.normalize(command[:, :2])
    dot_prod = torch.sum(dir_b * dir_c, dim=1)  # [-1,1] range, 1 when on target
    return (dot_prod + 1) / 2  # [0, 1] range, 1 when on target


def track_heading_unit(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of heading commands (yaw) using normalized value."""
    command: torch.Tensor = env.command_manager.get_command(command_name)

    heading_error = command[:, 2]  # [-pi, pi] range, 0 when on target
    return 1 - torch.abs(heading_error) / pi  # [0, 1] range, 1 when on target


def track_speed_xy_unit(
    env: ManagerBasedRLEnv, max_err: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of speed commands (xy axes) using normalized value. Normalization relative to max_err."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command: torch.Tensor = env.command_manager.get_command(command_name)
    speed_error = torch.linalg.norm(command[:, :2], dim=1) - torch.linalg.norm(
        asset.data.root_lin_vel_b[:, :2], dim=1
    )  # [-inf,inf] range, 0 when on target
    return 1 - torch.clip(torch.abs(speed_error), max=max_err) / max_err  # [0, 1] range, 1 when on target


# unit task penalties


def prolonged_contacts_unit(env: ManagerBasedRLEnv, max_time: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired prolonged contact received. using normalized value. Normalization relative to max_time."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = torch.amax(
        contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids], 1
    )  # [0,inf] range, 0 when on target
    return torch.clip(contact_time, max=max_time) / max_time  # [0,1] range, 0 when on target


def undesired_contacts_unit(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contact received. using normalized value. Either 1 for contact or 0 for none."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = torch.amax(
        torch.norm(contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids], dim=-1), (1, 2)
    )  # [0,inf] range, 0 when on target
    return (forces > threshold).float()  # [0,1] range, 0 when on target


def contact_forces_unit(env: ManagerBasedRLEnv, max_force: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize maximum contact force received. using normalized value. Normalization relative to max_force."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = torch.amax(
        torch.norm(contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids], dim=-1), (1, 2)
    )  # [0,inf] range, 0 when on target
    return torch.clip(forces, max=max_force) / max_force  # [0,1] range, 0 when on target


def joint_torques_unit(
    env: ManagerBasedRLEnv, max_torque: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize max joint torques applied on the articulation using normalized value. Normalization relative to max_torque.
    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the result.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    max_torques = torch.amax(torch.abs(torques), dim=1)  # [0,inf] range, 0 when on target
    return torch.clip(max_torques, max=max_torque) / max_torque  # [0,1] range, 0 when on target


def joint_acc_unit(
    env: ManagerBasedRLEnv, max_acc: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize max joint accelerations on the articulation using normalized value. Normalization relative to max_acc."""
    asset: Articulation = env.scene[asset_cfg.name]
    acc = asset.data.joint_acc[:, asset_cfg.joint_ids]
    max_accs = torch.amax(torch.abs(acc), dim=1)  # [0,inf] range, 0 when on target
    return torch.clip(max_accs, max=max_acc) / max_acc  # [0,1] range, 0 when on target


def joint_vel_unit(
    env: ManagerBasedRLEnv, max_vel: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize max joint velocities on the articulation using normalized value. Normalization relative to max_vel."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    max_vels = torch.amax(torch.abs(vel), dim=1)  # [0,inf] range, 0 when on target
    return torch.clip(max_vels, max=max_vel) / max_vel  # [0,1] range, 0 when on target


def action_rate_unit(env: ManagerBasedRLEnv, max_act: float) -> torch.Tensor:
    """Penalize the rate of change of the actions."""
    act = torch.norm(env.action_manager.action - env.action_manager.prev_action, dim=1)
    return torch.clip(act, max=max_act) / max_act  # [0,1] range, 0 when on target


def unflat_orientation_unit(
    env: ManagerBasedRLEnv, min_dot: float = -1.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize non-flat base orientation using normalized value."""
    asset: RigidObject = env.scene[asset_cfg.name]
    dir_g = torch.nn.functional.normalize(asset.data.projected_gravity_b)
    dot_prod = dir_g[:, 2]  # [-1,1] range, -1 when on target
    return (torch.clip(dot_prod, min=min_dot) - min_dot) / (1 - min_dot)  # [0,1] range, 0 when on target


# unit task rewards
# returns 1 if error is 0, 0 if error is max
# form : (maxerr - clip(|err|, maxerr)) / maxerr


def r_dir_xy_unit(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of direction commands (xy axes) using normalized value."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command: torch.Tensor = env.command_manager.get_command(command_name)
    dir_b = torch.nn.functional.normalize(asset.data.root_lin_vel_b[:, :2])
    dir_c = torch.nn.functional.normalize(command[:, :2])
    dot_prod = torch.sum(dir_b * dir_c, dim=1)  # [-1,1] range, 1 when on target
    return (dot_prod + 1) / 2  # [0, 1] range, 1 when on target


def r_heading_unit(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of heading commands (yaw) using normalized value."""
    command: torch.Tensor = env.command_manager.get_command(command_name)
    heading_error = command[:, 2]  # [-pi, pi] range, 0 when on target
    return 1 - torch.abs(heading_error) / pi  # [0, 1] range, 1 when on target


def r_speed_xy_unit(
    env: ManagerBasedRLEnv, a: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of speed commands (xy axes) using normalized value. Normalization relative to max_err."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command: torch.Tensor = env.command_manager.get_command(command_name)
    speed_error = torch.linalg.norm(command[:, :2], dim=1) - torch.linalg.norm(
        asset.data.root_lin_vel_b[:, :2], dim=1
    )  # [-inf,inf] range, 0 when on target
    return 1 - torch.clip(torch.abs(speed_error), max=a) / a  # [0, 1] range, 1 when on target


def r_flat_orientation_unit(
    env: ManagerBasedRLEnv, a: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward flat base orientation using normalized value. Normalization relative to max_dot."""
    asset: RigidObject = env.scene[asset_cfg.name]
    dir_g = torch.nn.functional.normalize(asset.data.projected_gravity_b, dim=-1)
    dot_prod = torch.sum(torch.abs(env.nZ - dir_g), dim=-1)
    return (a - torch.clip(dot_prod, max=a)) / (1 + a)  # [0,1] range, 1 when on target


def r_joint_pose_unit(env, a: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint positions that are close to the default one."""
    asset: Articulation = env.scene[asset_cfg.name]
    angle_dist = torch.norm(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids], dim=-1
    )  # [0,?pi?] range, 0 when on target
    return 1 - torch.clip(angle_dist, max=pi) / pi  # [0,1] range, 1 when on target


# exponential task rewards :  -----------------------------------------------------------------------------------------------------------------
# [0,1] range : 1 when err=0, 0.01 when err=maxerr
# form : exp( -(err / (errfac*maxerr))**2 )  ;  no clipping
errfac: float = 0.46  # gives approx 0.01 at err=maxerr


def r_joint_val_exp(
    env: ManagerBasedRLEnv, maxerr: float, val_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    err = torch.linalg.norm(getattr(asset.data, val_name)[:, asset_cfg.joint_ids], dim=-1)
    return torch.exp(-((err / (errfac * maxerr)) ** 2))


def r_joint_jerk_exp(
    env: ManagerBasedRLEnv, maxerr: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return r_joint_val_exp(env, maxerr, "joint_jerk", asset_cfg)


def r_joint_acc_exp(
    env: ManagerBasedRLEnv, maxerr: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return r_joint_val_exp(env, maxerr, "joint_acc", asset_cfg)


def r_joint_vel_exp(
    env: ManagerBasedRLEnv, maxerr: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return r_joint_val_exp(env, maxerr, "joint_vel", asset_cfg)


def r_joint_torque_exp(
    env: ManagerBasedRLEnv, maxerr: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return r_joint_val_exp(env, maxerr, "applied_torque", asset_cfg)


def r_action_exp(env: ManagerBasedRLEnv, maxerr: float) -> torch.Tensor:
    err = torch.linalg.norm(env.action_manager.action, dim=-1)
    return torch.exp(-((err / (errfac * maxerr)) ** 2))


def r_action_rate_exp(env: ManagerBasedRLEnv, maxerr: float) -> torch.Tensor:
    diff = env.action_manager.action - env.action_manager.prev_action
    err = torch.linalg.norm(diff, dim=-1)
    return torch.exp(-((err / (errfac * maxerr)) ** 2))


def r_flat_orientation_exp(
    env: ManagerBasedRLEnv, maxerr: float = 2.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    dir_g = torch.nn.functional.normalize(asset.data.projected_gravity_b, dim=-1)
    err = torch.linalg.norm(env.nZ - dir_g, dim=-1)  # 2 at max
    return torch.exp(-((err / (errfac * maxerr)) ** 2))


def r_unflat_orientation_exp(
    env: ManagerBasedRLEnv, maxerr: float = 2.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    dir_g = torch.nn.functional.normalize(asset.data.projected_gravity_b, dim=-1)
    err = torch.linalg.norm(env.Z - dir_g, dim=-1)  # 2 at max
    return torch.exp(-((err / (errfac * maxerr)) ** 2))


def r_velz_exp(
    env: ManagerBasedRLEnv, maxerr: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    err = asset.data.root_lin_vel_b[:, 2]
    return torch.exp(-((err / (errfac * maxerr)) ** 2))


def r_joint_pose_exp(
    env: ManagerBasedRLEnv, maxerr: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    diff = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    err = torch.linalg.norm(diff, dim=-1)
    return torch.exp(-((err / (errfac * maxerr)) ** 2))


def r_contact_dist_exp(env: ManagerBasedRLEnv, maxerr: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    err = torch.linalg.norm(contact_sensor.data.current_contact_distance[:, sensor_cfg.body_ids], dim=-1)
    return torch.exp(-((err / (errfac * maxerr)) ** 2))


def r_contact_time_exp(env: ManagerBasedRLEnv, maxerr: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    err = torch.linalg.norm(contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids], dim=-1)
    return torch.exp(-((err / (errfac * maxerr)) ** 2))


def r_dir_xy_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    maxerr: float = 2.0,
    threshold: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    command: torch.Tensor = env.command_manager.get_command(command_name)

    dir_b = torch.nn.functional.normalize(asset.data.root_lin_vel_b[:, :2], dim=1)
    speed_b = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    dir_c = torch.nn.functional.normalize(command[:, :2])
    err = torch.linalg.norm(dir_b - dir_c, dim=-1)  # 2 at max

    # no reward if movement is too slow
    res = torch.exp(-((err / (errfac * maxerr)) ** 2))
    res *= (speed_b > threshold).float()

    return res


def r_heading_exp(
    env: ManagerBasedRLEnv, command_name: str, maxerr: float = pi, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    # asset: RigidObject = env.scene[asset_cfg.name]
    command: torch.Tensor = env.command_manager.get_command(command_name)
    err = command[:, 2]  # pi at max
    return torch.exp(-((err / (errfac * maxerr)) ** 2))


def r_speed_xy_exp(
    env: ManagerBasedRLEnv, maxerr: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    command: torch.Tensor = env.command_manager.get_command(command_name)
    err = torch.linalg.norm(command[:, :2], dim=1) - torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    return torch.exp(-((err / (errfac * maxerr)) ** 2))


def r_vel_xy_exp(
    env: ManagerBasedRLEnv, maxerr: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    command: torch.Tensor = env.command_manager.get_command(command_name)

    diff = command[:, :2] - asset.data.root_lin_vel_b[:, :2]
    err = torch.linalg.norm(diff, dim=-1)
    return torch.exp(-((err / (errfac * maxerr)) ** 2))


# bi-linear task reward / penalties : ---------------------------------------------------------------------------------------------------------
# [-1,1] range : [-1,0] is |err|>minerr, [0,1] is |err|<minerr, -1 is maxerr, 1 is err=0
# form :
# 	err = clip(|err|)
# 	if err < minerr : rew = 1 - err / minerr
# 	else            : pen = - (err-minerr) / (maxerr-minerr)


def bilin(err: float, minerr: float, maxerr: float):
    # err should be >0
    err = torch.clip(err, max=maxerr)
    return ((1 - err / minerr) * (err < minerr), ((minerr - err) / (maxerr - minerr)) * (err > minerr))


def rp_joint_val_bilin(
    env: ManagerBasedRLEnv,
    minerr: float,
    maxerr: float,
    val_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    data = getattr(asset.data, val_name)[:, asset_cfg.joint_ids]
    err = torch.torch.linalg.norm(data, dim=-1)
    return bilin(err, minerr, maxerr)


def rp_joint_acc_bilin(
    env: ManagerBasedRLEnv, minerr: float, maxerr: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return rp_joint_val_bilin(env, minerr, maxerr, "joint_acc", asset_cfg)


def rp_joint_vel_bilin(
    env: ManagerBasedRLEnv, minerr: float, maxerr: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return rp_joint_val_bilin(env, minerr, maxerr, "joint_vel", asset_cfg)


def rp_joint_torque_bilin(
    env: ManagerBasedRLEnv, minerr: float, maxerr: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return rp_joint_val_bilin(env, minerr, maxerr, "applied_torque", asset_cfg)


def rp_flat_orientation_bilin(
    env: ManagerBasedRLEnv, minerr: float, maxerr: float = 2.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    dir_g = torch.nn.functional.normalize(asset.data.projected_gravity_b, dim=-1)
    err = torch.linalg.norm(env.nZ - dir_g, dim=-1)  # 2 at max
    return bilin(err, minerr, maxerr)


def rp_vel_xy_bilin(
    env: ManagerBasedRLEnv,
    minerr: float,
    maxerr: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    command: torch.Tensor = env.command_manager.get_command(command_name)
    diff = command[:, :2] - asset.data.root_lin_vel_b[:, :2]
    err = torch.linalg.norm(diff, dim=-1)
    return bilin(err, minerr, maxerr)


def rp_heading_bilin(
    env: ManagerBasedRLEnv,
    command_name: str,
    minerr: float,
    maxerr: float = pi,
    # asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # asset: RigidObject = env.scene[asset_cfg.name]
    command: torch.Tensor = env.command_manager.get_command(command_name)
    err = torch.abs(command[:, 2])  # pi at max
    return bilin(err, minerr, maxerr)


# sparse task penalties :  -----------------------------------------------------------------------------------------------------------------
# [-1,0] range : -1 when bad, 0 when good
def p_joint_val_sparse(
    env: ManagerBasedRLEnv, threshold: float, val_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    isbad = torch.linalg.norm(getattr(asset.data, val_name)[:, asset_cfg.joint_ids], dim=1) > threshold
    return -isbad.float()


def p_joint_acc_sparse(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return p_joint_val_sparse(env, threshold, "joint_acc", asset_cfg)


def p_joint_vel_sparse(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return p_joint_val_sparse(env, threshold, "joint_vel", asset_cfg)


def p_joint_torque_sparse(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return p_joint_val_sparse(env, threshold, "applied_torque", asset_cfg)


def p_contact_sparse(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    isbad = torch.amax(contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids], dim=-1) > 0.01
    return -isbad.float()


# sparse task rewards :  -----------------------------------------------------------------------------------------------------------------
# [0,1] range : 0 when bad, 1 when good
def r_alive_sparse(env: ManagerBasedRLEnv) -> torch.Tensor:
    return (~env.termination_manager.terminated).float()


def r_contact_sparse(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    isok = torch.amax(contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids], dim=-1) < 1e-3
    return isok.float()


# linear task rewards (rewards err closer to 0): ---------------------------------------------------------------------------------------------------------
# [0,1] range : 0 is maxerr, 1 is err=0
# form :
# 	err = clip(|err|)
# 	rew = 1 - err / maxerr

DEBUG_REW = False
if DEBUG_REW:
    import inspect


def lin(err: float, maxerr: float):
    if DEBUG_REW:
        print(inspect.currentframe().f_back.f_code.co_name, torch.amax(err, dim=0))

    # err should be >0
    err = torch.clip(err, max=maxerr)
    return 1 - err / maxerr


def linmax(err: float, maxerr: float):
    if DEBUG_REW:
        print(inspect.currentframe().f_back.f_code.co_name, torch.amax(err, dim=0))

    # err should be >0
    err = torch.clip(err, max=maxerr)
    return err / maxerr


def r_joint_acc_lin(
    env: ManagerBasedRLEnv, maxerr: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    data = asset.data.joint_acc[:, asset_cfg.joint_ids]
    err = torch.sum(torch.abs(data), dim=-1)
    return lin(err, maxerr)


def r_joint_vel_lin(
    env: ManagerBasedRLEnv, maxerr: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    data = asset.data.joint_vel[:, asset_cfg.joint_ids]
    err = torch.sum(torch.abs(data), dim=-1)
    return lin(err, maxerr)


def r_joint_torque_lin(
    env: ManagerBasedRLEnv, maxerr: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    data = asset.data.applied_torque[:, asset_cfg.joint_ids]
    err = torch.sum(torch.abs(data), dim=-1)
    return lin(err, maxerr)


def r_joint_power_lin(
    env: ManagerBasedRLEnv, maxerr: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    speed = asset.data.joint_vel[:, asset_cfg.joint_ids]
    err = torch.sum(torch.abs(torque * speed), dim=-1)
    return lin(err, maxerr)


def r_frame_val_lin(env: ManagerBasedRLEnv, maxerr: float, val_name: str, frame_cfg) -> torch.Tensor:
    frame: FrameTransformer = env.scene[frame_cfg.name]

    data = getattr(frame.data, val_name)
    err = torch.sum(torch.linalg.norm(data, dim=-1), dim=-1)
    return lin(err, maxerr)


def r_frame_vel_lin(env: ManagerBasedRLEnv, maxerr: float, frame_cfg) -> torch.Tensor:
    return r_frame_val_lin(env, maxerr, "target_vel_w", frame_cfg)


def r_frame_acc_lin(env: ManagerBasedRLEnv, maxerr: float, frame_cfg) -> torch.Tensor:
    return r_frame_val_lin(env, maxerr, "target_acc_w", frame_cfg)


def r_action_rate_lin(env: ManagerBasedRLEnv, maxerr: float) -> torch.Tensor:
    diff = env.action_manager.action - env.action_manager.prev_action
    err = torch.linalg.norm(diff, dim=-1)
    return lin(err, maxerr)


def r_action_lin(env: ManagerBasedRLEnv, maxerr: float) -> torch.Tensor:
    data = env.action_manager.action
    err = torch.sum(torch.abs(data))
    return lin(err, maxerr)


def r_velz_lin(
    env: ManagerBasedRLEnv, maxerr: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    err = torch.abs(asset.data.root_lin_vel_b[:, 2])

    return lin(err, maxerr)


def r_acc_lin(
    env: ManagerBasedRLEnv, maxerr: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    data = asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :]
    err = torch.sum(torch.norm(data, dim=-1), dim=-1)
    return lin(err, maxerr)


def r_flat_orientation_lin(
    env: ManagerBasedRLEnv, maxerr: float = 2.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    dir_g = torch.nn.functional.normalize(asset.data.projected_gravity_b, dim=-1)
    err = torch.linalg.norm(env.nZ - dir_g, dim=-1)  # 2 at max
    return lin(err, maxerr)


def r_joint_pose_lin(
    env: ManagerBasedRLEnv, maxerr: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    diff = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    err = torch.linalg.norm(diff, dim=-1)
    return lin(err, maxerr)


def r_contact_dist_lin(env: ManagerBasedRLEnv, maxerr: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    err = torch.sum(contact_sensor.data.current_contact_distance[:, sensor_cfg.body_ids], dim=-1)
    return lin(err, maxerr)


def r_feetair_velcom_lin(
    env: ManagerBasedRLEnv, target_t: float, maxerr: float, command_name: str, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # command: torch.Tensor = env.command_manager.get_command(command_name)

    times = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    err = torch.abs(times - target_t).sum(dim=-1)
    # err *= torch.norm(command[:, :2], dim=1) > 0.1
    return lin(err, maxerr)


def r_nbcontact_lin(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    iscontact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 1e-3
    err = iscontact.float().sum(dim=-1)
    maxerr = iscontact.size(dim=-1)
    return lin(err, maxerr)


def r_com_linvel_lin(
    env: ManagerBasedRLEnv, maxerr: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    command: torch.Tensor = env.command_manager.get_command(command_name)

    diff = command[:, :2] - asset.data.root_lin_vel_b[:, :2]
    err = torch.linalg.norm(diff, dim=-1)
    return lin(err, maxerr)


def r_com_angvel_lin(
    env: ManagerBasedRLEnv, maxerr: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    command: torch.Tensor = env.command_manager.get_command(command_name)

    diff = command[:, 2] - asset.data.root_ang_vel_b[:, 2]
    err = torch.abs(diff)
    return lin(err, maxerr)


def r_com_heading_lin(
    env: ManagerBasedRLEnv, command_name: str, maxerr: float = pi, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    command: torch.Tensor = env.command_manager.get_command(command_name)

    err = torch.abs(command[:, 2])  # pi at max
    return lin(err, maxerr)


def r_com_headingpos_lin(
    env: ManagerBasedRLEnv, command_name: str, maxerr: float = pi, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    command: torch.Tensor = env.command_manager.get_term(command_name).heading_command_b

    err = torch.abs(command[:])  # pi at max
    return lin(err, maxerr)


def r_com_eepos_lin(
    env: ManagerBasedRLEnv, maxerr: float, command_name: str, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    command: torch.Tensor = env.command_manager.get_term(command_name).pos_command_w

    diff = ee_frame.data.target_pos_w[:, 0, :] - command[:, :3]
    err = torch.linalg.norm(diff, dim=-1)
    return lin(err, maxerr)


def r_com_pos_lin(
    env: ManagerBasedRLEnv, maxerr: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    command: torch.Tensor = env.command_manager.get_term(command_name).pos_command_w

    diff = asset.data.root_pos_w[:, :3] - command[:, :3]
    err = torch.linalg.norm(diff, dim=-1)
    return lin(err, maxerr)


def r_com_releepos_lin(
    env: ManagerBasedRLEnv,
    maxerr: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripperStator"]),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)

    return lin(distance, maxerr)


# linear max task rewards (rewards err closer to errmax): ---------------------------------------------------------------------------------------------------------
# [0,1] range : 0 is err=0, 1 is err=errmax
# form :
# 	err = clip(|err|)
# 	rew = err / maxerr


def r_velx_linmax(
    env: ManagerBasedRLEnv, maxerr: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    err = torch.abs(asset.data.root_lin_vel_w[:, 0])
    return linmax(err, maxerr)


def r_height_linmax(
    env: ManagerBasedRLEnv, maxerr: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    err = torch.clip(asset.data.root_pos_w[:, 2], min=0)
    return linmax(err, maxerr)
