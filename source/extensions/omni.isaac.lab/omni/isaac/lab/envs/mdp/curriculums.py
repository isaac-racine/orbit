# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def modify_reward_weight(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, weight: float, num_steps: int):
    """Curriculum that modifies a reward weight a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The weight of the reward term.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)

def modify_constraint_pmax(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, pmax_ini: float, pmax_end: float, num_steps: int, num_steps_grad: int):
    """Curriculum that modifies a constraint pmax over a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the constraint term.
        pmax_ini: Initial value of pmax.
        pmax_end: Final value of pmax.
        num_steps: The number of steps after which the change should be applied.
        num_steps_grad: The number of steps over which pmax should be updated gradually.
    """
    if env.common_step_counter > num_steps:
        step_offset = env.common_step_counter - num_steps

        if step_offset <= num_steps_grad:
            fraction = step_offset / num_steps_grad
            
            # Calculate the new pmax value
            new_pmax = pmax_ini + fraction * (pmax_end - pmax_ini)

            # Obtain term settings
            term_cfg = env.constraint_manager.get_term_cfg(term_name)

            # Update term settings
            term_cfg.pmax = new_pmax

            # Apply the new configuration
            env.constraint_manager.set_term_cfg(term_name, term_cfg)
