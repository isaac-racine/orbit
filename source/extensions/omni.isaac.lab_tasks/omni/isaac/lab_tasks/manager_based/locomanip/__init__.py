# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, env_cfg

##
# Register Gym environments.
##

# track velocity command, heading velocity command, end effector position command
gym.register(
    id="Isaac-LocoManip-Flat",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.FlatEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PPORunnerCfg,
		"custom_rl_cfg_entry_point": agents.custom_rl_cfg.PPORunnerCfg,
    },
)
gym.register(
    id="Isaac-LocoManip-Curriculum",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.CurriculumEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PPORunnerCfg,
		"custom_rl_cfg_entry_point": agents.custom_rl_cfg.PPORunnerCfg,
    },
)
gym.register(
    id="Isaac-LocoManip-Flat-PlayControl",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.FlatEnvCfg_PlayControl,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PPORunnerCfg,
		"custom_rl_cfg_entry_point": agents.custom_rl_cfg.PPORunnerCfg,
    },
)
gym.register(
    id="Isaac-LocoManip-Curriculum-PlayControl",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.CurriculumEnvCfg_PlayControl,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PPORunnerCfg,
		"custom_rl_cfg_entry_point": agents.custom_rl_cfg.PPORunnerCfg,
    },
)