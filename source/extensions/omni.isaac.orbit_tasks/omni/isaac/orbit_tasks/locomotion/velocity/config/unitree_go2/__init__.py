# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg, custom_env_cfg, custom1_env_cfg, custom2_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.UnitreeGo2FlatEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UnitreeGo2FlatPPORunnerCfg,
    },
)
gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-Play-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.UnitreeGo2FlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UnitreeGo2FlatPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go2-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.UnitreeGo2RoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UnitreeGo2RoughPPORunnerCfg,
    },
)
gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go2-Play-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.UnitreeGo2RoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UnitreeGo2RoughPPORunnerCfg,
    },
)
gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go2-PlayControl-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.UnitreeGo2RoughEnvCfg_PLAYCONTROL,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UnitreeGo2RoughPPORunnerCfg,
    },
)

# track velocity command (with speed)
gym.register(
    id="Isaac-Velocity-Custom-Unitree-Go2-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": custom_env_cfg.UnitreeGo2VelCustomEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UnitreeGo2VelCustom1PPORunnerCfg,
    },
)
gym.register(
    id="Isaac-Velocity-Custom-Unitree-Go2-v1",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": custom1_env_cfg.UnitreeGo2VelCustomEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UnitreeGo2VelCustom1PPORunnerCfg,
    },
)
gym.register(
    id="Isaac-Velocity-Custom-Unitree-Go2-PlayControl-v1",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": custom1_env_cfg.UnitreeGo2VelCustomEnvCfg_PLAYCONTROL,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UnitreeGo2VelCustom1PPORunnerCfg,
    },
)

# track the direction part of velocity command
gym.register(
    id="Isaac-Velocity-Custom-Unitree-Go2-v2",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": custom2_env_cfg.UnitreeGo2VelCustomEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UnitreeGo2VelCustom2PPORunnerCfg,
    },
)
gym.register(
    id="Isaac-Velocity-Custom-Unitree-Go2-PlayControl-v2",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": custom2_env_cfg.UnitreeGo2VelCustomEnvCfg_PLAYCONTROL,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UnitreeGo2VelCustom2PPORunnerCfg,
    },
)
