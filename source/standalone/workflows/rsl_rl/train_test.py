# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.orbit.app import AppLauncher

# local imports
import cli_args	 # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go2-Play-v1", help="Name of the task.")
parser.add_argument("--device", choices=["gamepad","keyboard"], default="gamepad", help="Choose from options: gamepad, keyboard")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from omni.isaac.orbit.devices import Se2Gamepad, Se2Keyboard
from omni.isaac.orbit.envs.ui import ViewportCameraController
from omni.isaac.orbit.envs import ViewerCfg
from rsl_rl.runners import OnPolicyRunner
import omni.isaac.orbit_tasks
from omni.isaac.orbit_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_onnx


def main():
	"""Play with RSL-RL agent."""
	# parse configuration
	env_cfg = parse_env_cfg(
		args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
	)
	agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

	# create isaac environment
	env = gym.make(args_cli.task, cfg=env_cfg)
	# wrap around environment for rsl-rl
	env = RslRlVecEnvWrapper(env)

	# specify directory for logging experiments
	log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
	log_root_path = os.path.abspath(log_root_path)
	print(f"[INFO] Loading experiment from directory: {log_root_path}")
	resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
	print(f"[INFO]: Loading model checkpoint from: {resume_path}")

	# load previously trained model
	ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
	# write git state to logs
	runner.add_git_repo_to_log(__file__)
	# save resume path before creating a new log_dir
	if agent_cfg.resume:
		# get path to previous checkpoint
		resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
		print(f"[INFO]: Loading model checkpoint from: {resume_path}")
		# load previously trained model
		runner.load(resume_path)

	# set seed of the environment
	env.seed(agent_cfg.seed)

	# dump the configuration into log-directory
	dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
	dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
	dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
	dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

	# run training
	runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

	# close the simulator
	env.close()


if __name__ == "__main__":
	# run the main function
	main()
	# close sim app
	simulation_app.close()
