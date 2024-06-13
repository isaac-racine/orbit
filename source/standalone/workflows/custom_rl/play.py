# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate")
parser.add_argument("--task", type=str, default=None, help="Name of the task")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

parser.add_argument("--playcontrol", type=bool, default=False, help="Control command directly")
parser.add_argument("--device", choices=["gamepad", "keyboard"], default="gamepad", help="Choose from options: gamepad, keyboard")
parser.add_argument("--lin_sensi", type=float, default=2, help="Gamepad linear speed sensitivity")
parser.add_argument("--rot_sensi", type=float, default=3.14/2, help="Gamepad rotational speed sensitivity")

# append RSL-RL cli arguments
cli_args.add_custom_rl_args(parser)
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

from omni.isaac.lab.rl.custom_rl.runners import OnPolicyRunner
from omni.isaac.lab.devices import Se2Gamepad, Se2Keyboard
from omni.isaac.lab.envs import ViewerCfg
from omni.isaac.lab.envs.ui import ViewportCameraController

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.custom_rl import (
    CustomRlOnPolicyRunnerCfg,
    CustomRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)


def command_control(teleop_interface, env):
	com = teleop_interface.advance()
	com[1] *= -1 ; com[2] *= -1
	command = env.unwrapped.command_manager._terms["base_velocity"]
	command.vel_command_b[0, 0] = com[0]
	command.vel_command_b[0, 1] = com[1]
	command.vel_command_b[0, 2] = com[2]

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: CustomRlOnPolicyRunnerCfg = cli_args.parse_custom_rl_cfg(args_cli.task, args_cli)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for custom-rl
    env = CustomRlVecEnvWrapper(env)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "custom_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict_ignore(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(ppo_runner.alg.actor_critic, path=export_model_dir, filename="policy.onnx")
    
    if args_cli.playcontrol:
        # setup device control (gamepad, keyboard)
        if args_cli.device == "gamepad":
            teleop_interface = Se2Gamepad(
                v_x_sensitivity=args_cli.lin_sensi,
                v_y_sensitivity=args_cli.lin_sensi / 2,
                omega_z_sensitivity=args_cli.rot_sensi,
                dead_zone=0.05,
            )
        elif args_cli.device == "keyboard":
            teleop_interface = Se2Keyboard(
                # v_x_sensitivity     = args_cli.lin_sensi,
                # v_y_sensitivity     = args_cli.lin_sensi/2,
                # omega_z_sensitivity = args_cli.rot_sensi,
            )
    
        teleop_interface.reset()
        command_control(teleop_interface,env)
    
        # setup camera
        cam_controller = ViewportCameraController(env.unwrapped, ViewerCfg())
        cam_controller.update_view_to_asset_root("robot")
        cam_controller.update_view_location([0, -4, 3], [0, 2, 0])
    
    # run environment
    obs, _ = env.get_observations()
    while simulation_app.is_running():
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            
            if args_cli.playcontrol : command_control(teleop_interface,env)
            
            # env stepping
            obs, _, _, _ = env.step(actions)
    
    # reset environment
    obs, _ = env.get_observations()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
