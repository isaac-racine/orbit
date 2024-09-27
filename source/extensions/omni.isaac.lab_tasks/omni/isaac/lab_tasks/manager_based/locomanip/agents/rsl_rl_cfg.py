# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass
import math

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl_unified_policy import *

RESUME = True


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
	num_steps_per_env = 40
	max_iterations = 10000
	save_interval = 200
	experiment_name = "locomanip"
	empirical_normalization = False
	policy = RslRlPpoActorCriticCfg(
		init_noise_std= 1.0, #[[0.8, 1.0, 1.0] * 4 + [1.0] * 6],
		actor_hidden_dims=[128], 
		critic_hidden_dims=[128],
		activation="elu",

		leg_control_head_hidden_dims = [128, 128],
		arm_control_head_hidden_dims = [128, 128],
		priv_encoder_dims = [64, 20],
		num_leg_actions = 12,
		num_arm_actions = 7,
	)
	algorithm = RslRlPpoAlgorithmCfg(
		value_loss_coef=1.0,
		use_clipped_value_loss=True,
		clip_param=0.2,
		entropy_coef=0.01,
		num_learning_epochs=5,
		num_mini_batches=4,
		learning_rate=2.0e-4,
		schedule="adaptive",
		gamma=0.99,
		lam=0.95,
		desired_kl=0.01,
		max_grad_norm=1.0, ## ??
		min_policy_std = [[0.15, 0.25, 0.25] * 4 + [0.2] * 3 + [0.05] * 3],
		mixing_schedule=[1.0, 0, 3000] if not RESUME else [1.0, 0, 1],
		dagger_update_freq=20,
		priv_reg_coef_schedual=[0, 0.1, 3000, 7000] if not RESUME else [0, 1, 1000, 1000],
	)