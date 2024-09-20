# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass
import math

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl_unified_policy import *


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
	num_steps_per_env = 40
	max_iterations = 10000
	save_interval = 200
	experiment_name = "locomanip"
	empirical_normalization = False
	policy = RslRlPpoActorCriticCfg(
		init_noise_std=1.0,
		actor_hidden_dims=[512, 256, 128], # not like in the paper (suppose to be 128 then seperate in 2 head of 128 (one for th manip and one for the loco))
		critic_hidden_dims=[512, 256, 128],
		activation="elu",
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
	)