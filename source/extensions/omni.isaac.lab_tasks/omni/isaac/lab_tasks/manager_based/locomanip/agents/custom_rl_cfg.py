# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass
import math

from omni.isaac.lab_tasks.utils.wrappers.custom_rl import *

@configclass
class PPORunnerCfg(CustomRlOnPolicyRunnerCfg):
	num_steps_per_env = 24
	max_iterations = 50000
	save_interval = 100
	experiment_name = "locomanip"
	empirical_normalization = False
	policy = CustomRlPpoActorCriticCfg(
		#noise_scheduler="trig",
		#noise_sch_params={"valmin":1e-6, "valmax":1.0, "period1":20, "period2":200, "cyclesmax":2, "init_step":20},
		#noise_scheduler="pwm",
		#noise_sch_params={"valbase":1e-3, "valmax":0.25, "dutycycle":1.0, "period":200, "cyclesmax":1},
		init_noise_std=0.75,
		
		actor_hidden_dims=[512, 256, 128],
		critic_hidden_dims=[512, 256, 128],
		activation="elu",
		
		#actor_w_zero=True,
		#static_noise=True,
	)
	algorithm = CustomRlPpoAlgorithmCfg(
		num_learning_epochs=5,
		num_mini_batches=6,
		
		#lr_scheduler="trig",
		#lr_sch_params={"valmin":5e-5, "valmax":2e-3, "period1":50, "period2":500, "cyclesmax":1},
		lr_scheduler="fixed",
		init_learning_rate=2e-4,
		
		gamma=0.99,
		lam=0.95,
		desired_kl=0.01,
		max_grad_norm=1.0,
		value_loss_coef=1.0,
		use_clipped_value_loss=True,
		clip_param=0.2,
		entropy_coef=0.0, # 0.0 prevents the noise from increasing above init value
	)

#@configclass
#class PPORunnerCfg(CustomRlOnPolicyRunnerCfg):
#	num_steps_per_env = 24
#	max_iterations = 10000
#	save_interval = 100
#	experiment_name = "locomanip"
#	empirical_normalization = False
#	policy = CustomRlPpoActorCriticCfg(
#		init_noise_std=1.0,
#		actor_hidden_dims=[512, 256, 128],
#		critic_hidden_dims=[512, 256, 128],
#		activation="elu",
#	)
#	algorithm = CustomRlPpoAlgorithmCfg(
#		value_loss_coef=1.0,
#		use_clipped_value_loss=True,
#		clip_param=0.2,
#		entropy_coef=0.01,
#		num_learning_epochs=5,
#		num_mini_batches=4,
#		learning_rate=1.0e-3,
#		schedule="adaptive",
#		gamma=0.99,
#		lam=0.95,
#		desired_kl=0.01,
#		max_grad_norm=1.0,
#	)