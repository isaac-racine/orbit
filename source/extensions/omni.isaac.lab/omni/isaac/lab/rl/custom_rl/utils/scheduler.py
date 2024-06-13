#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch


class TrigScheduler:
	def __init__(self, valmin=0.0, valmax=1.0, period1=100, period2=500, cyclesmax=float("+inf"), init_step=0):
		self.n_cycles = 0
		self.valmin = valmin
		self.valmax = valmax
		
		self.p1 = period1
		self.p2 = period2
		self.ptot = period1 + period2 - 1
		self.cyclesmax = cyclesmax
		
		if period1 < 2 or period2 < 2:
			raise ValueError('Period must be >= 2')
		
		self.s1 = (valmax - valmin) / (self.p1 - 1)
		self.s2 = (valmin - valmax) / (self.p2 - 1)
		
		self.n_steps = init_step
	
	def step(self):
		if self.n_steps < self.p1 : s = self.s1 ; off = self.valmin
		else                      : s = self.s2 ; off = self.valmax - s*self.p1
		
		val = self.n_steps * s + off
		
		if self.n_cycles >= self.cyclesmax : return val
		
		self.n_steps += 1
		if self.n_steps >= self.ptot:
			self.n_steps = 0
			self.n_cycles += 1
		
		return val
class PWMScheduler:
	def __init__(self, valbase=0.0, valmax=1.0, dutycycle=0.1, period=300, cyclesmax=float("+inf"), init_rel=0):
		self.n_cycles = 0
		self.valbase = valbase
		self.valmax = valmax
		
		self.dc = dutycycle
		self.p1 = round((1-dutycycle) * period)
		self.p2 = period
		self.cyclesmax = cyclesmax
		
		self.n_steps = round(init_rel * period)
	
	def step(self):
		if self.n_cycles >= self.cyclesmax : return self.valbase
		
		if self.n_steps < self.p1 : val = self.valbase
		else                      : val = self.valmax
		
		self.n_steps += 1
		if self.n_steps >= self.p2:
			self.n_steps = 0
			self.n_cycles += 1
		
		return val

schedulers = {
	"trig": TrigScheduler,
	"pwm": PWMScheduler,
}
def get_scheduler(sch_str, params):
	if not sch_str in schedulers:
		raise ValueError(f"{sch_str} is not a valid scheduler")
	return schedulers[sch_str](**params)