# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="Augment input size of trained model")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--model_path", type=str, default="", help="Path of the model file")
parser.add_argument("--new_size", type=str, default="", help="New size of the input layer")
args_cli = parser.parse_args()

import os
import torch

def main():
	
	print("""
		Resizing network...
		Warning :: the new inputs must go at the ***end*** of the input tensor 
	""")
	
	newsize = 400
	model = torch.load(args_cli.model_path, map_location=torch.device('cpu'))
	
	model_states = model['model_state_dict']
	for key in model_states.keys():
		if not 'rnn.weight_ih_' in key : continue
		
		weights = model_states[key]
		if oldsize is None:
			oldsize = weights.shape[1]
			pad = torch.nn.ConstantPad1d((0,newsize-oldsize), 0.0)
			
		model_states[key] = pad(weights)
	
	


if __name__ == "__main__":
	# run the main function
	main()