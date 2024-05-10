# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import omni.isaac.orbit.terrains as terrain_gen

from ..terrain_generator_cfg import TerrainGeneratorCfg, FlatPatchSamplingCfg

import math

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
	size=(8.0, 8.0),
	border_width=20.0,
	num_rows=10,
	num_cols=20,
	horizontal_scale=0.1,
	vertical_scale=0.005,
	slope_threshold=0.75,
	use_cache=False,
	sub_terrains={
		"pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
			proportion=0.2,
			step_height_range=(0.05, 0.23),
			step_width=0.3,
			platform_width=3.0,
			border_width=1.0,
			holes=False,
		),
		"pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
			proportion=0.2,
			step_height_range=(0.05, 0.23),
			step_width=0.3,
			platform_width=3.0,
			border_width=1.0,
			holes=False,
		),
		"boxes": terrain_gen.MeshRandomGridTerrainCfg(
			proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
		),
		"random_rough": terrain_gen.HfRandomUniformTerrainCfg(
			proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
		),
		"hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
			proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
		),
		"hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
			proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
		),
	},
)
"""Rough terrains configuration."""

VEL_CUSTOM_TERRAIN_CFG = TerrainGeneratorCfg(
	size=(15.0, 15.0),
	border_width=2.0,
	num_rows=6,
	num_cols=7,
	horizontal_scale=0.1,
	vertical_scale=0.005,
	use_cache=False,
	curriculum=True,
	sub_terrains={
		"flat": terrain_gen.MeshPlaneTerrainCfg(
		),
		"random_rough": terrain_gen.HfRandomUniformTerrainCfg(
			downsampled_scale=0.2,
			noise_range=(0.0, 0.10),
			noise_step=0.01,
			border_width=2.0
		),
		"boxes": terrain_gen.MeshRandomGridTerrainCfg(
			grid_width=0.45,
			grid_height_range=(0.05, 0.2),
			platform_width=2.0,
		),
		"hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
			horizontal_scale=0.01,
			slope_range=(0.0, 0.7),
			platform_width=2.0,
			border_width=2.0,
		),
		"pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
			step_height_range=(0.05, 0.3),
			step_width=0.3,
			platform_width=3.0,
			holes=False,
			border_width=2.0,
		),
		"hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
			horizontal_scale=0.01,
			slope_range=(0.0, 0.7),
			platform_width=2.0,
			border_width=2.0,
		),
		"pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
			step_height_range=(0.05, 0.3),
			step_width=0.3,
			platform_width=3.0,
			holes=False,
			border_width=2.0,
		),
	},
)

TARGET_SAMPLING_CFG = FlatPatchSamplingCfg(
	num_patches=5,
	patch_radius=[0.5],
	max_height_diff=math.inf
)
POS_CUSTOM_TERRAIN_CFG = TerrainGeneratorCfg(
	size=(20.0, 20.0),
	border_width=2.0,
	num_rows=4,
	num_cols=7,
	horizontal_scale=0.1,
	vertical_scale=0.005,
	use_cache=False,
	curriculum=True,
	color_scheme='random',
	sub_terrains={
		"flat": terrain_gen.MeshPlaneTerrainCfg(
			flat_patch_sampling={'target': TARGET_SAMPLING_CFG},
		),
		"random_rough": terrain_gen.HfRandomUniformTerrainCfg(
			flat_patch_sampling={'target': TARGET_SAMPLING_CFG},
			downsampled_scale=0.2,
			noise_range=(0.0, 0.10),
			noise_step=0.01,
			border_width=2.0
		),
		"boxes": terrain_gen.MeshRandomGridTerrainCfg(
			flat_patch_sampling={'target': TARGET_SAMPLING_CFG},
			grid_width=0.45,
			grid_height_range=(0.05, 0.2),
			platform_width=2.0,
		),
		"hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
			flat_patch_sampling={'target': TARGET_SAMPLING_CFG},
			horizontal_scale=0.01,
			slope_range=(0.0, 0.7),
			platform_width=2.0,
			border_width=2.0,
		),
		"pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
			flat_patch_sampling={'target': TARGET_SAMPLING_CFG},
			step_height_range=(0.05, 0.3),
			step_width=0.3,
			platform_width=3.0,
			holes=False,
			border_width=2.0,
		),
		"hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
			flat_patch_sampling={'target': TARGET_SAMPLING_CFG},
			horizontal_scale=0.01,
			slope_range=(0.0, 0.7),
			platform_width=2.0,
			border_width=2.0,
		),
		"pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
			flat_patch_sampling={'target': TARGET_SAMPLING_CFG},
			step_height_range=(0.05, 0.3),
			step_width=0.3,
			platform_width=3.0,
			holes=False,
			border_width=2.0,
		),
	},
)
