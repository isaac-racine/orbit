from dataclasses import dataclass

import numpy as np
import scipy.spatial.transform as tf
import torch
import random
import trimesh
from utils import *

#from .utils import *  # noqa: F401, F403

@dataclass
class MeshPyramidSlopeTerrainCfg():
	"""Configuration for a pyramid stair mesh terrain."""
	
	size: tuple[float, float] = (15.0, 15.0)
	
	border_width: float = 2.0
	platform_width: float = 5.0
	platform_height: float = 5.0
	
	holes = False
	"""The width of the border around the terrain (in m). Defaults to 0.0.

	The border is a flat terrain with the same height as the terrain.
	"""
	slope_angle_range: tuple[float, float] = None
	"""The minimum and maximum slope angle (in radians)."""
@dataclass
class MeshInvertedPyramidSlopeTerrainCfg(MeshPyramidSlopeTerrainCfg) : pass

@dataclass
class MeshPyramidStairsRandTerrainCfg():
	size: tuple[float, float] = (15.0, 15.0)
	
	border_width: float = 2.0
	platform_width: float = 5.0
	platform_height: float = 5.0
	
	step_height_range: tuple[float, float] = (0.1, 0.3)
	step_height_maxincr: float = .0
	step_width_range: tuple[float, float] = (0.4, 2.0)
	step_width_maxincr: float = .0
	holes: bool = False
@dataclass
class MeshInvertedPyramidStairsRandTerrainCfg(MeshPyramidStairsRandTerrainCfg) : pass


def pyramid_slope_terrain(
	difficulty: float, cfg: MeshPyramidSlopeTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
	"""
	If the size is not square, the slope is respected only for the x-axis
	"""
	
	meshes_list = list()
	
	slope_size = np.array([
		cfg.size[0] - 2 * cfg.border_width - cfg.platform_width,
		cfg.size[1] - 2 * cfg.border_width - cfg.platform_height
	])/2.0
	angle = np.array([
		np.interp(difficulty, (0.0,1.0), cfg.slope_angle_range),
		0.0
	])
	height = np.tan(angle[0])*slope_size[0]
	angle[1] = np.arctan(height/slope_size[1])
	slope_length = slope_size / np.cos(angle)
	
	terrain_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
	
	# generate the border if needed
	if cfg.border_width > 0.0 and not cfg.holes:
		border_meshes = make_border(cfg.size, terrain_size, 0, [0,0,0])
		meshes_list += border_meshes
	
	# generate the terrain
	origin = [0.0, 0.0, height]
	
	center_mesh = make_plane((cfg.platform_width,cfg.platform_height), height, True)
	meshes_list.append(center_mesh)
	
	slope_mesh1 = make_slope(angle[0], (0.0,1.0,0.0), (slope_length[0],cfg.platform_height))
	slope_mesh1.apply_translation(np.array([ cfg.platform_width/2 - np.min(slope_mesh1.vertices[:,0]), 0, height - np.max(slope_mesh1.vertices[:,2]) ]))
	slope_mesh2 = make_slope(-angle[0], (0.0,1.0,0.0), (slope_length[0],cfg.platform_height))
	slope_mesh2.apply_translation(np.array([ -cfg.platform_width/2 + np.min(slope_mesh2.vertices[:,0]), 0, height - np.max(slope_mesh2.vertices[:,2]) ]))
	slope_mesh3 = make_slope(-angle[1], (1.0,0.0,0.0), (cfg.platform_width,slope_length[1]))
	slope_mesh3.apply_translation(np.array([ 0, cfg.platform_height/2 - np.min(slope_mesh3.vertices[:,1]), height - np.max(slope_mesh3.vertices[:,2]) ]))
	slope_mesh4 = make_slope(angle[1], (1.0,0.0,0.0), (cfg.platform_width,slope_length[1]))
	slope_mesh4.apply_translation(np.array([ 0, -cfg.platform_height/2 + np.min(slope_mesh4.vertices[:,1]), height - np.max(slope_mesh4.vertices[:,2]) ]))
	
	meshes_list += [slope_mesh1,slope_mesh2,slope_mesh3,slope_mesh4]
	
	# fill the holes
	if not cfg.holes:
		for sx in [-1.0,1.0]:
			for sy in [-1.0,1.0]:
				meshes_list.append(trimesh.Trimesh(
					vertices=np.array([
						(sx*cfg.platform_width/2, sy*cfg.platform_height/2, height), # bl
						(sx*terrain_size[0]/2, sy*cfg.platform_height/2, 0), # br
						(sx*terrain_size[0]/2, sy*terrain_size[1]/2, 0), # tr
						(sx*cfg.platform_width/2, sy*terrain_size[1]/2, 0), # tl
					]),
					faces=np.array([[0,1,2], [0,2,3]]) if sx*sy>0 else np.array([[1,0,2], [2,0,3]])
				))
	
	return meshes_list, origin
def inverted_pyramid_slope_terrain(
	difficulty: float, cfg: MeshInvertedPyramidSlopeTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
	meshes_list, origin = pyramid_slope_terrain(difficulty, cfg)
	# set symmetry
	for mesh in meshes_list:
		for vert in mesh.vertices : vert[2] *= -1
	# flip normals
	for mesh in meshes_list[-1:-4]:
		for face in mesh.faces : face[:] = [face[1],face[0],face[2]]
		
	origin[2] *= -1
	
	return meshes_list, origin

def gen_uniform_steps_totsize(size, step_range):
	vals = [0,]
	while True:
		vals.append(vals[-1] + random.uniform(*step_range))
		if vals[-1] >= size:
			vals[-1] = size
			break
	return vals
def gen_uniform_steps_totnum(num, step_range):
	vals = [0,]
	for i in range(num-1) : vals.append(vals[-1] + random.uniform(*step_range))
	return vals

def pyramid_stairsrand_terrain(
	difficulty: float, cfg: MeshPyramidStairsRandTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
	
	# Doesn't work with non square terrain
	
	step_width_max = cfg.step_width_range[0] + difficulty * (cfg.step_width_range[1] - cfg.step_width_range[0])
	step_height_max = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
	step_width_min = difficulty * cfg.step_width_maxincr + cfg.step_width_range[0]
	step_height_min = difficulty * cfg.step_height_maxincr + cfg.step_height_range[0]
	
	terrain_size = np.array((
		cfg.size[0] - 2 * cfg.border_width,
		cfg.size[1] - 2 * cfg.border_width
	))
	platform_size = np.array((cfg.platform_width, cfg.platform_height))
	stairs_size = terrain_size - platform_size
	
	# get steps, width first
	steps_width  = np.array(gen_uniform_steps_totsize(stairs_size[0]/2, (step_width_min, step_width_max)))
	steps_height = np.array(gen_uniform_steps_totnum(len(steps_width), (step_height_min, step_height_max)))
	height = steps_height[-1]

	meshes_list = list()

	# generate the border if needed
	if cfg.border_width > 0.0 and not cfg.holes:
		border_meshes = make_border(cfg.size, terrain_size, 0, [0,0,0])
		meshes_list += border_meshes
	
	# generate the terrain
	origin = [0.0, 0.0, height]

	center_mesh = make_plane((cfg.platform_width,cfg.platform_height), height, True)
	meshes_list.append(center_mesh)
	
	# stair profiles
	stairs_lt = [] ; stairs_lb = []
	stairs_bl = [] ; stairs_br = []
	for i in range(len(steps_width)):
		x = steps_width[i]-terrain_size[0]/2
		y = steps_width[i]-terrain_size[1]/2
		
		if i != 0:
			stairs_lt.append((x, cfg.platform_height/2, steps_height[i-1]))
			stairs_lb.append((x, -cfg.platform_height/2, steps_height[i-1]))
			stairs_br.append((cfg.platform_width/2, y, steps_height[i-1]))
			stairs_bl.append((-cfg.platform_width/2, y, steps_height[i-1]))
		
		stairs_lt.append((x, cfg.platform_height/2, steps_height[i]))
		stairs_lb.append((x, -cfg.platform_height/2, steps_height[i]))
		stairs_br.append((cfg.platform_width/2, y, steps_height[i]))
		stairs_bl.append((-cfg.platform_width/2, y, steps_height[i]))
	
	stairs_lt = np.array(stairs_lt) ; stairs_rt = np.copy(stairs_lt) ; stairs_rt[:,0] *= -1
	stairs_lb = np.array(stairs_lb) ; stairs_rb = np.copy(stairs_lb) ; stairs_rb[:,0] *= -1
	stairs_tr = np.array(stairs_br) ; stairs_tr = np.copy(stairs_br) ; stairs_tr[:,1] *= -1
	stairs_tl = np.array(stairs_bl) ; stairs_tl = np.copy(stairs_bl) ; stairs_tl[:,1] *= -1
	
	# faces
	faces = []
	for i in range(1, len(stairs_lt)):
		ind1 = i ; ind2 = len(stairs_lt)+i
		faces += ((ind1-1, ind2-1, ind2), (ind1-1, ind2, ind1))
	
	# meshes
	mesh_l = trimesh.Trimesh( vertices=np.concatenate((stairs_lt, stairs_lb)), faces=np.array(faces) ) # left
	mesh_b = trimesh.Trimesh( vertices=np.concatenate((stairs_bl, stairs_br)), faces=np.array(faces) ) # bottom
	mesh_r = trimesh.Trimesh( vertices=np.concatenate((stairs_rt, stairs_rb)), faces=np.array(faces) ) # right
	mesh_t = trimesh.Trimesh( vertices=np.concatenate((stairs_tr, stairs_tl)), faces=np.array(faces) ) # top
	meshes_list += [mesh_l, mesh_b, mesh_r, mesh_t]
	
	# fill the holes
	if not cfg.holes:
		# skewed stairs
		mesh_bl = trimesh.Trimesh( vertices=np.concatenate((np.array(stairs_lb), np.array(stairs_bl))), faces=np.array(faces) ) # bottom left
		mesh_br = trimesh.Trimesh( vertices=np.concatenate((np.array(stairs_br), np.array(stairs_rb))), faces=np.array(faces) ) # bottom right
		mesh_tl = trimesh.Trimesh( vertices=np.concatenate((np.array(stairs_lt), np.array(stairs_tl))), faces=np.array(faces) ) # top left
		mesh_tr = trimesh.Trimesh( vertices=np.concatenate((np.array(stairs_rt), np.array(stairs_tr))), faces=np.array(faces) ) # top right
		meshes_list += [mesh_bl, mesh_br, mesh_tl, mesh_tr]
		
		# triangle in each corner
		for sx in [-1.0,1.0]:
			for sy in [-1.0,1.0]:
				meshes_list.append(trimesh.Trimesh(
					vertices=np.array([
						(terrain_size[0]/2*sx, terrain_size[1]/2*sy, 0),
						(platform_size[0]/2*sx, terrain_size[1]/2*sy, 0),
						(terrain_size[0]/2*sx, platform_size[1]/2*sy, 0),
					]),
					faces=np.array((0,1,2))
				))
	
	
	return meshes_list, origin
def inverted_pyramid_stairsrand_terrain(
	difficulty: float, cfg: MeshInvertedPyramidStairsRandTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
	meshes_list, origin = pyramid_stairsrand_terrain(difficulty, cfg)
	# set symmetry
	for mesh in meshes_list:
		for vert in mesh.vertices : vert[2] *= -1
	# flip normals
	for mesh in meshes_list[-1:-4]:
		for face in mesh.faces : face[:] = [face[1],face[0],face[2]]
		
	origin[2] *= -1
	
	return meshes_list, origin

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.axis('equal')
ax.set_xlim(-5.0, 5.0)
ax.set_ylim(-5.0, 5.0)
ax.set_zlim(0.0, 5.0)

meshes = inverted_pyramid_stairsrand_terrain(1.0, MeshInvertedPyramidStairsRandTerrainCfg())[0]

for mesh in meshes : ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:,1], mesh.vertices[:,2], triangles=mesh.faces);
plt.show()
