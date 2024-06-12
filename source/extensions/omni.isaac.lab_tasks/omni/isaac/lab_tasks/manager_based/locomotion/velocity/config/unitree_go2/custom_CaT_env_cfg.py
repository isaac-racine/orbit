# Copyright (c) 2022-2024, The lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING



import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg, EventTermCfg, ObservationGroupCfg,ObservationTermCfg, ConstraintTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns, CameraCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from omni.isaac.lab.terrains.config.rough import VEL_CUSTOM_TERRAIN_CFG, ROUGH_TERRAINS_CFG
from omni.isaac.lab_assets.unitree import UNITREE_GO2_CFG


DEBUG_VIS=False
EPISODE_LENGTH=40.0
SIM_DT=0.005

UNWANTED_CONTACT_BODIES=["base",".*_hip","Head_.*",".*_calf"]
ILLEGAL_CONTACT_BODIES=["base", "Head_.*",".*_thigh"]
MAX_PUSHSPEED = 1.5
MAX_COM_LINSPEED = 1.0
MAX_COM_ANGSPEED = math.pi/2

#Constraint Hyperparameters
JOINT_TORQUE_LIMIT = 45 #N/m
JOINT_VEL_LIMIT = 16 #rad/s
JOINT_ACC_LIMIT = 800 #rad/s^2
ACTION_RATE_LIMIT = 110 #rad/s
BASE_ORIENTATION_LIMIT = 0.1 #rad
CONTACT_FORCE_LIMIT = 50 #N
HIP_ANGLE_LIMIT = 0.2 #rad
AIR_TIME_TARGET = 0.25 #s
NUMBER_OF_FOOT_CONTACT_TARGET = 2
VELOCITY_TRACKING = 0.2 #m/s or 1 rad/s


##
# Scene definition
##


@configclass
class CaTSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG, #VEL_CUSTOM_TERRAIN_CFG,
        min_init_terrain_level=0,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=DEBUG_VIS,
    )
    
    # robots
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=DEBUG_VIS,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(math.inf, math.inf),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=DEBUG_VIS,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-MAX_COM_LINSPEED, MAX_COM_LINSPEED),
            lin_vel_y=(-MAX_COM_LINSPEED, MAX_COM_LINSPEED),
            ang_vel_z=(-MAX_COM_ANGSPEED, MAX_COM_ANGSPEED),
            heading=(-math.pi, math.pi)
        )
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObservationTermCfg(func=mdp.last_action)
        height_scan = ObservationTermCfg(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base"), "mass_distribution_params": (-1.0, 3.0), "operation": "add"},
    )

    # reset
    reset_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }
        },
    )

    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    
    # interval
    push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, min(5.0, EPISODE_LENGTH)),
        params={"velocity_range": {"x": (-MAX_PUSHSPEED, MAX_PUSHSPEED), "y": (-MAX_PUSHSPEED, MAX_PUSHSPEED)}},
        curriculum_dependency = True,
        curriculum_row_range = (1,-1), # starting from 2nd level
        curriculum_col_range = (1,1), # random boxes ground
    )


@configclass
class ConstraintsCfg:
    """Configuration for contraints."""

    #  TODO: Trouver les probabilités de terminaison pour chacun des termes.(pmax=?)
    #  TODO: Vérifier les unités qui sont utilisé.
    
    ############
    # Option A #
    ############

    # Joint constraints
    c_joint_torque = ConstraintTermCfg(func=mdp.c_joint_torque, params={"limval": JOINT_TORQUE_LIMIT}, pmax=0.05)
    c_joint_vel = ConstraintTermCfg(func=mdp.c_joint_vel, params={"limval": JOINT_VEL_LIMIT}, pmax=0.05)
    c_joint_acc = ConstraintTermCfg(func=mdp.c_joint_acc, params={"limval": JOINT_ACC_LIMIT}, pmax=0.05) # Verifier la limite des moteurs

    # Other
    #c_action_rate = ConstraintTermCfg(func=mdp.c_action_rate, params={"limval": ACTION_RATE_LIMIT}, pmax=0.05) # based on paper

    # Style constraint (only on flat terrain)

    #c_base_ori = ConstraintTermCfg(func=mdp.c_base_ori_xy, params={"limval": BASE_ORIENTATION_LIMIT}, pmax=0.25)
    #c_hip_ori = ConstraintTermCfg(func=mdp.c_hip_ori, params={"limval": HIP_ANGLE_LIMIT}, pmax=0.25)
    #c_air_time = ConstraintTermCfg(func=mdp.c_air_time, params={"limval": AIR_TIME_TARGET, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"),\
    #                                                            "command_name": "base_velocity",}, pmax=0.25)
    #c_nb_foot_contact = ConstraintTermCfg(func=mdp.c_nb_foot_contact, params={"limval": NUMBER_OF_FOOT_CONTACT_TARGET}, pmax=0.25)
    #c_stand_still = ConstraintTermCfg(func=mdp.c_stand_still, params={"limval": ACTION_RATE_LIMIT}, pmax=0.25)


    ############
    # Option B #
    ############
    



@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Option A (task formulation through rewards)
    r_track_lin_vel_xy_exp = RewardTermCfg(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    r_track_ang_vel_z_exp = RewardTermCfg(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    #Option B (Task formulation through sof constraints)
    # r = 1

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    
    out_of_bounds = TerminationTermCfg(func=mdp.root_out_of_curriculum, time_out=True)
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    
    robot_illegal_contact = TerminationTermCfg(
    	func=mdp.illegal_contact,
    	params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=ILLEGAL_CONTACT_BODIES), "threshold": 1.0},
    )
    
    #Foot contact force




    #contact_full = TerminationTermCfg(
    #	func=mdp.illegal_contact,
    #	params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=UNWANTED_CONTACT_BODIES), "threshold": 1.0},
    #	curriculum_dependency = True,
    #	curriculum_row_range = (0,0), # only first level
    #)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    
    terrain_levels = CurriculumTermCfg(func=mdp.terrain_levels_vel)
    #terrain_levels = CurriculumTermCfg(func=mdp.terrain_levels_vel2)

    # action_rate = CurriculumTermCfg(
    #     func=mdp.modify_constraint_pmax, params={"term_name": "c_action_rate", "pmax_ini": 0.005, "pmax_end" : 0.25, "num_steps": 10000, "num_steps_grad": 30000}
    # )
    joint_vel = CurriculumTermCfg(
        func=mdp.modify_constraint_pmax, params={"term_name": "c_joint_vel", "pmax_ini": 0.05, "pmax_end" : 0.25, "num_steps": 10000, "num_steps_grad": 30000}
    )
    joint_acc = CurriculumTermCfg(
        func=mdp.modify_constraint_pmax, params={"term_name": "c_joint_acc", "pmax_ini": 0.05, "pmax_end" : 0.25, "num_steps": 10000, "num_steps_grad": 30000}
    )
    joint_torque = CurriculumTermCfg(
        func=mdp.modify_constraint_pmax, params={"term_name": "c_joint_torque", "pmax_ini": 0.05, "pmax_end" : 0.25, "num_steps": 10000, "num_steps_grad": 30000}
    )

##
# Environment configuration
##


@configclass
class UnitreeGo2VelCustomEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: CaTSceneCfg = CaTSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    constraints: ConstraintsCfg = ConstraintsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""    

        # general settings
        self.decimation = 4
        self.episode_length_s = EPISODE_LENGTH
        
        # simulation settings
        self.sim.dt = SIM_DT
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        
        # articulation settings
        self.scene.robot.spawn.articulation_props.solver_position_iteration_count = 20 # a high number prevents objects from going through the ground
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

@configclass
class UnitreeGo2VelCustomEnvCfg_PLAYCONTROL(UnitreeGo2VelCustomEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        self.episode_length_s = 3600
        
        self.scene.terrain.terrain_generator.num_rows = 5
        #self.scene.terrain.terrain_generator.size = (8.0,8.0)
        self.scene.terrain.min_init_terrain_level=0
        self.scene.terrain.max_init_terrain_level=0
        self.curriculum.terrain_levels = None
        self.terminations.out_of_bounds = None
        
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        
        self.episode_length_s = 3600
        
        self.commands.base_velocity = mdp.UserVelocityCommandCfg(
            asset_name="robot",
            debug_vis=DEBUG_VIS,
            resampling_time_range=(math.inf, math.inf), # not used 
        )