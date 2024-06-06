# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.orbit.sensors.ray_caster.patterns.patterns_cfg import OmniPatternCfg
from omni.isaac.orbit.sensors.ray_caster.ray_caster_cfg import RayCasterCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit_tasks.locomotion.position.position_env_cfg import LocomotionPositionRoughEnvCfg

# import omni.isaac.orbit_tasks.locomotion.velocity.mdp as mdp


##
# Pre-defined configs
##
from omni.isaac.orbit_assets.unitree import UNITREE_GO2_CFG  # isort: skip


@configclass
class UnitreeGo2RoughEnvCfg(LocomotionPositionRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_range"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_pos_xy_exp.weight = 1.5
        self.rewards.track_heading_exp.weight = 0.75
        self.rewards.track_speedxy_exp.weight = 1.0
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class UnitreeGo2RoughEnvCfg_PLAY(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class UnitreeGo2CustomEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # modify scanner
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=RayCasterCfg.OffsetCfg(pos=(0.28945, 0.0, -0.04682)),
            pattern_cfg=OmniPatternCfg(),
            max_distance=30,
            debug_vis=True,
            mesh_prim_paths=["/World/ground"],
            update_period=0,
        )


@configclass
class UnitreeGo2CustomEnvCfg_PLAY(UnitreeGo2CustomEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # receive commands from user
        # self.commands.base_velocity = mdp.UserVelocityCommandCfg(
        # 	asset_name="robot",
        # 	debug_vis=True,
        # )

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
