"""Training configs for ``RobotLab-Go2W-Symmetry-v1``.

Flat plane terrain (GPU savings), Walk These Ways augmented auxiliary rewards,
and sparse top-2 MoE gating for blind proprioceptive training.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass

import robot_lab.tasks.go2w.mdp as mdp
from robot_lab.tasks.go2.rsl_rl_cfg import MoECTSSymmetryRunnerCfg, RslRlMoeCtsActorCriticCfg
from robot_lab.tasks.go2w.env_cfg import (
    BASE_HEIGHT_TARGET,
    FOOT_LINK_NAME,
    STAND_STILL_JOINT_PENALTY_TERMS,
    WHEEL_RADIUS,
    Go2WEnvSymmetryCfg,
    Go2WSceneCfg,
    RewardsCfg,
)
from robot_lab.tasks.go2w.rsl_rl_cfg import Go2WMoeCtsSymmetryCfg

# WTW ``scripts/train.py`` scales for the six augmented auxiliary rewards (Table 1).
WTW_JUMP_WEIGHT = 100.0
WTW_ORIENTATION_CONTROL_WEIGHT = -20.0 # -5.0
WTW_RAIBERT_HEURISTIC_WEIGHT = -10.0
WTW_RAIBERT_HEURISTIC_IMBALANCE_X_WEIGHT = -15
WTW_RAIBERT_HEURISTIC_IMBALANCE_Y_WEIGHT = -15
WTW_FEET_CLEARANCE_CMD_LINEAR_WEIGHT = -30.0
WTW_FEET_CLEARANCE_CMD_LINEAR_IMBALANCE_WEIGHT = -10 # -40
WTW_TRACKING_CONTACTS_SHAPED_FORCE_WEIGHT = 4.0
WTW_TRACKING_CONTACTS_SHAPED_VEL_WEIGHT = 4.0

_WTW_FOOTSWING_HEIGHT_CMD = 0.04

_WTW_GAIT_TIMING_PARAMS = {
    "gait_frequency": 1.0,
    "gait_phase": 0.5,
    "gait_offset": 0.0,
    "gait_bound": 0.0,
    "gait_duration": 0.5,
    "kappa_gait_probs": 0.07,
    "footswing_height_cmd": _WTW_FOOTSWING_HEIGHT_CMD,
    # freq = base * (0.5 * |vy| + 1); planted gait (freq=0) stays planted.
    "scale_gait_frequency_by_vy": True,
    "gait_frequency_vy_coef": 3.0,
    # Footswing: 20% of base at iter 0 → 100% by iter 2500 (num_steps_per_iter=24).
    "footswing_height_curriculum_start_scale": 1,
    "footswing_height_curriculum_end_it": 2500,
    "num_steps_per_iter": 24,
}

# Used whenever both vy and yaw commands are ~0 (stand / straight-roll).
_WTW_ZERO_VY_YAW_GAIT_TIMING_PARAMS = {
    "gait_frequency": 0.0,
    "gait_phase": 0.5,
    "gait_offset": 0.0,
    "gait_bound": 0.0,
    "gait_duration": 0.5,
    "kappa_gait_probs": 0.07,
    "footswing_height_cmd": 0.0,
}

# Keep WTW rewards active at 0 commands; swap gait params instead of zeroing terms.
_WTW_VY_YAW_GATE_PARAMS = {
    "command_name": "base_velocity",
    "zero_when_vy_yaw_zero": False,
}

_WTW_GAIT_SWITCH_PARAMS = {
    "switch_gait_when_vy_yaw_zero": True,
    **{f"zero_vy_yaw_{key}": value for key, value in _WTW_ZERO_VY_YAW_GAIT_TIMING_PARAMS.items()},
}

# 2x reward magnitude for 0.75s after ``base_velocity`` resample.
_POST_RESAMPLE_BOOST_PARAMS = {
    "command_name": "base_velocity",
    "boost_duration_s": 0.75,
    "boost_scale": 2.0,
}


@configclass
class Go2WFlatSceneCfg(Go2WSceneCfg):
    """Go2W scene on a single infinite plane (no procedural terrain mesh)."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        class_type=TerrainImporter,
        terrain_type="plane",
        terrain_generator=None,
        env_spacing=0.5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.5, 0.5, 0.5),
        ),
        debug_vis=False,
    )


def disable_stand_still_terrain_gate(rewards) -> None:
    """Trigger ``stand_still_scale`` from command only (no procedural ``flat`` column)."""
    for term_name in STAND_STILL_JOINT_PENALTY_TERMS:
        term = getattr(rewards, term_name, None)
        if term is not None:
            term.params["require_flat_terrain"] = False


@configclass
class WalkTheseWaysSymmetryRewardsCfg(RewardsCfg):
    """Base Go2W rewards plus WTW augmented auxiliary terms."""

    # Plane terrain has no generator columns; bake the flag into static params so Hydra
    # ``to_dict`` / ``from_dict`` cannot fall back to the function default (True).
    hip_pos_penalty_l1 = RewTerm(
        func=mdp.joint_pos_penalty_l1,
        weight=-0.04,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint"),
            "stand_still_scale": 10.0,
            "stand_cmd_idxs": [0, 1],
            "require_flat_terrain": False,
        },
    )
    joint_pos_penalty_l1 = RewTerm(
        func=mdp.joint_pos_penalty_l1,
        weight=-0.008,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_(thigh|calf)_joint"),
            "stand_still_scale": 10.0,
            "stand_cmd_idxs": [0, 1],
            "require_flat_terrain": False,
        },
    )
    hip_pos_penalty_l1_leg_var = RewTerm(
        func=mdp.joint_pos_penalty_l1_leg_var,
        weight=-0.2,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint"),
            "stand_still_scale": 10.0,
            "stand_cmd_idxs": [0, 1],
            "require_flat_terrain": False,
            "window_s": 1.0,
        },
    )
    joint_pos_penalty_l1_leg_var = RewTerm(
        func=mdp.joint_pos_penalty_l1_leg_var,
        weight=-0.2,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_(thigh|calf)_joint"),
            "stand_still_scale": 10.0,
            "stand_cmd_idxs": [0, 1],
            "require_flat_terrain": False,
            "window_s": 1.0,
        },
    )

    # Same base weights as ``RewardsCfg``; 2x for 0.75s after command resample.
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2_post_resample_boost,
        weight=-1.0,
        params={**_POST_RESAMPLE_BOOST_PARAMS},
    )
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2_post_resample_boost,
        weight=-0.025,
        params={**_POST_RESAMPLE_BOOST_PARAMS},
    )
    lin_acc_z_l2 = RewTerm(
        func=mdp.lin_acc_z_l2_post_resample_boost,
        weight=-0.001,
        params={**_POST_RESAMPLE_BOOST_PARAMS},
    )
    ang_acc_xy_l2 = RewTerm(
        func=mdp.ang_acc_xy_l2_post_resample_boost,
        weight=-0.0000025,
        params={**_POST_RESAMPLE_BOOST_PARAMS},
    )

    wtw_jump = RewTerm(
        func=mdp.wtw_jump_post_resample_boost,
        weight=WTW_JUMP_WEIGHT,
        params={
            "base_height_target": BASE_HEIGHT_TARGET,
            **_WTW_VY_YAW_GATE_PARAMS,
            **_POST_RESAMPLE_BOOST_PARAMS,
        },
    )
    wtw_orientation_control = RewTerm(
        func=mdp.wtw_orientation_control,
        weight=WTW_ORIENTATION_CONTROL_WEIGHT,
        params={**_WTW_VY_YAW_GATE_PARAMS},
    )
    wtw_raibert_heuristic = RewTerm(
        func=mdp.wtw_raibert_heuristic,
        weight=WTW_RAIBERT_HEURISTIC_WEIGHT,
        params={
            "stance_width_cmd": 0.284,
            "stance_length_cmd": 0.387,
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINK_NAME),
            **_WTW_GAIT_TIMING_PARAMS,
            **_WTW_VY_YAW_GATE_PARAMS,
            **_WTW_GAIT_SWITCH_PARAMS,
        },
    )
    wtw_raibert_heuristic_imbalance_x = RewTerm(
        func=mdp.wtw_raibert_heuristic_imbalance,
        weight=WTW_RAIBERT_HEURISTIC_IMBALANCE_X_WEIGHT,
        params={
            "stance_width_cmd": 0.284,
            "stance_length_cmd": 0.387,
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINK_NAME),
            "axis": "x",
            **_WTW_GAIT_TIMING_PARAMS,
            **_WTW_VY_YAW_GATE_PARAMS,
            **_WTW_GAIT_SWITCH_PARAMS,
        },
    )
    wtw_raibert_heuristic_imbalance_y = RewTerm(
        func=mdp.wtw_raibert_heuristic_imbalance,
        weight=WTW_RAIBERT_HEURISTIC_IMBALANCE_Y_WEIGHT,
        params={
            "stance_width_cmd": 0.284,
            "stance_length_cmd": 0.387,
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINK_NAME),
            "axis": "y",
            **_WTW_GAIT_TIMING_PARAMS,
            **_WTW_VY_YAW_GATE_PARAMS,
            **_WTW_GAIT_SWITCH_PARAMS,
        },
    )
    wtw_feet_clearance_cmd_linear = RewTerm(
        func=mdp.wtw_feet_clearance_cmd_linear,
        weight=WTW_FEET_CLEARANCE_CMD_LINEAR_WEIGHT,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINK_NAME),
            "foot_radius": WHEEL_RADIUS,
            **_WTW_GAIT_TIMING_PARAMS,
            **_WTW_VY_YAW_GATE_PARAMS,
            **_WTW_GAIT_SWITCH_PARAMS,
        },
    )
    wtw_feet_clearance_cmd_linear_imbalance = RewTerm(
        func=mdp.wtw_feet_clearance_cmd_linear_imbalance,
        weight=WTW_FEET_CLEARANCE_CMD_LINEAR_IMBALANCE_WEIGHT,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINK_NAME),
            "foot_radius": WHEEL_RADIUS,
            **_WTW_GAIT_TIMING_PARAMS,
            **_WTW_VY_YAW_GATE_PARAMS,
            **_WTW_GAIT_SWITCH_PARAMS,
        },
    )
    wtw_tracking_contacts_shaped_force = RewTerm(
        func=mdp.wtw_tracking_contacts_shaped_force,
        weight=WTW_TRACKING_CONTACTS_SHAPED_FORCE_WEIGHT,
        params={
            "gait_force_sigma": 100.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK_NAME),
            **_WTW_GAIT_TIMING_PARAMS,
            **_WTW_VY_YAW_GATE_PARAMS,
            **_WTW_GAIT_SWITCH_PARAMS,
        },
    )
    wtw_tracking_contacts_shaped_vel = RewTerm(
        func=mdp.wtw_tracking_contacts_shaped_vel,
        weight=WTW_TRACKING_CONTACTS_SHAPED_VEL_WEIGHT,
        params={
            "gait_vel_sigma": 10.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINK_NAME),
            **_WTW_GAIT_TIMING_PARAMS,
            **_WTW_VY_YAW_GATE_PARAMS,
            **_WTW_GAIT_SWITCH_PARAMS,
        },
    )

    def __post_init__(self):
        super().__post_init__()
        disable_stand_still_terrain_gate(self)


@configclass
class Go2WSymmetryFlatWtwEnvCfg(Go2WEnvSymmetryCfg):
    """Blind Go2W symmetry training: flat plane + WTW augmented auxiliary rewards."""

    scene: Go2WFlatSceneCfg = Go2WFlatSceneCfg(num_envs=4096, env_spacing=2.5)
    rewards: WalkTheseWaysSymmetryRewardsCfg = WalkTheseWaysSymmetryRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Task-only fixed command magnitudes (no curriculum expansion).
        cmd = self.commands.base_velocity
        cmd.ranges.lin_vel_x = (-1.5, 1.5)
        cmd.ranges.lin_vel_y = (-0.75, 0.75)
        cmd.ranges.ang_vel_yaw = (-1.5, 1.5)
        cmd.command_range_max.lin_vel_x = (-1.5, 1.5)
        cmd.command_range_max.lin_vel_y = (-0.75, 0.75)
        cmd.command_range_max.ang_vel_yaw = (-1.5, 1.5)

        # No terrain mesh / curriculum on a plane.
        self.curriculum.terrain_levels = None
        if getattr(self.curriculum, "step_height_range", None) is not None:
            self.curriculum.step_height_range = None

        # Terrain-specific shaping is meaningless on flat ground.
        self.rewards.terrain_level_progress = None
        self.rewards.dont_wait = None
        self.rewards.wheels_not_in_contact = None
        self.rewards.local_terrain_tilt_angle = None
        self.rewards.base_tilt_angle.weight = -0.5
        self.rewards.base_height_l2.weight = 0.0
        self.rewards.wheel_lateral_drag.weight = 0.0
        self.rewards.feet_regulation.weight = -0.05
        self.rewards.action_rate_l2.weight = -0.05
        self.rewards.action_smoothness_l2.weight = -0.05
        if getattr(self.curriculum, "terrain_level_progress", None) is not None:
            self.curriculum.terrain_level_progress = None
        if getattr(self.curriculum, "local_terrain_tilt_angle", None) is not None:
            self.curriculum.local_terrain_tilt_angle = None
        if getattr(self.curriculum, "wheels_not_in_contact", None) is not None:
            self.curriculum.wheels_not_in_contact = None
        if getattr(self.curriculum, "wheel_lateral_drag", None) is not None:
            self.curriculum.wheel_lateral_drag = None
        if getattr(self.curriculum, "feet_regulation", None) is not None:
            self.curriculum.feet_regulation = None
        if getattr(self.curriculum, "base_height_l2", None) is not None:
            self.curriculum.base_height_l2 = None

        self.apply_stand_still_scale_without_terrain()

    def apply_stand_still_scale_without_terrain(self) -> None:
        """Re-apply after Hydra ``from_dict`` (same pattern as MGDP obs-group flags)."""
        disable_stand_still_terrain_gate(self.rewards)


@configclass
class Go2WSparseMoeCtsActorCriticCfg(RslRlMoeCtsActorCriticCfg):
    """Student MoE with top-2 sparse gating (30 other experts inactive per forward)."""

    expert_num = 12
    gating_top_k = 3


@configclass
class Go2WSymmetryFlatWtwRunnerCfg(MoECTSSymmetryRunnerCfg):
    """MoE-CTS runner for flat-plane blind Go2W with sparse gating."""

    experiment_name = "go2w_moe_cts_symmetry_flat_wtw"
    policy = Go2WSparseMoeCtsActorCriticCfg()

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.symmetry_cfg = Go2WMoeCtsSymmetryCfg()
        self.algorithm.symmetry_cfg.use_symmetric_augmentation = True
