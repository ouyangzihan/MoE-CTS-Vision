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
    Go2WEnvSymmetryCfg,
    Go2WSceneCfg,
    RewardsCfg,
)
from robot_lab.tasks.go2w.rsl_rl_cfg import Go2WMoeCtsSymmetryCfg

# WTW ``scripts/train.py`` scales for the six augmented auxiliary rewards (Table 1).
WTW_JUMP_WEIGHT = 10.0
WTW_ORIENTATION_CONTROL_WEIGHT = -5.0
WTW_RAIBERT_HEURISTIC_WEIGHT = -10.0
WTW_FEET_CLEARANCE_CMD_LINEAR_WEIGHT = -30.0
WTW_TRACKING_CONTACTS_SHAPED_FORCE_WEIGHT = 4.0
WTW_TRACKING_CONTACTS_SHAPED_VEL_WEIGHT = 4.0

_WTW_GAIT_TIMING_PARAMS = {
    "gait_frequency": 1.5,
    "gait_phase": 0.5,
    "gait_offset": 0.0,
    "gait_bound": 0.0,
    "gait_duration": 0.5,
    "kappa_gait_probs": 0.07,
}

# Disable WTW gait shaping on straight-line commands (vy=0 and yaw=0).
_WTW_VY_YAW_GATE_PARAMS = {
    "command_name": "base_velocity",
    "zero_when_vy_yaw_zero": True,
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


@configclass
class WalkTheseWaysSymmetryRewardsCfg(RewardsCfg):
    """Base Go2W rewards plus WTW augmented auxiliary terms."""

    wtw_jump = RewTerm(
        func=mdp.wtw_jump,
        weight=WTW_JUMP_WEIGHT,
        params={
            "base_height_target": BASE_HEIGHT_TARGET,
            **_WTW_VY_YAW_GATE_PARAMS,
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
        },
    )
    wtw_feet_clearance_cmd_linear = RewTerm(
        func=mdp.wtw_feet_clearance_cmd_linear,
        weight=WTW_FEET_CLEARANCE_CMD_LINEAR_WEIGHT,
        params={
            "footswing_height_cmd": 0.07,
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINK_NAME),
            **_WTW_GAIT_TIMING_PARAMS,
            **_WTW_VY_YAW_GATE_PARAMS,
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
        },
    )


@configclass
class Go2WSymmetryFlatWtwEnvCfg(Go2WEnvSymmetryCfg):
    """Blind Go2W symmetry training: flat plane + WTW augmented auxiliary rewards."""

    scene: Go2WFlatSceneCfg = Go2WFlatSceneCfg(num_envs=8192, env_spacing=0.5)
    rewards: WalkTheseWaysSymmetryRewardsCfg = WalkTheseWaysSymmetryRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # No terrain mesh / curriculum on a plane.
        self.curriculum.terrain_levels = None
        if getattr(self.curriculum, "step_height_range", None) is not None:
            self.curriculum.step_height_range = None

        # Terrain-specific shaping is meaningless on flat ground.
        self.rewards.terrain_level_progress = None
        self.rewards.dont_wait = None
        self.rewards.wheels_not_in_contact = None
        self.rewards.local_terrain_tilt_angle = None
        self.rewards.base_tilt_angle.weight = -0.2
        if getattr(self.curriculum, "terrain_level_progress", None) is not None:
            self.curriculum.terrain_level_progress = None
        if getattr(self.curriculum, "local_terrain_tilt_angle", None) is not None:
            self.curriculum.local_terrain_tilt_angle = None
        if getattr(self.curriculum, "wheels_not_in_contact", None) is not None:
            self.curriculum.wheels_not_in_contact = None

        # Flat plane has no procedural terrain columns; gate stand-still scale on command only.
        for term_name in ("hip_pos_penalty_l1", "joint_pos_penalty_l1"):
            term = getattr(self.rewards, term_name, None)
            if term is not None:
                term.params["require_flat_terrain"] = False


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
