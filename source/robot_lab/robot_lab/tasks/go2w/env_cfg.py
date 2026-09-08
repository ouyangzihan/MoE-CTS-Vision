import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import robot_lab.tasks.go2w.mdp as mdp
from robot_lab.assets.unitree import GO2W_CFG_UNITREE, GO2W_CFG_UNITREE_D435I
from robot_lab.sensors import DelayRayCasterCameraCfg, process_depth_image
from robot_lab.tasks.go2.mdp.terrains import TERRAIN_CFG, Go2TerrainImporter

###
# Constants
###

LEG_JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

WHEEL_JOINT_NAMES = [
    "FL_foot_joint", "FR_foot_joint", "RL_foot_joint", "RR_foot_joint",
]

ALL_JOINT_NAMES = LEG_JOINT_NAMES + WHEEL_JOINT_NAMES

BASE_LINK_NAME = "base"
FOOT_LINK_NAME = ".*_foot"
BASE_HEIGHT_TARGET = 0.408
WHEEL_RADIUS = 0.086

# --- Camera ---
D435I_DEPTH_WIDTH = 60
D435I_DEPTH_HEIGHT = 60
D435I_INTRINSIC_WIDTH = 106
D435I_INTRINSIC_HEIGHT = 60
D435I_HORIZONTAL_FOV_DEG = 87.0
D435I_VERTICAL_FOV_DEG = 58.0
D435I_FOV_RANDOMIZATION_DEG = 3.0
D435I_PRINCIPAL_POINT_RANDOMIZATION_PX = 1.0
D435I_DEPTH_MAX = 2.5
D435I_DEPTH_IMAGE_SHAPE = (D435I_DEPTH_HEIGHT, D435I_DEPTH_WIDTH)
D435I_GAUSSIAN_BLUR_SIGMA = 1.0
D435I_GAUSSIAN_BLUR_KERNEL_SIZE = 3
D435I_DEPTH_NUM_OUTPUT_FRAMES = 4
D435I_DEPTH_HISTORY_SKIP_FRAMES = 5
D435I_DEPTH_HISTORY_LENGTH = (D435I_DEPTH_NUM_OUTPUT_FRAMES - 1) * D435I_DEPTH_HISTORY_SKIP_FRAMES + 1
D435I_CAMERA_UPDATE_HZ = 50.0
D435I_CAMERA_UPDATE_PERIOD = 1.0 / D435I_CAMERA_UPDATE_HZ
# Depth age in sensor frames @ 50 Hz (1 frame = 20 ms) → 20–60 ms.
D435I_CAMERA_MIN_DELAY_FRAMES = 2
D435I_CAMERA_MAX_DELAY_FRAMES = 6
D435I_CAMERA_MIN_DELAY = D435I_CAMERA_MIN_DELAY_FRAMES * D435I_CAMERA_UPDATE_PERIOD
D435I_CAMERA_MAX_DELAY = D435I_CAMERA_MAX_DELAY_FRAMES * D435I_CAMERA_UPDATE_PERIOD
D435I_CAMERA_POS_BASE = (0.3264636, -0.00003, 0.0947706)
D435I_CAMERA_POS_RANDOMIZATION_M = 0.01
D435I_CAMERA_ROT_BASE = (0.9612616959383189, 0.0, 0.27563735581699916, 0.0)
D435I_CAMERA_RPY_RANDOMIZATION_DEG = 3.0

LEG_JOINT_SCENE_CFG = SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)
WHEEL_JOINT_SCENE_CFG = SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES, preserve_order=True)
ALL_JOINT_SCENE_CFG = SceneEntityCfg("robot", joint_names=ALL_JOINT_NAMES, preserve_order=True)

##
# Scene definition
##


@configclass
class Go2WSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with the Go2W robot."""

    terrain = TerrainImporterCfg(
        class_type=Go2TerrainImporter,
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TERRAIN_CFG,
        max_init_terrain_level=5,
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

    robot: ArticulationCfg = GO2W_CFG_UNITREE.replace(prim_path="{ENV_REGEX_NS}/Robot")

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    height_scanner_small = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.4, 0.3]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    local_terrain_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.8, 0.55]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    wheel_contact_points = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot",
        track_contact_points=True,
        filter_prim_paths_expr=["/World/ground/terrain"],
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
        ),
    )


@configclass
class Go2WD435iSceneCfg(Go2WSceneCfg):
    """Go2W scene variant with D435i camera meshes and a front depth camera sensor."""

    robot: ArticulationCfg = GO2W_CFG_UNITREE_D435I.replace(prim_path="{ENV_REGEX_NS}/Robot")

    front_depth_camera = DelayRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=D435I_CAMERA_UPDATE_PERIOD,
        offset=DelayRayCasterCameraCfg.OffsetCfg(
            pos=D435I_CAMERA_POS_BASE,
            rot=D435I_CAMERA_ROT_BASE,
            convention="world",
        ),
        mesh_prim_paths=["/World/ground"],
        data_types=["distance_to_image_plane"],
        max_distance=D435I_DEPTH_MAX,
        min_delay=D435I_CAMERA_MIN_DELAY,
        max_delay=D435I_CAMERA_MAX_DELAY,
        horizontal_fov_range=(
            D435I_HORIZONTAL_FOV_DEG - D435I_FOV_RANDOMIZATION_DEG,
            D435I_HORIZONTAL_FOV_DEG + D435I_FOV_RANDOMIZATION_DEG,
        ),
        vertical_fov_range=(
            D435I_VERTICAL_FOV_DEG - D435I_FOV_RANDOMIZATION_DEG,
            D435I_VERTICAL_FOV_DEG + D435I_FOV_RANDOMIZATION_DEG,
        ),
        intrinsic_width=D435I_INTRINSIC_WIDTH,
        intrinsic_height=D435I_INTRINSIC_HEIGHT,
        principal_point_jitter=D435I_PRINCIPAL_POINT_RANDOMIZATION_PX,
        randomize_intrinsics_on_reset=False,
        pos_randomization_range=(-D435I_CAMERA_POS_RANDOMIZATION_M, D435I_CAMERA_POS_RANDOMIZATION_M),
        randomize_pos_on_reset=True,
        rpy_randomization_deg=D435I_CAMERA_RPY_RANDOMIZATION_DEG,
        randomize_rot_on_reset=True,
        depth_clipping_behavior="max",
        depth_norm_max=D435I_DEPTH_MAX,
        depth_normalize=True,
        gaussian_blur_sigma=D435I_GAUSSIAN_BLUR_SIGMA,
        gaussian_blur_kernel_size=D435I_GAUSSIAN_BLUR_KERNEL_SIZE,
        enable_sensor_noise=True,
        use_env_cfg_noise_overrides=False,
        sensor_noise_std= 0, # 0.02,
        sensor_dropout_prob= 0, # 0.2,
        sensor_depth_dependent_noise_scale= 0, # 0.5,
        sensor_edge_speckle_prob= 0, # 0.04,
        sensor_temporal_flicker_std= 0, # 0.015,
        sensor_hole_blob_prob= 0, # 0.075,
        sensor_hole_blob_size_range=(3, 12),
        sensor_randomize_dropout_fill_value=True,
        sensor_dropout_fill_value=None,
        depth_history_length=D435I_DEPTH_HISTORY_LENGTH,
        depth_num_output_frames=D435I_DEPTH_NUM_OUTPUT_FRAMES,
        depth_history_skip_frames=D435I_DEPTH_HISTORY_SKIP_FRAMES,
        pattern_cfg=patterns.PinholeCameraPatternCfg.from_intrinsic_matrix(
            intrinsic_matrix=[
                D435I_INTRINSIC_WIDTH / (2.0 * math.tan(math.radians(D435I_HORIZONTAL_FOV_DEG) / 2.0)),
                0.0,
                D435I_DEPTH_WIDTH * 0.5,
                0.0,
                D435I_INTRINSIC_HEIGHT / (2.0 * math.tan(math.radians(D435I_VERTICAL_FOV_DEG) / 2.0)),
                D435I_DEPTH_HEIGHT * 0.5,
                0.0,
                0.0,
                1.0,
            ],
            width=D435I_DEPTH_WIDTH,
            height=D435I_DEPTH_HEIGHT,
            focal_length=1.0,
        ),
        debug_vis=False,
    )


##
# MDP settings
##


def make_pose_velocity_command_cfg() -> mdp.PoseVelocityCommandCfg:
    """Hiking-style edge-target PoseVelocityCommand (flat-patch boundary goals)."""
    return mdp.PoseVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        velocity_control_stiffness=1.0,
        heading_control_stiffness=1.0,
        only_positive_lin_vel_x=True,
        # 10% stand, 20% reverse-into-target (neg vx + heading+π), 70% forward.
        rel_standing_envs=0.1,
        rel_reverse_envs=0.45,
        reverse_lin_vel_x_abs_max=1.0,  # reverse clamp [-1, 0]; yaw unchanged
        ranges=mdp.PoseVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
        ),
        command_range_max=mdp.PoseVelocityCommandCfg.CommandRangeMaxCfg(
            lin_vel_x=(0.0, 1.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.5, 1.5),
        ),
        command_range_expand_interval=1000,
        command_range_expand=mdp.PoseVelocityCommandCfg.CommandRangeExpandCfg(
            lin_vel_x=0.1,
            lin_vel_y=0.0,
            ang_vel_z=0.1,
        ),
        # Per-terrain caps (positive vx, zero vy). Intersected with curriculum ranges.
        velocity_ranges={
            "wave": {"lin_vel_x": (0.0, 1.5), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.5, 1.5)},
            "slope_up": {"lin_vel_x": (0.0, 1.5), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.5, 1.5)},
            "slope_down": {"lin_vel_x": (0.0, 1.5), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.5, 1.5)},
            "rough_slope": {"lin_vel_x": (0.0, 1.5), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.5, 1.5)},
            "stairs_up": {"lin_vel_x": (0.0, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.5, 1.5)},
            "stairs_down": {"lin_vel_x": (0.0, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.5, 1.5)},
            "obstacles": {"lin_vel_x": (0.0, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.5, 1.5)},
            "stepping_stones": {"lin_vel_x": (0.0, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.5, 1.5)},
            "gap": {"lin_vel_x": (0.0, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.5, 1.5)},
            "flat": {"lin_vel_x": (0.0, 2.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-2.0, 2.0)},
        },
        debug_vis=False,
    )


def make_legacy_go2rl_gym_command_cfg() -> mdp.Go2RLGymCommandCfg:
    """Build fixed-range random-velocity Go2RLGym command config.

    Each of ``vx``, ``vy``, ``yaw`` is sampled independently: 60% zero, 10% range
    max, 10% range min, 20% uniform in range. Joint all-zero / limit-vel overrides
    are disabled.
    """
    fixed = (-1.0, 1.0)
    return mdp.Go2RLGymCommandCfg(
        resampling_time=5.0,
        resampling_time_range=(5.0, 5.0),
        dynamic_resample_commands=False,
        independent_axis_mixture=True,
        axis_zero_prob=0.6,
        axis_max_prob=0.1,
        axis_min_prob=0.1,
        zero_command_curriculum=None,
        limit_vel_prob=0.0,
        limit_ang_vel_at_zero_command_prob=0.0,
        ranges=mdp.Go2RLGymCommandCfg.Ranges(
            lin_vel_x=fixed,
            lin_vel_y=fixed,
            ang_vel_yaw=fixed,
        ),
        command_range_max=mdp.Go2RLGymCommandCfg.CommandRangeMaxCfg(
            lin_vel_x=fixed,
            lin_vel_y=fixed,
            ang_vel_yaw=fixed,
        ),
        command_range_expand_interval=None,
        command_range_curriculum=[],
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP.

    Default: Go2RLGym random ``(vx, vy, yaw)`` velocity commands.
    Enable Hiking-style edge-target PoseVelocityCommand via ``use_pose_velocity_command``.
    """

    base_velocity = make_legacy_go2rl_gym_command_cfg()


def disable_command_range_curriculum(command_cfg) -> None:
    """Fix command ranges at their maximum targets and disable iteration-based expansion."""
    for range_name in ("lin_vel_x", "lin_vel_y", "ang_vel_z", "ang_vel_yaw"):
        if hasattr(command_cfg.ranges, range_name) and hasattr(command_cfg.command_range_max, range_name):
            setattr(command_cfg.ranges, range_name, getattr(command_cfg.command_range_max, range_name))
    command_cfg.command_range_expand_interval = None
    if hasattr(command_cfg, "command_range_curriculum"):
        command_cfg.command_range_curriculum = []


def _set_stand_cmd_idxs(env_cfg: ManagerBasedRLEnvCfg, *, pose_velocity: bool) -> None:
    """Match stand-still penalty indices to the active command layout."""
    idxs = [0, 1] if pose_velocity else [1, 2]
    rewards = getattr(env_cfg, "rewards", None)
    if rewards is None:
        return
    for term_name in ("hip_pos_penalty_l1", "joint_pos_penalty_l1"):
        term = getattr(rewards, term_name, None)
        if term is not None and "stand_cmd_idxs" in term.params:
            term.params["stand_cmd_idxs"] = idxs


def _set_velocity_command_obs(env_cfg: ManagerBasedRLEnvCfg, *, pose_velocity: bool) -> None:
    """Use 3-dim ``(vx, vy, yaw)`` obs for Go2RLGym; 2-dim ``(vx, yaw)`` for PoseVelocity."""
    use_abs_yaw = pose_velocity or bool(getattr(env_cfg, "use_yaw_joint_symmetry", False))
    obs_func = mdp.generated_commands_abs_yaw if use_abs_yaw else mdp.generated_commands
    observations = getattr(env_cfg, "observations", None)
    if observations is None:
        return
    for group_name in ("policy", "critic", "single_obs"):
        group = getattr(observations, group_name, None)
        if group is not None and hasattr(group, "velocity_commands"):
            group.velocity_commands.func = obs_func


def _uses_plane_terrain(env_cfg: ManagerBasedRLEnvCfg) -> bool:
    """True when the scene has no procedural terrain generator (flat plane / USD)."""
    terrain = getattr(getattr(env_cfg, "scene", None), "terrain", None)
    if terrain is None:
        return False
    if getattr(terrain, "terrain_type", None) == "plane":
        return True
    return getattr(terrain, "terrain_generator", None) is None


def apply_legacy_velocity_command(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Use Go2RLGym random velocity commands and restore gym terrain curriculum."""
    env_cfg.commands.base_velocity = make_legacy_go2rl_gym_command_cfg()
    disable_command_range_curriculum(env_cfg.commands.base_velocity)
    if hasattr(env_cfg, "curriculum"):
        if _uses_plane_terrain(env_cfg):
            env_cfg.curriculum.terrain_levels = None
        else:
            env_cfg.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel_gym)
    _set_stand_cmd_idxs(env_cfg, pose_velocity=False)
    _set_velocity_command_obs(env_cfg, pose_velocity=False)


def apply_pose_velocity_command(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Enable Hiking-style edge-target PoseVelocityCommand (flat-patch goals)."""
    env_cfg.commands.base_velocity = make_pose_velocity_command_cfg()
    if not getattr(env_cfg, "use_reward_weight_curriculum", True):
        disable_command_range_curriculum(env_cfg.commands.base_velocity)
    if hasattr(env_cfg, "curriculum"):
        if _uses_plane_terrain(env_cfg):
            env_cfg.curriculum.terrain_levels = None
        else:
            env_cfg.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    _set_stand_cmd_idxs(env_cfg, pose_velocity=True)
    _set_velocity_command_obs(env_cfg, pose_velocity=True)


def configure_command_delivery(env_cfg: ManagerBasedRLEnvCfg, *, use_pose_velocity: bool) -> None:
    """Select random Go2RLGym velocity commands or edge-target PoseVelocityCommand."""
    if use_pose_velocity:
        apply_pose_velocity_command(env_cfg)
    else:
        apply_legacy_velocity_command(env_cfg)


def resolve_use_pose_velocity_command(env_cfg: ManagerBasedRLEnvCfg, args_cli) -> bool:
    """Resolve command mode from env cfg and optional CLI overrides."""
    use_pose = bool(getattr(env_cfg, "use_pose_velocity_command", False))
    if getattr(args_cli, "pose_velocity_command", False):
        use_pose = True
    if getattr(args_cli, "legacy_velocity_command", False):
        use_pose = False
    return use_pose


def enable_pose_velocity_target_vis(
    env_cfg: ManagerBasedRLEnvCfg, *, show_all_patches: bool = False
) -> bool:
    """Enable PoseVelocityCommand markers for the active edge target (red cylinder).

    Also draws goal/current velocity arrows. Returns True if PoseVelocityCommand is active.
    ``show_all_patches`` is ignored (targets are sampled on-the-fly, not precomputed).
    """
    del show_all_patches  # kept for API compatibility with earlier callers
    base_velocity = getattr(getattr(env_cfg, "commands", None), "base_velocity", None)
    if base_velocity is None or not hasattr(base_velocity, "patch_vis"):
        return False
    base_velocity.debug_vis = True
    base_velocity.patch_vis = False
    return True



@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        scale={".*_hip_joint": 0.25, "^(?!.*_hip_joint).*": 0.25},
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )
    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=WHEEL_JOINT_NAMES,
        scale=1.0,
        use_default_offset=False,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel_delayed,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity_delayed,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_yaw_sym_delayed,
            params={"asset_cfg": LEG_JOINT_SCENE_CFG},
            noise=Unoise(n_min=-0.03, n_max=0.03),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel_yaw_sym_delayed,
            params={"asset_cfg": ALL_JOINT_SCENE_CFG},
            noise=Unoise(n_min=-2.0, n_max=2.0),
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(
            # Stored in canonical (positive-yaw) space by ActionManagerGo2W.
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.history_length = 10
            self.enable_corruption = True
            self.concatenate_terms = True
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            scale=2.0,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel_delayed,
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity_delayed,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_yaw_sym_delayed,
            params={"asset_cfg": LEG_JOINT_SCENE_CFG},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel_yaw_sym_delayed,
            params={"asset_cfg": ALL_JOINT_SCENE_CFG},
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_acc = ObsTerm(
            func=mdp.joint_acc_yaw_sym,
            params={"asset_cfg": LEG_JOINT_SCENE_CFG},
            clip=(-100.0, 100.0),
            scale=1e-4,
        )
        joint_torque = ObsTerm(
            func=mdp.joint_effort_yaw_sym,
            params={"asset_cfg": LEG_JOINT_SCENE_CFG},
            clip=(-100.0, 100.0),
            scale=0.01,
        )
        contact_force = ObsTerm(
            func=mdp.foot_contact_force_norm,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK_NAME)},
            clip=(-100.0, 100.0),
            scale=1e-3,
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=2.5,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SingleObsCfg(PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.history_length = 1

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    single_obs: SingleObsCfg = SingleObsCfg()


@configclass
class D435iObservationsCfg(ObservationsCfg):
    @configclass
    class DepthCfg(ObsGroup):
        """Depth observation for the student/policy encoder."""

        depth_image = ObsTerm(
            func=process_depth_image,
            params={
                "sensor_cfg": SceneEntityCfg("front_depth_camera"),
                "data_type": "distance_to_image_plane",
                "image_shape": D435I_DEPTH_IMAGE_SHAPE,
                "max_depth": D435I_DEPTH_MAX,
                "normalize": False,
                "use_delay": True,
                "use_history_stack": True,
                "num_output_frames": D435I_DEPTH_NUM_OUTPUT_FRAMES,
            },
            clip=(0.0, 5.0),
            scale=0.5,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CleanDepthCfg(ObsGroup):
        """Undamped delayed depth target for denoising aux loss (not fed to actor)."""

        depth_image = ObsTerm(
            func=process_depth_image,
            params={
                "sensor_cfg": SceneEntityCfg("front_depth_camera"),
                "data_type": "distance_to_image_plane",
                "image_shape": D435I_DEPTH_IMAGE_SHAPE,
                "max_depth": D435I_DEPTH_MAX,
                "normalize": True,
                "use_delay": True,
                "use_history_stack": False,
            },
            clip=(0.0, 5.0),
            scale=0.5,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class HeightMapCfg(ObsGroup):
        """Privileged local elevation map (17x11) for height reconstruction / alignment."""

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=2.5,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    depth: DepthCfg = DepthCfg()
    # Populated only when Go2WD435iEnvCfg.use_mgdp_depth_aux is True.
    clean_depth: CleanDepthCfg | None = None
    height_map: HeightMapCfg | None = None


@configclass
class EventCfg:
    """Configuration for events."""

    randomize_rigid_body_mass_base = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    randomize_rigid_body_mass_others = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="^(?!.*base).*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    randomize_com_positions = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.03, 0.03)},
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.25),
            "damping_distribution_params": (0.8, 1.25),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    randomize_motor_zero_offset = EventTerm(
        func=mdp.randomize_action_joint_pos_offset,
        mode="reset",
        params={
            "action_term_name": "joint_pos",
            "offset_range": (-0.1, 0.1),
        },
    )
    randomize_push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "velocity_range": {
                "x": (-0.4, 0.4),
                "y": (-0.4, 0.4),
                "roll": (-0.6, 0.6),
                "pitch": (-0.6, 0.6),
                "yaw": (-0.6, 0.6),
            }
        },
    )
    randomize_rigid_body_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.5),
            "dynamic_friction_range": (0.2, 1.5),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.2), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp_post_resample_boost,
        # CurriculumCfg holds weight at 6.0 (no anneal); 2x std for 0.75s after resample.
        weight=6.0,
        params={
            "command_name": "base_velocity",
            "std": 0.707106781,
            "boost_duration_s": 0.75,
            "boost_scale": 2.0,
        },
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp_post_resample_boost,
        # CurriculumCfg holds weight at 3.0 (no anneal); 2x std for 0.75s after resample.
        weight=3.0,
        params={
            "command_name": "base_velocity",
            "std": 0.707106781,
            "boost_duration_s": 0.75,
            "boost_scale": 2.0,
        },
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)#-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.025)#-0.05)
    base_tilt_angle = RewTerm(func=mdp.base_tilt_angle, weight=0.0)
    local_terrain_tilt_angle = RewTerm(
        func=mdp.local_terrain_tilt_angle,
        weight=-0.2,
        params={
            "height_sensor_cfg": SceneEntityCfg("local_terrain_scanner"),
            "contact_sensor_cfg": SceneEntityCfg(
                "wheel_contact_points", body_names=FOOT_LINK_NAME
            ),
            # 0.8 x 0.55 m at 0.1 m resolution gives 54 rays; four
            # contacts therefore have the same total weight as all rays.
            "contact_point_weight": 20.,
            # Side-to-side (roll) tilt penalized 2x front-to-back (pitch).
            "lateral_scale": 2.0,
        },
    )
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-7,
        params={"asset_cfg": LEG_JOINT_SCENE_CFG},
    )
    joint_power = RewTerm(
        func=mdp.joint_power,
        weight=-2e-5,
        params={"asset_cfg": LEG_JOINT_SCENE_CFG},
    )
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1e-4,
        params={"asset_cfg": LEG_JOINT_SCENE_CFG},
    )
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "target_height": BASE_HEIGHT_TARGET,
            "sensor_cfg": SceneEntityCfg("height_scanner_small"),
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_smoothness_l2 = RewTerm(func=mdp.action_smoothness_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-5.0, #-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=".*_hip|.*_thigh|.*calf|Head_.*|.*_foot_motor|camera_base",
            ),
            "threshold": 5.0,
        },
    )
    wheels_not_in_contact = RewTerm(
        func=mdp.wheels_not_in_contact,
        weight=-.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK_NAME), "threshold": 1.0},
    )
    # Penalize sideways (body-y) slip at contacting wheels; curriculum ramps to -0.08.
    wheel_lateral_drag = RewTerm(
        func=mdp.wheel_lateral_drag,
        weight=-0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK_NAME),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINK_NAME),
            "threshold": 1.0,
        },
    )
    wheel_slip_ratio = RewTerm(
        func=mdp.wheel_slip_ratio,
        weight=-0.03,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK_NAME),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINK_NAME, preserve_order=True),
            "threshold": 1.0,
            "wheel_radius": 0.086,
            "contact_offset_body": (0.0, 0.0, -0.086),
            "terrain_static_friction": 1.0,
            "terrain_dynamic_friction": 1.0,
        },
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": LEG_JOINT_SCENE_CFG},
    )
    feet_regulation = RewTerm(
        func=mdp.feet_regulation,
        weight=-0.05,
        params={
            "base_height_target": BASE_HEIGHT_TARGET,
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINK_NAME),
            "sensor_cfg": SceneEntityCfg("height_scanner_small"),
            "wheel_radius": 0.086,
        },
    )
    hip_pos_penalty_l1 = RewTerm(
        func=mdp.joint_pos_penalty_l1,
        weight=-0.04,#-0.05,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint"),
            "stand_still_scale": 10.0,
            # PoseVelocity command is ``(vx, yaw)``.
            "stand_cmd_idxs": [0, 1],
            # Bake True for Hydra; Symmetry-v1 flat plane overrides to False.
            "require_flat_terrain": True,
        },
    )
    joint_pos_penalty_l1 = RewTerm(
        func=mdp.joint_pos_penalty_l1,
        weight=-0.008,#-0.01,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_(thigh|calf)_joint"),
            "stand_still_scale": 10.0,
            "stand_cmd_idxs": [0, 1],
            "require_flat_terrain": True,
        },
    )
    terrain_level_progress = RewTerm(func=mdp.terrain_level_progress, weight=6.)
    # Hiking-in-the-Wild: discourage freezing / backing up under forward commands.
    dont_wait = RewTerm(
        func=mdp.dont_wait,
        weight=-0.,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.3,
            "slow_threshold": 0.15,
            "reverse_threshold": -0.15,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=BASE_LINK_NAME),
            "threshold": 1.0,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP.

    Default uses ``terrain_levels_vel_gym`` with Go2RLGym random velocity commands.
    ``use_pose_velocity_command`` switches to ``terrain_levels_vel``.
    """

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel_gym)
    # Enabled in Go2WD435iEnvCfg when use_mgdp_depth_aux is True.
    depth_noise = None
    step_height_range = CurrTerm(
        func=mdp.step_height_range_curriculum,
        params={
            "start_it": 100000, # 10000,
            "interval": 5000, # 500,
            "delta": 0.01,
            "max_lower": 0.01,
            "initial_lower": 0.0,
            "upper": 0.2,
            "mesh_lower": 0.0,
            "sub_terrain_names": ("stairs_up", "stairs_down"),
            "num_steps_per_iter": 24,
        },
    )
    track_lin_vel_xy_exp = CurrTerm(
        mdp.gradual_reward_weight_modification,
        params={
            "term_name": "track_lin_vel_xy_exp",
            "initial_weight": 6.0,
            "final_weight": 6.0,
            "start_it": 0,
            "end_it": 1000,
        },
    )
    track_ang_vel_z_exp = CurrTerm(
        mdp.gradual_reward_weight_modification,
        params={
            "term_name": "track_ang_vel_z_exp",
            "initial_weight": 3.0,
            "final_weight": 3.0,
            "start_it": 0,
            "end_it": 1000,
        },
    )
    joint_pos_penalty_l1 = CurrTerm(
        mdp.gradual_reward_weight_modification,
        params={
            "term_name": "joint_pos_penalty_l1",
            "initial_weight": -0.008,
            "final_weight": -0.1,
            "start_it": 0,
            "end_it": 7500,
        },
    )
    hip_pos_penalty_l1 = CurrTerm(
        mdp.gradual_reward_weight_modification,
        params={
            "term_name": "hip_pos_penalty_l1",
            "initial_weight": -0.04,
            "final_weight": -0.5,
            "start_it": 0,
            "end_it": 7500,
        },
    )
    wheels_not_in_contact = CurrTerm(
        mdp.gradual_reward_weight_modification,
        params={
            "term_name": "wheels_not_in_contact",
            "initial_weight": -0.,
            "final_weight": -0.3, #-0.25
            "start_it": 0,
            "end_it": 5000,
        },
    )
    wheel_lateral_drag = CurrTerm(
        mdp.gradual_reward_weight_modification,
        params={
            "term_name": "wheel_lateral_drag",
            "initial_weight": -0.0,
            "final_weight": -0.08,
            "start_it": 0,
            "end_it": 5000,
        },
    )
    feet_regulation = CurrTerm(
        mdp.gradual_reward_weight_modification,
        params={
            "term_name": "feet_regulation",
            "initial_weight": -0.0,
            "final_weight": -0.0,
            "start_it": 0,
            "end_it": 5000,
        },
    )
    local_terrain_tilt_angle = CurrTerm(
        mdp.gradual_reward_weight_modification,
        params={
            "term_name": "local_terrain_tilt_angle",
            "initial_weight": -0.3,
            "final_weight": -0.6,
            "start_it": 0,
            "end_it": 5000,
        },
    )
    terrain_level_progress = CurrTerm(
        mdp.gradual_reward_weight_modification,
        params={
            "term_name": "terrain_level_progress",
            "initial_weight": 10.0,
            "final_weight": 5.0,
            "start_it": 0,
            "end_it": 5000,
        },
    )
    base_linear_velocity = CurrTerm(
        mdp.gradual_reward_weight_modification,
        params={"term_name": "lin_vel_z_l2", "initial_weight": -2.0, "final_weight": -0.0, "start_it": 0, "end_it": 1500},
    )
    base_height_l2 = CurrTerm(
        mdp.gradual_reward_weight_modification,
        params={"term_name": "base_height_l2", "initial_weight": -2.0, "final_weight": -40.0, "start_it": 0, "end_it": 5000},
    )


##
# Environment configuration
##


@configclass
class Go2WEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Go2W robot on rough terrain."""

    scene: Go2WSceneCfg = Go2WSceneCfg(num_envs=16384, env_spacing=0.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    # If disabled, reward weights and command ranges are fixed at their final
    # targets, and the step-height lower-bound ramp is off (stairs stay at the
    # baked (0.0, 0.2) range across all terrain rows). Terrain-level curriculum
    # is unchanged.
    use_reward_weight_curriculum: bool = True
    # When True: Hiking-style edge-target PoseVelocityCommand (flat-patch goals).
    # When False (default): Go2RLGym random (vx, vy, yaw) velocity commands.
    use_pose_velocity_command: bool = False
    # When True: network sees |yaw| and L/R-mirrored joints for yaw < 0;
    # policy actions are un-mirrored before apply. Command term stays signed.
    # Off: use MoE CTS offline L/R batch augmentation instead (see rsl_rl_cfg).
    use_yaw_joint_symmetry: bool = False
    # Proprio observation delay (IMU ang-vel / gravity, joint pos & vel).
    # Applied at physics rate (sim.dt), distinct from actuator cmd delay and
    # from D435i camera max_delay. Set max to 0 to disable.
    obs_delay_min_s: float = 0.0
    obs_delay_max_s: float = 0.005

    def __post_init__(self):
        """Post initialization."""
        configure_command_delivery(self, use_pose_velocity=self.use_pose_velocity_command)

        self.decimation = 4
        self.episode_length_s = 25.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = int(1 * 1024 * 1024)
        self.sim.physx.gpu_collision_stack_size = int(512 * 1024 * 1024)
        self.sim.physx.enable_external_forces_every_iteration = True

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.height_scanner_small is not None:
            self.scene.height_scanner_small.update_period = self.decimation * self.sim.dt
        if self.scene.local_terrain_scanner is not None:
            self.scene.local_terrain_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.wheel_contact_points is not None:
            self.scene.wheel_contact_points.update_period = self.sim.dt

        if not self.use_reward_weight_curriculum:
            disable_command_range_curriculum(self.commands.base_velocity)
            reward_weight_terms = (
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "joint_pos_penalty_l1",
                "hip_pos_penalty_l1",
                "wheels_not_in_contact",
                "wheel_lateral_drag",
                # "base_tilt_angle",
                "local_terrain_tilt_angle",
                "base_linear_velocity",
                "base_height_l2",
                "terrain_level_progress",
            )
            for curriculum_term_name in reward_weight_terms:
                curriculum_term = getattr(self.curriculum, curriculum_term_name)
                reward_term_name = curriculum_term.params["term_name"]
                getattr(self.rewards, reward_term_name).weight = curriculum_term.params["final_weight"]
                setattr(self.curriculum, curriculum_term_name, None)

            # Disable step-height lower-bound ramp; keep baked (0.0, 0.2) across all rows.
            if getattr(self.curriculum, "step_height_range", None) is not None:
                self.curriculum.step_height_range = None

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class Go2WD435iEnvCfg(Go2WEnvCfg):
    """Go2W environment variant with the D435i front depth camera enabled."""

    scene: Go2WD435iSceneCfg = Go2WD435iSceneCfg(num_envs=2048, env_spacing=0.5)
    observations: D435iObservationsCfg = D435iObservationsCfg()
    # Master switch for MGDP-style depth aux training (denoise / height recon /
    # geometry alignment / harsher noise curriculum). False = current pipeline.
    use_mgdp_depth_aux: bool = False
    # Runtime noise overrides written by depth_noise_curriculum when aux is on.
    depth_noise_std: float = 0.02
    depth_dropout_prob: float = 0.2
    depth_dependent_noise_scale: float = 0.0
    depth_edge_speckle_prob: float = 0.0
    depth_temporal_flicker_std: float = 0.0
    depth_hole_blob_prob: float = 0.0
    depth_hole_blob_size_range: tuple[int, int] = (3, 12)
    # Official MGDP dropout zeros holes; None keeps legacy far-fill when aux is off.
    depth_dropout_fill_value: float | None = None

    def apply_mgdp_depth_aux_settings(self) -> None:
        """Apply/re-apply MGDP aux obs groups from ``use_mgdp_depth_aux``.

        Hydra ``from_dict`` updates the flag after ``__post_init__`` without
        re-running it, so play/train scripts must call this after overrides.
        """
        depth_term = self.observations.depth.depth_image
        camera_cfg = self.scene.front_depth_camera
        if self.use_mgdp_depth_aux:
            self.observations.clean_depth = D435iObservationsCfg.CleanDepthCfg()
            self.observations.height_map = D435iObservationsCfg.HeightMapCfg()
            camera_cfg.use_env_cfg_noise_overrides = True
            camera_cfg.sensor_noise_std = self.depth_noise_std
            camera_cfg.sensor_dropout_prob = self.depth_dropout_prob
            camera_cfg.sensor_depth_dependent_noise_scale = self.depth_dependent_noise_scale
            camera_cfg.sensor_edge_speckle_prob = self.depth_edge_speckle_prob
            camera_cfg.sensor_temporal_flicker_std = self.depth_temporal_flicker_std
            camera_cfg.sensor_hole_blob_prob = self.depth_hole_blob_prob
            camera_cfg.sensor_hole_blob_size_range = self.depth_hole_blob_size_range
            if self.depth_dropout_fill_value is None:
                self.depth_dropout_fill_value = 0.0
            camera_cfg.sensor_randomize_dropout_fill_value = False
            camera_cfg.sensor_dropout_fill_value = self.depth_dropout_fill_value
            self.curriculum.depth_noise = CurrTerm(
                func=mdp.depth_noise_curriculum,
                params={
                    "start_it": 0,
                    "end_it": 10000,
                    "num_steps_per_iter": 24,
                    "noise_std_range": (0.02, 0.02), # (0.02, 0.06),
                    "dropout_prob_range": (0.2, 0.2),
                    "depth_dependent_noise_scale_range": (0.0, 0.0), # (0.0, 1.0),
                    "edge_speckle_prob_range": (0.0, 0.0), # (0.0, 0.08),
                    "temporal_flicker_std_range": (0.0, 0.0), # (0.0, 0.03),
                    "hole_blob_prob_range": (0.0, 0.0), # (0.0, 0.15),
                },
            )
        else:
            self.observations.clean_depth = None
            self.observations.height_map = None
            if camera_cfg is not None:
                camera_cfg.use_env_cfg_noise_overrides = False
                camera_cfg.sensor_randomize_dropout_fill_value = True
                camera_cfg.sensor_dropout_fill_value = None
            if getattr(self.curriculum, "depth_noise", None) is not None:
                self.curriculum.depth_noise = None
        del depth_term

    def __post_init__(self):
        super().__post_init__()
        if self.scene.front_depth_camera is not None:
            self.scene.front_depth_camera.update_period = D435I_CAMERA_UPDATE_PERIOD
        self.apply_mgdp_depth_aux_settings()


@configclass
class Go2WEnvSymmetryCfg(Go2WEnvCfg):
    """Environment configuration with symmetry augmentation for MoE CTS."""

    scene: Go2WSceneCfg = Go2WSceneCfg(num_envs=8192, env_spacing=0.5)
