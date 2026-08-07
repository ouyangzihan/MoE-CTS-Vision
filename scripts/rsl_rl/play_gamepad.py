# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Play a trained RSL-RL policy in Isaac Sim with a Logitech F710 gamepad.

Controls (F710 in X / Xbox mode):
    Left stick          : lin_vel_x, lin_vel_y  (scale via --lin_vel_x / --lin_vel_y)
    Right stick L/R     : ang_vel_z (yaw)       (scale via --ang_vel_z)
    Button X            : toggle camera follow (behind robot, facing forward)
    Button A            : in-place reset (same XY/yaw, upright, default joints/height) + policy recurrent state

On episode termination, the robot is also reset in-place (not teleported back to the spawn origin).

Required terrain selection (single block only, to save GPU memory):
    --terrain_type NAME --terrain_level N
    Play generates one 1x1 terrain patch of NAME at difficulty matching curriculum row N.
    Level N is 0-based (0 .. num_rows-1). Robot spawns at that patch's center.

Example:
    python scripts/rsl_rl/play_gamepad.py --task=RobotLab-Go2W-D435i-v0 --terrain_type stairs_up --terrain_level 5
    python scripts/rsl_rl/play_gamepad.py --task=RobotLab-Go2-v0 --terrain_type flat --terrain_level 0 \\
        --lin_vel_x 1.5 --lin_vel_y 0.8 --ang_vel_z 1.2
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

# Preload native extensions before isaacsim/Kit modules are imported to avoid
# Windows DLL loader conflicts when Isaac Lab/RSL-RL import them later.
import h5py  # noqa: F401
import tensordict  # noqa: F401

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
from utils import export_cts_policy_as_jit, export_cts_policy_as_onnx, export_cts_cnn_gru_policy_as_jit

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RSL-RL agent with a Logitech F710 gamepad.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=int(1e9), help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of environments to simulate (forced to 1 for gamepad teleop).",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--real-time",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Run in real-time (default: on). Use --no-real-time to disable.",
)
parser.add_argument(
    "--lin_vel_x",
    type=float,
    default=1.0,
    help="Max |lin_vel_x| command from left stick up/down (m/s). Default: 1.0.",
)
parser.add_argument(
    "--lin_vel_y",
    type=float,
    default=1.0,
    help="Max |lin_vel_y| command from left stick left/right (m/s). Default: 1.0.",
)
parser.add_argument(
    "--ang_vel_z",
    type=float,
    default=1.0,
    help="Max |ang_vel_z| (yaw) command from right stick left/right (rad/s). Default: 1.0.",
)
parser.add_argument(
    "--terrain_type",
    type=str,
    default=None,
    help="Required. Terrain type name to generate as a single 1x1 play patch "
    "(zero-proportion types allowed). Valid names are printed if missing/invalid.",
)
parser.add_argument(
    "--terrain_level",
    type=int,
    default=None,
    help="Required. Terrain difficulty level, 0-based curriculum row index. "
    "Valid range is printed if missing/invalid.",
)
parser.add_argument(
    "--legacy_velocity_command",
    action="store_true",
    default=False,
    help="Use the previous Go2RLGym random velocity command instead of flat-patch PoseVelocityCommand.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
# gamepad teleop always uses a single environment
if args_cli.num_envs is not None and args_cli.num_envs != 1:
    print(f"[WARN] play_gamepad forces num_envs=1 (got {args_cli.num_envs}).")
args_cli.num_envs = 1

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time

import carb
import carb.input
import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner, OnPolicyRunnerCTS

from isaaclab.assets import Articulation
from isaaclab.devices import Se2Gamepad, Se2GamepadCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedEnv,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_apply_yaw, yaw_quat
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import robot_lab.tasks  # noqa: F401

# Camera pose in robot yaw frame: behind (-x), elevated (+z), looking slightly ahead.
_CAMERA_EYE_B = (-3.0, 0.0, 1.2)
_CAMERA_LOOKAT_B = (0.5, 0.0, 0.3)

# Play-only terrain mesh resolution (meters). Larger = coarser mesh = less VRAM.
# Training defaults are typically horizontal_scale=0.1, vertical_scale=0.005.
_PLAY_TERRAIN_HORIZONTAL_SCALE = .01  # 5x training
_PLAY_TERRAIN_VERTICAL_SCALE = .0005  # 5x training


def _estimate_ground_z(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    height_sensor_name: str,
) -> torch.Tensor:
    """Estimate local terrain height under the robot (world-frame z)."""
    fallback = env.scene.env_origins[env_ids, 2]
    if height_sensor_name not in env.scene.sensors:
        return fallback

    sensor = env.scene.sensors[height_sensor_name]
    ray_hits_z = sensor.data.ray_hits_w[env_ids, ..., 2]
    invalid = (
        torch.isnan(ray_hits_z).any(dim=-1)
        | torch.isinf(ray_hits_z).any(dim=-1)
        | (torch.max(torch.abs(ray_hits_z), dim=-1).values > 1e6)
    )
    estimated = torch.mean(ray_hits_z, dim=-1)
    return torch.where(invalid, fallback, estimated)


def reset_root_state_inplace(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_sensor_name: str = "height_scanner_small",
) -> None:
    """Keep XY + yaw; stand upright at default height above local terrain; zero root vel."""
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w[env_ids].clone()
    root_quat = yaw_quat(asset.data.root_quat_w[env_ids].clone())
    default_height = asset.data.default_root_state[env_ids, 2]
    ground_z = _estimate_ground_z(env, env_ids, height_sensor_name)
    root_pos[:, 2] = ground_z + default_height
    velocities = torch.zeros((len(env_ids), 6), device=asset.device, dtype=root_pos.dtype)
    asset.write_root_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_joints_to_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset joints to exact default positions/velocities (no randomization)."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def _configure_inplace_reset(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Replace spawn-origin reset with in-place standing reset for gamepad play."""
    events = getattr(env_cfg, "events", None)
    if events is None:
        return
    if hasattr(events, "reset_base"):
        events.reset_base = EventTerm(
            func=reset_root_state_inplace,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "height_sensor_name": "height_scanner_small",
            },
        )
    if hasattr(events, "reset_robot_joints"):
        events.reset_robot_joints = EventTerm(
            func=reset_joints_to_default,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
    # Keep play resets deterministic (no actuator / motor offset noise).
    if hasattr(events, "randomize_actuator_gains"):
        events.randomize_actuator_gains = None
    if hasattr(events, "randomize_motor_zero_offset"):
        events.randomize_motor_zero_offset = None


def _apply_gamepad_command(env, command: torch.Tensor) -> None:
    """Write gamepad velocity into the env command term and block auto-resampling."""
    command_manager = getattr(env.unwrapped, "command_manager", None)
    if command_manager is None:
        return
    try:
        cmd_term = command_manager.get_term("base_velocity")
    except Exception:
        return
    command = command.view(1, -1).to(env.unwrapped.device)
    if hasattr(cmd_term, "set_external_command"):
        cmd_term.set_external_command(command)
    elif hasattr(cmd_term, "commands"):
        cmd_term.commands[:] = command
    elif hasattr(cmd_term, "vel_command_b"):
        cmd_term.vel_command_b[:] = command
    else:
        return
    if hasattr(cmd_term, "time_left"):
        cmd_term.time_left.fill_(1.0e6)


def _remap_gamepad_yaw_to_horizontal(gamepad: Se2Gamepad) -> None:
    """Bind yaw to right-stick left/right; flip lin_y and yaw to match robot frame."""
    v_y = gamepad.v_y_sensitivity
    sens = gamepad.omega_z_sensitivity
    mapping = gamepad._INPUT_STICK_VALUE_MAPPING
    for key in (
        carb.input.GamepadInput.LEFT_STICK_LEFT,
        carb.input.GamepadInput.LEFT_STICK_RIGHT,
        carb.input.GamepadInput.RIGHT_STICK_UP,
        carb.input.GamepadInput.RIGHT_STICK_DOWN,
        carb.input.GamepadInput.RIGHT_STICK_LEFT,
        carb.input.GamepadInput.RIGHT_STICK_RIGHT,
    ):
        mapping.pop(key, None)
    # Stick left → +lin_y, stick right → -lin_y (flipped vs Se2Gamepad default).
    mapping[carb.input.GamepadInput.LEFT_STICK_LEFT] = (0, 1, v_y)
    mapping[carb.input.GamepadInput.LEFT_STICK_RIGHT] = (1, 1, v_y)
    # Stick left → +yaw, stick right → -yaw (flipped vs Se2Gamepad default).
    mapping[carb.input.GamepadInput.RIGHT_STICK_LEFT] = (0, 2, sens)
    mapping[carb.input.GamepadInput.RIGHT_STICK_RIGHT] = (1, 2, sens)
    gamepad._base_command_raw[:, 1:3] = 0.0


def _update_heading_camera(env) -> None:
    """Place the viewport behind the robot, looking along its facing direction."""
    robot = env.unwrapped.scene["robot"]
    root_pos = robot.data.root_pos_w[0]
    root_quat = robot.data.root_quat_w[0]
    eye_b = torch.tensor(_CAMERA_EYE_B, device=root_pos.device, dtype=root_pos.dtype)
    lookat_b = torch.tensor(_CAMERA_LOOKAT_B, device=root_pos.device, dtype=root_pos.dtype)
    eye_w = root_pos + quat_apply_yaw(root_quat.unsqueeze(0), eye_b.unsqueeze(0)).squeeze(0)
    lookat_w = root_pos + quat_apply_yaw(root_quat.unsqueeze(0), lookat_b.unsqueeze(0)).squeeze(0)
    env.unwrapped.sim.set_camera_view(
        eye=eye_w.detach().cpu().tolist(),
        target=lookat_w.detach().cpu().tolist(),
    )


def _set_camera_follow(env, enabled: bool) -> None:
    """Enable/disable heading-aware camera follow."""
    vcc = getattr(env.unwrapped, "viewport_camera_controller", None)
    # Disable Isaac Lab's fixed-offset asset tracker so it does not fight our update.
    if vcc is not None:
        vcc.cfg.origin_type = "world"
        vcc.cfg.asset_name = None
    if enabled:
        _update_heading_camera(env)
        print("[INFO] Camera follow: ON (behind robot, facing forward)")
    else:
        if vcc is not None:
            vcc.update_view_to_world()
        print("[INFO] Camera follow: OFF")


def _terrain_generator_cfg(env_cfg: ManagerBasedRLEnvCfg):
    """Return (terrain_importer_cfg, terrain_generator_cfg) or (None, None)."""
    terrain_importer_cfg = getattr(getattr(env_cfg, "scene", None), "terrain", None)
    terrain_gen_cfg = getattr(terrain_importer_cfg, "terrain_generator", None)
    return terrain_importer_cfg, terrain_gen_cfg


def _list_terrain_types(env_cfg: ManagerBasedRLEnvCfg) -> list[str]:
    """All configured sub-terrain names, including zero-proportion types."""
    _, terrain_gen_cfg = _terrain_generator_cfg(env_cfg)
    if terrain_gen_cfg is None or terrain_gen_cfg.sub_terrains is None:
        return []
    return list(terrain_gen_cfg.sub_terrains.keys())


def _terrain_level_range(env_cfg: ManagerBasedRLEnvCfg) -> tuple[int, int] | None:
    """Inclusive (min_level, max_level) from the training curriculum row count."""
    _, terrain_gen_cfg = _terrain_generator_cfg(env_cfg)
    if terrain_gen_cfg is None:
        return None
    num_rows = int(terrain_gen_cfg.num_rows)
    if num_rows <= 0:
        return None
    return 0, num_rows - 1


def _print_terrain_choices(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Print valid --terrain_type names and --terrain_level range."""
    names = _list_terrain_types(env_cfg)
    level_range = _terrain_level_range(env_cfg)
    print(f"[INFO] Available --terrain_type values: {names if names else '(none)'}")
    if level_range is None:
        print("[INFO] Available --terrain_level range: (unavailable)")
    else:
        lo, hi = level_range
        print(f"[INFO] Available --terrain_level range: [{lo}, {hi}] (inclusive)")


def _validate_terrain_cli(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Require --terrain_type / --terrain_level; print choices on missing/invalid input."""
    terrain_type = args_cli.terrain_type
    terrain_level = args_cli.terrain_level
    names = _list_terrain_types(env_cfg)
    level_range = _terrain_level_range(env_cfg)

    missing = terrain_type is None or terrain_level is None
    invalid_type = terrain_type is not None and names and terrain_type not in names
    invalid_level = False
    if terrain_level is not None and level_range is not None:
        lo, hi = level_range
        invalid_level = terrain_level < lo or terrain_level > hi

    if missing or invalid_type or invalid_level or not names or level_range is None:
        _print_terrain_choices(env_cfg)
        if missing:
            raise ValueError("Both --terrain_type and --terrain_level are required.")
        if not names or level_range is None:
            raise ValueError("Environment has no terrain generator with sub-terrains / rows.")
        if invalid_type:
            raise ValueError(f"Unknown --terrain_type '{terrain_type}'. Valid: {names}")
        lo, hi = level_range
        raise ValueError(f"--terrain_level must be in [{lo}, {hi}], got {terrain_level}.")


def _configure_single_terrain_block(env_cfg: ManagerBasedRLEnvCfg, terrain_type: str, terrain_level: int) -> None:
    """Generate only one terrain patch of the chosen type at the chosen curriculum difficulty."""
    _, terrain_gen_cfg = _terrain_generator_cfg(env_cfg)
    if terrain_gen_cfg is None or terrain_gen_cfg.sub_terrains is None:
        raise ValueError("Cannot configure play terrain: terrain generator config is missing.")
    if terrain_type not in terrain_gen_cfg.sub_terrains:
        raise ValueError(f"Unknown terrain_type '{terrain_type}'.")

    orig_rows = int(terrain_gen_cfg.num_rows)
    lower, upper = terrain_gen_cfg.difficulty_range
    # Match Isaac Lab curriculum band for row ``terrain_level`` in the original grid:
    # difficulty ∈ [lower + (upper-lower)*level/rows, lower + (upper-lower)*(level+1)/rows).
    d_lo = lower + (upper - lower) * (terrain_level / orig_rows)
    d_hi = lower + (upper - lower) * ((terrain_level + 1) / orig_rows)
    if getattr(terrain_gen_cfg, "use_gym_difficulty", False):
        # Bake discrete gym difficulty (level / orig_rows); single-row gym remap would break it.
        d = terrain_level / orig_rows
        d_lo, d_hi = d, d
        terrain_gen_cfg.use_gym_difficulty = False

    chosen = terrain_gen_cfg.sub_terrains[terrain_type]
    chosen.proportion = 1.0
    terrain_gen_cfg.sub_terrains = {terrain_type: chosen}
    terrain_gen_cfg.num_rows = 1
    terrain_gen_cfg.num_cols = 1
    terrain_gen_cfg.curriculum = True
    terrain_gen_cfg.difficulty_range = (d_lo, d_hi)
    terrain_gen_cfg.border_width = 1.0
    terrain_gen_cfg.horizontal_scale = _PLAY_TERRAIN_HORIZONTAL_SCALE
    terrain_gen_cfg.vertical_scale = _PLAY_TERRAIN_VERTICAL_SCALE
    # Height-field noise uses int(noise_step / vertical_scale); keep step >= scale.
    if hasattr(chosen, "noise_step") and chosen.noise_step < _PLAY_TERRAIN_VERTICAL_SCALE:
        chosen.noise_step = _PLAY_TERRAIN_VERTICAL_SCALE

    # Keep command metadata consistent with the single remaining type.
    base_velocity = getattr(getattr(env_cfg, "commands", None), "base_velocity", None)
    if base_velocity is not None:
        velocity_ranges = getattr(base_velocity, "velocity_ranges", None)
        if isinstance(velocity_ranges, dict):
            if terrain_type in velocity_ranges:
                base_velocity.velocity_ranges = {terrain_type: velocity_ranges[terrain_type]}
            else:
                base_velocity.velocity_ranges = {}
        random_velocity_terrain = getattr(base_velocity, "random_velocity_terrain", None)
        if random_velocity_terrain is not None:
            base_velocity.random_velocity_terrain = [k for k in random_velocity_terrain if k == terrain_type]

    print(
        f"[INFO] Play terrain: single block type='{terrain_type}', "
        f"curriculum_level={terrain_level}/{orig_rows - 1}, "
        f"difficulty_range=({d_lo:.4f}, {d_hi:.4f}), border_width=1.0 m, "
        f"horizontal_scale={_PLAY_TERRAIN_HORIZONTAL_SCALE}, "
        f"vertical_scale={_PLAY_TERRAIN_VERTICAL_SCALE}"
    )


def _spawn_at_single_terrain_center(env, terrain_type: str, terrain_level: int) -> None:
    """Teleport env 0 to the center of the single generated terrain patch."""
    raw_env = env.unwrapped
    terrain = getattr(raw_env.scene, "terrain", None)
    if terrain is None or terrain.terrain_origins is None:
        raise RuntimeError("Cannot spawn at terrain center: terrain origins are unavailable.")

    origin = terrain.terrain_origins[0, 0]
    terrain.terrain_levels[0] = 0
    terrain.terrain_types[0] = 0
    terrain.env_origins[0] = origin

    env_ids = torch.tensor([0], device=raw_env.device, dtype=torch.long)
    asset: Articulation = raw_env.scene["robot"]
    root_states = asset.data.default_root_state[env_ids].clone()
    positions = root_states[:, 0:3] + origin.unsqueeze(0)
    orientations = root_states[:, 3:7]
    velocities = torch.zeros((1, 6), device=asset.device, dtype=root_states.dtype)
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
    reset_joints_to_default(raw_env, env_ids)

    print(
        f"[INFO] Initial spawn at terrain_type='{terrain_type}', "
        f"terrain_level={terrain_level}, origin={origin.detach().cpu().tolist()}"
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent using an F710 gamepad."""
    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if getattr(args_cli, "legacy_velocity_command", False):
        from robot_lab.tasks.go2w.env_cfg import apply_legacy_velocity_command

        apply_legacy_velocity_command(env_cfg)
        print("[INFO] Using legacy Go2RLGymCommand (--legacy_velocity_command).")
    else:
        from robot_lab.tasks.go2w.env_cfg import enable_pose_velocity_target_vis

        if enable_pose_velocity_target_vis(env_cfg):
            print("[INFO] PoseVelocityCommand target flat-patch visualization enabled.")
    env_cfg.scene.num_envs = 1
    # PhysX GPU memory reservation is large by default for high-throughput training.
    # For play (single env), drastically shrink the collision stack reservation to reduce VRAM usage.
    if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "physx"):
        if hasattr(env_cfg.sim.physx, "gpu_collision_stack_size"):
            env_cfg.sim.physx.gpu_collision_stack_size = int(64 * 1024 * 1024)  # 64MB

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Keep depth camera extrinsics deterministic for teleop/play:
    # use the configured default position/orientation with no DR on reset.
    front_depth_camera = getattr(getattr(env_cfg, "scene", None), "front_depth_camera", None)
    if front_depth_camera is not None:
        if hasattr(front_depth_camera, "pos_randomization_range"):
            front_depth_camera.pos_randomization_range = None
        if hasattr(front_depth_camera, "randomize_pos_on_reset"):
            front_depth_camera.randomize_pos_on_reset = False
        if hasattr(front_depth_camera, "rpy_randomization_deg"):
            front_depth_camera.rpy_randomization_deg = None
        if hasattr(front_depth_camera, "randomize_rot_on_reset"):
            front_depth_camera.randomize_rot_on_reset = False

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.randomize_push_robot = None
    # Keep XY/yaw on termination / A-reset; restore upright default pose above terrain.
    _configure_inplace_reset(env_cfg)
    if hasattr(env_cfg, "curriculum"):
        if hasattr(env_cfg.curriculum, "command_levels_lin_vel"):
            env_cfg.curriculum.command_levels_lin_vel = None
        if hasattr(env_cfg.curriculum, "command_levels_ang_vel"):
            env_cfg.curriculum.command_levels_ang_vel = None
        # Avoid curriculum relocating the single play env after initial spawn.
        if hasattr(env_cfg.curriculum, "terrain_levels"):
            env_cfg.curriculum.terrain_levels = None

    _validate_terrain_cli(env_cfg)
    _configure_single_terrain_block(env_cfg, args_cli.terrain_type, args_cli.terrain_level)

    # camera follow is handled manually each step (heading-aware); keep VCC in world mode
    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.env_index = 0

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        raise NotImplementedError("Pre-trained checkpoint retrieval is disabled temporarily.")
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        import imageio

        video_path = os.path.join(log_dir, "videos", "play_gamepad", time.strftime("%Y-%m-%d_%H-%M-%S") + ".mp4")
        writer = imageio.get_writer(video_path, fps=int(1 / env.unwrapped.step_dt))

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    with torch.inference_mode():
        _spawn_at_single_terrain_center(env, args_cli.terrain_type, args_cli.terrain_level)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerCTS":
        runner = OnPolicyRunnerCTS(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    is_recurrent = bool(getattr(policy_nn, "is_recurrent", False))
    if is_recurrent and hasattr(policy_nn, "student_cnn_gru"):
        export_cts_cnn_gru_policy_as_jit(
            policy_nn,
            actor_obs_normalizer=policy_nn.actor_obs_normalizer,
            single_obs_normalizer=policy_nn.single_obs_normalizer,
            path=export_model_dir,
            filename="policy.pt",
        )
        print(f"[INFO] Exported CNN-GRU CTS policy to: {export_model_dir}/policy.pt")
        print("[INFO] Copy to rl_sar/policy/go2w/moe_cts_d435i/policy.pt for deploy.")
    elif is_recurrent:
        print(
            "[WARN] Skipping JIT/ONNX export: recurrent CTS policy without student_cnn_gru "
            "is not supported by the exporters. Play will continue with in-sim inference only."
        )
    elif agent_cfg.class_name == "OnPolicyRunnerCTS":
        export_cts_policy_as_jit(
            policy_nn,
            actor_obs_normalizer=policy_nn.actor_obs_normalizer,
            single_obs_normalizer=policy_nn.single_obs_normalizer,
            path=export_model_dir,
            filename="policy.pt",
        )
        export_cts_policy_as_onnx(
            policy_nn,
            actor_obs_normalizer=policy_nn.actor_obs_normalizer,
            single_obs_normalizer=policy_nn.single_obs_normalizer,
            path=export_model_dir,
            filename="policy.onnx",
        )
    else:
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    # ---- gamepad setup ----
    gamepad = Se2Gamepad(
        Se2GamepadCfg(
            v_x_sensitivity=args_cli.lin_vel_x,
            v_y_sensitivity=args_cli.lin_vel_y,
            omega_z_sensitivity=args_cli.ang_vel_z,
            dead_zone=0.05,
            sim_device=str(env.unwrapped.device),
        )
    )
    _remap_gamepad_yaw_to_horizontal(gamepad)
    print(gamepad)
    print(
        f"[INFO] Command scales: lin_vel_x=±{args_cli.lin_vel_x:.2f} m/s, "
        f"lin_vel_y=±{args_cli.lin_vel_y:.2f} m/s, ang_vel_z=±{args_cli.ang_vel_z:.2f} rad/s"
    )
    print(
        "[INFO] Controls: Left stick = lin_x/lin_y | Right stick L/R = yaw | "
        "X = camera follow | A = in-place reset"
    )
    print("[INFO] Put the F710 switch in X (Xbox) mode if the pad is not detected.")
    print("[INFO] Termination / A reset: keep XY+yaw, upright, default joints, default height above terrain.")

    camera_follow = True
    reset_requested = False
    button_pressed = {
        carb.input.GamepadInput.X: False,
        carb.input.GamepadInput.A: False,
    }
    _set_camera_follow(env, True)

    _orig_on_gamepad_event = gamepad._on_gamepad_event

    def _on_gamepad_event(event: carb.input.GamepadEvent, *args, **kwargs):
        nonlocal camera_follow, reset_requested
        result = _orig_on_gamepad_event(event, *args, **kwargs)
        if event.input in button_pressed:
            pressed = event.value > 0.5
            # rising edge only
            if pressed and not button_pressed[event.input]:
                if event.input == carb.input.GamepadInput.X:
                    camera_follow = not camera_follow
                    _set_camera_follow(env, camera_follow)
                elif event.input == carb.input.GamepadInput.A:
                    reset_requested = True
            button_pressed[event.input] = pressed
        return result

    gamepad._on_gamepad_event = _on_gamepad_event

    dt = env.unwrapped.step_dt
    obs = env.get_observations()

    while simulation_app.is_running():
        start_time = time.time()

        if reset_requested:
            reset_requested = False
            with torch.inference_mode():
                obs, _ = env.reset()
                dones = torch.ones(env.num_envs, dtype=torch.long, device=env.device)
                policy_nn.reset(dones)
            print("[INFO] Robot reset in-place.")

        command = gamepad.advance()
        with torch.inference_mode():
            # Keep policy obs and env rewards on the same teleop command.
            _apply_gamepad_command(env, command)
            obs = env.get_observations()
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)
            # Re-assert after env command.compute() so the buffer stays teleop-driven.
            _apply_gamepad_command(env, command)

        if camera_follow:
            _update_heading_camera(env)

        if args_cli.video:
            writer.append_data(env.env.render())

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()
    if args_cli.video:
        writer.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
