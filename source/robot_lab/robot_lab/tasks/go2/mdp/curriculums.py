# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

from .utils import is_robot_on_terrain

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from robot_lab.tasks.go2.mdp.commands import Go2RLGymCommand


def command_levels_lin_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_lin_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # Update ranges
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)


def command_levels_ang_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_ang_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original angular velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device)
        env._initial_ang_vel_z = env._original_ang_vel_z * range_multiplier[0]
        env._final_ang_vel_z = env._original_ang_vel_z * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.ang_vel_z = env._initial_ang_vel_z.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_ang_vel_z = torch.clamp(new_ang_vel_z, min=env._final_ang_vel_z[0], max=env._final_ang_vel_z[1])

            # Update ranges
            base_velocity_ranges.ang_vel_z = new_ang_vel_z.tolist()

    return torch.tensor(base_velocity_ranges.ang_vel_z[1], device=env.device)

def command_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_term_name: str,
    num_steps_per_iter: int = 24,
) -> float:
    """
    阶跃式指令课程 (数据存储在 CommandCfg 中)。
    """
    try:
        cmd_term = env.command_manager.get_term(command_term_name)
        cmd_cfg = cmd_term.cfg
    except LookupError:
        return 0.0

    current_iter = env.common_step_counter // num_steps_per_iter
    schedule = cmd_cfg.curriculum_schedule
    
    for i in range(len(schedule) - 1, -1, -1):
        stage = schedule[i]
        
        if current_iter >= stage['iter']:
            for t_name, active_range in cmd_cfg.ranges.items():
                
                hard_limit = cmd_cfg.terrain_max_ranges.get(t_name)
                if hard_limit is None:
                    continue

                def get_intersection(target_val, limit_val):
                    new_min = max(target_val[0], limit_val[0])
                    new_max = min(target_val[1], limit_val[1])
                    return (new_min, new_max)

                if 'lin_vel_x' in stage:
                    new_range = get_intersection(stage['lin_vel_x'], hard_limit.lin_vel_x)
                    active_range.lin_vel_x = new_range
                
                if 'lin_vel_y' in stage:
                    new_range = get_intersection(stage['lin_vel_y'], hard_limit.lin_vel_y)
                    active_range.lin_vel_y = new_range
                    
                if 'ang_vel_yaw' in stage:
                    new_range = get_intersection(stage['ang_vel_yaw'], hard_limit.ang_vel_z)
                    active_range.ang_vel_z = new_range
                    
                if 'heading' in stage and hasattr(active_range, 'heading'):
                    new_range = get_intersection(stage['heading'], hard_limit.heading)
                    active_range.heading = new_range

            schedule.pop(i)
            break

    last_key = list(cmd_cfg.ranges.keys())[-1]
    return cmd_cfg.ranges[last_key].lin_vel_x[1]

def gradual_ref_stand_modification(
    env: ManagerBasedRLEnv, 
    env_ids: Sequence[int],
    term_name: str,
    initial: float,
    final: float,
    start_it: int,
    end_it: int,
):
    current_it = env.common_step_counter // 24
    if current_it < start_it:
        return

    if current_it >= end_it:
        new = final
    else:
        new = (current_it - start_it) / (end_it - start_it) * (final - initial) + initial

    term = env.command_manager.get_term(term_name)
    term.cfg.rel_standing_envs = new
    

def gradual_reward_weight_modification(
    env: ManagerBasedRLEnv, 
    env_ids: Sequence[int],
    term_name: str,
    initial_weight: float,
    final_weight: float,
    start_it: int,
    end_it: int,
):
    """Curriculum that gradually modifies a reward weight between an initial and final value over a range of steps."""
    current_it = env.common_step_counter // 24
    if current_it < start_it:
        return

    if current_it >= end_it:
        new_weight = final_weight
    else:
        new_weight = (current_it - start_it) / (end_it - start_it) * (final_weight - initial_weight) + initial_weight

    term_cfg = env.reward_manager.get_term_cfg(term_name)
    term_cfg.weight = new_weight
    env.reward_manager.set_term_cfg(term_name, term_cfg)


def step_height_range_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    start_it: int = 10000,
    interval: int = 500,
    delta: float = 0.01,
    max_lower: float = 0.15,
    initial_lower: float = 0.0,
    upper: float = 0.2,
    mesh_lower: float = 0.0,
    sub_terrain_names: Sequence[str] = ("stairs_up", "stairs_down"),
    num_steps_per_iter: int = 24,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> dict[str, float]:
    """Raise effective stair step-height lower bound on a fixed iteration schedule.

    Isaac Lab bakes terrain meshes once and cannot safely remesh PhysX collision
    mid-run. This curriculum keeps a single mesh with range ``(mesh_lower, upper)``
    and raises the minimum terrain level so robots only see stairs whose height is
    at least the scheduled lower bound.
    """
    from robot_lab.tasks.go2.mdp.terrains import (
        compute_step_height_lower_from_iteration,
        step_height_lower_to_min_terrain_level,
    )

    terrain: TerrainImporter = env.scene.terrain
    current_it = env.common_step_counter // num_steps_per_iter
    target_lower = compute_step_height_lower_from_iteration(
        current_it,
        start_it=start_it,
        interval=interval,
        delta=delta,
        max_lower=max_lower,
        initial_lower=initial_lower,
    )

    # Log the scheduled range on the cfg (mesh itself stays at mesh_lower..upper).
    gen_cfg = terrain.cfg.terrain_generator
    if gen_cfg is not None:
        for name in sub_terrain_names:
            sub_cfg = gen_cfg.sub_terrains.get(name)
            if sub_cfg is not None and hasattr(sub_cfg, "step_height_range"):
                sub_cfg.step_height_range = (target_lower, upper)

    num_rows = int(gen_cfg.num_rows) if gen_cfg is not None else int(terrain.max_terrain_level)
    min_level = step_height_lower_to_min_terrain_level(
        target_lower, upper=upper, num_rows=num_rows, mesh_lower=mesh_lower
    )

    if hasattr(terrain, "set_step_height_min_level"):
        prev_level = getattr(terrain, "step_height_min_level", 0)
        if min_level != prev_level or abs(getattr(terrain, "step_height_active_lower", -1.0) - target_lower) > 1e-9:
            old_origins = terrain.env_origins.clone()
            terrain.set_step_height_min_level(min_level, active_lower=target_lower)
            delta_origins = terrain.env_origins - old_origins
            moved_ids = torch.nonzero(torch.any(delta_origins != 0.0, dim=-1), as_tuple=False).squeeze(-1)
            if moved_ids.numel() > 0:
                asset: Articulation = env.scene[asset_cfg.name]
                root_state = asset.data.root_state_w[moved_ids].clone()
                root_state[:, :3] += delta_origins[moved_ids]
                root_state[:, 7:] = 0.0
                asset.write_root_state_to_sim(root_state, env_ids=moved_ids)
                try:
                    command_term = env.command_manager.get_term("base_velocity")
                except Exception:
                    command_term = None
                if command_term is not None and hasattr(command_term, "pos_command_w"):
                    command_term.pos_command_w[moved_ids, :3] += delta_origins[moved_ids]

    return {"lower": target_lower, "min_level": float(min_level)}
    
def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> dict[str, torch.Tensor]:
    """Terrain curriculum based on walked distance vs commanded linear speed.

    Compatible with PoseVelocityCommand (``(vx, yaw)``) and 3D velocity commands
    (``(vx, vy, yaw)``). Uses usable sub-terrain length (size minus border).

    Returns a dict so CurriculumManager logs ``Curriculum/terrain_levels/mean``
    and ``Curriculum/terrain_levels/<terrain_name>``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    terrain_cfg = terrain.cfg.terrain_generator
    border = getattr(terrain_cfg, "sub_terrain_border_width", 0.0) or 0.0
    terrain_length = max(0.0, terrain_cfg.size[0] - 2.0 * border)
    move_up = distance > terrain_length / 2
    # PoseVelocityCommand.command is (vx, yaw); Go2RLGym / Uniform is (vx, vy, yaw).
    if command.shape[-1] == 2:
        cmd_lin_speed = torch.abs(command[env_ids, 0])
    else:
        cmd_lin_speed = torch.norm(command[env_ids, :2], dim=1)
    move_down = distance < cmd_lin_speed * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)

    levels = terrain.terrain_levels.float()
    extras: dict[str, torch.Tensor] = {"mean": torch.mean(levels)}
    if terrain_cfg is not None and terrain_cfg.sub_terrains is not None:
        for terrain_name, sub_cfg in terrain_cfg.sub_terrains.items():
            if getattr(sub_cfg, "proportion", 0.0) <= 0.0:
                continue
            mask = is_robot_on_terrain(env, terrain_name)
            extras[terrain_name] = torch.mean(levels[mask]) if mask.any() else levels.new_zeros(())
    return extras


def terrain_levels_vel_gym(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> dict[str, torch.Tensor]:
    """
    使用 max_move_distance 而非 reset 时的瞬间位移, 比较标准基于 commands_xy_accumulation.

    Returns a dict so CurriculumManager logs overall and per-terrain mean levels as
    ``Curriculum/terrain_levels/mean`` and ``Curriculum/terrain_levels/<name>``.
    """
    terrain = env.scene.terrain
    command: Go2RLGymCommand = env.command_manager.get_term("base_velocity")

    max_move_dist = command.max_move_distance[env_ids]
    cmd_accum = command.commands_xy_accumulation[env_ids]
    
    resampling_time = command.cfg.resampling_time
    zero_prob = command.zero_command_prob
    
    terrain_cfg = terrain.cfg.terrain_generator
    sub_terrain_border_width = getattr(terrain_cfg, "sub_terrain_border_width", 0.0) or 0.0
    terrain_length = max(0.0, terrain_cfg.size[0] - 2.0 * sub_terrain_border_width)
    move_up = max_move_dist > terrain_length / 2
    target_dist = torch.norm(cmd_accum, dim=1) * (resampling_time * (1 - zero_prob))
    move_down = (max_move_dist < target_dist * 0.5) * ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)

    levels = terrain.terrain_levels.float()
    extras: dict[str, torch.Tensor] = {"mean": torch.mean(levels)}

    if terrain_cfg is not None and terrain_cfg.sub_terrains is not None:
        for terrain_name, sub_cfg in terrain_cfg.sub_terrains.items():
            if getattr(sub_cfg, "proportion", 0.0) <= 0.0:
                continue
            mask = is_robot_on_terrain(env, terrain_name)
            extras[terrain_name] = torch.mean(levels[mask]) if mask.any() else levels.new_zeros(())

    return extras


def resume_iteration_curriculum(
    env: ManagerBasedRLEnv,
    learning_iteration: int,
    num_steps_per_iter: int = 24,
    skip_terms: Sequence[str] = ("terrain_levels",),
) -> int:
    """Restore iteration-based curriculum as if training had reached ``learning_iteration``.

    Sets ``env.common_step_counter`` so later updates stay on the same schedule, then
    applies reward-weight, depth-noise, and step-height terms immediately. Terrain-level
    curriculum is skipped because it is performance-based. Command-range expansion is
    applied when the active command term supports it.
    """
    steps = max(int(num_steps_per_iter), 1)
    env.common_step_counter = int(learning_iteration) * steps
    env_ids = torch.arange(env.num_envs, device=env.device)
    skip = set(skip_terms)
    curriculum_manager = getattr(env, "curriculum_manager", None)
    if curriculum_manager is not None:
        for name, term_cfg in zip(curriculum_manager._term_names, curriculum_manager._term_cfgs):
            if name in skip:
                continue
            state = term_cfg.func(env, env_ids, **term_cfg.params)
            curriculum_manager._curriculum_state[name] = state
    try:
        command_term = env.command_manager.get_term("base_velocity")
    except Exception:
        command_term = None
    if command_term is not None and hasattr(command_term, "_update_command_range_curriculum"):
        command_term._update_command_range_curriculum()
    return env.common_step_counter


def _lerp(start: float, end: float, alpha: float) -> float:
    return start + (end - start) * float(np.clip(alpha, 0.0, 1.0))


def depth_noise_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    start_it: int = 0,
    end_it: int = 5000,
    num_steps_per_iter: int = 24,
    noise_std_range: tuple[float, float] = (0.02, 0.06),
    dropout_prob_range: tuple[float, float] = (0.2, 0.2),
    depth_dependent_noise_scale_range: tuple[float, float] = (0.0, 1.0),
    edge_speckle_prob_range: tuple[float, float] = (0.0, 0.08),
    temporal_flicker_std_range: tuple[float, float] = (0.0, 0.03),
    hole_blob_prob_range: tuple[float, float] = (0.0, 0.15),
) -> dict[str, float]:
    """Ramp depth-image corruption severity for MGDP-style outdoor robustness.

    Writes the scheduled values onto ``env.cfg`` fields consumed by
    ``process_depth_image(..., use_cfg_noise_overrides=True)``.
    """
    del env_ids  # curriculum applies globally
    current_it = env.common_step_counter // max(int(num_steps_per_iter), 1)
    if current_it <= start_it:
        alpha = 0.0
    elif current_it >= end_it:
        alpha = 1.0
    else:
        alpha = (current_it - start_it) / max(end_it - start_it, 1)

    env.cfg.depth_noise_std = _lerp(noise_std_range[0], noise_std_range[1], alpha)
    env.cfg.depth_dropout_prob = _lerp(dropout_prob_range[0], dropout_prob_range[1], alpha)
    env.cfg.depth_dependent_noise_scale = _lerp(
        depth_dependent_noise_scale_range[0], depth_dependent_noise_scale_range[1], alpha
    )
    env.cfg.depth_edge_speckle_prob = _lerp(edge_speckle_prob_range[0], edge_speckle_prob_range[1], alpha)
    env.cfg.depth_temporal_flicker_std = _lerp(
        temporal_flicker_std_range[0], temporal_flicker_std_range[1], alpha
    )
    env.cfg.depth_hole_blob_prob = _lerp(hole_blob_prob_range[0], hole_blob_prob_range[1], alpha)

    return {
        "noise_std": env.cfg.depth_noise_std,
        "dropout_prob": env.cfg.depth_dropout_prob,
        "depth_dependent_noise_scale": env.cfg.depth_dependent_noise_scale,
        "edge_speckle_prob": env.cfg.depth_edge_speckle_prob,
        "temporal_flicker_std": env.cfg.depth_temporal_flicker_std,
        "hole_blob_prob": env.cfg.depth_hole_blob_prob,
    }
