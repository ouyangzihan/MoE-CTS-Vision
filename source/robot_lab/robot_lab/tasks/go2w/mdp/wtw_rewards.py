"""Walk These Ways augmented auxiliary rewards (Table 1, MoB).

Ports the six behavior-conditioned reward terms from
``Reference/walk-these-ways/.../go1_gym/envs/rewards/corl_rewards.py`` for Go2W.
Gait timing uses fixed trotting defaults from the WTW training script unless
overridden via reward-term params. When ``switch_gait_when_vy_yaw_zero`` is set,
envs with both vy and yaw ~0 use the ``zero_vy_yaw_*`` gait targets instead.

Optional behaviors (enabled from ``symmetry_training_cfg``):
- ``scale_gait_frequency_by_vy``: target freq = base * (coef * |vy| + 1).
- ``footswing_height_curriculum_start_scale``: linear ramp of swing height to full
  by ``footswing_height_curriculum_end_it`` learning iterations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_from_euler_xyz, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@dataclass
class WalkTheseWaysGaitState:
    gait_indices: torch.Tensor
    foot_indices: torch.Tensor
    desired_contact_states: torch.Tensor


_BEHAVIOR_OVERRIDE_KEYS = (
    "gait_frequency",
    "gait_phase",
    "gait_offset",
    "gait_bound",
    "gait_duration",
    "footswing_height_cmd",
)


def _gait_params_dict(
    gait_frequency: float,
    gait_phase: float,
    gait_offset: float,
    gait_bound: float,
    gait_duration: float,
    kappa_gait_probs: float,
    command_name: str = "base_velocity",
    command_threshold: float = 1e-3,
    switch_gait_when_vy_yaw_zero: bool = False,
    zero_vy_yaw_gait_frequency: float = 0.0,
    zero_vy_yaw_gait_phase: float = 0.5,
    zero_vy_yaw_gait_offset: float = 0.0,
    zero_vy_yaw_gait_bound: float = 0.0,
    zero_vy_yaw_gait_duration: float = 0.5,
    zero_vy_yaw_kappa_gait_probs: float = 0.07,
    footswing_height_cmd: float = 0.19,
    zero_vy_yaw_footswing_height_cmd: float = 0.0,
    scale_gait_frequency_by_vy: bool = False,
    gait_frequency_vy_coef: float = 0.5,
    footswing_height_curriculum_start_scale: float | None = None,
    footswing_height_curriculum_end_it: int = 2500,
    num_steps_per_iter: int = 24,
) -> dict:
    return {
        "gait_frequency": gait_frequency,
        "gait_phase": gait_phase,
        "gait_offset": gait_offset,
        "gait_bound": gait_bound,
        "gait_duration": gait_duration,
        "kappa_gait_probs": kappa_gait_probs,
        "command_name": command_name,
        "command_threshold": command_threshold,
        "switch_gait_when_vy_yaw_zero": switch_gait_when_vy_yaw_zero,
        "zero_vy_yaw_gait_frequency": zero_vy_yaw_gait_frequency,
        "zero_vy_yaw_gait_phase": zero_vy_yaw_gait_phase,
        "zero_vy_yaw_gait_offset": zero_vy_yaw_gait_offset,
        "zero_vy_yaw_gait_bound": zero_vy_yaw_gait_bound,
        "zero_vy_yaw_gait_duration": zero_vy_yaw_gait_duration,
        "zero_vy_yaw_kappa_gait_probs": zero_vy_yaw_kappa_gait_probs,
        "footswing_height_cmd": footswing_height_cmd,
        "zero_vy_yaw_footswing_height_cmd": zero_vy_yaw_footswing_height_cmd,
        "scale_gait_frequency_by_vy": scale_gait_frequency_by_vy,
        "gait_frequency_vy_coef": gait_frequency_vy_coef,
        "footswing_height_curriculum_start_scale": footswing_height_curriculum_start_scale,
        "footswing_height_curriculum_end_it": footswing_height_curriculum_end_it,
        "num_steps_per_iter": num_steps_per_iter,
    }


def _vy_yaw_command_active_mask(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    command_threshold: float = 1e-3,
) -> torch.Tensor:
    """True when lateral ``vy`` or yaw-rate command is non-zero."""
    cmd = env.command_manager.get_command(command_name)
    if cmd.shape[1] > 2:
        vy = cmd[:, 1]
    else:
        vy = torch.zeros(env.num_envs, device=env.device)
    yaw = cmd[:, -1]
    return (vy.abs() > command_threshold) | (yaw.abs() > command_threshold)


def _footswing_height_curriculum_scale(env: ManagerBasedRLEnv, params: dict) -> float:
    """Linear ramp of footswing height: ``start_scale`` at iter 0 → 1.0 at ``end_it``."""
    start_scale = params.get("footswing_height_curriculum_start_scale")
    if start_scale is None:
        return 1.0
    end_it = max(int(params.get("footswing_height_curriculum_end_it", 2500)), 1)
    num_steps = max(int(params.get("num_steps_per_iter", 24)), 1)
    current_it = int(env.common_step_counter) // num_steps
    progress = min(1.0, max(0.0, current_it / float(end_it)))
    start = float(start_scale)
    return start + (1.0 - start) * progress


def _resolve_behavior_params(env: ManagerBasedRLEnv, params: dict) -> dict[str, torch.Tensor]:
    scalar_defaults = {
        "gait_frequency": 3.0,
        "gait_phase": 0.5,
        "gait_offset": 0.0,
        "gait_bound": 0.0,
        "gait_duration": 0.5,
        "body_height_cmd": 0.0,
        "footswing_height_cmd": 0.19,
        "body_pitch_cmd": 0.0,
        "body_roll_cmd": 0.0,
        "stance_width_cmd": 0.3,
        "stance_length_cmd": 0.45,
    }
    footswing_scale = _footswing_height_curriculum_scale(env, params)
    resolved = {}
    for key, scalar_default in scalar_defaults.items():
        value = float(params.get(key, scalar_default))
        if key == "footswing_height_cmd":
            value *= footswing_scale
        resolved[key] = torch.full((env.num_envs,), value, device=env.device)

    command_name = str(params.get("command_name", "base_velocity"))
    command_threshold = float(params.get("command_threshold", 1e-3))

    if params.get("switch_gait_when_vy_yaw_zero", False):
        stand_mask = ~_vy_yaw_command_active_mask(env, command_name, command_threshold)
        for key in _BEHAVIOR_OVERRIDE_KEYS:
            src = f"zero_vy_yaw_{key}"
            if src not in params:
                continue
            override = float(params[src])
            override_t = torch.full((env.num_envs,), override, device=env.device)
            resolved[key] = torch.where(stand_mask, override_t, resolved[key])

    # Target frequency = base * (coef * |vy| + 1). Planted gait (freq≈0) stays planted.
    if params.get("scale_gait_frequency_by_vy", False):
        cmd = env.command_manager.get_command(command_name)
        if cmd.shape[1] > 2:
            vy_abs = cmd[:, 1].abs()
        else:
            vy_abs = torch.zeros(env.num_envs, device=env.device)
        coef = float(params.get("gait_frequency_vy_coef", 0.5))
        resolved["gait_frequency"] = resolved["gait_frequency"] * (coef * vy_abs + 1.0)

    return resolved


def _update_wtw_gait_state(env: ManagerBasedRLEnv, params: dict) -> WalkTheseWaysGaitState:
    """Update gait clocks once per env step (shared across WTW reward terms)."""
    step = int(env.common_step_counter)
    if (
        getattr(env, "_wtw_gait_last_step", -1) == step
        and hasattr(env, "_wtw_gait_state")
        and hasattr(env, "_wtw_behavior_params")
    ):
        return env._wtw_gait_state

    behavior = _resolve_behavior_params(env, params)
    kappa = float(params.get("kappa_gait_probs", 0.07))

    if not hasattr(env, "_wtw_gait_indices"):
        env._wtw_gait_indices = torch.zeros(env.num_envs, device=env.device)

    env._wtw_gait_indices = torch.remainder(
        env._wtw_gait_indices + env.step_dt * behavior["gait_frequency"], 1.0
    )

    gait_indices = env._wtw_gait_indices
    phases = behavior["gait_phase"]
    offsets = behavior["gait_offset"]
    bounds = behavior["gait_bound"]
    durations = behavior["gait_duration"]

    foot_index_terms = [
        gait_indices + phases + offsets + bounds,
        gait_indices + offsets,
        gait_indices + bounds,
        gait_indices + phases,
    ]
    foot_indices = torch.remainder(torch.stack(foot_index_terms, dim=1), 1.0)

    normed_foot_indices = foot_indices.clone()
    for idx in range(4):
        term = normed_foot_indices[:, idx]
        stance_mask = torch.remainder(term, 1.0) < durations
        swing_mask = ~stance_mask
        term_stance = torch.remainder(term, 1.0) * (0.5 / durations.clamp_min(1e-6))
        term_swing = 0.5 + (torch.remainder(term, 1.0) - durations) * (
            0.5 / (1.0 - durations).clamp_min(1e-6)
        )
        normed_foot_indices[:, idx] = torch.where(stance_mask, term_stance, term_swing)

    smoothing_cdf_start = torch.distributions.Normal(0.0, kappa).cdf

    def _smoothing_multiplier(foot_term: torch.Tensor) -> torch.Tensor:
        rem = torch.remainder(foot_term, 1.0)
        return smoothing_cdf_start(rem) * (1.0 - smoothing_cdf_start(rem - 0.5)) + smoothing_cdf_start(rem - 1.0) * (
            1.0 - smoothing_cdf_start(rem - 0.5 - 1.0)
        )

    desired_contact_states = torch.stack(
        [_smoothing_multiplier(foot_index_terms[i]) for i in range(4)],
        dim=1,
    )

    swing_phases = 1.0 - torch.abs(1.0 - torch.clip(normed_foot_indices * 2.0 - 1.0, 0.0, 1.0) * 2.0)

    # Frequency 0 (vy=yaw≈0) is a planted gait: all wheels in stance, not a frozen trot.
    plant_mask = behavior["gait_frequency"] <= 1e-6
    if plant_mask.any():
        env._wtw_gait_indices = torch.where(
            plant_mask, torch.zeros_like(env._wtw_gait_indices), env._wtw_gait_indices
        )
        gait_indices = env._wtw_gait_indices
        desired_contact_states = torch.where(
            plant_mask.unsqueeze(1), torch.ones_like(desired_contact_states), desired_contact_states
        )
        swing_phases = torch.where(plant_mask.unsqueeze(1), torch.zeros_like(swing_phases), swing_phases)

    env._wtw_gait_state = WalkTheseWaysGaitState(
        gait_indices=gait_indices,
        foot_indices=swing_phases,
        desired_contact_states=desired_contact_states,
    )
    env._wtw_behavior_params = behavior
    env._wtw_gait_last_step = step
    return env._wtw_gait_state


def _apply_wtw_vy_yaw_gate(
    env: ManagerBasedRLEnv,
    reward: torch.Tensor,
    command_name: str,
    zero_when_vy_yaw_zero: bool,
    command_threshold: float,
) -> torch.Tensor:
    if not zero_when_vy_yaw_zero:
        return reward
    return reward * _vy_yaw_command_active_mask(env, command_name, command_threshold)


def _foot_positions_body_frame(
    env: ManagerBasedRLEnv, asset: Articulation, foot_body_ids: list[int] | slice
) -> torch.Tensor:
    foot_pos_w = asset.data.body_pos_w[:, foot_body_ids, :]
    num_feet = foot_pos_w.shape[1]
    cur = foot_pos_w - asset.data.root_pos_w.unsqueeze(1)
    footsteps = torch.zeros(env.num_envs, num_feet, 3, device=env.device)
    root_quat_conj = quat_conjugate(asset.data.root_quat_w)
    for i in range(num_feet):
        footsteps[:, i, :] = quat_apply(root_quat_conj, cur[:, i, :])
    return footsteps


def wtw_jump(
    env: ManagerBasedRLEnv,
    base_height_target: float,
    body_height_cmd: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    zero_when_vy_yaw_zero: bool = False,
    command_threshold: float = 1e-3,
) -> torch.Tensor:
    """Track commanded body height (WTW ``jump``)."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_height = asset.data.root_pos_w[:, 2]
    target = body_height_cmd + base_height_target
    reward = -torch.square(body_height - target)
    return _apply_wtw_vy_yaw_gate(env, reward, command_name, zero_when_vy_yaw_zero, command_threshold)


def wtw_orientation_control(
    env: ManagerBasedRLEnv,
    body_pitch_cmd: float = 0.0,
    body_roll_cmd: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    zero_when_vy_yaw_zero: bool = False,
    command_threshold: float = 1e-3,
) -> torch.Tensor:
    """Track commanded body pitch / roll via projected gravity (WTW ``orientation_control``)."""
    asset: Articulation = env.scene[asset_cfg.name]
    gravity = asset.data.GRAVITY_VEC_W
    zeros = torch.zeros(env.num_envs, device=env.device)
    quat_roll = quat_from_euler_xyz(-torch.full((env.num_envs,), body_roll_cmd, device=env.device), zeros, zeros)
    quat_pitch = quat_from_euler_xyz(zeros, -torch.full((env.num_envs,), body_pitch_cmd, device=env.device), zeros)
    desired_quat = quat_mul(quat_roll, quat_pitch)
    desired_projected_gravity = quat_apply_inverse(desired_quat, gravity)
    projected_gravity = asset.data.projected_gravity_b
    reward = torch.sum(torch.square(projected_gravity[:, :2] - desired_projected_gravity[:, :2]), dim=1)
    return _apply_wtw_vy_yaw_gate(env, reward, command_name, zero_when_vy_yaw_zero, command_threshold)


def _gait_switch_kwargs(
    command_name: str,
    command_threshold: float,
    switch_gait_when_vy_yaw_zero: bool,
    zero_vy_yaw_gait_frequency: float,
    zero_vy_yaw_gait_phase: float,
    zero_vy_yaw_gait_offset: float,
    zero_vy_yaw_gait_bound: float,
    zero_vy_yaw_gait_duration: float,
    zero_vy_yaw_kappa_gait_probs: float,
    footswing_height_cmd: float = 0.19,
    zero_vy_yaw_footswing_height_cmd: float = 0.0,
    scale_gait_frequency_by_vy: bool = False,
    gait_frequency_vy_coef: float = 0.5,
    footswing_height_curriculum_start_scale: float | None = None,
    footswing_height_curriculum_end_it: int = 2500,
    num_steps_per_iter: int = 24,
) -> dict:
    return {
        "command_name": command_name,
        "command_threshold": command_threshold,
        "switch_gait_when_vy_yaw_zero": switch_gait_when_vy_yaw_zero,
        "zero_vy_yaw_gait_frequency": zero_vy_yaw_gait_frequency,
        "zero_vy_yaw_gait_phase": zero_vy_yaw_gait_phase,
        "zero_vy_yaw_gait_offset": zero_vy_yaw_gait_offset,
        "zero_vy_yaw_gait_bound": zero_vy_yaw_gait_bound,
        "zero_vy_yaw_gait_duration": zero_vy_yaw_gait_duration,
        "zero_vy_yaw_kappa_gait_probs": zero_vy_yaw_kappa_gait_probs,
        "footswing_height_cmd": footswing_height_cmd,
        "zero_vy_yaw_footswing_height_cmd": zero_vy_yaw_footswing_height_cmd,
        "scale_gait_frequency_by_vy": scale_gait_frequency_by_vy,
        "gait_frequency_vy_coef": gait_frequency_vy_coef,
        "footswing_height_curriculum_start_scale": footswing_height_curriculum_start_scale,
        "footswing_height_curriculum_end_it": footswing_height_curriculum_end_it,
        "num_steps_per_iter": num_steps_per_iter,
    }


def _wtw_raibert_squared_xy_err(
    env: ManagerBasedRLEnv,
    command_name: str,
    stance_width_cmd: float,
    stance_length_cmd: float,
    asset_cfg: SceneEntityCfg,
    gait_params: dict,
) -> torch.Tensor:
    """Per-leg squared Raibert xy error. Shape ``(num_envs, 4, 2)``."""
    gait = _update_wtw_gait_state(env, gait_params)
    asset: Articulation = env.scene[asset_cfg.name]
    footsteps = _foot_positions_body_frame(env, asset, asset_cfg.body_ids)

    desired_ys = torch.tensor(
        [stance_width_cmd / 2, -stance_width_cmd / 2, stance_width_cmd / 2, -stance_width_cmd / 2],
        device=env.device,
    ).unsqueeze(0).expand(env.num_envs, -1)
    desired_xs = torch.tensor(
        [stance_length_cmd / 2, stance_length_cmd / 2, -stance_length_cmd / 2, -stance_length_cmd / 2],
        device=env.device,
    ).unsqueeze(0).expand(env.num_envs, -1)

    cmd = env.command_manager.get_command(command_name)
    x_vel_des = cmd[:, 0:1]
    yaw_vel_des = cmd[:, 2:3] if cmd.shape[1] > 2 else cmd[:, 1:2]
    behavior = env._wtw_behavior_params
    frequencies = behavior["gait_frequency"].unsqueeze(1)
    # Frequency 0 (stand / straight-roll gait) must not inflate Raibert lead.
    step_period = torch.where(
        frequencies > 1e-6,
        0.5 / frequencies.clamp_min(1e-6),
        torch.zeros_like(frequencies),
    )

    phases = torch.abs(1.0 - gait.foot_indices * 2.0) * 1.0 - 0.5
    y_vel_des = yaw_vel_des * stance_length_cmd / 2.0
    desired_ys_offset = phases * y_vel_des * step_period
    desired_ys_offset[:, 2:4] *= -1.0
    desired_xs_offset = phases * x_vel_des * step_period

    desired_ys = desired_ys + desired_ys_offset
    desired_xs = desired_xs + desired_xs_offset
    desired_footsteps = torch.stack((desired_xs, desired_ys), dim=2)
    err = torch.abs(desired_footsteps - footsteps[:, :, 0:2])
    return torch.square(err)


def wtw_raibert_heuristic(
    env: ManagerBasedRLEnv,
    command_name: str,
    stance_width_cmd: float,
    stance_length_cmd: float,
    asset_cfg: SceneEntityCfg,
    gait_frequency: float = 3.0,
    gait_phase: float = 0.5,
    gait_offset: float = 0.0,
    gait_bound: float = 0.0,
    gait_duration: float = 0.5,
    kappa_gait_probs: float = 0.07,
    zero_when_vy_yaw_zero: bool = False,
    command_threshold: float = 1e-3,
    switch_gait_when_vy_yaw_zero: bool = False,
    zero_vy_yaw_gait_frequency: float = 0.0,
    zero_vy_yaw_gait_phase: float = 0.5,
    zero_vy_yaw_gait_offset: float = 0.0,
    zero_vy_yaw_gait_bound: float = 0.0,
    zero_vy_yaw_gait_duration: float = 0.5,
    zero_vy_yaw_kappa_gait_probs: float = 0.07,
    footswing_height_cmd: float = 0.19,
    zero_vy_yaw_footswing_height_cmd: float = 0.0,
    scale_gait_frequency_by_vy: bool = False,
    gait_frequency_vy_coef: float = 0.5,
    footswing_height_curriculum_start_scale: float | None = None,
    footswing_height_curriculum_end_it: int = 2500,
    num_steps_per_iter: int = 24,
) -> torch.Tensor:
    """Raibert foot-placement heuristic (WTW ``raibert_heuristic``)."""
    gait_params = _gait_params_dict(
        gait_frequency,
        gait_phase,
        gait_offset,
        gait_bound,
        gait_duration,
        kappa_gait_probs,
        **_gait_switch_kwargs(
            command_name,
            command_threshold,
            switch_gait_when_vy_yaw_zero,
            zero_vy_yaw_gait_frequency,
            zero_vy_yaw_gait_phase,
            zero_vy_yaw_gait_offset,
            zero_vy_yaw_gait_bound,
            zero_vy_yaw_gait_duration,
            zero_vy_yaw_kappa_gait_probs,
            footswing_height_cmd,
            zero_vy_yaw_footswing_height_cmd,
            scale_gait_frequency_by_vy,
            gait_frequency_vy_coef,
            footswing_height_curriculum_start_scale,
            footswing_height_curriculum_end_it,
            num_steps_per_iter,
        ),
    )
    err_sq = _wtw_raibert_squared_xy_err(
        env, command_name, stance_width_cmd, stance_length_cmd, asset_cfg, gait_params
    )
    reward = torch.sum(err_sq, dim=(1, 2))
    return _apply_wtw_vy_yaw_gate(env, reward, command_name, zero_when_vy_yaw_zero, command_threshold)


def wtw_raibert_heuristic_imbalance(
    env: ManagerBasedRLEnv,
    command_name: str,
    stance_width_cmd: float,
    stance_length_cmd: float,
    asset_cfg: SceneEntityCfg,
    axis: str,
    gait_frequency: float = 3.0,
    gait_phase: float = 0.5,
    gait_offset: float = 0.0,
    gait_bound: float = 0.0,
    gait_duration: float = 0.5,
    kappa_gait_probs: float = 0.07,
    zero_when_vy_yaw_zero: bool = False,
    command_threshold: float = 1e-3,
    switch_gait_when_vy_yaw_zero: bool = False,
    zero_vy_yaw_gait_frequency: float = 0.0,
    zero_vy_yaw_gait_phase: float = 0.5,
    zero_vy_yaw_gait_offset: float = 0.0,
    zero_vy_yaw_gait_bound: float = 0.0,
    zero_vy_yaw_gait_duration: float = 0.5,
    zero_vy_yaw_kappa_gait_probs: float = 0.07,
    footswing_height_cmd: float = 0.19,
    zero_vy_yaw_footswing_height_cmd: float = 0.0,
    scale_gait_frequency_by_vy: bool = False,
    gait_frequency_vy_coef: float = 0.5,
    footswing_height_curriculum_start_scale: float | None = None,
    footswing_height_curriculum_end_it: int = 2500,
    num_steps_per_iter: int = 24,
) -> torch.Tensor:
    """Std of per-leg squared Raibert error along one axis (``x`` or ``y``)."""
    axis_idx = {"x": 0, "y": 1}.get(axis)
    if axis_idx is None:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    gait_params = _gait_params_dict(
        gait_frequency,
        gait_phase,
        gait_offset,
        gait_bound,
        gait_duration,
        kappa_gait_probs,
        **_gait_switch_kwargs(
            command_name,
            command_threshold,
            switch_gait_when_vy_yaw_zero,
            zero_vy_yaw_gait_frequency,
            zero_vy_yaw_gait_phase,
            zero_vy_yaw_gait_offset,
            zero_vy_yaw_gait_bound,
            zero_vy_yaw_gait_duration,
            zero_vy_yaw_kappa_gait_probs,
            footswing_height_cmd,
            zero_vy_yaw_footswing_height_cmd,
            scale_gait_frequency_by_vy,
            gait_frequency_vy_coef,
            footswing_height_curriculum_start_scale,
            footswing_height_curriculum_end_it,
            num_steps_per_iter,
        ),
    )
    err_sq = _wtw_raibert_squared_xy_err(
        env, command_name, stance_width_cmd, stance_length_cmd, asset_cfg, gait_params
    )
    # Population std: values are already squared errors.
    reward = err_sq[:, :, axis_idx].std(dim=1, unbiased=False)
    return _apply_wtw_vy_yaw_gate(env, reward, command_name, zero_when_vy_yaw_zero, command_threshold)


def wtw_feet_clearance_cmd_linear(
    env: ManagerBasedRLEnv,
    footswing_height_cmd: float,
    asset_cfg: SceneEntityCfg,
    foot_radius: float = 0.02,
    gait_frequency: float = 3.0,
    gait_phase: float = 0.5,
    gait_offset: float = 0.0,
    gait_bound: float = 0.0,
    gait_duration: float = 0.5,
    kappa_gait_probs: float = 0.07,
    command_name: str = "base_velocity",
    zero_when_vy_yaw_zero: bool = False,
    command_threshold: float = 1e-3,
    switch_gait_when_vy_yaw_zero: bool = False,
    zero_vy_yaw_gait_frequency: float = 0.0,
    zero_vy_yaw_gait_phase: float = 0.5,
    zero_vy_yaw_gait_offset: float = 0.0,
    zero_vy_yaw_gait_bound: float = 0.0,
    zero_vy_yaw_gait_duration: float = 0.5,
    zero_vy_yaw_kappa_gait_probs: float = 0.07,
    zero_vy_yaw_footswing_height_cmd: float = 0.0,
    scale_gait_frequency_by_vy: bool = False,
    gait_frequency_vy_coef: float = 0.5,
    footswing_height_curriculum_start_scale: float | None = None,
    footswing_height_curriculum_end_it: int = 2500,
    num_steps_per_iter: int = 24,
) -> torch.Tensor:
    """Track commanded swing height of the wheel hub (WTW ``feet_clearance_cmd_linear``).

    ``foot_radius`` is the wheel radius for Go2W (hub ``z`` on the ground). Target hub
    height is ``footswing_height_cmd + foot_radius``. Default ``0.02`` is the WTW Go1
    rubber foot; pass ``WHEEL_RADIUS`` (0.086) for Go2W.
    """
    gait_params = _gait_params_dict(
        gait_frequency,
        gait_phase,
        gait_offset,
        gait_bound,
        gait_duration,
        kappa_gait_probs,
        **_gait_switch_kwargs(
            command_name,
            command_threshold,
            switch_gait_when_vy_yaw_zero,
            zero_vy_yaw_gait_frequency,
            zero_vy_yaw_gait_phase,
            zero_vy_yaw_gait_offset,
            zero_vy_yaw_gait_bound,
            zero_vy_yaw_gait_duration,
            zero_vy_yaw_kappa_gait_probs,
            footswing_height_cmd,
            zero_vy_yaw_footswing_height_cmd,
            scale_gait_frequency_by_vy,
            gait_frequency_vy_coef,
            footswing_height_curriculum_start_scale,
            footswing_height_curriculum_end_it,
            num_steps_per_iter,
        ),
    )
    gait = _update_wtw_gait_state(env, gait_params)
    asset: Articulation = env.scene[asset_cfg.name]
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    target_height = env._wtw_behavior_params["footswing_height_cmd"] + foot_radius
    swing_mask = 1.0 - gait.desired_contact_states
    rew = torch.square(target_height.unsqueeze(1) - foot_height) * swing_mask
    reward = torch.sum(rew, dim=1)
    return _apply_wtw_vy_yaw_gate(env, reward, command_name, zero_when_vy_yaw_zero, command_threshold)


def wtw_feet_clearance_cmd_linear_imbalance(
    env: ManagerBasedRLEnv,
    footswing_height_cmd: float,
    asset_cfg: SceneEntityCfg,
    foot_radius: float = 0.02,
    gait_frequency: float = 3.0,
    gait_phase: float = 0.5,
    gait_offset: float = 0.0,
    gait_bound: float = 0.0,
    gait_duration: float = 0.5,
    kappa_gait_probs: float = 0.07,
    command_name: str = "base_velocity",
    zero_when_vy_yaw_zero: bool = False,
    command_threshold: float = 1e-3,
    switch_gait_when_vy_yaw_zero: bool = False,
    zero_vy_yaw_gait_frequency: float = 0.0,
    zero_vy_yaw_gait_phase: float = 0.5,
    zero_vy_yaw_gait_offset: float = 0.0,
    zero_vy_yaw_gait_bound: float = 0.0,
    zero_vy_yaw_gait_duration: float = 0.5,
    zero_vy_yaw_kappa_gait_probs: float = 0.07,
    zero_vy_yaw_footswing_height_cmd: float = 0.0,
    scale_gait_frequency_by_vy: bool = False,
    gait_frequency_vy_coef: float = 0.5,
    footswing_height_curriculum_start_scale: float | None = None,
    footswing_height_curriculum_end_it: int = 2500,
    num_steps_per_iter: int = 24,
) -> torch.Tensor:
    """Variance of per-leg clearance error (target minus actual, all four legs).

    Actual clearance is hub height minus wheel radius. Target is commanded swing
    height in swing and 0 in stance (wheel on ground ⇒ hub at ``foot_radius``).
    Stance/swing uses gait ``desired_contact_states`` (1 = stance).
    """
    gait_params = _gait_params_dict(
        gait_frequency,
        gait_phase,
        gait_offset,
        gait_bound,
        gait_duration,
        kappa_gait_probs,
        **_gait_switch_kwargs(
            command_name,
            command_threshold,
            switch_gait_when_vy_yaw_zero,
            zero_vy_yaw_gait_frequency,
            zero_vy_yaw_gait_phase,
            zero_vy_yaw_gait_offset,
            zero_vy_yaw_gait_bound,
            zero_vy_yaw_gait_duration,
            zero_vy_yaw_kappa_gait_probs,
            footswing_height_cmd,
            zero_vy_yaw_footswing_height_cmd,
            scale_gait_frequency_by_vy,
            gait_frequency_vy_coef,
            footswing_height_curriculum_start_scale,
            footswing_height_curriculum_end_it,
            num_steps_per_iter,
        ),
    )
    gait = _update_wtw_gait_state(env, gait_params)
    asset: Articulation = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    actual_clearance = foot_z - foot_radius
    swing_frac = 1.0 - gait.desired_contact_states
    target_clearance = swing_frac * env._wtw_behavior_params["footswing_height_cmd"].unsqueeze(1)
    per_leg = target_clearance - actual_clearance
    reward = per_leg.var(dim=1, unbiased=False)
    return _apply_wtw_vy_yaw_gate(env, reward, command_name, zero_when_vy_yaw_zero, command_threshold)


def wtw_tracking_contacts_shaped_force(
    env: ManagerBasedRLEnv,
    gait_force_sigma: float,
    sensor_cfg: SceneEntityCfg,
    gait_frequency: float = 3.0,
    gait_phase: float = 0.5,
    gait_offset: float = 0.0,
    gait_bound: float = 0.0,
    gait_duration: float = 0.5,
    kappa_gait_probs: float = 0.07,
    command_name: str = "base_velocity",
    zero_when_vy_yaw_zero: bool = False,
    command_threshold: float = 1e-3,
    switch_gait_when_vy_yaw_zero: bool = False,
    zero_vy_yaw_gait_frequency: float = 0.0,
    zero_vy_yaw_gait_phase: float = 0.5,
    zero_vy_yaw_gait_offset: float = 0.0,
    zero_vy_yaw_gait_bound: float = 0.0,
    zero_vy_yaw_gait_duration: float = 0.5,
    zero_vy_yaw_kappa_gait_probs: float = 0.07,
    footswing_height_cmd: float = 0.19,
    zero_vy_yaw_footswing_height_cmd: float = 0.0,
    scale_gait_frequency_by_vy: bool = False,
    gait_frequency_vy_coef: float = 0.5,
    footswing_height_curriculum_start_scale: float | None = None,
    footswing_height_curriculum_end_it: int = 2500,
    num_steps_per_iter: int = 24,
) -> torch.Tensor:
    """Shaped contact-force tracking (WTW ``tracking_contacts_shaped_force``)."""
    from isaaclab.sensors import ContactSensor

    gait_params = _gait_params_dict(
        gait_frequency,
        gait_phase,
        gait_offset,
        gait_bound,
        gait_duration,
        kappa_gait_probs,
        **_gait_switch_kwargs(
            command_name,
            command_threshold,
            switch_gait_when_vy_yaw_zero,
            zero_vy_yaw_gait_frequency,
            zero_vy_yaw_gait_phase,
            zero_vy_yaw_gait_offset,
            zero_vy_yaw_gait_bound,
            zero_vy_yaw_gait_duration,
            zero_vy_yaw_kappa_gait_probs,
            footswing_height_cmd,
            zero_vy_yaw_footswing_height_cmd,
            scale_gait_frequency_by_vy,
            gait_frequency_vy_coef,
            footswing_height_curriculum_start_scale,
            footswing_height_curriculum_end_it,
            num_steps_per_iter,
        ),
    )
    gait = _update_wtw_gait_state(env, gait_params)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    desired_contact = gait.desired_contact_states
    reward = torch.zeros(env.num_envs, device=env.device)
    for i in range(foot_forces.shape[1]):
        reward += -(1.0 - desired_contact[:, i]) * (
            1.0 - torch.exp(-foot_forces[:, i].square() / gait_force_sigma)
        )
    reward = reward / foot_forces.shape[1]
    # Planted gait (freq=0): original swing term is 0; penalize missing contact force.
    plant_mask = env._wtw_behavior_params["gait_frequency"] <= 1e-6
    no_force = torch.exp(-foot_forces.square() / gait_force_sigma).mean(dim=1)
    reward = torch.where(plant_mask, -no_force, reward)
    return _apply_wtw_vy_yaw_gate(env, reward, command_name, zero_when_vy_yaw_zero, command_threshold)


def wtw_tracking_contacts_shaped_vel(
    env: ManagerBasedRLEnv,
    gait_vel_sigma: float,
    asset_cfg: SceneEntityCfg,
    gait_frequency: float = 3.0,
    gait_phase: float = 0.5,
    gait_offset: float = 0.0,
    gait_bound: float = 0.0,
    gait_duration: float = 0.5,
    kappa_gait_probs: float = 0.07,
    command_name: str = "base_velocity",
    zero_when_vy_yaw_zero: bool = False,
    command_threshold: float = 1e-3,
    switch_gait_when_vy_yaw_zero: bool = False,
    zero_vy_yaw_gait_frequency: float = 0.0,
    zero_vy_yaw_gait_phase: float = 0.5,
    zero_vy_yaw_gait_offset: float = 0.0,
    zero_vy_yaw_gait_bound: float = 0.0,
    zero_vy_yaw_gait_duration: float = 0.5,
    zero_vy_yaw_kappa_gait_probs: float = 0.07,
    footswing_height_cmd: float = 0.19,
    zero_vy_yaw_footswing_height_cmd: float = 0.0,
    scale_gait_frequency_by_vy: bool = False,
    gait_frequency_vy_coef: float = 0.5,
    footswing_height_curriculum_start_scale: float | None = None,
    footswing_height_curriculum_end_it: int = 2500,
    num_steps_per_iter: int = 24,
) -> torch.Tensor:
    """Shaped foot-velocity tracking during stance (WTW ``tracking_contacts_shaped_vel``)."""
    gait_params = _gait_params_dict(
        gait_frequency,
        gait_phase,
        gait_offset,
        gait_bound,
        gait_duration,
        kappa_gait_probs,
        **_gait_switch_kwargs(
            command_name,
            command_threshold,
            switch_gait_when_vy_yaw_zero,
            zero_vy_yaw_gait_frequency,
            zero_vy_yaw_gait_phase,
            zero_vy_yaw_gait_offset,
            zero_vy_yaw_gait_bound,
            zero_vy_yaw_gait_duration,
            zero_vy_yaw_kappa_gait_probs,
            footswing_height_cmd,
            zero_vy_yaw_footswing_height_cmd,
            scale_gait_frequency_by_vy,
            gait_frequency_vy_coef,
            footswing_height_curriculum_start_scale,
            footswing_height_curriculum_end_it,
            num_steps_per_iter,
        ),
    )
    gait = _update_wtw_gait_state(env, gait_params)
    asset: Articulation = env.scene[asset_cfg.name]
    foot_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    foot_vel_world = torch.norm(foot_vel_w, dim=-1)
    # Rolling planted wheels move with the base; penalize hub speed relative to the body.
    foot_vel_rel = torch.norm(foot_vel_w - asset.data.root_lin_vel_w.unsqueeze(1), dim=-1)
    plant_mask = env._wtw_behavior_params["gait_frequency"] <= 1e-6
    foot_velocities = torch.where(plant_mask.unsqueeze(1), foot_vel_rel, foot_vel_world)
    desired_contact = gait.desired_contact_states
    reward = torch.zeros(env.num_envs, device=env.device)
    for i in range(foot_velocities.shape[1]):
        reward += -desired_contact[:, i] * (
            1.0 - torch.exp(-foot_velocities[:, i].square() / gait_vel_sigma)
        )
    reward = reward / foot_velocities.shape[1]
    return _apply_wtw_vy_yaw_gate(env, reward, command_name, zero_when_vy_yaw_zero, command_threshold)
