# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Privileged targets for the Go2W state estimator."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def foot_contact_state(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
    slip_speed_threshold: float = 0.15,
    contact_offset_body: tuple[float, float, float] = (0.0, 0.0, -0.087),
) -> torch.Tensor:
    """Per-foot contact class in FL, FR, RL, RR order.

    Returns one float per foot: 0 airborne, 1 contacting and gripping (tangential
    contact-patch speed at or below ``slip_speed_threshold``), 2 contacting and
    sliding. Slip matches ``wheel_slip_ratio``: velocity of the contact point
    with the wheel-axle component removed, so pure rolling stays near zero.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_ids = asset_cfg.body_ids
    num_wheels = len(body_ids)

    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > contact_threshold
    )

    body_quat = asset.data.body_quat_w[:, body_ids, :]
    axis_b = torch.tensor([0.0, 1.0, 0.0], device=env.device, dtype=torch.float32)
    offset_b = torch.tensor(contact_offset_body, device=env.device, dtype=torch.float32)
    axis_w = quat_apply(
        body_quat.reshape(-1, 4),
        axis_b.unsqueeze(0).expand(env.num_envs * num_wheels, 3),
    ).reshape(env.num_envs, num_wheels, 3)
    offset_w = quat_apply(
        body_quat.reshape(-1, 4),
        offset_b.view(1, 1, 3).expand(env.num_envs, num_wheels, 3).reshape(-1, 3),
    ).reshape(env.num_envs, num_wheels, 3)

    v_contact = asset.data.body_lin_vel_w[:, body_ids, :] + torch.cross(
        asset.data.body_ang_vel_w[:, body_ids, :], offset_w, dim=-1
    )
    v_axial = (v_contact * axis_w).sum(dim=-1, keepdim=True) * axis_w
    slip_speed = torch.linalg.vector_norm(v_contact - v_axial, dim=-1)

    state = torch.zeros(env.num_envs, num_wheels, device=env.device, dtype=torch.float32)
    gripping = contacts & (slip_speed <= slip_speed_threshold)
    sliding = contacts & (slip_speed > slip_speed_threshold)
    state = torch.where(gripping, torch.ones_like(state), state)
    state = torch.where(sliding, torch.full_like(state, 2.0), state)
    return state
