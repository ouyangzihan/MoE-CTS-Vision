# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Go2W-specific reward terms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def dont_wait(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    command_threshold: float = 0.3,
    slow_threshold: float = 0.15,
    reverse_threshold: float = -0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize freezing / reversing when a forward velocity is commanded.

    Hiking-in-the-Wild ``Don't Wait`` (Table IV):

    ``I(vx* > command_threshold) * (I(vx < slow) + I(vx < 0) + I(vx < reverse))``

    where ``vx*`` is commanded linear-x and ``vx`` is body-frame base linear-x.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd_vx = env.command_manager.get_command(command_name)[:, 0]
    actual_vx = asset.data.root_lin_vel_b[:, 0]
    forward_cmd = (cmd_vx > command_threshold).float()
    return forward_cmd * (
        (actual_vx < slow_threshold).float()
        + (actual_vx < 0.0).float()
        + (actual_vx < reverse_threshold).float()
    )


def wheels_not_in_contact(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize wheels that are not in contact with any surface.

    Returns the number of selected bodies whose net contact force stays below
    ``threshold`` over the contact-sensor history (airborne wheel count).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(~is_contact, dim=1).float()


def wheel_lateral_drag(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize lateral (body-y) slip at wheels in contact.

    Compares each wheel's body-frame velocity to the rigid-body velocity at that
    point (base linear + angular cross offset). Residual y slip indicates
    sideways dragging / skidding on the ground. Only contacting wheels count.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_ids = asset_cfg.body_ids
    num_wheels = len(body_ids)

    root_quat = asset.data.root_quat_w.unsqueeze(1).expand(-1, num_wheels, -1).reshape(-1, 4)
    wheel_vel_b = quat_apply_inverse(
        root_quat, asset.data.body_lin_vel_w[:, body_ids, :].reshape(-1, 3)
    ).reshape(env.num_envs, num_wheels, 3)

    wheel_pos_w = asset.data.body_pos_w[:, body_ids, :] - asset.data.root_pos_w.unsqueeze(1)
    wheel_pos_b = quat_apply_inverse(root_quat, wheel_pos_w.reshape(-1, 3)).reshape(
        env.num_envs, num_wheels, 3
    )

    v_expected = asset.data.root_lin_vel_b.unsqueeze(1) + torch.cross(
        asset.data.root_ang_vel_b.unsqueeze(1).expand(-1, num_wheels, -1),
        wheel_pos_b,
        dim=-1,
    )
    lateral_slip = wheel_vel_b[:, :, 1] - v_expected[:, :, 1]

    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > threshold
    )
    return torch.sum(torch.square(lateral_slip) * contacts.float(), dim=1)


class wheel_slip_ratio(ManagerTermBase):
    """Penalize tangential slip at contacting wheels, weighted by effective friction.

    For each wheel in contact, compares the contact-patch velocity (in the plane
    perpendicular to the wheel axle) to zero slip against a static surface. The
    per-wheel penalty is ``w * slip^2`` where ``w = (mu_s + mu_d) / 2`` uses
    average-combined wheel and terrain material coefficients.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.sensor_cfg: SceneEntityCfg = cfg.params.get(
            "sensor_cfg", SceneEntityCfg("contact_forces", body_names=".*_foot")
        )
        contact_offset_body = cfg.params.get("contact_offset_body", (0.0, 0.0, -0.086))
        self._contact_offset_b = torch.tensor(contact_offset_body, device=env.device, dtype=torch.float32)
        self._wheel_axis_b = torch.tensor([0.0, 1.0, 0.0], device=env.device, dtype=torch.float32)
        self.terrain_static_friction = float(cfg.params.get("terrain_static_friction", 1.0))
        self.terrain_dynamic_friction = float(cfg.params.get("terrain_dynamic_friction", 1.0))

        asset: Articulation = env.scene[self.asset_cfg.name]
        self._asset = asset
        self._num_shapes_per_body = self._compute_num_shapes_per_body(asset)
        self._wheel_shape_ids = self._shape_ids_for_bodies(asset, self.asset_cfg.body_ids)
        self._wheel_friction_w: torch.Tensor | None = None

    @staticmethod
    def _compute_num_shapes_per_body(asset: Articulation) -> list[int]:
        num_shapes_per_body = []
        for link_path in asset.root_physx_view.link_paths[0]:
            link_physx_view = asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore[attr-defined]
            num_shapes_per_body.append(link_physx_view.max_shapes)
        return num_shapes_per_body

    def _shape_ids_for_bodies(self, asset: Articulation, body_ids: list[int] | slice) -> list[list[int]]:
        if isinstance(body_ids, slice):
            body_ids = list(range(asset.num_bodies))
        shape_ids_per_body = []
        for body_id in body_ids:
            start_idx = sum(self._num_shapes_per_body[:body_id])
            end_idx = start_idx + self._num_shapes_per_body[body_id]
            shape_ids_per_body.append(list(range(start_idx, end_idx)))
        return shape_ids_per_body

    def _ensure_friction_weights(self) -> torch.Tensor:
        if self._wheel_friction_w is not None:
            return self._wheel_friction_w

        materials = self._asset.root_physx_view.get_material_properties().to(self.device)
        friction_weights = []
        for shape_ids in self._wheel_shape_ids:
            wheel_materials = materials[:, shape_ids, :]
            mu_s_robot = wheel_materials[..., 0].mean(dim=-1)
            mu_d_robot = wheel_materials[..., 1].mean(dim=-1)
            mu_s_contact = 0.5 * (mu_s_robot + self.terrain_static_friction)
            mu_d_contact = 0.5 * (mu_d_robot + self.terrain_dynamic_friction)
            friction_weights.append(0.5 * (mu_s_contact + mu_d_contact))
        self._wheel_friction_w = torch.stack(friction_weights, dim=1)
        return self._wheel_friction_w

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        threshold: float,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        wheel_radius: float = 0.086,
        contact_offset_body: tuple[float, float, float] = (0.0, 0.0, -0.086),
        terrain_static_friction: float = 1.0,
        terrain_dynamic_friction: float = 1.0,
    ) -> torch.Tensor:
        del wheel_radius, contact_offset_body, terrain_static_friction, terrain_dynamic_friction
        asset: Articulation = env.scene[asset_cfg.name]
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        body_ids = asset_cfg.body_ids
        num_wheels = len(body_ids)

        contacts = (
            contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
            .norm(dim=-1)
            .max(dim=1)[0]
            > threshold
        )

        body_quat = asset.data.body_quat_w[:, body_ids, :]
        axis_w = quat_apply(
            body_quat.reshape(-1, 4),
            self._wheel_axis_b.unsqueeze(0).expand(env.num_envs * num_wheels, 3),
        ).reshape(env.num_envs, num_wheels, 3)

        contact_offset_b = self._contact_offset_b.view(1, 1, 3).expand(env.num_envs, num_wheels, 3)
        contact_offset_w = quat_apply(body_quat.reshape(-1, 4), contact_offset_b.reshape(-1, 3)).reshape(
            env.num_envs, num_wheels, 3
        )

        v_lin = asset.data.body_lin_vel_w[:, body_ids, :]
        v_ang = asset.data.body_ang_vel_w[:, body_ids, :]
        v_contact = v_lin + torch.cross(v_ang, contact_offset_w, dim=-1)
        v_axial = (v_contact * axis_w).sum(dim=-1, keepdim=True) * axis_w
        v_slip = v_contact - v_axial
        slip_sq = torch.sum(torch.square(v_slip), dim=-1)

        friction_w = self._ensure_friction_weights()
        return torch.sum(slip_sq * friction_w * contacts.float(), dim=1)


def base_tilt_angle(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return the base tilt from the horizontal plane in radians."""
    asset: Articulation = env.scene[asset_cfg.name]
    gravity_b = asset.data.projected_gravity_b
    return torch.atan2(torch.linalg.vector_norm(gravity_b[:, :2], dim=1), -gravity_b[:, 2])


def local_terrain_tilt_angle(
    env: ManagerBasedRLEnv,
    contact_point_weight: float,
    lateral_scale: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner_small"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("wheel_contact_points"),
) -> torch.Tensor:
    """Return base tilt relative to a weighted local terrain plane.

    The plane is fitted to height-scanner ray hits and measured wheel-terrain
    contact positions. Each valid ray has unit weight and each valid contact
    has the fixed ``contact_point_weight``; airborne wheels contribute nothing.
    Invalid or degenerate fits fall back to the world-up normal.

    Side-to-side (body-y / roll) tilt is scaled by ``lateral_scale`` relative to
    front-to-back (body-x / pitch) before forming the tilt magnitude.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    height_sensor: RayCaster = env.scene.sensors[height_sensor_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]

    ray_hits_w = height_sensor.data.ray_hits_w
    contact_pos_w = contact_sensor.data.contact_pos_w
    if contact_pos_w is None:
        raise RuntimeError(
            f"Contact sensor '{contact_sensor_cfg.name}' must enable track_contact_points."
        )

    # Contact points have shape (env, wheel, filtered body, xyz). The terrain
    # filter has one body, but flattening also supports future additional filters.
    contact_pos_w = contact_pos_w[:, contact_sensor_cfg.body_ids].flatten(1, 2)
    points_w = torch.cat((ray_hits_w, contact_pos_w), dim=1)

    ray_valid = torch.isfinite(ray_hits_w).all(dim=-1) & (torch.abs(ray_hits_w) < 1.0e6).all(dim=-1)
    contact_valid = torch.isfinite(contact_pos_w).all(dim=-1) & (torch.abs(contact_pos_w) < 1.0e6).all(dim=-1)
    weights = torch.cat(
        (
            ray_valid.to(points_w.dtype),
            contact_valid.to(points_w.dtype) * contact_point_weight,
        ),
        dim=1,
    )

    # Fit z = a*x + b*y + c after weighted centering. Centering keeps the
    # two-by-two normal equations well-conditioned in large world coordinates.
    safe_points_w = torch.where((weights > 0.0).unsqueeze(-1), points_w, 0.0)
    weight_sum = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    centroid_w = (safe_points_w * weights.unsqueeze(-1)).sum(dim=1) / weight_sum
    centered = safe_points_w - centroid_w.unsqueeze(1)
    dx, dy, dz = centered.unbind(dim=-1)

    s_xx = (weights * dx * dx).sum(dim=1)
    s_xy = (weights * dx * dy).sum(dim=1)
    s_yy = (weights * dy * dy).sum(dim=1)
    s_xz = (weights * dx * dz).sum(dim=1)
    s_yz = (weights * dy * dz).sum(dim=1)

    determinant = s_xx * s_yy - s_xy.square()
    valid_fit = (weights > 0.0).sum(dim=1) >= 3
    valid_fit &= determinant > 1.0e-8
    safe_determinant = torch.where(valid_fit, determinant, torch.ones_like(determinant))
    slope_x = (s_yy * s_xz - s_xy * s_yz) / safe_determinant
    slope_y = (s_xx * s_yz - s_xy * s_xz) / safe_determinant

    terrain_normal_w = torch.stack((-slope_x, -slope_y, torch.ones_like(slope_x)), dim=1)
    terrain_normal_w = torch.nn.functional.normalize(terrain_normal_w, dim=1)
    world_up = torch.zeros_like(terrain_normal_w)
    world_up[:, 2] = 1.0
    terrain_normal_w = torch.where(valid_fit.unsqueeze(1), terrain_normal_w, world_up)

    terrain_normal_b = quat_apply_inverse(asset.data.root_quat_w, terrain_normal_w)
    # Body-x = pitch (fore-aft); body-y = roll (lateral). Scale roll heavier.
    tilt_xy = torch.stack(
        (terrain_normal_b[:, 0], lateral_scale * terrain_normal_b[:, 1]),
        dim=1,
    )
    return torch.atan2(
        torch.linalg.vector_norm(tilt_xy, dim=1),
        terrain_normal_b[:, 2],
    )


class terrain_level_progress(ManagerTermBase):
    """Dense reward for progress toward the terrain-level raise threshold.

    Each step returns the increase in max planar distance from the env origin,
    divided by half the usable terrain length (the ``move_up`` trigger distance).
    Values are scaled by ``1 / step_dt`` so that after RewardManager's ``* dt``
    the episode sum equals the coverage fraction (may exceed 1.0).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._prev_max_move_distance = torch.zeros(env.num_envs, device=env.device)
        terrain_cfg = env.scene.terrain.cfg.terrain_generator
        border = getattr(terrain_cfg, "sub_terrain_border_width", 0.0) or 0.0
        terrain_length = max(0.0, terrain_cfg.size[0] - 2.0 * border)
        self._half_terrain_length = max(terrain_length / 2.0, 1e-6)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._prev_max_move_distance[env_ids] = 0.0

    def __call__(
        self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        asset: RigidObject = env.scene[asset_cfg.name]
        current_dist = torch.norm(asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2], dim=1)
        new_max = torch.maximum(self._prev_max_move_distance, current_dist)
        delta = new_max - self._prev_max_move_distance
        self._prev_max_move_distance.copy_(new_max)
        # Compensate RewardManager's ``* dt`` so episode sum ≈ coverage fraction.
        return delta / (self._half_terrain_length * env.step_dt)
