"""Edge-target based position → velocity command (Hiking-style)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.terrains import TerrainImporter
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, yaw_quat
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class PoseVelocityCommand(CommandTerm):
    """Velocity command from sub-terrain edge targets.

    On each resample, picks a point on the boundary of the env's terrain cell
    (one axis fixed at ±half-extent, the other uniform in ``[-half, half]``,
    corners allowed), raycasts for height, then converts body-frame pose error
    into ``(vx, ωz)`` (``vy`` forced to 0).

    Env modes (mutually exclusive): standing (zero cmd), reverse-into-target
    (negative ``vx``, heading error + π), or forward (positive ``vx``).
    """

    cfg: PoseVelocityCommandCfg

    def __init__(self, cfg: PoseVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.terrain: TerrainImporter = env.scene["terrain"]

        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.max_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_reverse_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._use_external_command = False
        self._terrain_wp_mesh = None

        if self.cfg.rel_standing_envs + self.cfg.rel_reverse_envs > 1.0:
            raise ValueError(
                "rel_standing_envs + rel_reverse_envs must be <= 1.0 "
                f"(got {self.cfg.rel_standing_envs} + {self.cfg.rel_reverse_envs})."
            )

        terrain_size = self.terrain.cfg.terrain_generator.size
        self._half_extent_x = 0.5 * float(terrain_size[0])
        self._half_extent_y = 0.5 * float(terrain_size[1])

        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["tracking_exp_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["tracking_exp_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["max_command_x"] = torch.zeros(self.num_envs, device=self.device)

        self.lin_vel_x_range = torch.zeros(self.num_envs, 2, device=self.device)
        self.lin_vel_y_range = torch.zeros(self.num_envs, 2, device=self.device)
        self.ang_vel_z_range = torch.zeros(self.num_envs, 2, device=self.device)

        self.random_lin_vel_x_range = torch.zeros(self.num_envs, 2, device=self.device)
        self.random_lin_vel_y_range = torch.zeros(self.num_envs, 2, device=self.device)
        self.random_ang_vel_z_range = torch.zeros(self.num_envs, 2, device=self.device)
        self.random_velocity_indices = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.random_lin_vel_x = torch.zeros(self.num_envs, device=self.device)
        self.random_lin_vel_y = torch.zeros(self.num_envs, device=self.device)
        self.random_ang_vel_z = torch.zeros(self.num_envs, device=self.device)

        self.command_ranges = {
            "lin_vel_x": [float(cfg.ranges.lin_vel_x[0]), float(cfg.ranges.lin_vel_x[1])],
            "lin_vel_y": [float(cfg.ranges.lin_vel_y[0]), float(cfg.ranges.lin_vel_y[1])],
            "ang_vel_z": [float(cfg.ranges.ang_vel_z[0]), float(cfg.ranges.ang_vel_z[1])],
        }
        self._start_command_ranges = {k: list(v) for k, v in self.command_ranges.items()}
        self._max_command_ranges = {
            "lin_vel_x": [float(cfg.command_range_max.lin_vel_x[0]), float(cfg.command_range_max.lin_vel_x[1])],
            "lin_vel_y": [float(cfg.command_range_max.lin_vel_y[0]), float(cfg.command_range_max.lin_vel_y[1])],
            "ang_vel_z": [float(cfg.command_range_max.ang_vel_z[0]), float(cfg.command_range_max.ang_vel_z[1])],
        }
        self._last_command_range_expand_count = -1
        self._terrain_cap_ranges: dict[str, dict[str, list[float]]] | None = None

        self._init_terrain_velocity_ranges()
        self._refresh_env_velocity_ranges()

    def __str__(self) -> str:
        msg = "PoseVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tEdge half-extent (x, y): ({self._half_extent_x}, {self._half_extent_y})\n"
        msg += f"\tCurrent ranges: {self.command_ranges}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """Policy / MDP command ``(vx, yaw)``. Shape (num_envs, 2)."""
        return self.vel_command_b[:, [0, 2]]

    @property
    def commands(self) -> torch.Tensor:
        """Internal 3D buffer ``(vx, vy=0, yaw)`` for teleop writers."""
        return self.vel_command_b

    def set_external_command(self, command: torch.Tensor) -> None:
        """Override pose-derived commands (used by gamepad / teleop).

        Accepts ``(vx, yaw)`` or ``(vx, vy, yaw)``; lateral ``vy`` is always zeroed.
        """
        self._use_external_command = True
        command = command.to(self.device).view(command.shape[0], -1)
        if command.shape[-1] == 2:
            self.vel_command_b[:, 0] = command[:, 0]
            self.vel_command_b[:, 2] = command[:, 1]
        elif command.shape[-1] == 3:
            self.vel_command_b[:] = command
        else:
            raise ValueError(f"Expected command dim 2 or 3, got {command.shape[-1]}.")
        self.vel_command_b[:, 1] = 0.0
        self.is_standing_env[:] = False
        self.is_reverse_env[:] = False

    def _ensure_terrain_wp_mesh(self):
        """Lazy-load the terrain warp mesh (shared with RayCaster when available)."""
        if self._terrain_wp_mesh is not None:
            return

        mesh_prim_path = self.terrain.cfg.prim_path
        try:
            from isaaclab.sensors import RayCaster

            if mesh_prim_path in RayCaster.meshes:
                self._terrain_wp_mesh = RayCaster.meshes[mesh_prim_path]
                return
        except Exception:
            pass

        import omni
        from pxr import UsdGeom

        mesh_prim = sim_utils.get_first_matching_child_prim(
            mesh_prim_path, lambda prim: prim.GetTypeName() == "Mesh"
        )
        if mesh_prim is None or not mesh_prim.IsValid():
            raise RuntimeError(f"PoseVelocityCommand could not find a Mesh under '{mesh_prim_path}'.")
        usd_mesh = UsdGeom.Mesh(mesh_prim)
        points = np.asarray(usd_mesh.GetPointsAttr().Get())
        transform_matrix = np.array(omni.usd.get_world_transform_matrix(mesh_prim)).T
        points = (transform_matrix[:3, :3] @ points.T + transform_matrix[:3, 3:4]).T
        indices = np.asarray(usd_mesh.GetFaceVertexIndicesAttr().Get())
        self._terrain_wp_mesh = convert_to_warp_mesh(points, indices, device=self.device)

    def _sample_edge_targets(self, env_ids: Sequence[int]) -> None:
        """Sample targets on the full sub-terrain cell boundary and raycast for z."""
        n = len(env_ids)
        origins = self._env.scene.env_origins[env_ids]

        # Fix either x or y at ±half-extent; sample the other in [-half, half] (corners kept).
        fix_x = torch.rand(n, device=self.device) < 0.5
        sign = torch.where(torch.rand(n, device=self.device) < 0.5, -1.0, 1.0)
        other = torch.empty(n, device=self.device).uniform_(-1.0, 1.0)

        offset_x = torch.where(fix_x, sign * self._half_extent_x, other * self._half_extent_x)
        offset_y = torch.where(fix_x, other * self._half_extent_y, sign * self._half_extent_y)

        self.pos_command_w[env_ids, 0] = origins[:, 0] + offset_x
        self.pos_command_w[env_ids, 1] = origins[:, 1] + offset_y
        self.pos_command_w[env_ids, 2] = origins[:, 2]

        self._ensure_terrain_wp_mesh()
        ray_starts = self.pos_command_w[env_ids].clone()
        ray_starts[:, 2] = origins[:, 2] + 100.0
        ray_dirs = torch.zeros_like(ray_starts)
        ray_dirs[:, 2] = -1.0
        hits = raycast_mesh(ray_starts, ray_dirs, self._terrain_wp_mesh)[0]
        valid = torch.isfinite(hits[:, 2])
        self.pos_command_w[env_ids, 2] = torch.where(valid, hits[:, 2], origins[:, 2])

    def _init_terrain_velocity_ranges(self):
        """Build per-column terrain velocity caps from ``cfg.velocity_ranges``."""
        if self.cfg.velocity_ranges is None:
            return

        terrain_generator_cfg = self.terrain.cfg.terrain_generator
        proportions = np.array([sub_cfg.proportion for sub_cfg in terrain_generator_cfg.sub_terrains.values()])
        proportions = proportions / np.sum(proportions)

        sub_indices = []
        for index in range(terrain_generator_cfg.num_cols):
            sub_index = np.min(np.where(index / terrain_generator_cfg.num_cols + 0.001 < np.cumsum(proportions))[0])
            sub_indices.append(sub_index)
        sub_indices = np.array(sub_indices, dtype=np.int32)
        sub_terrains_names = list(terrain_generator_cfg.sub_terrains.keys())

        self._terrain_cap_ranges = {}
        for key, value in self.cfg.velocity_ranges.items():
            if key not in sub_terrains_names:
                # Play scripts may drop zero-proportion terrains while leaving caps in cfg.
                print(f"[WARN] Skipping velocity_ranges for missing terrain type '{key}'.")
                continue
            self._terrain_cap_ranges[key] = {
                "lin_vel_x": [float(value["lin_vel_x"][0]), float(value["lin_vel_x"][1])],
                "lin_vel_y": [float(value["lin_vel_y"][0]), float(value["lin_vel_y"][1])],
                "ang_vel_z": [float(value["ang_vel_z"][0]), float(value["ang_vel_z"][1])],
            }

        if self.cfg.random_velocity_terrain is not None:
            for key in self.cfg.random_velocity_terrain:
                if key not in sub_terrains_names:
                    print(f"[WARN] Skipping random_velocity_terrain for missing terrain type '{key}'.")
                    continue
                terrain_type_index = sub_terrains_names.index(key)
                type_indices = np.where(sub_indices == terrain_type_index)[0]
                for type_indice in type_indices:
                    env_indices = torch.where(self.terrain.terrain_types == type_indice)[0]
                    self.random_velocity_indices[env_indices] = True

        self._col_terrain_names = []
        for col in range(terrain_generator_cfg.num_cols):
            self._col_terrain_names.append(sub_terrains_names[sub_indices[col]])

    def _refresh_env_velocity_ranges(self):
        """Apply global curriculum ranges, then intersect with per-terrain caps."""
        self.lin_vel_x_range[:, 0] = self.command_ranges["lin_vel_x"][0]
        self.lin_vel_x_range[:, 1] = self.command_ranges["lin_vel_x"][1]
        self.lin_vel_y_range[:, 0] = self.command_ranges["lin_vel_y"][0]
        self.lin_vel_y_range[:, 1] = self.command_ranges["lin_vel_y"][1]
        self.ang_vel_z_range[:, 0] = self.command_ranges["ang_vel_z"][0]
        self.ang_vel_z_range[:, 1] = self.command_ranges["ang_vel_z"][1]

        if self._terrain_cap_ranges is not None:
            for name, caps in self._terrain_cap_ranges.items():
                col_ids = [i for i, n in enumerate(self._col_terrain_names) if n == name]
                if not col_ids:
                    continue
                col_ids_t = torch.tensor(col_ids, device=self.device, dtype=torch.long)
                env_mask = torch.isin(self.terrain.terrain_types, col_ids_t)
                if not torch.any(env_mask):
                    continue
                self.lin_vel_x_range[env_mask, 0] = torch.maximum(
                    self.lin_vel_x_range[env_mask, 0],
                    torch.full((), caps["lin_vel_x"][0], device=self.device),
                )
                self.lin_vel_x_range[env_mask, 1] = torch.minimum(
                    self.lin_vel_x_range[env_mask, 1],
                    torch.full((), caps["lin_vel_x"][1], device=self.device),
                )
                self.lin_vel_y_range[env_mask, 0] = torch.maximum(
                    self.lin_vel_y_range[env_mask, 0],
                    torch.full((), caps["lin_vel_y"][0], device=self.device),
                )
                self.lin_vel_y_range[env_mask, 1] = torch.minimum(
                    self.lin_vel_y_range[env_mask, 1],
                    torch.full((), caps["lin_vel_y"][1], device=self.device),
                )
                self.ang_vel_z_range[env_mask, 0] = torch.maximum(
                    self.ang_vel_z_range[env_mask, 0],
                    torch.full((), caps["ang_vel_z"][0], device=self.device),
                )
                self.ang_vel_z_range[env_mask, 1] = torch.minimum(
                    self.ang_vel_z_range[env_mask, 1],
                    torch.full((), caps["ang_vel_z"][1], device=self.device),
                )

        self.random_lin_vel_x_range[:, 0] = self.command_ranges["lin_vel_x"][0]
        self.random_lin_vel_x_range[:, 1] = self.command_ranges["lin_vel_x"][1]
        self.random_lin_vel_y_range[:, 0] = self.command_ranges["lin_vel_y"][0]
        self.random_lin_vel_y_range[:, 1] = self.command_ranges["lin_vel_y"][1]
        self.random_ang_vel_z_range[:, 0] = self.command_ranges["ang_vel_z"][0]
        self.random_ang_vel_z_range[:, 1] = self.command_ranges["ang_vel_z"][1]

    def _update_command_range_curriculum(self):
        """Expand global command ranges over training iterations."""
        interval = self.cfg.command_range_expand_interval
        if interval is None or interval <= 0:
            return

        current_iter = self._env.common_step_counter // self.cfg.num_steps_per_iter
        expand_count = current_iter // interval
        if expand_count == self._last_command_range_expand_count:
            return
        self._last_command_range_expand_count = expand_count

        expand = self.cfg.command_range_expand
        updated = False
        for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
            delta = float(getattr(expand, key)) * expand_count
            if key == "lin_vel_x" and self.cfg.only_positive_lin_vel_x:
                low = self._start_command_ranges[key][0]
                high = self._start_command_ranges[key][1] + delta
            elif key == "lin_vel_y" and self.cfg.only_positive_lin_vel_x:
                low = self._start_command_ranges[key][0]
                high = self._start_command_ranges[key][1]
            else:
                low = self._start_command_ranges[key][0] - delta
                high = self._start_command_ranges[key][1] + delta
            low = max(low, float(self._max_command_ranges[key][0]))
            high = min(high, float(self._max_command_ranges[key][1]))
            new_range = [low, high]
            if new_range != list(self.command_ranges[key]):
                self.command_ranges[key] = new_range
                updated = True

        if updated:
            self.cfg.ranges.lin_vel_x = tuple(self.command_ranges["lin_vel_x"])
            self.cfg.ranges.lin_vel_y = tuple(self.command_ranges["lin_vel_y"])
            self.cfg.ranges.ang_vel_z = tuple(self.command_ranges["ang_vel_z"])
            self._refresh_env_velocity_ranges()
            print(f"[PoseVelocityCommand] ranges updated at iter {current_iter}: {self.command_ranges}")

    def _update_metrics(self):
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )
        lin_vel_error = torch.sum(
            torch.square(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2]),
            dim=1,
        )
        self.metrics["tracking_exp_vel_xy"] += (
            torch.exp(-lin_vel_error / self.cfg.lin_vel_metrics_std**2) / self._env.max_episode_length
        )
        angular_vel_error = torch.square(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        self.metrics["tracking_exp_vel_yaw"] += (
            torch.exp(-angular_vel_error / self.cfg.ang_vel_metrics_std**2) / self._env.max_episode_length
        )
        self.metrics["max_command_x"][:] = self.command_ranges["lin_vel_x"][1]

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        self._update_command_range_curriculum()
        self._sample_edge_targets(env_ids)

        r = torch.empty(len(env_ids), device=self.device)
        self.max_command_b[env_ids, 0] = self.lin_vel_x_range[env_ids, 0] + r.uniform_(0.0, 1.0) * (
            self.lin_vel_x_range[env_ids, 1] - self.lin_vel_x_range[env_ids, 0]
        )
        self.max_command_b[env_ids, 1] = self.lin_vel_y_range[env_ids, 0] + r.uniform_(0.0, 1.0) * (
            self.lin_vel_y_range[env_ids, 1] - self.lin_vel_y_range[env_ids, 0]
        )
        self.max_command_b[env_ids, 2] = self.ang_vel_z_range[env_ids, 0] + r.uniform_(0.0, 1.0) * (
            self.ang_vel_z_range[env_ids, 1] - self.ang_vel_z_range[env_ids, 0]
        )
        self.max_command_b[env_ids, 0] = torch.abs(self.max_command_b[env_ids, 0])
        self.max_command_b[env_ids, 1] = torch.abs(self.max_command_b[env_ids, 1])
        self.max_command_b[env_ids, 2] = torch.abs(self.max_command_b[env_ids, 2])

        # Mutually exclusive modes: standing, reverse-into-target, forward.
        # e.g. rel_standing=0.1, rel_reverse=0.2 → 10% stand, 20% reverse, 70% forward.
        mode_u = r.uniform_(0.0, 1.0)
        standing = mode_u < self.cfg.rel_standing_envs
        reverse = (~standing) & (mode_u < self.cfg.rel_standing_envs + self.cfg.rel_reverse_envs)
        self.is_standing_env[env_ids] = standing
        self.is_reverse_env[env_ids] = reverse

        current_batch_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        current_batch_mask[env_ids] = True
        update_mask = current_batch_mask & self.random_velocity_indices
        random_velocity_env_ids = update_mask.nonzero(as_tuple=False).flatten()

        if len(random_velocity_env_ids) > 0:
            self.random_lin_vel_x[random_velocity_env_ids] = self.random_lin_vel_x_range[
                random_velocity_env_ids, 0
            ] + torch.rand(len(random_velocity_env_ids), device=self.device) * (
                self.random_lin_vel_x_range[random_velocity_env_ids, 1]
                - self.random_lin_vel_x_range[random_velocity_env_ids, 0]
            )
            self.random_lin_vel_y[random_velocity_env_ids] = self.random_lin_vel_y_range[
                random_velocity_env_ids, 0
            ] + torch.rand(len(random_velocity_env_ids), device=self.device) * (
                self.random_lin_vel_y_range[random_velocity_env_ids, 1]
                - self.random_lin_vel_y_range[random_velocity_env_ids, 0]
            )
            self.random_ang_vel_z[random_velocity_env_ids] = self.random_ang_vel_z_range[
                random_velocity_env_ids, 0
            ] + torch.rand(len(random_velocity_env_ids), device=self.device) * (
                self.random_ang_vel_z_range[random_velocity_env_ids, 1]
                - self.random_ang_vel_z_range[random_velocity_env_ids, 0]
            )
            self.random_ang_vel_z *= torch.abs(self.random_ang_vel_z) > 0.5

    def _update_command(self):
        if self._use_external_command:
            return

        target_vec = self.pos_command_w - self.robot.data.root_pos_w[:, :3]
        target_dist = torch.norm(target_vec[:, :2], dim=1)
        self.pos_command_b[:] = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec)

        # Forward: +body-x toward target. Reverse: always command −|body-x| so the
        # robot backs while yaw aligns the rear toward the target.
        vx_body = self.pos_command_b[:, 0] * self.cfg.velocity_control_stiffness
        self.vel_command_b[:, 0] = torch.where(
            self.is_reverse_env, -torch.abs(vx_body), vx_body
        )
        self.vel_command_b[:, 1] = 0.0

        target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
        # Reverse uses opposite heading so the body −x axis points at the target.
        heading_error = wrap_to_pi(target_direction - self.robot.data.heading_w)
        heading_error = torch.where(
            self.is_reverse_env, wrap_to_pi(heading_error + torch.pi), heading_error
        )
        self.heading_command_w = heading_error
        self.vel_command_b[:, 2] = self.heading_command_w * self.cfg.heading_control_stiffness

        if self.cfg.only_positive_lin_vel_x:
            # Forward envs: [0, +max]. Reverse envs: [−reverse_cap, 0] (yaw uncapped vs forward).
            reverse_max_x = torch.minimum(
                self.max_command_b[:, 0],
                torch.full_like(self.max_command_b[:, 0], self.cfg.reverse_lin_vel_x_abs_max),
            )
            min_x = torch.where(
                self.is_reverse_env,
                -reverse_max_x,
                torch.zeros_like(self.max_command_b[:, 0]),
            )
            max_x = torch.where(
                self.is_reverse_env,
                torch.zeros_like(self.max_command_b[:, 0]),
                self.max_command_b[:, 0],
            )
        else:
            min_x = -self.max_command_b[:, 0]
            max_x = self.max_command_b[:, 0]
        self.vel_command_b[:, 0] = torch.clamp(self.vel_command_b[:, 0], min=min_x, max=max_x)
        self.vel_command_b[:, 2] = torch.clamp(
            self.vel_command_b[:, 2],
            -self.max_command_b[:, 2],
            self.max_command_b[:, 2],
        )
        self.vel_command_b[:] *= (target_dist > self.cfg.target_dis_threshold).unsqueeze(-1)
        self.vel_command_b[:, 0] *= (torch.abs(self.vel_command_b[:, 0]) > self.cfg.lin_vel_threshold).float()
        self.vel_command_b[:, 2] *= (torch.abs(self.vel_command_b[:, 2]) > self.cfg.ang_vel_threshold).float()
        self.vel_command_b[:, 1] = 0.0

        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

        random_velocity_env_ids = self.random_velocity_indices.nonzero(as_tuple=False).flatten()
        self.vel_command_b[random_velocity_env_ids, 0] = self.random_lin_vel_x[random_velocity_env_ids]
        self.vel_command_b[random_velocity_env_ids, 1] = 0.0
        self.vel_command_b[random_velocity_env_ids, 2] = self.random_ang_vel_z[random_velocity_env_ids]

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "flat_patch_visualizer"):
                self.cfg.flat_patch_visualizer_cfg.markers["Goal"].radius = self.cfg.target_dis_threshold
                self.flat_patch_visualizer = VisualizationMarkers(self.cfg.flat_patch_visualizer_cfg)
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.flat_patch_visualizer.set_visibility(True)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "flat_patch_visualizer"):
                self.flat_patch_visualizer.set_visibility(False)
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        marker_indices = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.flat_patch_visualizer.visualize(self.pos_command_w, marker_indices=marker_indices)

        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.vel_command_b[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 8.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        arrow_quat = math_utils.quat_mul(self.robot.data.root_quat_w, arrow_quat)
        return arrow_scale, arrow_quat


@configclass
class PoseVelocityCommandCfg(CommandTermCfg):
    """Configuration for edge-target based velocity commands."""

    class_type: type = PoseVelocityCommand

    asset_name: str = "robot"
    velocity_control_stiffness: float = 1.0
    heading_control_stiffness: float = 1.0
    only_positive_lin_vel_x: bool = True

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = MISSING
        lin_vel_y: tuple[float, float] = MISSING
        ang_vel_z: tuple[float, float] = MISSING

    ranges: Ranges = MISSING

    @configclass
    class CommandRangeExpandCfg:
        lin_vel_x: float = 0.3
        lin_vel_y: float = 0.0
        ang_vel_z: float = 0.2

    command_range_expand: CommandRangeExpandCfg = CommandRangeExpandCfg()
    command_range_expand_interval: int | None = 2000

    @configclass
    class CommandRangeMaxCfg:
        lin_vel_x: tuple[float, float] = (0.0, 2.0)
        lin_vel_y: tuple[float, float] = (0.0, 0.0)
        ang_vel_z: tuple[float, float] = (-2.0, 2.0)

    command_range_max: CommandRangeMaxCfg = CommandRangeMaxCfg()
    num_steps_per_iter: int = 24

    random_velocity_terrain: list[str] | None = None
    velocity_ranges: dict | None = None

    lin_vel_threshold: float = 0.15
    ang_vel_threshold: float = 0.15
    lin_vel_metrics_std: float = 0.5
    ang_vel_metrics_std: float = 0.5
    rel_standing_envs: float = 0.0
    """Fraction of envs that receive zero velocity commands (stand still)."""
    rel_reverse_envs: float = 0.0
    """Fraction of envs that reverse into the edge target (negative vx, heading + π).

    Standing and reverse are mutually exclusive and sampled from the same uniform draw,
    so forward fraction is ``1 - rel_standing_envs - rel_reverse_envs``.
    """
    reverse_lin_vel_x_abs_max: float = 1.0
    """Absolute |vx| cap for reverse envs (clamp to [-this, 0]). Yaw uses the normal max."""
    target_dis_threshold: float = 0.2

    flat_patch_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/TerrainFlatPatches",
        markers={
            "Goal": sim_utils.CylinderCfg(
                radius=0.15,
                height=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
    )
    patch_vis: bool = False
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
