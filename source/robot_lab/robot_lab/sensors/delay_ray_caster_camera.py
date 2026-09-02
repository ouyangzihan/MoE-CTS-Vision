"""Ray-caster camera with randomized output delay and Hiking-style depth processing."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCameraCfg
from isaaclab.sensors.ray_caster.ray_caster_camera import RayCasterCamera
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul


@dataclass
class DepthNoiseRuntime:
    """Mutable depth-corruption parameters (sensor-level)."""

    enabled: bool = False
    noise_std: float = 0.0
    dropout_prob: float = 0.0
    depth_dependent_noise_scale: float = 0.0
    edge_speckle_prob: float = 0.0
    temporal_flicker_std: float = 0.0
    hole_blob_prob: float = 0.0
    hole_blob_size_range: tuple[int, int] = (3, 12)
    dropout_fill_value: float | None = None
    randomize_dropout_fill_value: bool = False


def _gaussian_kernel1d(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (coords / max(sigma, 1e-6)) ** 2)
    return kernel / kernel.sum()


def gaussian_blur_depth(
    depth: torch.Tensor,
    kernel_size: int = 3,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Separable Gaussian blur for depth maps ``[B, H, W]``."""
    if kernel_size <= 1 or sigma <= 0.0:
        return depth
    radius = kernel_size // 2
    kernel_1d = _gaussian_kernel1d(kernel_size, sigma, depth.device, depth.dtype)
    x = depth.unsqueeze(1)
    x = F.pad(x, (radius, radius, radius, radius), mode="replicate")
    x = F.conv2d(x, kernel_1d.view(1, 1, 1, -1), padding=0)
    x = F.conv2d(x, kernel_1d.view(1, 1, -1, 1), padding=0)
    return x.squeeze(1)


def add_depth_noise(
    depth: torch.Tensor,
    max_depth: float,
    noise_std: float = 0.0,
    dropout_prob: float = 0.0,
    depth_dependent_noise_scale: float = 0.0,
    edge_speckle_prob: float = 0.0,
    temporal_flicker_std: float = 0.0,
    prev_depth: torch.Tensor | None = None,
    hole_blob_prob: float = 0.0,
    hole_blob_size_range: tuple[int, int] = (3, 12),
    dropout_fill_value: float | None = None,
    randomize_dropout_fill_value: bool = False,
) -> torch.Tensor:
    """Apply synthetic depth noise (Gaussian, dropout, and optional outdoor artifacts).

    Args:
        depth: Depth images ``[B, H, W]`` in meters (pre-normalization).
        max_depth: Far-plane / invalid fill value.
        noise_std: Base Gaussian std (meters).
        dropout_prob: Per-pixel Bernoulli hole probability.
        dropout_fill_value: Value written into dropout holes. ``None`` → ``max_depth``
            (legacy). Official MGDP uses ``0.0`` (near / invalid).
        randomize_dropout_fill_value: When True, each env independently uses ``0.0`` or
            ``max_depth`` (legacy ``None`` fill) for dropout holes with 50/50 probability.
        depth_dependent_noise_scale: Extra Gaussian std scale as
            ``noise_std * scale * (depth / max_depth)``.
        edge_speckle_prob: Probability of writing far values along strong depth edges.
        temporal_flicker_std: Blend previous noisy frame with Gaussian jitter.
        prev_depth: Previous noisy depth (same shape) for temporal flicker.
        hole_blob_prob: Per-env probability of stamping a contiguous far hole.
        hole_blob_size_range: Inclusive ``(min, max)`` blob side length in pixels.
    """
    noisy = depth
    if noise_std > 0.0 or depth_dependent_noise_scale > 0.0:
        gauss = torch.randn_like(noisy) * noise_std
        if depth_dependent_noise_scale > 0.0:
            gauss = gauss + torch.randn_like(noisy) * (
                noise_std * depth_dependent_noise_scale * (noisy / max(max_depth, 1e-6))
            )
        noisy = noisy + gauss

    if edge_speckle_prob > 0.0:
        dx = (noisy[..., :, 1:] - noisy[..., :, :-1]).abs()
        dy = (noisy[..., 1:, :] - noisy[..., :-1, :]).abs()
        edge = torch.zeros_like(noisy)
        edge[..., :, 1:] = torch.maximum(edge[..., :, 1:], dx)
        edge[..., :, :-1] = torch.maximum(edge[..., :, :-1], dx)
        edge[..., 1:, :] = torch.maximum(edge[..., 1:, :], dy)
        edge[..., :-1, :] = torch.maximum(edge[..., :-1, :], dy)
        edge_mask = (edge > 0.05 * max_depth) & (torch.rand_like(noisy) < edge_speckle_prob)
        noisy = torch.where(edge_mask, torch.full_like(noisy, max_depth), noisy)

    if hole_blob_prob > 0.0:
        b, h, w = noisy.shape
        blob_hit = torch.rand(b, device=noisy.device) < hole_blob_prob
        if torch.any(blob_hit):
            lo, hi = hole_blob_size_range
            hi = max(hi, lo)
            for env_i in torch.nonzero(blob_hit, as_tuple=False).flatten().tolist():
                side = int(torch.randint(lo, hi + 1, (1,), device=noisy.device).item())
                side = max(1, min(side, h, w))
                y0 = int(torch.randint(0, h - side + 1, (1,), device=noisy.device).item())
                x0 = int(torch.randint(0, w - side + 1, (1,), device=noisy.device).item())
                noisy[env_i, y0 : y0 + side, x0 : x0 + side] = max_depth

    if dropout_prob > 0.0:
        dropout_mask = torch.rand_like(noisy) < dropout_prob
        if randomize_dropout_fill_value:
            use_zero_fill = torch.rand(noisy.shape[0], 1, 1, device=noisy.device) < 0.5
            fill = torch.where(
                use_zero_fill,
                torch.zeros_like(noisy),
                torch.full_like(noisy, max_depth),
            )
            noisy = torch.where(dropout_mask, fill, noisy)
        else:
            fill = max_depth if dropout_fill_value is None else float(dropout_fill_value)
            noisy = noisy.masked_fill(dropout_mask, fill)

    if temporal_flicker_std > 0.0 and prev_depth is not None and prev_depth.shape == noisy.shape:
        mix = torch.rand(noisy.shape[0], 1, 1, device=noisy.device).clamp(0.0, 0.35)
        flicker = torch.randn_like(noisy) * temporal_flicker_std
        noisy = (1.0 - mix) * noisy + mix * (prev_depth + flicker)

    return noisy.clamp_(0.0, max_depth)


class DelayRayCasterCamera(RayCasterCamera):
    """RayCasterCamera with delay, optional sensor-level depth processing, and frame history."""

    cfg: "DelayRayCasterCameraCfg"

    def _initialize_rays_impl(self):
        super()._initialize_rays_impl()
        self._offset_pos_base = self._offset_pos[0].clone()
        self._offset_quat_base = self._offset_quat[0].clone()
        if self._uses_random_intrinsics():
            self._randomize_intrinsics(self._ALL_INDICES)
        if self.cfg.pos_randomization_range is not None:
            self._randomize_pos(self._ALL_INDICES)
        if self.cfg.rpy_randomization_deg is not None:
            self._randomize_rot(self._ALL_INDICES)

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)
        if not hasattr(self, "_ALL_INDICES"):
            return

        env_ids = self._resolve_env_ids(env_ids)
        if self.cfg.randomize_intrinsics_on_reset and self._uses_random_intrinsics():
            self._randomize_intrinsics(env_ids)
        if self.cfg.pos_randomization_range is not None and self.cfg.randomize_pos_on_reset:
            self._randomize_pos(env_ids)
        if self.cfg.rpy_randomization_deg is not None and self.cfg.randomize_rot_on_reset:
            self._randomize_rot(env_ids)
        if hasattr(self, "_delay_steps"):
            self._delay_steps[env_ids] = self._sample_delay_steps(len(env_ids))
            self._delay_write_index[env_ids] = 0
            self._delay_history_initialized[env_ids] = False
        if hasattr(self, "_processed_write_index"):
            self._processed_write_index[env_ids] = 0
            self._processed_history_initialized[env_ids] = False
            if hasattr(self, "_prev_noisy_depth"):
                self._prev_noisy_depth[env_ids] = 0.0

    def _create_buffers(self):
        super()._create_buffers()

        self._raw_output = {name: torch.zeros_like(value) for name, value in self._data.output.items()}
        sensor_dt = self.cfg.update_period if self.cfg.update_period > 0.0 else self._sim_physics_dt
        self._delay_sensor_dt = sensor_dt
        self._min_delay_steps = int(math.floor(self.cfg.min_delay / sensor_dt + 1e-9))
        self._max_delay_steps = int(math.floor(self.cfg.max_delay / sensor_dt + 1e-9))
        self._delay_history_length = self._max_delay_steps + 1
        self._delay_write_index = torch.zeros(self._view.count, dtype=torch.long, device=self._device)
        self._delay_steps = self._sample_delay_steps(self._view.count)
        self._delay_history_initialized = torch.zeros(self._view.count, dtype=torch.bool, device=self._device)
        self._delay_history = {
            name: torch.zeros(
                self._delay_history_length,
                self._view.count,
                *value.shape[1:],
                device=self._device,
                dtype=value.dtype,
            )
            for name, value in self._data.output.items()
        }

        self._runtime_noise = DepthNoiseRuntime(
            enabled=self.cfg.enable_sensor_noise,
            noise_std=self.cfg.sensor_noise_std,
            dropout_prob=self.cfg.sensor_dropout_prob,
            depth_dependent_noise_scale=self.cfg.sensor_depth_dependent_noise_scale,
            edge_speckle_prob=self.cfg.sensor_edge_speckle_prob,
            temporal_flicker_std=self.cfg.sensor_temporal_flicker_std,
            hole_blob_prob=self.cfg.sensor_hole_blob_prob,
            hole_blob_size_range=tuple(self.cfg.sensor_hole_blob_size_range),
            dropout_fill_value=self.cfg.sensor_dropout_fill_value,
            randomize_dropout_fill_value=self.cfg.sensor_randomize_dropout_fill_value,
        )
        self._env_cfg_ref = None

        if self.cfg.depth_history_length > 0:
            sample_shape = next(iter(self._data.output.values())).shape[1:]
            if len(sample_shape) == 1 and sample_shape[0] == 1:
                h = int(self.cfg.pattern_cfg.height)
                w = int(self.cfg.pattern_cfg.width)
                processed_shape = (h, w)
            elif len(sample_shape) >= 2:
                processed_shape = sample_shape[-2], sample_shape[-1]
            else:
                processed_shape = (int(self.cfg.pattern_cfg.height), int(self.cfg.pattern_cfg.width))

            self._processed_shape = processed_shape
            self._processed_history_length = int(self.cfg.depth_history_length)
            self._processed_write_index = torch.zeros(self._view.count, dtype=torch.long, device=self._device)
            self._processed_history_initialized = torch.zeros(self._view.count, dtype=torch.bool, device=self._device)
            self._processed_history = torch.zeros(
                self._processed_history_length,
                self._view.count,
                *processed_shape,
                device=self._device,
                dtype=torch.float32,
            )
            self._prev_noisy_depth = torch.zeros(self._view.count, *processed_shape, device=self._device, dtype=torch.float32)
            offsets = torch.arange(
                0,
                self.cfg.depth_num_output_frames * self.cfg.depth_history_skip_frames,
                self.cfg.depth_history_skip_frames,
                device=self._device,
            )
            self._processed_frame_offsets = torch.flip(offsets, dims=(0,))

    def bind_env_cfg(self, env_cfg) -> None:
        """Optional hook so sensor noise curriculum can read ``env.cfg`` overrides."""
        self._env_cfg_ref = env_cfg

    def sync_noise_from_env_cfg(self, env_cfg) -> None:
        """Copy depth-noise fields from the environment cfg onto runtime sensor params."""
        if not self.cfg.use_env_cfg_noise_overrides:
            return
        self._runtime_noise.enabled = True
        self._runtime_noise.noise_std = float(getattr(env_cfg, "depth_noise_std", self._runtime_noise.noise_std))
        self._runtime_noise.dropout_prob = float(
            getattr(env_cfg, "depth_dropout_prob", self._runtime_noise.dropout_prob)
        )
        self._runtime_noise.depth_dependent_noise_scale = float(
            getattr(env_cfg, "depth_dependent_noise_scale", self._runtime_noise.depth_dependent_noise_scale)
        )
        self._runtime_noise.edge_speckle_prob = float(
            getattr(env_cfg, "depth_edge_speckle_prob", self._runtime_noise.edge_speckle_prob)
        )
        self._runtime_noise.temporal_flicker_std = float(
            getattr(env_cfg, "depth_temporal_flicker_std", self._runtime_noise.temporal_flicker_std)
        )
        self._runtime_noise.hole_blob_prob = float(
            getattr(env_cfg, "depth_hole_blob_prob", self._runtime_noise.hole_blob_prob)
        )
        self._runtime_noise.hole_blob_size_range = tuple(
            getattr(env_cfg, "depth_hole_blob_size_range", self._runtime_noise.hole_blob_size_range)
        )
        if hasattr(env_cfg, "depth_dropout_fill_value"):
            self._runtime_noise.dropout_fill_value = getattr(env_cfg, "depth_dropout_fill_value")

    def _resolve_noise_params(self) -> DepthNoiseRuntime:
        if self.cfg.use_env_cfg_noise_overrides and self._env_cfg_ref is not None:
            self.sync_noise_from_env_cfg(self._env_cfg_ref)
        return self._runtime_noise

    def _depth_to_bhw(self, depth: torch.Tensor) -> torch.Tensor:
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth.squeeze(-1)
        elif depth.ndim == 4 and depth.shape[1] == 1:
            depth = depth.squeeze(1)
        elif depth.ndim == 2:
            depth = depth.reshape(depth.shape[0], *self._processed_shape)
        if depth.ndim != 3:
            raise ValueError(f"Expected depth [B,H,W], got {tuple(depth.shape)}")
        return depth.float()

    def _apply_sensor_pipeline(
        self,
        depth_bhw: torch.Tensor,
        env_ids: torch.Tensor,
        *,
        apply_noise: bool,
    ) -> torch.Tensor:
        norm_max = float(self.cfg.depth_norm_max)
        depth_bhw = torch.nan_to_num(depth_bhw, nan=norm_max, posinf=norm_max, neginf=0.0).clamp_(0.0, norm_max)

        if apply_noise:
            noise = self._resolve_noise_params()
            if noise.enabled:
                prev = self._prev_noisy_depth[env_ids] if noise.temporal_flicker_std > 0.0 else None
                depth_bhw = add_depth_noise(
                    depth_bhw,
                    max_depth=norm_max,
                    noise_std=noise.noise_std,
                    dropout_prob=noise.dropout_prob,
                    depth_dependent_noise_scale=noise.depth_dependent_noise_scale,
                    edge_speckle_prob=noise.edge_speckle_prob,
                    temporal_flicker_std=noise.temporal_flicker_std,
                    prev_depth=prev,
                    hole_blob_prob=noise.hole_blob_prob,
                    hole_blob_size_range=noise.hole_blob_size_range,
                    dropout_fill_value=noise.dropout_fill_value,
                    randomize_dropout_fill_value=noise.randomize_dropout_fill_value,
                )
                if noise.temporal_flicker_std > 0.0:
                    self._prev_noisy_depth[env_ids] = depth_bhw

        if self.cfg.gaussian_blur_sigma > 0.0:
            depth_bhw = gaussian_blur_depth(
                depth_bhw,
                kernel_size=self.cfg.gaussian_blur_kernel_size,
                sigma=self.cfg.gaussian_blur_sigma,
            )

        if self.cfg.depth_normalize:
            depth_bhw = depth_bhw / max(norm_max, 1e-6)
        return depth_bhw

    def _push_processed_history(self, processed_bhw: torch.Tensor, env_ids: torch.Tensor) -> None:
        write_index = self._processed_write_index[env_ids]
        uninitialized_mask = ~self._processed_history_initialized[env_ids]
        if uninitialized_mask.any():
            init_env_ids = env_ids[uninitialized_mask]
            init_values = processed_bhw[uninitialized_mask].unsqueeze(0).expand(
                self._processed_history_length,
                -1,
                *processed_bhw.shape[1:],
            )
            self._processed_history[:, init_env_ids] = init_values
        self._processed_history[write_index, env_ids] = processed_bhw
        self._processed_history_initialized[env_ids] = True
        self._processed_write_index[env_ids] = (write_index + 1) % self._processed_history_length

    def get_depth_history_stack(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Return normalized depth history ``[B, T, H, W]`` (oldest → newest)."""
        if not hasattr(self, "_processed_history"):
            raise RuntimeError("Processed depth history is disabled on this camera cfg.")
        env_ids = self._resolve_env_ids(env_ids)
        latest_index = (self._processed_write_index[env_ids] - 1) % self._processed_history_length
        frame_indices = (
            latest_index.unsqueeze(1) - self._processed_frame_offsets.unsqueeze(0)
        ) % self._processed_history_length
        batch_indices = env_ids.unsqueeze(1).expand(-1, frame_indices.shape[1])
        return self._processed_history[frame_indices, batch_indices]

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        super()._update_buffers_impl(env_ids)
        env_ids = self._resolve_env_ids(env_ids)
        write_index = self._delay_write_index[env_ids]
        read_index = (write_index - self._delay_steps[env_ids]) % self._delay_history_length
        uninitialized_mask = ~self._delay_history_initialized[env_ids]

        for name, history in self._delay_history.items():
            current = self._data.output[name][env_ids].clone()
            self._raw_output[name][env_ids] = current
            if uninitialized_mask.any():
                init_env_ids = env_ids[uninitialized_mask]
                init_current = current[uninitialized_mask].unsqueeze(0).expand(
                    self._delay_history_length,
                    -1,
                    *current.shape[1:],
                )
                history[:, init_env_ids] = init_current
            history[write_index, env_ids] = current
            self._data.output[name][env_ids] = history[read_index, env_ids]

        self._delay_history_initialized[env_ids] = True
        self._delay_write_index[env_ids] = (write_index + 1) % self._delay_history_length

        if hasattr(self, "_processed_history"):
            raw_depth = self._depth_to_bhw(self._raw_output["distance_to_image_plane"][env_ids])
            processed = self._apply_sensor_pipeline(raw_depth, env_ids, apply_noise=True)
            self._push_processed_history(processed, env_ids)

    def _resolve_env_ids(self, env_ids: Sequence[int] | None) -> torch.Tensor:
        if env_ids is None:
            return self._ALL_INDICES
        if isinstance(env_ids, slice):
            return self._ALL_INDICES[env_ids]
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self._device, dtype=torch.long)
        return torch.tensor(env_ids, device=self._device, dtype=torch.long)

    def _sample_delay_steps(self, num_envs: int) -> torch.Tensor:
        if self._max_delay_steps <= 0:
            return torch.zeros(num_envs, dtype=torch.long, device=self._device)
        low = max(0, self._min_delay_steps)
        high = self._max_delay_steps + 1
        if low >= high:
            return torch.full((num_envs,), self._max_delay_steps, dtype=torch.long, device=self._device)
        return torch.randint(low, high, (num_envs,), device=self._device)

    def get_output(self, data_type: str, use_delay: bool = True) -> torch.Tensor:
        """Return delayed or raw output for an enabled ray-caster camera data type."""
        data = self.data
        if use_delay or not hasattr(self, "_raw_output"):
            return data.output[data_type]
        return self._raw_output[data_type]

    def _uses_random_intrinsics(self) -> bool:
        return self.cfg.horizontal_fov_range is not None or self.cfg.vertical_fov_range is not None

    def _randomize_pos(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        env_ids = self._resolve_env_ids(env_ids)
        low, high = self.cfg.pos_randomization_range
        noise = torch.empty(len(env_ids), 3, device=self._device).uniform_(low, high)
        self._offset_pos[env_ids] = self._offset_pos_base + noise

    def _randomize_rot(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        env_ids = self._resolve_env_ids(env_ids)
        num_envs = len(env_ids)
        deg = float(self.cfg.rpy_randomization_deg)
        delta_rad = torch.empty(num_envs, 3, device=self._device).uniform_(
            -math.radians(deg),
            math.radians(deg),
        )
        delta_quat = quat_from_euler_xyz(delta_rad[:, 0], delta_rad[:, 1], delta_rad[:, 2])
        base = self._offset_quat_base.unsqueeze(0).expand(num_envs, -1)
        self._offset_quat[env_ids] = quat_mul(base, delta_quat)

    def _randomize_intrinsics(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        env_ids = self._resolve_env_ids(env_ids)
        intrinsic_matrices = self._sample_intrinsic_matrices(len(env_ids))
        self.set_intrinsic_matrices(intrinsic_matrices, focal_length=1.0, env_ids=env_ids)

    def _sample_intrinsic_matrices(self, num_envs: int) -> torch.Tensor:
        horizontal_fov_range = self.cfg.horizontal_fov_range
        vertical_fov_range = self.cfg.vertical_fov_range
        if horizontal_fov_range is None or vertical_fov_range is None:
            raise ValueError("Both horizontal_fov_range and vertical_fov_range must be set for FOV randomization.")

        horizontal_fov = torch.empty(num_envs, device=self._device).uniform_(*horizontal_fov_range)
        vertical_fov = torch.empty(num_envs, device=self._device).uniform_(*vertical_fov_range)
        intrinsic_width = float(self.cfg.intrinsic_width or self.cfg.pattern_cfg.width)
        intrinsic_height = float(self.cfg.intrinsic_height or self.cfg.pattern_cfg.height)
        crop_width = float(self.cfg.pattern_cfg.width)
        crop_height = float(self.cfg.pattern_cfg.height)

        fx = intrinsic_width / (2.0 * torch.tan(torch.deg2rad(horizontal_fov) * 0.5))
        fy = intrinsic_height / (2.0 * torch.tan(torch.deg2rad(vertical_fov) * 0.5))
        cx = crop_width * 0.5
        cy = crop_height * 0.5
        if self.cfg.principal_point_jitter > 0.0:
            cx = (crop_width + torch.empty(num_envs, device=self._device).uniform_(
                -self.cfg.principal_point_jitter,
                self.cfg.principal_point_jitter,
            )) * 0.5
            cy = (crop_height + torch.empty(num_envs, device=self._device).uniform_(
                -self.cfg.principal_point_jitter,
                self.cfg.principal_point_jitter,
            )) * 0.5
        else:
            cx = torch.full((num_envs,), cx, device=self._device)
            cy = torch.full((num_envs,), cy, device=self._device)

        intrinsic_matrices = torch.zeros(num_envs, 3, 3, device=self._device)
        intrinsic_matrices[:, 0, 0] = fx
        intrinsic_matrices[:, 0, 2] = cx
        intrinsic_matrices[:, 1, 1] = fy
        intrinsic_matrices[:, 1, 2] = cy
        intrinsic_matrices[:, 2, 2] = 1.0
        return intrinsic_matrices


@configclass
class DelayRayCasterCameraCfg(RayCasterCameraCfg):
    """Configuration for :class:`DelayRayCasterCamera`."""

    class_type: type = DelayRayCasterCamera

    min_delay: float = 0.0
    max_delay: float = 0.0
    horizontal_fov_range: tuple[float, float] | None = None
    vertical_fov_range: tuple[float, float] | None = None
    intrinsic_width: int | None = None
    intrinsic_height: int | None = None
    principal_point_jitter: float = 0.0
    randomize_intrinsics_on_reset: bool = False
    pos_randomization_range: tuple[float, float] | None = None
    randomize_pos_on_reset: bool = True
    rpy_randomization_deg: float | None = None
    randomize_rot_on_reset: bool = True

    depth_norm_max: float = 10.0
    """Clip/noise far-plane in meters before normalization."""
    depth_normalize: bool = True
    """Normalize clipped depth by ``depth_norm_max`` at the sensor."""
    gaussian_blur_sigma: float = 0.0
    gaussian_blur_kernel_size: int = 3

    enable_sensor_noise: bool = False
    use_env_cfg_noise_overrides: bool = False
    sensor_noise_std: float = 0.0
    sensor_dropout_prob: float = 0.0
    sensor_depth_dependent_noise_scale: float = 0.0
    sensor_edge_speckle_prob: float = 0.0
    sensor_temporal_flicker_std: float = 0.0
    sensor_hole_blob_prob: float = 0.0
    sensor_hole_blob_size_range: tuple[int, int] = (3, 12)
    sensor_dropout_fill_value: float | None = None
    sensor_randomize_dropout_fill_value: bool = False

    depth_history_length: int = 0
    """Ring buffer length for processed depth frames. ``0`` disables history stacking."""
    depth_num_output_frames: int = 1
    depth_history_skip_frames: int = 1

    def __post_init__(self):
        super().__post_init__()
        if self.min_delay < 0.0:
            raise ValueError(f"min_delay must be non-negative, got {self.min_delay}.")
        if self.max_delay < 0.0:
            raise ValueError(f"max_delay must be non-negative, got {self.max_delay}.")
        if self.max_delay < self.min_delay:
            raise ValueError(f"max_delay ({self.max_delay}) must be >= min_delay ({self.min_delay}).")
        if (self.horizontal_fov_range is None) != (self.vertical_fov_range is None):
            raise ValueError("horizontal_fov_range and vertical_fov_range must be set together.")
        if self.horizontal_fov_range is not None:
            if self.horizontal_fov_range[0] <= 0.0 or self.horizontal_fov_range[1] <= self.horizontal_fov_range[0]:
                raise ValueError(f"Invalid horizontal_fov_range: {self.horizontal_fov_range}.")
            if self.vertical_fov_range[0] <= 0.0 or self.vertical_fov_range[1] <= self.vertical_fov_range[0]:
                raise ValueError(f"Invalid vertical_fov_range: {self.vertical_fov_range}.")
        if self.principal_point_jitter < 0.0:
            raise ValueError(f"principal_point_jitter must be non-negative, got {self.principal_point_jitter}.")
        if self.pos_randomization_range is not None:
            if self.pos_randomization_range[1] < self.pos_randomization_range[0]:
                raise ValueError(f"Invalid pos_randomization_range: {self.pos_randomization_range}.")
        if self.rpy_randomization_deg is not None and self.rpy_randomization_deg < 0.0:
            raise ValueError(f"rpy_randomization_deg must be non-negative, got {self.rpy_randomization_deg}.")
        if self.depth_history_length > 0:
            min_history = (self.depth_num_output_frames - 1) * self.depth_history_skip_frames + 1
            if self.depth_history_length < min_history:
                raise ValueError(
                    f"depth_history_length ({self.depth_history_length}) must be >= {min_history} for "
                    f"{self.depth_num_output_frames} output frames with skip {self.depth_history_skip_frames}."
                )


DelayRayCaster = DelayRayCasterCamera
DelayRayCasterCfg = DelayRayCasterCameraCfg


def process_depth_image(
    env,
    sensor_cfg: SceneEntityCfg,
    data_type: str = "distance_to_image_plane",
    image_shape: tuple[int, int] | None = None,
    max_depth: float = 10.0,
    normalize: bool = True,
    use_delay: bool = True,
    enable_noise: bool = False,
    enable_augmentation: bool | None = None,
    use_history_stack: bool = True,
    num_output_frames: int | None = None,
    noise_std: float = 0.0,
    dropout_prob: float = 0.0,
    depth_dependent_noise_scale: float = 0.0,
    edge_speckle_prob: float = 0.0,
    temporal_flicker_std: float = 0.0,
    hole_blob_prob: float = 0.0,
    hole_blob_size_range: tuple[int, int] = (3, 12),
    use_cfg_noise_overrides: bool = False,
    dropout_fill_value: float | None = None,
    randomize_dropout_fill_value: bool = False,
) -> torch.Tensor:
    """Read ray-caster depth and flatten for the policy or aux losses.

    When the camera maintains a processed history stack, returns ``[B, T*H*W]`` with
    ``T`` oldest→newest frames already normalized/blurred/noised at sensor update.
    Observation-level noise is deprecated; corruption belongs on the sensor.
    """
    del (
        enable_noise,
        enable_augmentation,
        noise_std,
        dropout_prob,
        depth_dependent_noise_scale,
        edge_speckle_prob,
        temporal_flicker_std,
        hole_blob_prob,
        hole_blob_size_range,
        use_cfg_noise_overrides,
        dropout_fill_value,
        randomize_dropout_fill_value,
    )

    camera = env.scene.sensors[sensor_cfg.name]
    if hasattr(camera, "bind_env_cfg"):
        camera.bind_env_cfg(env.cfg)

    if use_history_stack and hasattr(camera, "get_depth_history_stack"):
        depth = camera.get_depth_history_stack()
        if num_output_frames is not None and depth.shape[1] != num_output_frames:
            raise ValueError(
                f"Expected {num_output_frames} history frames from sensor, got {depth.shape[1]}."
            )
        if image_shape is not None and tuple(depth.shape[-2:]) != tuple(image_shape):
            raise ValueError(f"Expected depth image shape {image_shape}, got {tuple(depth.shape[-2:])}.")
        return depth.flatten(1)

    if hasattr(camera, "get_output"):
        depth = camera.get_output(data_type, use_delay=use_delay).float()
    else:
        depth = camera.data.output[data_type].float()

    if depth.ndim == 4:
        if depth.shape[-1] == 1:
            depth = depth.squeeze(-1)
        elif depth.shape[1] == 1:
            depth = depth.squeeze(1)
    elif depth.ndim == 2 and image_shape is not None and depth.shape[-1] == image_shape[0] * image_shape[1]:
        depth = depth.reshape(depth.shape[0], *image_shape)
    if depth.ndim != 3:
        raise ValueError(f"Expected depth image with shape [B,H,W], got {tuple(depth.shape)}")

    depth = torch.nan_to_num(depth, nan=max_depth, posinf=max_depth, neginf=0.0).clamp_(0.0, max_depth)
    if image_shape is not None and tuple(depth.shape[-2:]) != tuple(image_shape):
        raise ValueError(f"Expected depth image shape {image_shape}, got {tuple(depth.shape[-2:])}.")

    if normalize:
        depth = depth / max_depth
    return depth.flatten(1)
