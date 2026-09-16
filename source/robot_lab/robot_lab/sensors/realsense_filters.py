"""Batched PyTorch port of Intel librealsense D400 post-processing.

Mirrors ``src/proc/{spatial,temporal,hole-filling,disparity-transform}-filter``
and the deploy order in ``rl_sar`` ``RealSenseDepthSource::GrabPreprocessed``:

    [decimation] → disparity → spatial → temporal → depth → [hole filling]

Invalid pixels are depth == 0 (RealSense Z16 zero). Far-plane values stay valid.
Remaining holes after the stack are filled with ``max_depth``, matching deploy.

CUDA uses blocked Triton kernels for the recursive 1D scans; CPU uses a
PyTorch fallback with the same math.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

# Intel spatial holes_fill mode → pixel radius (0=off, 5=unlimited).
_SPATIAL_HOLE_RADIUS = (0, 2, 4, 8, 16, 255)
# disparity = (baseline_m * fx_px * 32) / depth_m  (5 fractional bits).
_DISPARITY_FRACTIONS = 32.0
_TRITON_BLOCK = 128


def spatial_holes_fill_radius(mode: int) -> int:
    mode = int(mode)
    if mode <= 0:
        return 0
    if mode >= len(_SPATIAL_HOLE_RADIUS):
        return _SPATIAL_HOLE_RADIUS[-1]
    return _SPATIAL_HOLE_RADIUS[mode]


def build_persistence_map(persistence_param: int) -> list[int]:
    """Port of ``temporal_filter::recalc_persistence_map`` (256-entry LUT)."""
    persistence_param = int(persistence_param)
    raw = [0] * 256
    for i in range(256):
        last_7 = int(bool(i & 1))
        last_6 = int(bool(i & 2))
        last_5 = int(bool(i & 4))
        last_4 = int(bool(i & 8))
        last_3 = int(bool(i & 16))
        last_2 = int(bool(i & 32))
        last_1 = int(bool(i & 64))
        last_frame = int(bool(i & 128))
        ok = False
        if persistence_param == 1:
            ok = (last_frame + last_1 + last_2 + last_3 + last_4 + last_5 + last_6 + last_7) >= 8
        elif persistence_param == 2:
            ok = (last_frame + last_1 + last_2) >= 2
        elif persistence_param == 3:
            ok = (last_frame + last_1 + last_2 + last_3) >= 2
        elif persistence_param == 4:
            ok = (last_frame + last_1 + last_2 + last_3 + last_4 + last_5 + last_6 + last_7) >= 2
        elif persistence_param == 5:
            ok = (last_frame + last_1) >= 1
        elif persistence_param == 6:
            ok = (last_frame + last_1 + last_2 + last_3 + last_4) >= 1
        elif persistence_param == 7:
            ok = (last_frame + last_1 + last_2 + last_3 + last_4 + last_5 + last_6 + last_7) >= 1
        elif persistence_param == 8:
            ok = True
        raw[i] = 1 if ok else 0

    credible = [0] * 256
    for phase in range(8):
        mask = 1 << phase
        shift = 8 - phase
        for i in range(256):
            pos = ((i << shift) | (i >> phase)) & 255
            if raw[pos]:
                credible[i] |= mask
    return credible


def disparity_convert_factor(baseline_m: float, fx_px: float) -> float:
    """``d2d_convert_factor`` such that ``disparity = factor / depth_m``."""
    return float(baseline_m) * float(fx_px) * _DISPARITY_FRACTIONS


def depth_to_disparity(depth_m: torch.Tensor, factor: float) -> torch.Tensor:
    """Meters → DISPARITY32. Zero/non-finite depth becomes 0 (invalid)."""
    valid = torch.isfinite(depth_m) & (depth_m > 0)
    return torch.where(valid, depth_m.new_tensor(factor) / depth_m.clamp(min=1e-8), torch.zeros_like(depth_m))


def disparity_to_depth(disparity: torch.Tensor, factor: float) -> torch.Tensor:
    """DISPARITY32 → meters. Zero/non-finite disparity becomes 0 (invalid)."""
    valid = torch.isfinite(disparity) & (disparity > 0)
    return torch.where(
        valid, disparity.new_tensor(factor) / disparity.clamp(min=1e-8), torch.zeros_like(disparity)
    )


def _use_triton(tensor: torch.Tensor) -> bool:
    return _TRITON_AVAILABLE and tensor.is_cuda


if _TRITON_AVAILABLE:

    @triton.jit
    def _ema_row_bidir_kernel(ptr, n_rows, length, alpha, delta, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BLOCK + tl.arange(0, BLOCK)
        mask = rows < n_rows
        row_off = rows * length
        first = tl.load(ptr + row_off, mask=mask, other=0.0)
        state = first
        prev_orig = first
        in_valid = first > 0
        for i in range(1, length):
            curr = tl.load(ptr + row_off + i, mask=mask, other=0.0)
            curr_valid = curr > 0
            close = in_valid & curr_valid & (tl.abs(prev_orig - curr) < delta)
            filtered = curr * alpha + state * (1.0 - alpha)
            out_val = tl.where(close, filtered, curr)
            tl.store(ptr + row_off + i, out_val, mask=mask)
            state = tl.where(close, filtered, tl.where(curr_valid, curr, state))
            in_valid = curr_valid
            prev_orig = curr
        last = tl.load(ptr + row_off + (length - 1), mask=mask, other=0.0)
        state = last
        prev_orig = last
        in_valid = last > 0
        for i in range(1, length):
            idx = length - 1 - i
            curr = tl.load(ptr + row_off + idx, mask=mask, other=0.0)
            curr_valid = curr > 0
            close = in_valid & curr_valid & (tl.abs(prev_orig - curr) < delta)
            filtered = curr * alpha + state * (1.0 - alpha)
            out_val = tl.where(close, filtered, curr)
            tl.store(ptr + row_off + idx, out_val, mask=mask)
            state = tl.where(close, filtered, tl.where(curr_valid, curr, state))
            in_valid = curr_valid
            prev_orig = curr

    @triton.jit
    def _ema_col_bidir_kernel(ptr, n_cols, height, width, alpha, delta, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        cols = pid * BLOCK + tl.arange(0, BLOCK)
        mask = cols < n_cols
        batch = cols // width
        x = cols - batch * width
        base = batch * height * width + x
        first = tl.load(ptr + base, mask=mask, other=0.0)
        state = first
        prev_orig = first
        in_valid = first > 0
        for y in range(1, height):
            curr = tl.load(ptr + base + y * width, mask=mask, other=0.0)
            curr_valid = curr > 0
            close = in_valid & curr_valid & (tl.abs(prev_orig - curr) < delta)
            filtered = curr * alpha + state * (1.0 - alpha)
            out_val = tl.where(close, filtered, curr)
            tl.store(ptr + base + y * width, out_val, mask=mask)
            state = tl.where(close, filtered, tl.where(curr_valid, curr, state))
            in_valid = curr_valid
            prev_orig = curr
        last = tl.load(ptr + base + (height - 1) * width, mask=mask, other=0.0)
        state = last
        prev_orig = last
        in_valid = last > 0
        for y in range(1, height):
            idx = height - 1 - y
            curr = tl.load(ptr + base + idx * width, mask=mask, other=0.0)
            curr_valid = curr > 0
            close = in_valid & curr_valid & (tl.abs(prev_orig - curr) < delta)
            filtered = curr * alpha + state * (1.0 - alpha)
            out_val = tl.where(close, filtered, curr)
            tl.store(ptr + base + idx * width, out_val, mask=mask)
            state = tl.where(close, filtered, tl.where(curr_valid, curr, state))
            in_valid = curr_valid
            prev_orig = curr

    @triton.jit
    def _inertial_fill_kernel(ptr, n_rows, width, radius, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BLOCK + tl.arange(0, BLOCK)
        mask = rows < n_rows
        row_off = rows * width
        cur_fill = tl.zeros([BLOCK], dtype=tl.int32)
        for x in range(1, width):
            val = tl.load(ptr + row_off + x, mask=mask, other=1.0)
            empty = val == 0
            cur_fill = tl.where(empty, cur_fill + 1, tl.zeros([BLOCK], dtype=tl.int32))
            fill = empty & (cur_fill < radius)
            left = tl.load(ptr + row_off + (x - 1), mask=mask, other=0.0)
            tl.store(ptr + row_off + x, tl.where(fill, left, val), mask=mask)
        cur_fill = tl.zeros([BLOCK], dtype=tl.int32)
        for x in range(1, width):
            idx = width - 1 - x
            val = tl.load(ptr + row_off + idx, mask=mask, other=1.0)
            empty = val == 0
            cur_fill = tl.where(empty, cur_fill + 1, tl.zeros([BLOCK], dtype=tl.int32))
            fill = empty & (cur_fill < radius)
            right = tl.load(ptr + row_off + (idx + 1), mask=mask, other=0.0)
            tl.store(ptr + row_off + idx, tl.where(fill, right, val), mask=mask)

    @triton.jit
    def _holes_fill_left_kernel(ptr, n_rows, width, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BLOCK + tl.arange(0, BLOCK)
        mask = rows < n_rows
        row_off = rows * width
        for x in range(1, width):
            val = tl.load(ptr + row_off + x, mask=mask, other=1.0)
            left = tl.load(ptr + row_off + (x - 1), mask=mask, other=0.0)
            tl.store(ptr + row_off + x, tl.where(val == 0, left, val), mask=mask)

    @triton.jit
    def _holes_fill_farest_kernel(ptr, batch, height, width, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        batches = pid * BLOCK + tl.arange(0, BLOCK)
        mask = batches < batch
        for y in range(1, height - 1):
            for x in range(1, width):
                p = batches * height * width + y * width + x
                val = tl.load(ptr + p, mask=mask, other=1.0)
                tmp = tl.load(ptr + p - width, mask=mask, other=0.0)
                tmp = tl.maximum(tmp, tl.load(ptr + p - width - 1, mask=mask, other=0.0))
                tmp = tl.maximum(tmp, tl.load(ptr + p - 1, mask=mask, other=0.0))
                tmp = tl.maximum(tmp, tl.load(ptr + p + width - 1, mask=mask, other=0.0))
                tmp = tl.maximum(tmp, tl.load(ptr + p + width, mask=mask, other=0.0))
                tl.store(ptr + p, tl.where(val == 0, tmp, val), mask=mask)

    @triton.jit
    def _holes_fill_nearest_kernel(ptr, batch, height, width, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        batches = pid * BLOCK + tl.arange(0, BLOCK)
        mask = batches < batch
        for y in range(1, height - 1):
            for x in range(1, width):
                p = batches * height * width + y * width + x
                val = tl.load(ptr + p, mask=mask, other=1.0)
                tmp = tl.load(ptr + p - width, mask=mask, other=0.0)
                q = tl.load(ptr + p - width - 1, mask=mask, other=0.0)
                tmp = tl.where((q != 0) & (q < tmp), q, tmp)
                q = tl.load(ptr + p - 1, mask=mask, other=0.0)
                tmp = tl.where((q != 0) & (q < tmp), q, tmp)
                q = tl.load(ptr + p + width - 1, mask=mask, other=0.0)
                tmp = tl.where((q != 0) & (q < tmp), q, tmp)
                q = tl.load(ptr + p + width, mask=mask, other=0.0)
                tmp = tl.where((q != 0) & (q < tmp), q, tmp)
                tl.store(ptr + p, tl.where(val == 0, tmp, val), mask=mask)


def _launch_blocked(n_items: int, kernel, *args):
    grid = (triton.cdiv(n_items, _TRITON_BLOCK),)
    kernel[grid](*args, BLOCK=_TRITON_BLOCK)


def _ema_scan_torch(lines: torch.Tensor, alpha: float, delta: float) -> torch.Tensor:
    """One-way domain-transform EMA on ``[N, L]`` (Intel ``recursive_filter_*_fp``)."""
    length = lines.shape[-1]
    if length <= 1:
        return lines.clone()
    src = lines
    out = lines.clone()
    valid = src > 0
    state = src[:, 0].clone()
    in_valid = valid[:, 0].clone()
    prev_orig = src[:, 0].clone()
    one_minus_alpha = 1.0 - alpha
    for i in range(1, length):
        curr = src[:, i]
        curr_valid = valid[:, i]
        close = in_valid & curr_valid & ((prev_orig - curr).abs() < delta)
        filtered = curr * alpha + state * one_minus_alpha
        out[:, i] = torch.where(close, filtered, curr)
        state = torch.where(close, filtered, torch.where(curr_valid, curr, state))
        in_valid = curr_valid
        prev_orig = curr
    return out


def _ema_bidirectional_torch(lines: torch.Tensor, alpha: float, delta: float) -> torch.Tensor:
    lines = _ema_scan_torch(lines, alpha, delta)
    return _ema_scan_torch(lines.flip(-1), alpha, delta).flip(-1)


def inertial_holes_fill_torch(image: torch.Tensor, radius: int) -> torch.Tensor:
    """Horizontal inertial hole fill (Intel ``intertial_holes_fill``)."""
    if radius <= 0:
        return image
    out = image.clone()
    _, _, width = out.shape
    zero_fill = torch.zeros(out.shape[0], out.shape[1], device=out.device, dtype=torch.int32)
    cur_fill = zero_fill.clone()
    for x in range(1, width):
        empty = out[:, :, x] == 0
        cur_fill = torch.where(empty, cur_fill + 1, zero_fill)
        out[:, :, x] = torch.where(empty & (cur_fill < radius), out[:, :, x - 1], out[:, :, x])
    cur_fill = zero_fill.clone()
    for x in range(width - 2, -1, -1):
        empty = out[:, :, x] == 0
        cur_fill = torch.where(empty, cur_fill + 1, zero_fill)
        out[:, :, x] = torch.where(empty & (cur_fill < radius), out[:, :, x + 1], out[:, :, x])
    return out


def spatial_filter_disparity(
    disparity: torch.Tensor,
    alpha: float,
    delta: float,
    iterations: int,
    holes_fill_mode: int,
) -> torch.Tensor:
    """Edge-preserving spatial filter in disparity domain. ``disparity`` is ``[B,H,W]``."""
    batch, height, width = disparity.shape
    iterations = max(1, int(iterations))
    radius = spatial_holes_fill_radius(holes_fill_mode)
    if _use_triton(disparity) and height > 1 and width > 1:
        out = disparity.clone().contiguous()
        n_rows = batch * height
        n_cols = batch * width
        for _ in range(iterations):
            _launch_blocked(n_rows, _ema_row_bidir_kernel, out.view(-1), n_rows, width, float(alpha), float(delta))
            _launch_blocked(
                n_cols, _ema_col_bidir_kernel, out, n_cols, height, width, float(alpha), float(delta)
            )
        if radius > 0:
            _launch_blocked(n_rows, _inertial_fill_kernel, out.view(-1), n_rows, width, int(radius))
        return out

    out = disparity
    for _ in range(iterations):
        out = _ema_bidirectional_torch(out.reshape(batch * height, width), alpha, delta).reshape(batch, height, width)
        out = (
            _ema_bidirectional_torch(out.permute(0, 2, 1).reshape(batch * width, height), alpha, delta)
            .reshape(batch, width, height)
            .permute(0, 2, 1)
            .contiguous()
        )
    if radius > 0:
        out = inertial_holes_fill_torch(out, radius)
    return out


def holes_fill_left(depth: torch.Tensor) -> torch.Tensor:
    """Mode 0: fill holes from the left neighbor."""
    out = depth.contiguous().clone()
    if _use_triton(out):
        batch, _, width = out.shape
        _launch_blocked(batch * out.shape[1], _holes_fill_left_kernel, out.view(-1), batch * out.shape[1], width)
        return out
    width = out.shape[-1]
    for x in range(1, width):
        hole = out[:, :, x] == 0
        out[:, :, x] = torch.where(hole, out[:, :, x - 1], out[:, :, x])
    return out


def holes_fill_farest(depth: torch.Tensor) -> torch.Tensor:
    """Mode 1: max of {up, up-left, left, down-left, down} (Intel ``holes_fill_farest``)."""
    batch, height, width = depth.shape
    if height < 3 or width < 2:
        return depth.clone()
    out = depth.contiguous().clone()
    if _use_triton(out):
        _launch_blocked(batch, _holes_fill_farest_kernel, out, batch, height, width)
        return out
    for y in range(1, height - 1):
        for x in range(1, width):
            hole = out[:, y, x] == 0
            tmp = out[:, y - 1, x]
            tmp = torch.maximum(tmp, out[:, y - 1, x - 1])
            tmp = torch.maximum(tmp, out[:, y, x - 1])
            tmp = torch.maximum(tmp, out[:, y + 1, x - 1])
            tmp = torch.maximum(tmp, out[:, y + 1, x])
            out[:, y, x] = torch.where(hole, tmp, out[:, y, x])
    return out


def holes_fill_nearest(depth: torch.Tensor) -> torch.Tensor:
    """Mode 2: nearest valid of the same 5-neighborhood (Intel ``holes_fill_nearest``)."""
    batch, height, width = depth.shape
    if height < 3 or width < 2:
        return depth.clone()
    out = depth.contiguous().clone()
    if _use_triton(out):
        _launch_blocked(batch, _holes_fill_nearest_kernel, out, batch, height, width)
        return out

    def _take_nearer(tmp: torch.Tensor, neighbor: torch.Tensor) -> torch.Tensor:
        nonempty = neighbor != 0
        return torch.where(nonempty & (neighbor < tmp), neighbor, tmp)

    for y in range(1, height - 1):
        for x in range(1, width):
            hole = out[:, y, x] == 0
            tmp = out[:, y - 1, x]
            tmp = _take_nearer(tmp, out[:, y - 1, x - 1])
            tmp = _take_nearer(tmp, out[:, y, x - 1])
            tmp = _take_nearer(tmp, out[:, y + 1, x - 1])
            tmp = _take_nearer(tmp, out[:, y + 1, x])
            out[:, y, x] = torch.where(hole, tmp, out[:, y, x])
    return out


def hole_filling_filter(depth: torch.Tensor, mode: int) -> torch.Tensor:
    if mode == 0:
        return holes_fill_left(depth)
    if mode == 2:
        return holes_fill_nearest(depth)
    return holes_fill_farest(depth)


class RealSenseDepthPostprocess:
    """Per-env RealSense D400 filter stack (temporal state is per environment)."""

    def __init__(
        self,
        num_envs: int,
        height: int,
        width: int,
        device: torch.device,
        *,
        use_spatial: bool = True,
        spatial_magnitude: int = 2,
        spatial_alpha: float = 0.5,
        spatial_delta: float = 20.0,
        spatial_holes_fill: int = 4,
        use_temporal: bool = True,
        temporal_alpha: float = 0.4,
        temporal_delta: float = 20.0,
        temporal_persistence: int = 2,
        use_hole_filling: bool = True,
        hole_filling_mode: int = 1,
        stereo_baseline_m: float = 0.05,
        disparity_fx: float = 223.4,
    ):
        self.use_spatial = bool(use_spatial)
        self.spatial_magnitude = int(spatial_magnitude)
        self.spatial_alpha = float(spatial_alpha)
        self.spatial_delta = float(spatial_delta)
        self.spatial_holes_fill = int(spatial_holes_fill)
        self.use_temporal = bool(use_temporal)
        self.temporal_alpha = float(temporal_alpha)
        self.temporal_delta = float(temporal_delta)
        self.use_hole_filling = bool(use_hole_filling)
        self.hole_filling_mode = int(hole_filling_mode)
        self.convert_factor = disparity_convert_factor(stereo_baseline_m, disparity_fx)

        self._last_disparity = torch.zeros(num_envs, height, width, device=device, dtype=torch.float32)
        self._history = torch.zeros(num_envs, height, width, device=device, dtype=torch.uint8)
        self._frame_index = torch.zeros(num_envs, device=device, dtype=torch.long)
        persistence = build_persistence_map(temporal_persistence)
        self._persistence_map = torch.tensor(persistence, device=device, dtype=torch.uint8)

    def reset(self, env_ids: torch.Tensor) -> None:
        self._last_disparity[env_ids] = 0.0
        self._history[env_ids] = 0
        self._frame_index[env_ids] = 0

    def _temporal_filter(self, disparity: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        """Intel ``temp_jw_smooth`` in disparity domain, vectorized over pixels."""
        last = self._last_disparity[env_ids]
        history = self._history[env_ids]
        frame_index = self._frame_index[env_ids]
        mask = (1 << frame_index).to(dtype=torch.uint8).view(-1, 1, 1)
        alpha = self.temporal_alpha
        one_minus_alpha = 1.0 - alpha
        delta = self.temporal_delta

        cur_valid = disparity > 0
        prev_valid = last > 0
        diff = (disparity - last).abs()
        agree = cur_valid & prev_valid & (diff < delta)

        filtered = alpha * disparity + one_minus_alpha * last
        out = disparity.clone()
        out = torch.where(agree, filtered, out)

        new_last = last.clone()
        new_hist = history.clone()
        seed = cur_valid & ~prev_valid
        new_last = torch.where(seed, disparity, new_last)
        new_hist = torch.where(seed, mask.expand_as(new_hist), new_hist)
        new_last = torch.where(agree, filtered, new_last)
        new_hist = torch.where(agree, new_hist | mask, new_hist)
        edge = cur_valid & prev_valid & ~agree
        new_last = torch.where(edge, disparity, new_last)
        new_hist = torch.where(edge, mask.expand_as(new_hist), new_hist)
        persist = (~cur_valid) & prev_valid
        classification = self._persistence_map[history.long()]
        persist_ok = persist & ((classification & mask) != 0)
        out = torch.where(persist_ok, last, out)
        new_hist = torch.where(~cur_valid, new_hist & (~mask), new_hist)

        self._last_disparity[env_ids] = new_last
        self._history[env_ids] = new_hist
        self._frame_index[env_ids] = (frame_index + 1) % 8
        return out

    def __call__(self, depth_m: torch.Tensor, env_ids: torch.Tensor, max_depth: float) -> torch.Tensor:
        """Apply the stack to ``[B,H,W]`` meters for ``env_ids`` (same length as B)."""
        depth = depth_m
        if self.use_spatial or self.use_temporal:
            disp = depth_to_disparity(depth, self.convert_factor)
            if self.use_spatial:
                disp = spatial_filter_disparity(
                    disp,
                    alpha=self.spatial_alpha,
                    delta=self.spatial_delta,
                    iterations=self.spatial_magnitude,
                    holes_fill_mode=self.spatial_holes_fill,
                )
            if self.use_temporal:
                disp = self._temporal_filter(disp, env_ids)
            depth = disparity_to_depth(disp, self.convert_factor)
        if self.use_hole_filling:
            depth = hole_filling_filter(depth, self.hole_filling_mode)
        far = depth.new_tensor(max_depth)
        valid = torch.isfinite(depth) & (depth > 0)
        return torch.where(valid, depth.clamp(min=0.0, max=max_depth), far.expand_as(depth))
