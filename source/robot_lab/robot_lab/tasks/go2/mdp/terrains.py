from __future__ import annotations

import numpy as np
import torch

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator import TerrainGenerator
from isaaclab.terrains.terrain_importer import TerrainImporter
from isaaclab.utils import configclass
from isaaclab.terrains.height_field import hf_terrains
from isaaclab.terrains.height_field.utils import height_field_to_mesh


def compute_step_height_lower_from_iteration(
    iteration: int,
    start_it: int = 10000,
    interval: int = 500,
    delta: float = 0.01,
    max_lower: float = 0.15,
    initial_lower: float = 0.0,
) -> float:
    """Step-height lower bound schedule used by the go2w curriculum.

    Holds ``initial_lower`` until ``start_it``, then raises by ``delta`` every
    ``interval`` iterations (first bump at ``start_it``), clamping at ``max_lower``.
    """
    if iteration < start_it:
        return float(initial_lower)
    steps = (iteration - start_it) // interval + 1
    return float(min(initial_lower + steps * delta, max_lower))


def step_height_lower_to_min_terrain_level(
    lower: float,
    upper: float,
    num_rows: int,
    mesh_lower: float = 0.0,
) -> int:
    """Map a desired step-height lower bound to a minimum curriculum terrain row.

    Terrains are baked once with ``step_height = mesh_lower + difficulty * (upper - mesh_lower)``.
    Raising the effective lower bound is approximated by forbidding easy rows whose
    heights fall below ``lower``.
    """
    span = max(float(upper) - float(mesh_lower), 1e-8)
    min_difficulty = max(0.0, (float(lower) - float(mesh_lower)) / span)
    min_level = int(np.ceil(min_difficulty * num_rows - 1e-9))
    return int(np.clip(min_level, 0, max(num_rows - 1, 0)))


class PerSubTerrainSlopeThresholdGenerator(TerrainGenerator):
    """Terrain generator that lets each height-field sub-terrain override slope_threshold."""

    def __init__(self, cfg, device: str = "cpu"):
        self._apply_sub_terrain_border_width(cfg)
        super().__init__(cfg, device)

    def _apply_sub_terrain_border_width(self, cfg):
        border_width = getattr(cfg, "sub_terrain_border_width", None)
        if border_width is None:
            return

        for sub_cfg in cfg.sub_terrains.values():
            if hasattr(sub_cfg, "border_width"):
                sub_cfg.border_width = border_width

    def _get_terrain_mesh(self, difficulty, cfg):
        difficulty = self._maybe_use_gym_difficulty(difficulty)

        override = getattr(cfg, "slope_threshold_override", None)
        if override is None:
            return super()._get_terrain_mesh(difficulty, cfg)

        original_slope_threshold = getattr(cfg, "slope_threshold", None)
        cfg.slope_threshold = override
        try:
            return super()._get_terrain_mesh(difficulty, cfg)
        finally:
            cfg.slope_threshold = original_slope_threshold

    def _maybe_use_gym_difficulty(self, difficulty: float) -> float:
        if not getattr(self.cfg, "use_gym_difficulty", False):
            return difficulty

        lower, upper = self.cfg.difficulty_range
        if upper > lower:
            normalized_difficulty = (float(difficulty) - lower) / (upper - lower)
        else:
            normalized_difficulty = float(difficulty)
        normalized_difficulty = np.clip(normalized_difficulty, 0.0, 1.0)

        num_levels = self.cfg.num_rows
        level = min(int(np.floor(normalized_difficulty * num_levels)), num_levels - 1)
        return level / num_levels


class Go2TerrainImporter(TerrainImporter):
    """TerrainImporter with a raisable minimum curriculum terrain level."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.step_height_min_level = 0
        self.step_height_active_lower = 0.0

    def set_step_height_min_level(self, min_level: int, active_lower: float | None = None) -> int:
        """Raise the floor on terrain levels and remap origins for all envs."""
        if self.terrain_origins is None or not hasattr(self, "terrain_levels"):
            self.step_height_min_level = int(min_level)
            if active_lower is not None:
                self.step_height_active_lower = float(active_lower)
            return self.step_height_min_level

        min_level = int(np.clip(min_level, 0, self.max_terrain_level - 1))
        self.step_height_min_level = min_level
        if active_lower is not None:
            self.step_height_active_lower = float(active_lower)

        below = self.terrain_levels < min_level
        if below.any():
            self.terrain_levels[below] = min_level
            self.env_origins[below] = self.terrain_origins[self.terrain_levels[below], self.terrain_types[below]]
        return min_level

    def update_env_origins(self, env_ids: torch.Tensor, move_up: torch.Tensor, move_down: torch.Tensor):
        """Same as parent, then enforce ``step_height_min_level``."""
        super().update_env_origins(env_ids, move_up, move_down)
        if self.step_height_min_level <= 0:
            return
        levels = self.terrain_levels[env_ids]
        clamped = torch.clamp(levels, min=self.step_height_min_level)
        if torch.equal(levels, clamped):
            return
        self.terrain_levels[env_ids] = clamped
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]


@configclass
class Go2TerrainGeneratorCfg(terrain_gen.TerrainGeneratorCfg):
    class_type: type = PerSubTerrainSlopeThresholdGenerator
    sub_terrain_border_width: float | None = None
    use_gym_difficulty: bool = False


def with_slope_threshold(sub_terrain_cfg, slope_threshold: float | None):
    """Attach a per-sub-terrain slope-threshold override to a terrain config."""
    sub_terrain_cfg.slope_threshold_override = slope_threshold
    return sub_terrain_cfg


# -----------------------------------------------------------------------------
# Default RobotLab terrain setup
# -----------------------------------------------------------------------------

DEFAULT_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    class_type=PerSubTerrainSlopeThresholdGenerator,
    size=(8.0, 8.0),
    border_width=25.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.25),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.30,
            step_height_range=(0.05, 0.25),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15,
            grid_width=0.45,
            grid_height_range=(0.025, 0.1),
            platform_width=2.0,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.0, 0.5),
            platform_width=2.0,
            border_width=0.25,
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.0, 0.5),
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)


# -----------------------------------------------------------------------------
# Go2 Terrain setup
# -----------------------------------------------------------------------------
@height_field_to_mesh
def wave_terrain(difficulty: float, cfg) -> np.ndarray:
    """wave terrain: wave plus random uniform roughness."""
    wave = hf_terrains.wave_terrain.__wrapped__(difficulty, cfg)
    rough = hf_terrains.random_uniform_terrain.__wrapped__(difficulty, cfg)
    return np.rint(wave + rough).astype(np.int16)


@height_field_to_mesh
def rough_slope_terrain(difficulty: float, cfg) -> np.ndarray:
    """rough slope terrain: slope plus random uniform roughness."""
    slope = hf_terrains.pyramid_sloped_terrain.__wrapped__(difficulty, cfg)
    rough = hf_terrains.random_uniform_terrain.__wrapped__(difficulty, cfg)
    return np.rint(slope + rough).astype(np.int16)


@height_field_to_mesh
def pyramid_stairs_random_width_terrain(difficulty: float, cfg) -> np.ndarray:
    """Pyramid stairs with step width sampled uniformly from step_width_range."""
    original_step_width = cfg.step_width
    lo, hi = cfg.step_width_range
    cfg.step_width = float(np.random.uniform(lo, hi))
    try:
        return hf_terrains.pyramid_stairs_terrain.__wrapped__(difficulty, cfg)
    finally:
        cfg.step_width = original_step_width


@configclass
class WaveTerrainCfg(terrain_gen.HfWaveTerrainCfg):
    function = wave_terrain
    amplitude_range: tuple[float, float] = (0.1, 0.28)
    num_waves: int = 5
    noise_range: tuple[float, float] = (-0.05, 0.05)
    # Must be >= TERRAIN_CFG.vertical_scale so int(noise_step / vertical_scale) >= 1.
    noise_step: float = 0.005
    downsampled_scale: float = 0.2


@configclass
class RoughSlopeTerrainCfg(terrain_gen.HfPyramidSlopedTerrainCfg):
    function = rough_slope_terrain
    slope_range: tuple[float, float] = (0.1, 0.568)
    platform_width: float = 3.0
    noise_range: tuple[float, float] = (-0.05, 0.05)
    noise_step: float = 0.005
    downsampled_scale: float = 0.2


@configclass
class RandomWidthPyramidStairsTerrainCfg(terrain_gen.HfPyramidStairsTerrainCfg):
    function = pyramid_stairs_random_width_terrain
    step_width_range: tuple[float, float] = (0.25, 0.30)
    # Placeholder only; overwritten each generation from step_width_range.
    step_width: float = 0.275


@configclass
class RandomWidthInvertedPyramidStairsTerrainCfg(RandomWidthPyramidStairsTerrainCfg):
    inverted: bool = True


TERRAIN_CFG = Go2TerrainGeneratorCfg(
    size=(9.0, 9.0),  # 8.0 terrain + 0.5*2 sub_terrain_border_width
    border_width=25.0,
    sub_terrain_border_width=0.5,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    # slope correction = 0.75 ~ 36.9 degrees by default,
    # but recommended to set for each terrain type separately using with_slope_threshold
    slope_threshold=0.75,
    use_gym_difficulty=False,
    use_cache=False,
    sub_terrains={
        "wave": with_slope_threshold(
            WaveTerrainCfg(
                proportion=0.1,
            ),
            10.0,  # effectively disable slope correction for wave terrain
        ),
        "slope_up": with_slope_threshold(
            terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=0.,
                slope_range=(0.1, 0.568),
                platform_width=3.0,
            ),
            10.0,  # effectively disable slope correction for slope_up terrain
        ),
        "slope_down": with_slope_threshold(
            terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.,
                slope_range=(0.1, 0.568),
                platform_width=3.0,
            ),
            10.0,  # effectively disable slope correction for slope_down terrain
        ),
        "rough_slope": with_slope_threshold(
            RoughSlopeTerrainCfg(
                proportion=0.05,
            ),
            10.0,  # effectively disable slope correction for rough_slope terrain
        ),
        "stairs_up": with_slope_threshold(
            RandomWidthInvertedPyramidStairsTerrainCfg(
                proportion=0.65,
                # Baked mesh range. go2w curriculum raises effective lower bound via min terrain level.
                step_height_range=(0., 0.2),
                step_width_range=(0.29, 0.34),
                platform_width=3.0,
            ),
            0.25,  # ~14 deg slope correction recommended for stairs
        ),
        "stairs_down": with_slope_threshold(
            RandomWidthPyramidStairsTerrainCfg(
                proportion=0.05,
                step_height_range=(0., 0.2),
                step_width_range=(0.29, 0.34),
                platform_width=3.0,
            ),
            0.25,
        ),
        "obstacles": with_slope_threshold(
            terrain_gen.HfDiscreteObstaclesTerrainCfg(
                proportion=0.1,
                obstacle_width_range=(1.0, 2.0),
                obstacle_height_range=(0.05, 0.275),
                num_obstacles=20,
                platform_width=3.0,
            ),
            0.25,
        ),
        "stepping_stones": with_slope_threshold(
            terrain_gen.HfSteppingStonesTerrainCfg(
                proportion=0.0,
                stone_height_max=0.0,
                stone_width_range=(0.075, 1.575),
                stone_distance_range=(0.05, 0.10),
                holes_depth=-10.0,
                platform_width=4.0,
            ),
            0.25,
        ),
        "gap": terrain_gen.MeshGapTerrainCfg(
            proportion=0.0,
            gap_width_range=(0.0, 0.9),
            platform_width=3.0,
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.05),
    },
)
