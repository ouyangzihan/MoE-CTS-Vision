# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, NoReturn

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.modules.actor_critic_moe_cts import ActorCriticMoECTS, StudentMoEEncoder
from rsl_rl.networks import EmpiricalNormalization, L2Norm, SimNorm
from rsl_rl.networks.moe import MLP
from rsl_rl.utils import unpad_trajectories


class DepthCNNGRUEncoder(nn.Module):
    """CNN + GRU encoder for flattened depth images."""

    def __init__(
        self,
        image_shape: tuple[int, int] = (60, 60),
        in_channels: int = 1,
        cnn_channels: tuple[int, int, int] = (16, 32, 64),
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        pooled_shape: tuple[int, int] = (15, 15),
        gru_hidden_dim: int = 225,
        gru_num_layers: int = 1,
        activation: str = "elu",
        depth_num_frames: int = 1,
        enable_depth_aux: bool = False,
        height_map_shape: tuple[int, int] = (17, 11),
        align_dim: int = 32,
        # Match Isaac ObsTerm scales: depth clip→scale 0.5; height clip→scale 2.5.
        depth_obs_scale: float = 0.5,
        height_obs_scale: float = 2.5,
    ) -> None:
        super().__init__()
        self.image_shape = tuple(image_shape)
        self.in_channels = max(int(depth_num_frames), 1)
        self.pooled_shape = tuple(pooled_shape)
        self.cnn_channels = tuple(cnn_channels)
        self.cnn_feature_dim = self.pooled_shape[0] * self.pooled_shape[1]
        self.hidden_dim = gru_hidden_dim
        self.num_layers = gru_num_layers
        self.enable_depth_aux = enable_depth_aux
        self.height_map_shape = tuple(height_map_shape)
        self.height_map_dim = self.height_map_shape[0] * self.height_map_shape[1]
        self.align_dim = align_dim
        self.depth_obs_scale = float(depth_obs_scale)
        self.height_obs_scale = float(height_obs_scale)

        act = nn.ELU if activation == "elu" else nn.ReLU
        layers: list[nn.Module] = []
        last_channels = self.in_channels
        for channels in cnn_channels:
            layers.extend(
                [
                    nn.Conv2d(
                        last_channels,
                        channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    act(),
                ]
            )
            last_channels = channels
        self.cnn = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(self.pooled_shape)
        self.gru = nn.GRU(self.cnn_feature_dim, gru_hidden_dim, gru_num_layers)
        self.hidden_state: torch.Tensor | None = None

        self.depth_decoder: nn.Module | None = None
        self.height_decoder: nn.Module | None = None
        self.depth_align_proj: nn.Module | None = None
        self.height_align_encoder: nn.Module | None = None
        if enable_depth_aux:
            # MGDP uses Sigmoid depth / Tanh height; we scale into ObsTerm space.
            self.depth_decoder = nn.Sequential(
                nn.Conv2d(last_channels, 32, kernel_size=3, padding=1),
                act(),
                nn.Upsample(size=self.image_shape, mode="bilinear", align_corners=False),
                nn.Conv2d(32, 16, kernel_size=3, padding=1),
                act(),
                nn.Conv2d(16, 1, kernel_size=1),
                nn.Sigmoid(),
            )
            pooled_flat = last_channels * self.pooled_shape[0] * self.pooled_shape[1]
            self.height_decoder = nn.Sequential(
                nn.Linear(pooled_flat, 256),
                act(),
                nn.Linear(256, self.height_map_dim),
                nn.Tanh(),
            )
            # Align CNN (pre-GRU) visual token with height, matching MGDP encoder-token InfoNCE.
            cnn_flat = last_channels * self.pooled_shape[0] * self.pooled_shape[1]
            self.depth_align_proj = nn.Sequential(
                nn.Linear(cnn_flat, align_dim),
                L2Norm(),
            )
            self.height_align_encoder = nn.Sequential(
                nn.Linear(self.height_map_dim, 128),
                act(),
                nn.Linear(128, align_dim),
                L2Norm(),
            )

    def reset(self, dones: torch.Tensor | None = None) -> None:
        if dones is None:
            self.hidden_state = None
        elif self.hidden_state is not None:
            self.hidden_state[:, dones == 1, :] = 0.0

    def _format_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 2:
            if self.in_channels > 1:
                image = image.reshape(image.shape[0], self.in_channels, *self.image_shape)
            else:
                image = image.reshape(image.shape[0], self.in_channels, *self.image_shape)
        elif image.ndim == 3:
            image = image.reshape(image.shape[0] * image.shape[1], self.in_channels, *self.image_shape)
        elif image.ndim == 4:
            pass
        elif image.ndim == 5:
            image = image.reshape(image.shape[0] * image.shape[1], *image.shape[2:])
        else:
            raise ValueError(f"Unsupported depth image shape: {tuple(image.shape)}")
        return image

    def _cnn_feature_maps(self, image: torch.Tensor) -> torch.Tensor:
        """Return pooled CNN maps shaped ``[N, C, H_p, W_p]``."""
        x = self._format_image(image)
        x = self.cnn(x)
        return self.avgpool(x)

    def _encode_cnn(self, image: torch.Tensor, leading_shape: tuple[int, ...]) -> torch.Tensor:
        x = self._cnn_feature_maps(image)
        x = x.mean(dim=1).flatten(1)
        return x.reshape(*leading_shape, self.cnn_feature_dim)

    def forward(
        self,
        image: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
        update_hidden: bool = False,
    ) -> torch.Tensor:
        if masks is not None:
            leading_shape = image.shape[:2]
            features = self._encode_cnn(image, leading_shape)
            out, _ = self.gru(features, hidden_state)
            return unpad_trajectories(out, masks)

        if image.ndim in (3, 5):
            leading_shape = image.shape[:2]
            features = self._encode_cnn(image, leading_shape)
            out, next_hidden_state = self.gru(features, hidden_state if hidden_state is not None else self.hidden_state)
            if update_hidden:
                self.hidden_state = next_hidden_state.detach()
            return out

        leading_shape = (image.shape[0],)
        features = self._encode_cnn(image, leading_shape).unsqueeze(0)
        out, next_hidden_state = self.gru(features, hidden_state if hidden_state is not None else self.hidden_state)
        if update_hidden:
            self.hidden_state = next_hidden_state.detach()
        return out.squeeze(0)

    def decode_depth(self, image: torch.Tensor) -> torch.Tensor:
        """Predict clean depth in obs space ``[N, H*W]`` (Sigmoid × depth_obs_scale)."""
        if self.depth_decoder is None:
            raise RuntimeError("Depth decoder is disabled (enable_depth_aux=False).")
        maps = self._cnn_feature_maps(image)
        pred = self.depth_decoder(maps).flatten(1) * self.depth_obs_scale
        return pred

    def decode_height(self, image: torch.Tensor) -> torch.Tensor:
        """Predict local height map in obs space ``[N, H_h*W_h]`` (Tanh × height_obs_scale)."""
        if self.height_decoder is None:
            raise RuntimeError("Height decoder is disabled (enable_depth_aux=False).")
        maps = self._cnn_feature_maps(image)
        return self.height_decoder(maps.flatten(1)) * self.height_obs_scale

    def align_latents(
        self,
        depth_cnn_feat: torch.Tensor,
        height_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project depth CNN maps and height maps into a shared unit sphere (MGDP-style)."""
        if self.depth_align_proj is None or self.height_align_encoder is None:
            raise RuntimeError("Alignment heads are disabled (enable_depth_aux=False).")
        return self.depth_align_proj(depth_cnn_feat), self.height_align_encoder(height_map)

    def cnn_align_features(self, image: torch.Tensor) -> torch.Tensor:
        """Flattened pooled CNN maps used for contrastive alignment."""
        return self._cnn_feature_maps(image).flatten(1)


class ActorCriticMoECTSCNNGRU(ActorCriticMoECTS):
    """MoE CTS actor-critic with a depth CNN-GRU encoder for the student."""

    is_recurrent: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = (512, 256, 128),
        critic_hidden_dims: tuple[int] | list[int] = (512, 256, 128),
        teacher_encoder_hidden_dims: tuple[int] | list[int] = (512, 256),
        student_encoder_hidden_dims: tuple[int] | list[int] = (512, 256, 256),
        expert_num: int = 8,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        latent_dim: int = 32,
        norm_type: str = "l2norm",
        actor_image_obs_groups: Iterable[str] | None = None,
        image_shape: tuple[int, int] = (60, 60),
        cnn_channels: tuple[int, int, int] = (16, 32, 64),
        cnn_kernel_size: int = 3,
        cnn_stride: int = 2,
        cnn_padding: int = 1,
        cnn_pooled_shape: tuple[int, int] = (15, 15),
        gru_hidden_dim: int = 225,
        gru_num_layers: int = 1,
        depth_num_frames: int = 1,
        enable_depth_aux: bool = False,
        height_map_shape: tuple[int, int] = (17, 11),
        depth_align_dim: int = 32,
        clean_depth_obs_group: str = "clean_depth",
        height_map_obs_group: str = "height_map",
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCriticMoECTSCNNGRU.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        assert norm_type in ["l2norm", "simnorm"], f"Normalization type {norm_type} not supported!"
        assert "policy" in obs.keys() and "critic" in obs.keys() and "single_obs" in obs.keys(), (
            "obs must contain 'policy', 'critic' and 'single_obs' keys for ActorCriticMoECTSCNNGRU."
        )
        nn.Module.__init__(self)

        self.num_actions = num_actions
        self.obs_groups = obs_groups
        self.actor_image_obs_groups = list(actor_image_obs_groups or self._infer_image_groups(obs, obs_groups["policy"]))
        if not self.actor_image_obs_groups:
            raise ValueError("ActorCriticMoECTSCNNGRU requires at least one actor image observation group.")

        self.actor_obs_groups_1d = [group for group in obs_groups["policy"] if group not in self.actor_image_obs_groups]
        self.critic_obs_groups_1d = list(obs_groups["critic"])
        self.num_actor_obs = sum(obs[group].shape[-1] for group in self.actor_obs_groups_1d)
        self.num_critic_obs = sum(obs[group].shape[-1] for group in self.critic_obs_groups_1d)
        self.num_single_obs = obs["single_obs"].shape[-1]
        self.image_shape = tuple(image_shape)
        self.enable_depth_aux = bool(enable_depth_aux)
        self.clean_depth_obs_group = clean_depth_obs_group
        self.height_map_obs_group = height_map_obs_group
        resolved_height_map_shape = tuple(height_map_shape)
        if self.enable_depth_aux:
            missing = [g for g in (clean_depth_obs_group, height_map_obs_group) if g not in obs.keys()]
            if missing:
                raise ValueError(
                    "enable_depth_aux=True requires observation groups "
                    f"{clean_depth_obs_group!r} and {height_map_obs_group!r} in the env TensorDict. "
                    f"Missing: {missing}. Set Go2WD435iEnvCfg.use_mgdp_depth_aux=True."
                )
            height_dim = int(obs[height_map_obs_group].shape[-1])
            if math.prod(resolved_height_map_shape) != height_dim:
                resolved_height_map_shape = (1, height_dim)

        self.student_cnn_gru = DepthCNNGRUEncoder(
            image_shape=self.image_shape,
            depth_num_frames=depth_num_frames,
            cnn_channels=tuple(cnn_channels),
            kernel_size=cnn_kernel_size,
            stride=cnn_stride,
            padding=cnn_padding,
            pooled_shape=tuple(cnn_pooled_shape),
            gru_hidden_dim=gru_hidden_dim,
            gru_num_layers=gru_num_layers,
            activation=activation,
            enable_depth_aux=self.enable_depth_aux,
            height_map_shape=resolved_height_map_shape,
            align_dim=depth_align_dim,
            depth_obs_scale=0.5,
            height_obs_scale=2.5,
        )

        self.teacher_encoder = nn.Sequential(
            MLP(self.num_critic_obs, latent_dim, list(teacher_encoder_hidden_dims), activation=activation),
            L2Norm() if norm_type == "l2norm" else SimNorm(),
        )
        self.student_moe_encoder = StudentMoEEncoder(
            expert_num=expert_num,
            input_dim=self.num_actor_obs + gru_hidden_dim,
            hidden_dims=list(student_encoder_hidden_dims),
            output_dim=latent_dim,
            activation=activation,
            norm_type=norm_type,
        )
        print(f"Student CNN-GRU: {self.student_cnn_gru}")
        print(f"Teacher Encoder: {self.teacher_encoder}")
        print(f"Student MoE Encoder: {self.student_moe_encoder}")

        self.state_dependent_std = state_dependent_std
        actor_input_dim = latent_dim + self.num_single_obs
        if self.state_dependent_std:
            self.actor = MLP(actor_input_dim, [2, num_actions], list(actor_hidden_dims), activation)
        else:
            self.actor = MLP(actor_input_dim, num_actions, list(actor_hidden_dims), activation)
        print(f"Actor MLP: {self.actor}")

        self.critic = MLP(latent_dim + self.num_critic_obs, 1, list(critic_hidden_dims), activation)
        print(f"Critic MLP: {self.critic}")

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(self.num_actor_obs)
            self.single_obs_normalizer = EmpiricalNormalization(self.num_single_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
            self.single_obs_normalizer = torch.nn.Identity()

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(self.num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    @staticmethod
    def _infer_image_groups(obs: TensorDict, groups: list[str]) -> list[str]:
        inferred = []
        for group in groups:
            value = obs[group]
            if len(value.shape) == 4:
                inferred.append(group)
            elif len(value.shape) == 2:
                side = math.isqrt(value.shape[-1])
                if side * side == value.shape[-1] and ("depth" in group or "image" in group):
                    inferred.append(group)
                elif ("depth" in group or "image" in group) and value.shape[-1] % (60 * 60) == 0:
                    inferred.append(group)
        return inferred

    @staticmethod
    def _image_channels(obs: TensorDict, groups: list[str]) -> int:
        channels = 0
        for group in groups:
            value = obs[group]
            channels += value.shape[1] if len(value.shape) == 4 else 1
        return channels

    def _image_obs(self, obs: TensorDict, groups: list[str]) -> torch.Tensor:
        images = []
        for group in groups:
            image = obs[group]
            if image.ndim in (2, 3):
                images.append(image)
            elif image.ndim == 4:
                images.append(image.flatten(1))
            elif image.ndim == 5:
                images.append(image.flatten(2))
            else:
                raise ValueError(f"Unsupported image observation shape for {group}: {tuple(image.shape)}")
        return torch.cat(images, dim=-1)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        self.student_cnn_gru.reset(dones)

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _update_distribution(self, latent_and_obs: torch.Tensor) -> None:
        if self.state_dependent_std:
            mean_and_std = self.actor(latent_and_obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            mean = self.actor(latent_and_obs)
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[group] for group in self.actor_obs_groups_1d], dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[group] for group in self.critic_obs_groups_1d], dim=-1)

    def _normalize_actor_obs(self, obs: TensorDict, masks: torch.Tensor | None = None) -> torch.Tensor:
        obs_a = self.actor_obs_normalizer(self.get_actor_obs(obs))
        if masks is not None:
            obs_a = unpad_trajectories(obs_a, masks)
        return obs_a

    def _normalize_critic_obs(self, obs: TensorDict, masks: torch.Tensor | None = None) -> torch.Tensor:
        obs_c = self.critic_obs_normalizer(self.get_critic_obs(obs))
        if masks is not None:
            obs_c = unpad_trajectories(obs_c, masks)
        return obs_c

    def student_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
        update_memory: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_a = self._normalize_actor_obs(obs, masks)
        image = self._image_obs(obs, self.actor_image_obs_groups)
        image_feature = self.student_cnn_gru(image, masks=masks, hidden_state=hidden_state, update_hidden=update_memory)
        moe_input = torch.cat([obs_a, image_feature], dim=-1)
        if moe_input.ndim > 2:
            leading_shape = moe_input.shape[:-1]
            latent, weights = self.student_moe_encoder(moe_input.reshape(-1, moe_input.shape[-1]))
            return latent.reshape(*leading_shape, latent.shape[-1]), weights.reshape(*leading_shape, weights.shape[-1])
        return self.student_moe_encoder(moe_input)

    def compute_depth_aux_losses(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
        align_loss_type: str = "infonce",
        align_temperature: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        """Denoise / height-from-depth / depth↔height alignment losses (MGDP-inspired).

        Notes vs official MGDP:
        - Depth recon: noisy→clean (same idea); shallow CNN+Sigmoid (not full UNet+skips).
        - Height term: depth→height prediction (lite), not MGDP's height autoencoder.
        - Align: InfoNCE on CNN token ↔ height encoder (MGDP: height query, depth key).
        """
        del hidden_state  # align uses CNN maps; GRU remains for the policy path only
        if not self.enable_depth_aux:
            zero = torch.zeros((), device=obs[self.actor_image_obs_groups[0]].device)
            return {"depth_denoise": zero, "height_recon": zero, "depth_align": zero}

        image = self._image_obs(obs, self.actor_image_obs_groups)
        clean = obs[self.clean_depth_obs_group]
        height = obs[self.height_map_obs_group]

        if masks is not None:
            clean = unpad_trajectories(clean, masks)
            height = unpad_trajectories(height, masks)
            image_unpadded = unpad_trajectories(image, masks)
        else:
            image_unpadded = image

        # Recurrent batches are [T, N, D] after unpad; CNN heads expect [T*N, D].
        if image_unpadded.ndim >= 3:
            image_unpadded = image_unpadded.reshape(-1, image_unpadded.shape[-1])
            clean = clean.reshape(-1, clean.shape[-1])
            height = height.reshape(-1, height.shape[-1])

        depth_pred = self.student_cnn_gru.decode_depth(image_unpadded)
        height_pred = self.student_cnn_gru.decode_height(image_unpadded)
        cnn_feat = self.student_cnn_gru.cnn_align_features(image_unpadded)

        denoise_loss = (depth_pred - clean).pow(2).mean()
        height_loss = (height_pred - height).pow(2).mean()

        depth_z, height_z = self.student_cnn_gru.align_latents(cnn_feat, height)
        if align_loss_type == "mse":
            # Train both towers (official MGDP does not detach either side).
            align_loss = (depth_z - height_z).pow(2).mean()
        else:
            # MGDP: sim = height @ depth.T / T, CE(height→depth). Keep that one-way form.
            if depth_z.shape[0] < 2:
                align_loss = depth_z.new_zeros(())
            else:
                logits = height_z @ depth_z.transpose(0, 1) / max(align_temperature, 1e-6)
                labels = torch.arange(logits.shape[0], device=logits.device)
                align_loss = torch.nn.functional.cross_entropy(logits, labels)

        return {
            "depth_denoise": denoise_loss,
            "height_recon": height_loss,
            "depth_align": align_loss,
        }

    def teacher_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
        update_memory: bool = False,
    ) -> torch.Tensor:
        obs_c = self._normalize_critic_obs(obs, masks)
        return self.teacher_encoder(obs_c)

    def act(
        self,
        obs: TensorDict,
        is_teacher: bool,
        masks: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
        **kwargs: dict[str, Any],
    ) -> torch.Tensor:
        single_obs = self.single_obs_normalizer(obs["single_obs"])
        if masks is not None:
            single_obs = unpad_trajectories(single_obs, masks)
        if is_teacher:
            latent = self.teacher_latent(obs, masks=masks, hidden_state=hidden_state, update_memory=masks is None)
        else:
            with torch.no_grad():
                latent, _ = self.student_latent(obs, masks=masks, hidden_state=hidden_state, update_memory=masks is None)
        self._update_distribution(torch.cat([latent, single_obs], dim=-1))
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        single_obs = self.single_obs_normalizer(obs["single_obs"])
        latent, _ = self.student_latent(obs, update_memory=True)
        latent_and_obs = torch.cat([latent, single_obs], dim=-1)
        if self.state_dependent_std:
            return self.actor(latent_and_obs)[..., 0, :]
        return self.actor(latent_and_obs)

    def act_and_evaluate_cts(
        self,
        obs: TensorDict,
        teacher_env_idxs: torch.Tensor,
        student_env_idxs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        single_obs = self.single_obs_normalizer(obs["single_obs"])
        teacher_latent = self.teacher_latent(obs, update_memory=True)
        with torch.no_grad():
            student_latent, _ = self.student_latent(obs, update_memory=True)

        latent = torch.empty_like(teacher_latent)
        latent[teacher_env_idxs] = teacher_latent[teacher_env_idxs]
        latent[student_env_idxs] = student_latent[student_env_idxs]
        self._update_distribution(torch.cat([latent, single_obs], dim=-1))
        actions = self.distribution.sample()

        obs_c = self._normalize_critic_obs(obs)
        value_latent = latent.detach()
        values = self.critic(torch.cat([value_latent, obs_c], dim=-1))
        return actions, values, self.get_actions_log_prob(actions), self.action_mean, self.action_std

    def evaluate(
        self,
        obs: TensorDict,
        is_teacher: bool,
        masks: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
        **kwargs: dict[str, Any],
    ) -> torch.Tensor:
        obs_c = self._normalize_critic_obs(obs, masks)
        if is_teacher:
            latent = self.teacher_latent(obs, masks=masks, hidden_state=hidden_state)
        else:
            latent, _ = self.student_latent(obs, masks=masks, hidden_state=hidden_state)
        return self.critic(torch.cat([latent.detach(), obs_c], dim=-1))

    def evaluate_cts(
        self,
        obs: TensorDict,
        teacher_env_idxs: torch.Tensor,
        student_env_idxs: torch.Tensor,
    ) -> torch.Tensor:
        teacher_latent = self.teacher_latent(obs)
        student_latent, _ = self.student_latent(obs)
        latent = torch.empty_like(teacher_latent)
        latent[teacher_env_idxs] = teacher_latent[teacher_env_idxs]
        latent[student_env_idxs] = student_latent[student_env_idxs]
        obs_c = self._normalize_critic_obs(obs)
        return self.critic(torch.cat([latent.detach(), obs_c], dim=-1))

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_hidden_states(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        student_hidden_state = (
            None if self.student_cnn_gru.hidden_state is None else self.student_cnn_gru.hidden_state.detach()
        )
        return student_hidden_state, None

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
            self.single_obs_normalizer.update(obs["single_obs"])
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def ppo_parameters(self) -> list[dict[str, Any]]:
        noise_params: list[nn.Parameter] = []
        if hasattr(self, "std"):
            noise_params.append(self.std)
        if hasattr(self, "log_std"):
            noise_params.append(self.log_std)
        return [
            {"params": self.teacher_encoder.parameters()},
            {"params": self.critic.parameters()},
            {"params": self.actor.parameters()},
            {"params": noise_params},
        ]

    def student_encoder_parameters(self):
        return list(self.student_cnn_gru.parameters()) + list(self.student_moe_encoder.parameters())

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True
