# base version: IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/exporter.py

import copy
import os
import torch
import re
import os
import sys


def resolve_cts_single_obs_feature_dims(num_single_obs: int, num_actions: int) -> list[int]:
    """Infer per-term dims for CTS single_obs history layout.

    Supported layouts:
    - Go2: ``[ang(3), grav(3), cmd(3|2), joints(A), joints(A), actions(A)]``
    - Go2W: ``[ang(3), grav(3), cmd(3|2), legs(12), all_joints(A), actions(A)]``
    """
    candidates = [
        [3, 3, 3, num_actions, num_actions, num_actions],
        [3, 3, 2, num_actions, num_actions, num_actions],
        [3, 3, 3, 12, num_actions, num_actions],
        [3, 3, 2, 12, num_actions, num_actions],
    ]
    for dims in candidates:
        if sum(dims) == num_single_obs:
            return dims
    raise ValueError(
        "Unsupported single_obs layout: "
        f"num_single_obs={num_single_obs}, num_actions={num_actions}."
    )


"Script to log terminal output to a file, stripping ANSI escape codes."
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.log = open(filename, 'w', encoding='utf-8')

        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def write(self, message):
        clean_message = self.ansi_escape.sub('', message)

        self.terminal.write(message)
        self.log.write(clean_message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def export_cts_policy_as_jit(policy: object, actor_obs_normalizer: object | None, single_obs_normalizer: object | None, path: str, filename="policy.pt"):
    """Export CTS policy into a Torch JIT file.

    Args:
        policy: The CTS policy torch module.
        actor_obs_normalizer: The empirical normalizer module for actor observations. If None, Identity is used.
        single_obs_normalizer: The empirical normalizer module for single observations. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    policy_exporter = _TorchPolicyExporter(policy, actor_obs_normalizer, single_obs_normalizer)
    policy_exporter.export(path, filename)


def export_cts_policy_as_onnx(
    policy: object, path: str, actor_obs_normalizer: object | None = None, single_obs_normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    """Export CTS policy into a Torch ONNX file.

    Args:
        policy: The CTS policy torch module.
        actor_obs_normalizer: The empirical normalizer module for actor observations. If None, Identity is used.
        single_obs_normalizer: The empirical normalizer module for single observations. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(policy, actor_obs_normalizer, single_obs_normalizer, verbose)
    policy_exporter.export(path, filename)


"""
Helper Classes - Private.
"""


class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file."""

    def __init__(self, policy, actor_obs_normalizer=None, single_obs_normalizer=None):
        """Initialize a TorchScript exporter for CTS policy inference.

        The exported model consumes only the current `single_obs` frame and maintains
        internal stacked history to reconstruct actor observations expected by the student encoder.

        Args:
            policy: Source CTS policy module to export.
            actor_obs_normalizer: Normalizer applied to stacked actor observations.
            single_obs_normalizer: Normalizer applied to current single-frame observations.
        """
        assert not policy.is_recurrent, "CTS policy should not be recurrent"
        super().__init__()
        
        # copy policy parameters
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
        else:
            raise ValueError("Policy does not have an actor/student module.")
        self.student_moe_encoder = copy.deepcopy(policy.student_moe_encoder)
        self.state_dependent_std = policy.state_dependent_std
        self.num_actions = int(policy.num_actions)
        self.num_single_obs = int(policy.num_single_obs)
        self.num_actor_obs = int(policy.num_actor_obs)
        if self.num_actor_obs % self.num_single_obs != 0:
            raise ValueError(
                f"num_actor_obs ({self.num_actor_obs}) must be divisible by num_single_obs ({self.num_single_obs})."
            )
        self.history_len = self.num_actor_obs // self.num_single_obs
        # Keep the same per-term history layout as deploy-side push_obs_history.
        self.feature_dims = resolve_cts_single_obs_feature_dims(self.num_single_obs, self.num_actions)
        self.register_buffer("obs_history", torch.zeros(1, self.num_actor_obs, dtype=torch.float32))
        
        # copy normalizer if exists
        if actor_obs_normalizer:
            self.actor_obs_normalizer = copy.deepcopy(actor_obs_normalizer)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        if single_obs_normalizer:
            self.single_obs_normalizer = copy.deepcopy(single_obs_normalizer)
        else:
            self.single_obs_normalizer = torch.nn.Identity()

    def forward(self, single_obs: torch.Tensor):
        """Compute policy action from one current observation frame.

        The exporter keeps an internal FIFO history buffer and shifts it by one frame
        on each forward call before appending the latest observation.

        Args:
            single_obs: Current-step observation tensor with shape `[B, num_single_obs]`.

        Returns:
            The policy action tensor.
        """
        if single_obs.dim() == 1:
            single_obs = single_obs.unsqueeze(0)
        if single_obs.shape[-1] != self.num_single_obs:
            raise ValueError(
                f"Expected single_obs last dimension {self.num_single_obs}, got {single_obs.shape[-1]}."
            )
        if single_obs.shape[0] != 1:
            raise ValueError("TorchScript CTS deployment currently supports batch size 1 only.")

        next_history = self.obs_history.clone()
        history_offset = 0
        single_offset = 0
        for dim in self.feature_dims:
            block_size = dim * self.history_len
            block_end = history_offset + block_size
            single_end = single_offset + dim
            block = self.obs_history[:, history_offset:block_end]
            shifted_block = torch.cat([block[:, dim:], single_obs[:, single_offset:single_end]], dim=-1)
            next_history[:, history_offset:block_end] = shifted_block
            history_offset = block_end
            single_offset = single_end
        self.obs_history.copy_(next_history)

        single_obs = self.single_obs_normalizer(single_obs)
        obs_a = self.actor_obs_normalizer(self.obs_history)
        latent, _ = self.student_moe_encoder(obs_a)
        latent_and_obs = torch.cat([latent, single_obs], dim=-1)
        if self.state_dependent_std:
            return self.actor(latent_and_obs)[..., 0, :]
        else:
            return self.actor(latent_and_obs)
        
    @torch.jit.export
    def reset(self):
        """Reset internal observation history state."""
        self.obs_history.zero_()

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, policy, actor_obs_normalizer=None, single_obs_normalizer=None, verbose=False):
        assert not policy.is_recurrent, "CTS policy should not be recurrent"
        super().__init__()
        self.verbose = verbose
        
        # copy policy parameters
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
        else:
            raise ValueError("Policy does not have an actor/student module.")
        self.student_moe_encoder = copy.deepcopy(policy.student_moe_encoder)
        self.num_actions = int(policy.num_actions)
        self.num_single_obs = int(policy.num_single_obs)
        self.num_actor_obs = int(policy.num_actor_obs)
        if self.num_actor_obs % self.num_single_obs != 0:
            raise ValueError(
                f"num_actor_obs ({self.num_actor_obs}) must be divisible by num_single_obs ({self.num_single_obs})."
            )
        self.history_len = self.num_actor_obs // self.num_single_obs
        self.feature_dims = resolve_cts_single_obs_feature_dims(self.num_single_obs, self.num_actions)
        self.state_dependent_std = policy.state_dependent_std
        
        # copy normalizer if exists
        if actor_obs_normalizer:
            self.actor_obs_normalizer = copy.deepcopy(actor_obs_normalizer)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        if single_obs_normalizer:
            self.single_obs_normalizer = copy.deepcopy(single_obs_normalizer)
        else:
            self.single_obs_normalizer = torch.nn.Identity()

    def _extract_single_obs_from_history(self, history: torch.Tensor) -> torch.Tensor:
        if history.dim() == 1:
            history = history.unsqueeze(0)
        if history.shape[-1] != self.num_actor_obs:
            raise ValueError(
                f"Expected history last dimension {self.num_actor_obs}, got {history.shape[-1]}."
            )

        single_obs_terms = []
        offset = 0
        for dim in self.feature_dims:
            end = offset + dim * self.history_len
            single_obs_terms.append(history[:, end - dim:end])
            offset = end
        return torch.cat(single_obs_terms, dim=-1)

    def forward(self, history):
        if history.dim() == 1:
            history = history.unsqueeze(0)
        single_obs = self._extract_single_obs_from_history(history)
        single_obs = self.single_obs_normalizer(single_obs)
        obs_a = self.actor_obs_normalizer(history)
        latent, _ = self.student_moe_encoder(obs_a)
        latent_and_obs = torch.cat([latent, single_obs], dim=-1)
        if self.state_dependent_std:
            return self.actor(latent_and_obs)[..., 0, :]
        else:
            return self.actor(latent_and_obs)

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        opset_version = 18  # was 11, but it caused problems with linux-aarch, and 18 worked well across all systems.
        torch.onnx.export(
            self,
            torch.zeros(1, self.num_actor_obs),
            os.path.join(path, filename),
            export_params=True,
            opset_version=opset_version,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
        )


def export_cts_cnn_gru_policy_as_jit(
    policy: object,
    actor_obs_normalizer: object | None,
    single_obs_normalizer: object | None,
    path: str,
    filename: str = "policy.pt",
):
    """Export recurrent MoE-CTS CNN-GRU (D435i) student to TorchScript.

    Deploy interface (matches rl_sar moe_cts_d435i InferActions):
      forward(single_obs[1,53], depth[1,3600]) -> actions[1,16]
    Internal: term-major proprio history + GRU hidden state.
    """
    exporter = _TorchCNNGRUPolicyExporter(policy, actor_obs_normalizer, single_obs_normalizer)
    exporter.export(path, filename)


class _TorchCNNGRUPolicyExporter(torch.nn.Module):
    """JIT wrapper for ActorCriticMoECTSCNNGRU student deploy."""

    def __init__(self, policy, actor_obs_normalizer=None, single_obs_normalizer=None):
        assert bool(getattr(policy, "is_recurrent", False)), "Expected recurrent CNN-GRU CTS policy"
        assert hasattr(policy, "student_cnn_gru"), "Policy missing student_cnn_gru"
        super().__init__()

        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
        else:
            raise ValueError("Policy does not have an actor/student module.")

        self.student_moe_encoder = copy.deepcopy(policy.student_moe_encoder)
        self.cnn = copy.deepcopy(policy.student_cnn_gru.cnn)
        self.avgpool = copy.deepcopy(policy.student_cnn_gru.avgpool)
        self.gru = copy.deepcopy(policy.student_cnn_gru.gru)

        self.state_dependent_std = policy.state_dependent_std
        self.num_actions = int(policy.num_actions)
        self.num_single_obs = int(policy.num_single_obs)
        self.num_actor_obs = int(policy.num_actor_obs)
        if self.num_actor_obs % self.num_single_obs != 0:
            raise ValueError(
                f"num_actor_obs ({self.num_actor_obs}) must be divisible by num_single_obs ({self.num_single_obs})."
            )
        self.history_len = self.num_actor_obs // self.num_single_obs
        self.feature_dims = resolve_cts_single_obs_feature_dims(self.num_single_obs, self.num_actions)

        self.image_h = int(policy.student_cnn_gru.image_shape[0])
        self.image_w = int(policy.student_cnn_gru.image_shape[1])
        self.depth_dim = self.image_h * self.image_w
        self.gru_hidden_dim = int(policy.student_cnn_gru.hidden_dim)
        self.gru_num_layers = int(policy.student_cnn_gru.num_layers)
        self.cnn_feature_dim = int(policy.student_cnn_gru.cnn_feature_dim)

        self.register_buffer("obs_history", torch.zeros(1, self.num_actor_obs, dtype=torch.float32))
        self.register_buffer(
            "gru_hidden",
            torch.zeros(self.gru_num_layers, 1, self.gru_hidden_dim, dtype=torch.float32),
        )

        if actor_obs_normalizer:
            self.actor_obs_normalizer = copy.deepcopy(actor_obs_normalizer)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        if single_obs_normalizer:
            self.single_obs_normalizer = copy.deepcopy(single_obs_normalizer)
        else:
            self.single_obs_normalizer = torch.nn.Identity()

    def _update_obs_history(self, single_obs: torch.Tensor) -> torch.Tensor:
        next_history = self.obs_history.clone()
        history_offset = 0
        single_offset = 0
        for dim in self.feature_dims:
            block_size = dim * self.history_len
            block_end = history_offset + block_size
            single_end = single_offset + dim
            block = self.obs_history[:, history_offset:block_end]
            shifted_block = torch.cat([block[:, dim:], single_obs[:, single_offset:single_end]], dim=-1)
            next_history[:, history_offset:block_end] = shifted_block
            history_offset = block_end
            single_offset = single_end
        self.obs_history.copy_(next_history)
        return self.obs_history

    def _encode_depth(self, depth: torch.Tensor) -> torch.Tensor:
        # depth: [1, H*W] meters already preprocessed like training obs (normalize+scale applied in C++)
        image = depth.reshape(1, 1, self.image_h, self.image_w)
        x = self.cnn(image)
        x = self.avgpool(x)
        features = x.mean(dim=1).flatten(1)  # [1, cnn_feature_dim]
        features = features.unsqueeze(0)  # [1, 1, F] for GRU (seq, batch, feat)
        out, next_hidden = self.gru(features, self.gru_hidden)
        self.gru_hidden.copy_(next_hidden)
        return out.squeeze(0)  # [1, F]

    def forward(self, single_obs: torch.Tensor, depth: torch.Tensor):
        if single_obs.dim() == 1:
            single_obs = single_obs.unsqueeze(0)
        if depth.dim() == 1:
            depth = depth.unsqueeze(0)
        if single_obs.shape[-1] != self.num_single_obs:
            raise ValueError(
                f"Expected single_obs last dimension {self.num_single_obs}, got {single_obs.shape[-1]}."
            )
        if depth.shape[-1] != self.depth_dim:
            raise ValueError(f"Expected depth last dimension {self.depth_dim}, got {depth.shape[-1]}.")
        if single_obs.shape[0] != 1 or depth.shape[0] != 1:
            raise ValueError("TorchScript CNN-GRU CTS deployment currently supports batch size 1 only.")

        obs_history = self._update_obs_history(single_obs)
        image_feature = self._encode_depth(depth)
        obs_a = self.actor_obs_normalizer(obs_history)
        moe_input = torch.cat([obs_a, image_feature], dim=-1)
        latent, _ = self.student_moe_encoder(moe_input)
        single_obs_n = self.single_obs_normalizer(single_obs)
        latent_and_obs = torch.cat([latent, single_obs_n], dim=-1)
        if self.state_dependent_std:
            return self.actor(latent_and_obs)[..., 0, :]
        return self.actor(latent_and_obs)

    @torch.jit.export
    def reset(self):
        self.obs_history.zero_()
        self.gru_hidden.zero_()

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        self.eval()
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
