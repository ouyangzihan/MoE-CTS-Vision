import copy
import os
import torch


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


def export_cts_policy_as_jit(
    policy: object,
    actor_obs_normalizer: object | None,
    single_obs_normalizer: object | None,
    path: str,
    filename: str = "policy.pt",
) -> None:
    """Export CTS policy into a Torch JIT file with single_obs input."""
    policy_exporter = _TorchPolicyExporter(policy, actor_obs_normalizer, single_obs_normalizer)
    policy_exporter.export(path, filename)


class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of CTS actor-critic into JIT file."""

    def __init__(self, policy, actor_obs_normalizer=None, single_obs_normalizer=None):
        assert not policy.is_recurrent, "CTS policy should not be recurrent"
        super().__init__()

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
        self.feature_dims = resolve_cts_single_obs_feature_dims(self.num_single_obs, self.num_actions)
        self.register_buffer("obs_history", torch.zeros(1, self.num_actor_obs, dtype=torch.float32))

        if actor_obs_normalizer:
            self.actor_obs_normalizer = copy.deepcopy(actor_obs_normalizer)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        if single_obs_normalizer:
            self.single_obs_normalizer = copy.deepcopy(single_obs_normalizer)
        else:
            self.single_obs_normalizer = torch.nn.Identity()

    def forward(self, single_obs: torch.Tensor):
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
        return self.actor(latent_and_obs)

    @torch.jit.export
    def reset(self):
        self.obs_history.zero_()

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
