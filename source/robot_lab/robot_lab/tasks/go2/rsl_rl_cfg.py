from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 300000
    save_interval = 100
    experiment_name = "go2_rough" 
    
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class MoeCtsSymmetryCfg:
    use_symmetric_augmentation = False
    symmetry_class = "robot_lab.tasks.go2.mdp.symmetry:Go2MoECTSSymmetry"

@configclass
class RslRlMoeCtsActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name = "ActorCriticMoECTS"
    init_noise_std = 1.0
    expert_num = 8 # number of experts in the student model
    gating_top_k: int | None = None  # sparse gating: only top-k experts active (None = dense)
    latent_dim = 32
    norm_type = 'l2norm' # normalization type for encoders: l2norm, simnorm
    teacher_encoder_hidden_dims = [512, 256]
    student_encoder_hidden_dims = [512, 256, 256]
    actor_hidden_dims=[512, 256, 128]
    critic_hidden_dims=[512, 256, 128]
    activation="elu"
    actor_obs_normalization = False
    critic_obs_normalization = False

@configclass
class RslRlMoeCtsCnnGruActorCriticCfg(RslRlMoeCtsActorCriticCfg):
    class_name = "ActorCriticMoECTSCNNGRU"
    actor_image_obs_groups = ["depth"]
    image_shape = (60, 60)
    depth_num_frames = 4
    cnn_channels = (16, 32, 64)
    cnn_kernel_size = 3
    cnn_stride = 2
    cnn_padding = 1
    cnn_pooled_shape = (15, 15)
    gru_hidden_dim = 225
    gru_num_layers = 1
    # MGDP-style aux heads (denoise / height recon / align). Synced from
    # Go2WD435iEnvCfg.use_mgdp_depth_aux in train.py; keep False for current pipeline.
    enable_depth_aux = False
    height_map_shape = (17, 11)
    depth_align_dim = 32
    clean_depth_obs_group = "clean_depth"
    height_map_obs_group = "height_map"


@configclass
class RslRlRedoCfg:
    """Recycling Dormant Neurons (ReDo) hyperparameters."""

    enabled = False
    reset_period = 10_000  # 200_000 # gradient steps between recycle events
    reset_start_step = 0
    reset_end_step = 2_500_000
    logging_period = 1_000 # 20_000
    recycle_rate = 0.3
    score_type = "redo"  # redo | random | redo_inverted | threshold
    dead_neurons_threshold = 0.0
    init_method_outgoing = "zero"  # zero | random
    weight_scaling = False
    incoming_scale = 1.0
    outgoing_scale = 1.0
    sub_mean_score = False
    batch_size_statistics = 256
    module_names = (
        "actor",
        "critic",
        "teacher_encoder",
        "student_moe_encoder",
        "student_cnn_gru",
    )
    reset_start_layer_idx = 0


@configclass
class RslRlMoeCtsAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name = "MoECTS"
    value_loss_coef = 1.0
    load_balance_coef = 0.01  # coefficient for load balance loss
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 5e-4 # 1e-3
    student_encoder_learning_rate = 5e-4 # 1e-3
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    betas = (0.9, 0.999)
    weight_decay = 0.0
    desired_kl = 0.01
    max_grad_norm = 1.0
    teacher_env_ratio = 0.75  # percentage of envs assigned to teacher
    symmetry_cfg = MoeCtsSymmetryCfg()
    # Depth aux losses (active when enable_depth_aux and coefs > 0).
    depth_denoise_coef = 0.0
    height_recon_coef = 0.0
    depth_align_coef = 0.0
    depth_align_loss_type = "infonce"  # MGDP default; also supports "mse"
    depth_align_temperature = 0.1
    redo_cfg = RslRlRedoCfg()

@configclass
class MoECTSRunnerCfg(RslRlOnPolicyRunnerCfg):
    experiment_name = "go2_moe_cts"
    class_name = "OnPolicyRunnerCTS"
    num_steps_per_env = 24
    max_iterations = 300000
    save_interval = 100
    policy = RslRlMoeCtsActorCriticCfg()
    algorithm = RslRlMoeCtsAlgorithmCfg()

@configclass
class MoECTSD435iRunnerCfg(MoECTSRunnerCfg):
    experiment_name = "go2_moe_cts_d435i"
    policy = RslRlMoeCtsCnnGruActorCriticCfg()
    obs_groups = {
        "policy": ["policy", "depth"],
        "critic": ["critic"],
    }

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.symmetry_cfg.use_symmetric_augmentation = True

@configclass
class MoECTSSymmetryRunnerCfg(MoECTSRunnerCfg):
    experiment_name = "go2_moe_cts_symmetry"
    def __post_init__(self):
        super().__post_init__()
        self.algorithm.symmetry_cfg.use_symmetric_augmentation = True

# concat elu inspired by concat relu from https://arxiv.org/pdf/2303.07507
@configclass
class MoECTSCatELURunnerCfg(MoECTSRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.activation = 'cat_elu'
