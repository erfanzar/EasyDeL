import os
import sys
from typing import Literal, Optional

import numpy as np
from transformers.utils import flatten_dict


class PPOConfig:
    def __init__(
            self,
            exp_name: str = os.path.basename(sys.argv[0])[: -len(".py")],
            seed: int = 0,
            task_name: Optional[str] = None,
            model_name: Optional[str] = None,
            query_dataset: Optional[str] = None,
            reward_model: Optional[str] = None,
            remove_unused_columns: bool = True,
            tracker_kwargs: Optional[dict] = None,
            accelerator_kwargs: Optional[dict] = None,
            project_kwargs: Optional[dict] = None,
            tracker_project_name: str = "trl",
            push_to_hub_if_best_kwargs: Optional[dict] = None,
            steps: int = 20000,
            learning_rate: float = 1e-5,
            adap_kl_ctrl: bool = True,
            init_kl_coef: Optional[float] = 0.2,
            kl_penalty: Literal["kl", "abs", "mse", "full"] = "kl",
            target: Optional[float] = 6,
            horizon: Optional[float] = 10000,
            gamma: float = 1,
            lam: float = 0.95,
            cliprange: float = 0.2,
            cliprange_value: float = 0.2,
            vf_coef: float = 0.1,
            batch_size: int = 256,
            gradient_accumulation_steps: int = 1,
            ppo_epochs: int = 4,
            max_grad_norm: Optional[float] = None,
            target_kl: float = 1,
            compare_steps: int = 1,
            ratio_threshold: float = 10.0,
            use_score_scaling: bool = False,
            use_score_norm: bool = False,
            score_clip: Optional[float] = None,
            whiten_rewards: bool = False,
            is_encoder_decoder: Optional[bool] = None,
            warmup_steps: Optional[int] = 0,
            learning_rate_end: float = 1e-5,
            extra_optimizer_kwargs: dict | None = None,
            weight_decay: Optional[float] = 0.01,
    ):
        """
    Configuration class for PPOTrainer
    :param exp_name: str : the name of this experiment (by default is the file name without the extension name)
    :param seed: int :Seed value for random generations
    :param task_name: Optional[str] : Name of task to use - used only for tracking purposes
    :param model_name: Optional[str] :Name of model to use - used only for tracking purposes
    :param query_dataset: Optional[str] :Name of dataset to query - used only for tracking purposes
    :param reward_model: Optional[str] :The reward model to use - used only for tracking purposes
    :param remove_unused_columns: bool : Remove unused columns from the dataset if `datasets.Dataset` is used
    :param tracker_kwargs: Optional[dict] : Keyword arguments for the tracker
    :param accelerator_kwargs: Optional[dict] :Keyword arguments for the accelerator
    :param project_kwargs: Optional[dict] : Keyword arguments for the accelerator project config (e.g. `logging_dir`)
    :param tracker_project_name: str :Name of project to use for tracking
    :param push_to_hub_if_best_kwargs: Optional[dict] :Keyword arguments for pushing model to the hub during training
    (e.g. pretrained_model_name_or_path).
    :param steps: int : Number of training steps
    :param learning_rate: float :Adam learning rate
    :param adap_kl_ctrl: bool :Use adaptive KL control, otherwise linear
    :param init_kl_coef: Optional[float] : Initial KL penalty coefficient (used for adaptive and linear control)
    :param kl_penalty: Literal["kl", "abs", "mse", "full"] : kl penalty options: 'kl': model_logp - ref_logp,
    'abs': abs(kl),  'mse': mean squared error mse(kl) and 'full': the actual kl for all tokens in the distribution
    :param target: Optional[float] :Target KL value for adaptive KL control
    :param horizon: Optional[float] :Horizon for adaptive KL control
    :param gamma: float :Gamma parameter for advantage calculation
    :param lam: float : Lambda parameter for advantage calculation
    :param cliprange: float : Range for clipping in PPO policy gradient loss
    :param cliprange_value: float : Range for clipping values in loss calculation
    :param vf_coef: float : Scaling factor for value loss
    :param batch_size: int :Number of samples per optimisation step
    :param gradient_accumulation_steps: int :The number of gradient accumulation steps
    :param ppo_epochs: int : Number of optimisation epochs per batch of samples
    :param max_grad_norm: Optional[float] :Maximum gradient norm for gradient clipping
    :param target_kl: float :Stop early if we exceed this value by over 50%
    :param compare_steps: int : Number of steps between comparison of the current reward with the best seen so far
    :param ratio_threshold : float :Skip mini-batches with high PPO ratios that can cause loss spikes
    :param use_score_scaling: bool : Use score scaling
    :param use_score_norm: bool : Use score normalization. Only applicable if use_score_scaling is True
    :param score_clip: Optional[float] :Score clipping
    :param whiten_rewards: bool :Whiten the rewards before compute advantages
    :param is_encoder_decoder: Optional[bool] :TO BE FILLED In RUNTIME: Whether the model is an encoder-decoder model
    :param warmup_steps: Optional[int]:
    :param learning_rate_end: float :
    :param extra_optimizer_kwargs: dict | None :
    :param weight_decay: Optional[float] : Weight decay is Optimizer Weight decay :\
        """

        tracker_kwargs = tracker_kwargs if tracker_kwargs is not None else {}
        accelerator_kwargs = accelerator_kwargs if accelerator_kwargs is not None else {}
        project_kwargs = project_kwargs if project_kwargs is not None else {}
        push_to_hub_if_best_kwargs = push_to_hub_if_best_kwargs if push_to_hub_if_best_kwargs is not None else {}
        self.exp_name = exp_name
        self.seed = seed
        self.task_name = task_name
        self.model_name = model_name
        self.query_dataset = query_dataset
        self.reward_model = reward_model
        self.remove_unused_columns = remove_unused_columns
        self.tracker_kwargs = tracker_kwargs
        self.accelerator_kwargs = accelerator_kwargs
        self.project_kwargs = project_kwargs
        self.tracker_project_name = tracker_project_name
        self.push_to_hub_if_best_kwargs = push_to_hub_if_best_kwargs
        self.steps = steps
        self.learning_rate = learning_rate
        self.adap_kl_ctrl = adap_kl_ctrl
        self.init_kl_coef = init_kl_coef
        self.kl_penalty = kl_penalty
        self.target = target
        self.horizon = horizon
        self.gamma = gamma
        self.lam = lam
        self.cliprange = cliprange
        self.cliprange_value = cliprange_value
        self.vf_coef = vf_coef
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.ppo_epochs = ppo_epochs
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.compare_steps = compare_steps
        self.ratio_threshold = ratio_threshold
        self.use_score_scaling = use_score_scaling
        self.use_score_norm = use_score_norm
        self.score_clip = score_clip
        self.whiten_rewards = whiten_rewards
        self.is_encoder_decoder = is_encoder_decoder
        self.warmup_steps = warmup_steps
        self.learning_rate_end = learning_rate_end
        self.extra_optimizer_kwargs = extra_optimizer_kwargs
        self.weight_decay = weight_decay
        self.total_ppo_epochs = int(np.ceil(self.steps / (self.batch_size * self.gradient_accumulation_steps)))
        assert self.kl_penalty in ["kl", "abs", "mse", "full"]

    def to_dict(self):
        output_dict = {}
        for key, value in self.__dict__.items():
            output_dict[key] = value
        return flatten_dict(output_dict)
