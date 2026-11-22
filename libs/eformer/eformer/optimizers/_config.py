# Copyright 2026 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import typing as tp
import warnings
from dataclasses import asdict, dataclass, fields

import jax.numpy as jnp

T = tp.TypeVar("T", bound="SerializationMixin")


class SerializationMixin:
    """
    Mixin class providing serialization capabilities for configuration classes.

    This class provides methods to convert instances to and from dictionaries and JSON strings,
    making it easy to serialize and deserialize configuration objects.

    Methods:
        to_dict: Convert the instance to a dictionary, filtering out private fields.
        from_dict: Create an instance from a dictionary with error checking.
        to_json: Serialize the instance to a JSON string.
        from_json: Create an instance from a JSON string.
    """

    def to_dict(self) -> dict[str, tp.Any]:
        """
        Convert the instance to a dictionary, filtering out private fields.

        Returns:
            dict: A dictionary representation of the instance, excluding private fields.
        """
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls: type[T], data: dict[str, tp.Any]) -> T:
        """
        Create an instance from a dictionary with error checking.

        Args:
            data (dict): A dictionary containing the data to populate the instance.

        Returns:
            T: An instance of the class populated with the provided data.

        Raises:
            Warning: If unexpected keys are present in the input dictionary.
        """
        valid_fields = {f.name for f in fields(cls)}
        extra_keys = set(data.keys()) - valid_fields
        if extra_keys:
            warnings.warn(
                f"Ignoring unexpected keys {extra_keys} for {cls.__name__}",
                stacklevel=2,
            )

        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def to_json(self) -> str:
        """
        Serialize the instance to a JSON string.

        Returns:
            str: A JSON string representation of the instance.
        """
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls: type[T], json_str: str) -> T:
        """
        Create an instance from a JSON string.

        Args:
            json_str (str): A JSON string containing the data to populate the instance.

        Returns:
            T: An instance of the class populated with the data from the JSON string.
        """
        return cls.from_dict(json.loads(json_str))


@dataclass
class SchedulerConfig(SerializationMixin):
    """
    Configuration class for learning rate schedulers.

    Attributes:
        scheduler_type (Optional[Literal["linear", "cosine"]]): Type of scheduler to use.
        learning_rate (float): Initial learning rate. Defaults to 5e-5.
        learning_rate_end (Optional[float]): Final learning rate for linear scheduler.
        warmup_steps (Optional[int]): Number of warmup steps.
        steps (Optional[int]): Total number of steps. Required for non-constant schedulers.
        exponent (float): Exponent for polynomial decay. Defaults to 1.0.

    Methods:
        __post_init__: Validates the configuration after initialization.
        _validate: Performs validation checks on the configuration.
    """

    scheduler_type: tp.Literal["linear", "cosine"] | None = None
    learning_rate: float = 5e-5
    learning_rate_end: float | None = None
    warmup_steps: int | None = None
    steps: int | None = None
    exponent: float = 1.0

    def __post_init__(self):
        """
        Validates the configuration after initialization.
        """
        self._validate()

    def _validate(self):
        """
        Performs validation checks on the configuration.

        Raises:
            ValueError: If the configuration is invalid.
        """

        if self.scheduler_type is not None and self.steps is None:
            raise ValueError("Steps must be specified for non-constant schedulers")

        if self.scheduler_type == "linear":
            if self.learning_rate_end is None:
                raise ValueError("Linear scheduler requires learning_rate_end")

        if self.warmup_steps is not None:
            if self.steps is None:
                raise ValueError("Steps required when using warmup")
            if self.warmup_steps >= self.steps:
                raise ValueError("Warmup steps must be less than total steps")


@dataclass
class AdafactorConfig(SerializationMixin):
    """
    Configuration class for the Adafactor optimizer.

    Attributes:
        min_dim_size_to_factor (int): Minimum dimension size for factoring. Defaults to 128.
        decay_rate (float): Decay rate for second-moment estimator. Defaults to 0.8.
        decay_offset (int): Decay offset. Defaults to 0.
        multiply_by_parameter_scale (bool): Whether to multiply by parameter scale. Defaults to True.
        clipping_threshold (Optional[float]): Clipping threshold for updates. Defaults to 1.0.
        momentum (Optional[float]): Momentum factor. Defaults to None.
        dtype_momentum (jnp.dtype): Data type for momentum. Defaults to jnp.float32.
        weight_decay_rate (Optional[float]): Weight decay rate. Defaults to None.
        eps (float): Small constant for numerical stability. Defaults to 1e-30.
        factored (bool): Whether to use factored second-moment estimates. Defaults to True.
    """

    min_dim_size_to_factor: int = 128
    decay_rate: float = 0.8
    decay_offset: int = 0
    multiply_by_parameter_scale: bool = True
    clipping_threshold: float | None = 1.0
    momentum: float | None = None
    dtype_momentum: jnp.dtype = jnp.float32
    weight_decay_rate: float | None = None
    eps: float = 1e-30
    factored: bool = True


@dataclass
class AdamWConfig(SerializationMixin):
    """
    Configuration class for the AdamW optimizer.

    Attributes:
        b1 (float): Exponential decay rate for the first moment estimates. Defaults to 0.9.
        b2 (float): Exponential decay rate for the second moment estimates. Defaults to 0.999.
        eps (float): Small constant for numerical stability. Defaults to 1e-8.
        eps_root (float): Small constant for root calculations. Defaults to 0.0.
        mu_dtype (Optional[jnp.dtype]): Data type for momentum. Defaults to None.
    """

    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    eps_root: float = 0.0
    mu_dtype: jnp.dtype | None = None


@dataclass
class LionConfig(SerializationMixin):
    """
    Configuration class for the Lion optimizer.

    Attributes:
        b1 (float): Exponential decay rate for the first moment estimates. Defaults to 0.9.
        b2 (float): Exponential decay rate for the second moment estimates. Defaults to 0.99.
        mu_dtype (Optional[jnp.dtype]): Data type for momentum. Defaults to None.
    """

    b1: float = 0.9
    b2: float = 0.99
    mu_dtype: jnp.dtype | None = None


@dataclass
class RMSPropConfig(SerializationMixin):
    """
    Configuration class for the RMSProp optimizer.

    Attributes:
        decay (float): Decay rate for the moving average. Defaults to 0.9.
        initial_scale (float): Initial scale for the moving average. Defaults to 0.0.
        momentum (Optional[float]): Momentum factor. Defaults to None.
        nesterov (bool): Whether to use Nesterov momentum. Defaults to False.
        eps (float): Small constant for numerical stability. Defaults to 1e-8.
    """

    decay: float = 0.9
    initial_scale: float = 0.0
    momentum: float | None = None
    nesterov: bool = False
    eps: float = 1e-8


@dataclass
class MuonConfig(SerializationMixin):
    """
    Configuration class for the Muon (Momentum Orthogonalized by Newton-schulz) optimizer.

    Muon is designed for 2D parameters (matrices) and uses Newton-Schulz method to
    orthogonalize momentum. Non-2D parameters are processed through an Adam optimizer.

    Attributes:
        ns_coeffs (tuple[float, float, float]): Coefficients for the Newton-schulz method.
            Defaults to (3.4445, -4.775, 2.0315).
        ns_steps (int): Number of Newton-schulz iterations. Defaults to 5.
        beta (float): Decay rate for the exponentially weighted average of grads. Defaults to 0.95.
        eps (float): Term added to the denominator to improve numerical stability. Defaults to 1e-8.
        weight_decay (float): Strength of the weight decay regularization. Defaults to 0.0.
        weight_decay_mask (Any | None): Weight decay mask. Defaults to None.
        mu_dtype (jnp.dtype | None): Data type for momentum computation. Defaults to None.
        nesterov (bool): Whether to use Nesterov momentum. Defaults to True.
        adaptive (bool): Whether to scale the updates by the dual norm of the original updates.
            Defaults to False.
        adam_b1 (float): Exponential decay rate for first moment estimates in Adam (for non-2D params).
            Defaults to 0.9.
        adam_b2 (float): Exponential decay rate for second moment estimates in Adam (for non-2D params).
            Defaults to 0.999.
        adam_eps_root (float): Small constant for root calculations in Adam. Defaults to 0.0.
        adam_weight_decay (float): Weight decay for Adam optimizer (for non-2D params). Defaults to 0.0.
    """

    ns_coeffs: tuple[float, float, float] = (3.4445, -4.775, 2.0315)
    ns_steps: int = 5
    beta: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0
    weight_decay_mask: tp.Any | None = None
    mu_dtype: jnp.dtype | None = None
    nesterov: bool = True
    adaptive: bool = False
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    adam_eps_root: float = 0.0
    adam_weight_decay: float = 0.0


@dataclass
class SoapConfig(SerializationMixin):
    """
    Configuration class for the SOAP (Shampoo with Orthogonal and Adaptive Preconditioning) optimizer.

    SOAP combines Shampoo's second-order optimization with orthogonal preconditioning
    and adaptive scheduling for improved convergence.

    Attributes:
        weight_decay (float): Weight decay coefficient. Defaults to 0.0.
        beta1 (float): Momentum parameter for first moment estimates. Defaults to 0.95.
        beta2 (float): Momentum parameter for second moment estimates. Defaults to 0.95.
        shampoo_beta (float): Beta parameter for Shampoo preconditioning. Defaults to 0.95.
        epsilon (float): Small constant for numerical stability. Defaults to 1e-8.
        max_grad_norm (float | None): Maximum gradient norm for clipping. Defaults to 1.0.
        haps (list[int] | None): HAP schedule parameters. Defaults to None.
        schedule_list (list[str] | None): Schedule list. Defaults to None.
        precondition_frequency (int): Frequency of preconditioner updates. Defaults to 10.
        max_precond_dim (int): Maximum dimension for preconditioning. Defaults to 10000.
        merge_small_dims (bool): Whether to merge small dimensions. Defaults to True.
        one_diag (bool): Whether to use diagonal preconditioning only. Defaults to False.
        target_merged_dim_size (int): Target size for merged dimensions. Defaults to 2048.
        mu_dtype (jnp.dtype | None): Data type for momentum computation. Defaults to None.
        precond_dtype (jnp.dtype | None): Data type for preconditioners. Defaults to None.
        partition_grads_into_blocks (bool): Whether to partition gradients into blocks. Defaults to True.
        block_size (int): Block size for gradient partitioning. Defaults to 256.
    """

    weight_decay: float = 0.0
    beta1: float = 0.95
    beta2: float = 0.95
    shampoo_beta: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float | None = 1.0
    haps: list[int] | None = None
    schedule_list: list[str] | None = None
    precondition_frequency: int = 10
    max_precond_dim: int = 10000
    merge_small_dims: bool = True
    one_diag: bool = False
    target_merged_dim_size: int = 2048
    mu_dtype: jnp.dtype | None = None
    precond_dtype: jnp.dtype | None = None
    partition_grads_into_blocks: bool = True
    block_size: int = 256


@dataclass
class MarsConfig(SerializationMixin):
    """
    Configuration class for the Mars optimizer.

    Mars (Matrix-wise Adaptive Regularized Scaling) optimizer improves upon Adam
    by using matrix-wise adaptive regularization.

    Reference: https://arxiv.org/abs/2411.10438

    Attributes:
        weight_decay (float): Weight decay coefficient. Defaults to 0.1.
        beta1 (float): Exponential decay rate for first moment estimates. Defaults to 0.95.
        beta2 (float): Exponential decay rate for second moment estimates. Defaults to 0.99.
        gamma (float): Decay rate for exponentially weighted average of gradient from previous step.
            Defaults to 0.025.
        epsilon (float): Small constant for numerical stability. Defaults to 1e-8.
        max_grad_norm (float | None): Maximum gradient norm for clipping. Defaults to 1.0.
    """

    weight_decay: float = 0.1
    beta1: float = 0.95
    beta2: float = 0.99
    gamma: float = 0.025
    epsilon: float = 1e-8
    max_grad_norm: float | None = 1.0


@dataclass
class KronConfig(SerializationMixin):
    """
    Configuration class for the Kron (PSGD Kron) optimizer.

    Kron uses Kronecker-factored preconditioners for efficient second-order optimization,
    particularly effective for neural network training.

    Attributes:
        beta1 (float): Momentum parameter. Common values are 0.9 or 0.95. Defaults to 0.9.
        weight_decay (float): Weight decay coefficient. Defaults to 0.1.
        max_grad_norm (float | None): Optional gradient norm clipping value. Defaults to 1.0.
        normalize_grads (bool): Whether to normalize incoming gradients to unit norm layer-wise.
            Can help with stability. Defaults to False.
        preconditioner_update_probability (float): Final probability of updating the preconditioner.
            Defaults to 0.05 (update every 20 steps on average).
        update_prob_flat_start (int): Number of steps to keep update probability at 1.0 before
            annealing. Defaults to 500.
        max_size_triangular (int): Maximum size for triangular factorization. Defaults to 25000.
        min_ndim_triangular (int): Minimum number of dimensions for triangular factorization.
            Defaults to 2.
        memory_save_mode (str | None): Memory saving mode. Can be None, "one_diag", or "all_diag".
            Defaults to None.
        preconditioner_lr (float): Learning rate for preconditioner updates. Defaults to 0.1.
        preconditioner_init_scale (float): Initial scale for preconditioner. Defaults to 1.0.
        mu_dtype (jnp.dtype | None): Data type for momentum computation. Defaults to None.
        precond_dtype (jnp.dtype | None): Data type for preconditioners. Defaults to None.
        precond_update_precision (str | None): Precision for preconditioner updates. Defaults to "tensorfloat32".
        precond_grads_precision (str | None): Precision for gradient preconditioning. Defaults to None.
        lax_map_scanned_layers (bool): Whether to use lax.map for scanned layers. Defaults to False.
        lax_map_batch_size (int): Batch size for lax.map. Defaults to 8.
        merge_small_dims (bool): Whether to merge small dimensions. Defaults to True.
        target_merged_dim_size (int): Target size for merged dimensions. Defaults to 8192.
        partition_grads_into_blocks (bool): Whether to partition gradients into blocks. Defaults to True.
        block_size (int): Block size for gradient partitioning. Defaults to 256.
    """

    beta1: float = 0.9
    weight_decay: float = 0.1
    max_grad_norm: float | None = 1.0
    normalize_grads: bool = False
    preconditioner_update_probability: float = 0.05
    update_prob_flat_start: int = 500
    max_size_triangular: int = 25000
    min_ndim_triangular: int = 2
    memory_save_mode: str | None = None
    preconditioner_lr: float = 0.1
    preconditioner_init_scale: float = 1.0
    mu_dtype: jnp.dtype | None = None
    precond_dtype: jnp.dtype | None = None
    precond_update_precision: str | None = "tensorfloat32"
    precond_grads_precision: str | None = None
    lax_map_scanned_layers: bool = False
    lax_map_batch_size: int = 8
    merge_small_dims: bool = True
    target_merged_dim_size: int = 8192
    partition_grads_into_blocks: bool = True
    block_size: int = 256


@dataclass
class ScionConfig(SerializationMixin):
    """
    Configuration class for the Scion optimizer.

    Scion combines spectral normalization with sign-based updates for different parameter types,
    providing efficient training for neural networks.

    Reference: https://arxiv.org/abs/2502.07529

    Attributes:
        momentum (float): Momentum parameter for both spectral and sign methods. Defaults to 0.95.
        backend_steps (int): Number of steps for Newton-Schulz orthogonalization. Defaults to 10.
        beta1 (float): Beta1 parameter for sign method. Defaults to 0.9.
        epsilon (float): Small constant for numerical stability. Defaults to 1e-8.
        unconstrained (bool): Whether to use unconstrained version. Defaults to False.
        spectral_radius (float): Scaling factor for spectral method. Defaults to 50.
        sign_radius (float): Scaling factor for sign method. Defaults to 3000.
    """

    momentum: float = 0.95
    backend_steps: int = 10
    beta1: float = 0.9
    epsilon: float = 1e-8
    unconstrained: bool = False
    spectral_radius: float = 50
    sign_radius: float = 3000


@dataclass
class WhiteKronConfig(SerializationMixin):
    """
    Configuration class for the White Kron optimizer.

    White Kron is a Kronecker-factored preconditioned optimizer that uses different
    update styles (skew or quad) for efficient second-order optimization.

    Attributes:
        lr_style (str | None): Learning rate style. Defaults to "adam".
        b1 (float): Exponential decay rate for first moment estimates. Defaults to 0.95.
        normalize_grads (bool): Whether to normalize gradients. Defaults to False.
        max_size_dense (int): Maximum size for dense preconditioning. Defaults to 16384.
        preconditioner_lr (float): Learning rate for preconditioner updates. Defaults to 0.7.
        preconditioner_init_scale (float): Initial scale for preconditioner. Defaults to 1.0.
        dtype (str | jnp.dtype): Data type for computations. Defaults to jnp.bfloat16.
        scanned_layers (Any | None): Scanned layers configuration. Defaults to None.
        block_size (int): Block size for matrix operations. Defaults to 256.
        pipeline_axis_name (str | None): Name of pipeline axis for sharding. Defaults to None.
        pipeline_axis_size (int): Size of pipeline axis. Defaults to 1.
        params_partition_specs (Any | None): Parameter partition specifications. Defaults to None.
        noise_scale (float): Scale of noise added for numerical stability. Defaults to 1e-9.
        weight_decay (float): Weight decay coefficient. Defaults to 0.1.
        weight_decay_mask (Any | None): Weight decay mask. Defaults to None.
    """

    lr_style: str | None = "adam"
    b1: float = 0.95
    normalize_grads: bool = False
    max_size_dense: int = 16384
    preconditioner_lr: float = 0.7
    preconditioner_init_scale: float = 1.0
    dtype: str | jnp.dtype = jnp.bfloat16
    scanned_layers: tp.Any | None = None
    block_size: int = 256
    pipeline_axis_name: str | None = None
    pipeline_axis_size: int = 1
    params_partition_specs: tp.Any | None = None
    noise_scale: float = 1e-9
    weight_decay: float = 0.1
    weight_decay_mask: tp.Any | None = None
