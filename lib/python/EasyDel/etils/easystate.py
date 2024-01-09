import os
from typing import Any, Callable, Optional, Mapping

import fjformer
import jax.tree_util
from flax import core
from flax import struct
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT
import optax
from .auto_tx import get_optimizer_and_scheduler
from ..etils import AVAILABLE_SCHEDULERS, AVAILABLE_OPTIMIZERS
from ..modules.easydel_modelling_utils import EasyDelFlaxPretrainedModel, EasyDelPretrainedConfig


class EasyDelState(struct.PyTreeNode):
    step: int
    module: Optional[EasyDelFlaxPretrainedModel] = struct.field(pytree_node=False)
    module_config: Optional[EasyDelPretrainedConfig] = struct.field(pytree_node=False)
    module_config_args: Optional[dict] = struct.field(pytree_node=True)
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: Optional[optax.OptState] = struct.field(pytree_node=True)
    tx_name: str
    sc_name: str
    model_type: str
    tx_init: Optional[dict] = struct.field(pytree_node=True)
    hyperparameters: Optional[dict] = struct.field(pytree_node=True)

    def apply_gradients(self, *, grads, **kwargs):
        if OVERWRITE_WITH_GRADIENT in grads:
            grads_with_opt = grads['params']
            params_with_opt = self.params['params']
        else:
            grads_with_opt = grads
            params_with_opt = self.params

        updates, new_opt_state = self.tx.update(
            grads_with_opt, self.opt_state, params_with_opt
        )
        new_params_with_opt = optax.apply_updates(params_with_opt, updates)
        if OVERWRITE_WITH_GRADIENT in grads:
            new_params = {
                'params': new_params_with_opt,
                OVERWRITE_WITH_GRADIENT: grads[OVERWRITE_WITH_GRADIENT]
            }
        else:
            new_params = new_params_with_opt
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    @classmethod
    def create(
            cls,
            *,
            apply_fn: Callable,
            params: core.FrozenDict[str, Any] | Mapping[str, Any],
            tx: optax.GradientTransformation,
            tx_name: str,
            sc_name: str,
            model_type: str,
            tx_init: Optional[dict] = None,
            hyperparameters: Optional[dict] = None,
            module: Optional[EasyDelFlaxPretrainedModel] = None,
            module_config: Optional[EasyDelPretrainedConfig] = None,
            module_config_args: Optional[dict] = None,
            **kwargs
    ):
        if tx_init is None:
            tx_init = {}
        params_with_opt = (
            params['params'] if OVERWRITE_WITH_GRADIENT in params else params
        )
        opt_state = tx.init(params_with_opt)

        if module_config is not None and module_config_args is None:
            module_config_args = module_config.to_dict()

        if module_config_args is not None:
            module_config_args = {
                k: v for k, v in module_config_args.items() if isinstance(
                    v, (
                        int, float, str, list, tuple, bool, dict
                    )
                ) and not isinstance(
                    v, jax.sharding.PartitionSpec
                )
            }
        return cls(
            step=0,
            apply_fn=apply_fn,
            module=module,
            params=params,
            tx=tx,
            opt_state=opt_state,
            tx_name=tx_name,
            sc_name=sc_name,
            tx_init=tx_init,
            model_type=model_type,
            hyperparameters=hyperparameters,
            module_config=module_config,
            module_config_args=module_config_args,
            **kwargs,
        )

    @classmethod
    def load(
            cls,
            *,
            step: int,
            apply_fn: Callable,
            params: core.FrozenDict[str, Any] | Mapping[str, Any],
            opt_state: Optional[optax.OptState],
            tx_name: AVAILABLE_OPTIMIZERS,
            sc_name: AVAILABLE_SCHEDULERS,
            model_type: str,
            tx_init: Optional[dict] = None,
            hyperparameters: Optional[dict] = None,
            module: Optional[EasyDelFlaxPretrainedModel] = None,
            module_config: Optional[EasyDelPretrainedConfig] = None,
            module_config_args: Optional[dict] = None,
            **kwargs
    ):
        if tx_init is None:
            tx_init = {}

        tx, sc = get_optimizer_and_scheduler(
            optimizer=tx_name,
            scheduler=sc_name,
            **tx_init
        )
        return cls(
            step=step,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            tx_name=tx_name,
            sc_name=sc_name,
            tx_init=tx_init,
            hyperparameters=hyperparameters,
            model_type=model_type,
            module=module,
            module_config=module_config,
            module_config_args=module_config_args,
            **kwargs,
        )

    def __str__(self):
        params_size = sum(n.size for n in jax.tree_util.tree_flatten(self.params)[0])
        opt_state_size = sum(n.size for n in jax.tree_util.tree_flatten(self.opt_state)[0])
        module_config_string = self.module_config.__str__().replace("\n",
                                                                    "\n\t\t"
                                                                    "") if self.module_config is not None else None
        string = f"""
{self.__class__.__name__}(
    step: int = {self.step}
    module: Optional[EasyDelFlaxPretrainedModel] = {self.module}
    module_config: Optional[EasyDelPretrainedConfig] = {module_config_string}
    apply_fn: Callable = {self.apply_fn}
    params: core.FrozenDict[str, Any] = {params_size} Parameters
    tx: optax.GradientTransformation = {self.tx_name} Optimizer with {self.sc_name} Scheduler
    opt_state: Optional[optax.OptState] = {opt_state_size} Parameters
    model_type: str = {self.model_type}
    hyperparameters: Optional[dict] = {self.hyperparameters}

)
"""
        return string

    @classmethod
    def load_state(
            cls,
            checkpoint_path: str | os.PathLike,
            init_optimizer: bool = False,
            state_shard_fns: Optional[Mapping[str, Callable]] = None,
            verbose: bool = False
    ):

        from ..modules.auto_easydel_model import get_modules_by_type

        checkpoint = fjformer.CheckpointManager.load_checkpoint(
            path=checkpoint_path,
            shard_fns=state_shard_fns,
            verbose=verbose,
        )
        cfg, module, convertor = get_modules_by_type(model_type=checkpoint["model_type"])
        module_config = checkpoint.pop("module_config", None)
        if checkpoint["module_config_args"] is not None:
            module_config = EasyDelPretrainedConfig.from_dict(checkpoint["module_config_args"])
        state = cls.load(
            apply_fn=module.__call__,
            module=module,
            module_config=module_config,
            **checkpoint
        )
        if init_optimizer:
            state = state.init_opt_state()
        return state

    def save_state(
            self,
            filename: str | os.PathLike,
            save_optimizer: bool = False,
            checkpoint_dir: Optional[str | os.PathLike] = None,
            verbose: bool = False,
            gather_fns: dict[Callable] = None,
            float_dtype: str | jax.numpy.dtype = None,
    ):
        state = self
        if not save_optimizer:
            state = self.replace(
                opt_state=None
            )
        fjformer.CheckpointManager.save_state_to_file(
            state=state,
            path=os.path.join(checkpoint_dir, filename) if checkpoint_dir is not None else filename,
            verbose=verbose,
            gather_fns=gather_fns,
            float_dtype=float_dtype,
        )

    def free_opt_state(self):
        return self.replace(
            opt_state=None
        )

    def init_opt_state(self):
        if self.opt_state is None:
            params_with_opt = (
                self.params['params'] if OVERWRITE_WITH_GRADIENT in self.params else self.params
            )
            opt_state = self.tx.init(params_with_opt)

            return self.replace(
                opt_state=opt_state
            )
        return self
