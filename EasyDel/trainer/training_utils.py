import dataclasses
import os
import typing

import wandb

from EasyDel.trainer.config import TrainArguments

import jax
import flax
import optax
from transformers import FlaxAutoModelForCausalLM, AutoConfig
from IPython.display import clear_output
from tqdm import tqdm
from EasyDel.utils.utils import Timers

from jax.experimental.pjit import pjit, with_sharding_constraint
from flax.training import train_state
from jax import numpy as jnp
from torch.utils.data import DataLoader
from fjutils import match_partition_rules, make_shard_and_gather_fns, StreamingCheckpointer, count_params


def fsdp_train_step(state, batch, label_in_the_field=False, scope_logits=True):
    batch = with_sharding_constraint(batch, PartitionSpec(('dp', 'fsdp')))

    def calculate_loss(params):
        logits = state.apply_fn(params=params, **batch,
                                return_dict=True).logits
        if label_in_the_field:
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits[..., :-1, :] if scope_logits else logits,
                labels=batch['label'])
        else:
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits[..., :-1, :] if scope_logits else logits,
                labels=batch['input_ids'][..., 1:])
        loss = jnp.mean(loss)
        return loss

    grad_fn = jax.value_and_grad(calculate_loss, has_aux=False)
    loss__, grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return state, loss__


def fsdp_eval_step(state, batch_eval):
    batch_eval = with_sharding_constraint(
        batch_eval,
        PartitionSpec(
            ('dp', 'fsdp'))
    )

    def calculate_loss(params):
        logits = state.apply_fn(params=params, **batch_eval,
                                return_dict=True).logits
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits[..., :-1, :],
                                                               labels=batch_eval['input_ids'][..., 1:])
        loss = jnp.mean(loss)
        return loss

    loss__ = calculate_loss(state.params)
    return loss__


def predict(state, input_ids):
    input_ids = with_sharding_constraint(input_ids, PartitionSpec(('dp', 'fsdp')))
    pred = state.apply_fn(params=state.params, input_ids=input_ids, return_dict=True)
    token = jnp.argmax(jax.nn.softmax(pred.logits)[:, -1, :])
    input_ids = jnp.concatenate([input_ids, token.reshape(1, -1)], axis=-1)
    return input_ids


@dataclasses.dataclass
class OutputFineTuner:
    train_state: typing.Any
    predict_fun: typing.Any
    mesh: typing.Any
    ckpt_stream: typing.Any
    gather_fns: typing.Any
    shard_fns: typing.Any
    last_save_file_name: str


@dataclasses.dataclass
class OutputType:
    train_state: typing.Any
    predict_fun: typing.Any
    mesh: typing.Any
    ckpt_stream: typing.Any
    gather_fns: typing.Any
    shard_fns: typing.Any
    dataloader_train: typing.Any
    dataloader_eval: typing.Any
    tx: typing.Any
    scheduler: typing.Any
    model: typing.Any


def get_training_modules(
        dataset_train,
        ckpt_path,
        training_arguments: TrainArguments,
        use_pjit_attention_force: bool = True,
        dataset_eval=None,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        fully_fsdp=True,
        custom_rule=None,
        extra_configs=None,
        remove_ckpt_after_load: bool = False,
        model_class=None,
        configs_to_init_model_class=None,
        model_parameters=None,
        do_shard_fns: bool = True,
) -> OutputType:
    if extra_configs is None:
        extra_configs = {}
    timer = Timers(
        use_wandb=False,
        tensorboard_writer=training_arguments.get_board()
    )
    max_length = training_arguments.max_length

    def collate_fn(batch):
        rs = {}
        for key in batch[0].keys():
            ssp = [jnp.array(f[key])[..., -max_length:] for f in batch]
            rs[key] = jnp.stack(ssp).reshape(-1, ssp[0].shape[-1])
        return rs

    timer(
        'configuring data loaders'
    ).start()
    dataloader_train = DataLoader(dataset_train, collate_fn=collate_fn,
                                  batch_size=training_arguments.total_batch_size, drop_last=True)
    max_steps_train = training_arguments.num_train_epochs * len(
        dataloader_train) if training_arguments.max_steps is None else training_arguments.max_steps
    if dataset_eval is not None and training_arguments.do_eval:
        dataloader_eval = DataLoader(dataset_eval, collate_fn=collate_fn,
                                     batch_size=training_arguments.total_batch_size, drop_last=True)
        max_steps_eval = len(dataloader_eval) if training_arguments.max_steps is None else training_arguments.max_steps
    timer(
        'configuring data loaders'
    ).stop()
    timer.log(['configuring data loaders'])
    timer(
        'loading / creating config, model, optimizers'
    ).start()

    if model_class is None:
        config = AutoConfig.from_pretrained(training_arguments.model_id, trust_remote_code=True
                                            , gradient_checkpointing=training_arguments.gradient_checkpointing,
                                            use_pjit_attention_force=use_pjit_attention_force, **extra_configs
                                            )

        assert hasattr(config, 'get_partition_rules')
        model = FlaxAutoModelForCausalLM.from_config(config, trust_remote_code=True, dtype=dtype,
                                                     param_dtype=param_dtype,
                                                     _do_init=False)

    else:
        assert custom_rule is not None, 'if you are using custom model to init you must' \
                                        ' pass custom_rule for partition rules '
        model = model_class(
            **configs_to_init_model_class,
            _do_init=False
        )

    tx, scheduler = training_arguments.get_optimizer_and_scheduler(max_steps_train)
    timer(
        'loading / creating config, model, optimizers'
    ).stop()
    timer.log(['loading / creating config, model, optimizers'])

    def init_fn():
        # from flax.training import train_state
        params__ = model.init_weights(jax.random.PRNGKey(0), (1, max_length))
        if dtype == jnp.bfloat16:
            params__ = model.to_bf16(params__)
        elif dtype == jnp.float16:
            params__ = model.to_fp16(params__)
        return train_state.TrainState.create(
            tx=tx,
            params=flax.core.freeze({'params': params__}),
            apply_fn=model.__call__
        )

    def create_train_state_from_params(params_):
        # from flax.training import train_state
        return train_state.TrainState.create(
            tx=tx,
            apply_fn=model.__call__,
            params=params_
        )

    timer(
        'creating functions'
    ).start()
    train_state_shape = jax.eval_shape(init_fn)
    train_state_partition_spec = match_partition_rules(
        config.get_partition_rules(fully_fsdp=fully_fsdp) if custom_rule is None else custom_rule,
        train_state_shape)
    sharded_create_from_params_fn = pjit(
        create_train_state_from_params,
        in_shardings=(train_state_partition_spec.params,),
        out_shardings=train_state_partition_spec,
        donate_argnums=(0,)
    )
    sharded_train_step_fn = pjit(
        fsdp_train_step,
        in_shardings=(train_state_partition_spec, PartitionSpec(), PartitionSpec(), PartitionSpec()),
        out_shardings=(train_state_partition_spec, PartitionSpec()),
        donate_argnums=(0, 0, 0),
    )
    sharded_predict = pjit(predict, out_shardings=PartitionSpec(),
                           in_shardings=(train_state_partition_spec, PartitionSpec()))
    mesh = training_arguments.get_mesh()
    training_arguments.ckpt_path_exists()
    ckpt_streamer = training_arguments.get_streaming_checkpointer()
    timer(
        'creating functions'
    ).stop()
    timer.log(['creating functions'])
    with mesh:
        timer(
            'loading parameters'
        ).start()
        shard_fns, gather_fns = make_shard_and_gather_fns(train_state_partition_spec, dtype_specs=dtype)
        if model_parameters is None:
            _, params = StreamingCheckpointer.load_trainstate_checkpoint(
                f'params::{ckpt_path}', train_state_shape, shard_fns
            )
        else:
            params = model_parameters if not do_shard_fns else jax.tree_util.tree_map(lambda f, x: f(x), shard_fns,
                                                                                      model_parameters)

        if remove_ckpt_after_load:
            os.remove(ckpt_path)

        sharded_train_state_ = sharded_create_from_params_fn(params)
        timer(
            'loading parameters'
        ).stop()
        timer.log(['loading parameters'])
        count_params(sharded_train_state_.params)

        timer.write(timer.timers.keys(), 0)

    return OutputType(
        train_state=sharded_train_step_fn,
        dataloader_train=dataloader_train,
        dataloader_eval=dataloader_eval,
        shard_fns=shard_fns,
        gather_fns=gather_fns,
        scheduler=scheduler,
        tx=tx,
        predict_fun=sharded_predict,
        ckpt_stream=ckpt_streamer,
        mesh=mesh,
        model=model
    )
