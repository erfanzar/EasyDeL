import flax.core

from EasyDel.transform import llama_from_pretrained

from EasyDel import TrainArguments, CausalLMTrainer
from datasets import load_dataset
from huggingface_hub import HfApi
from EasyDel import configs
from jax import numpy as jnp
import EasyDel
from absl import flags, app
from fjutils import get_float_dtype_by_name

FLAGS = flags.FLAGS

flags.DEFINE_string(
    name='dataset_name',
    required=True,
    help='dataset from huggingface and must contains input_ids and attention_mask'
         ' or other things that model might need to be passed into',
    default=None
)

flags.DEFINE_string(
    name='ckpt_path',
    required=False,
    help='path to model weights for example (ckpt/llama_easydel_format)',
    default=None
)
flags.DEFINE_string(
    name='repo_id',
    required=True,
    help='repo to get model from',
    default=None
)
flags.DEFINE_string(
    name='model_type',
    default='7b',
    help='which model type of llama 1 to train example [13b , 7b , 3b ,...] (default is 7b model)'
)

flags.DEFINE_bool(
    name='use_flash_attention',
    default=False,
    help='use_flash_attention or no'
)

flags.DEFINE_bool(
    name='use_sacn_mlp',
    default=False,
    help='use_sacn_mlp or no'
)

flags.DEFINE_bool(
    name='remove_ckpt_after_load',
    default=False,
    help='remove_ckpt_after_load or no'
)

flags.DEFINE_bool(
    name='do_train',
    default=True,
    help='do_train or no'
)

flags.DEFINE_bool(
    name='do_eval',
    default=False,
    help='do_eval or no'
)

flags.DEFINE_bool(
    name='do_test',
    default=False,
    help='do_test or no'
)

flags.DEFINE_integer(
    name='max_sequence_length',
    default=2048,
    help='max sequence length for model to train'
)

flags.DEFINE_integer(
    name='batch_size',
    default=10,
    help='the batch size to use to train model (will be multiply to gradient_accumulation_steps)'
)

flags.DEFINE_integer(
    name='gradient_accumulation_steps',
    default=8,
    help='the gradient accumulation steps to use to train model (will be multiply to batch_size)'
)

flags.DEFINE_integer(
    name='num_train_epochs',
    default=10,
    help='number of training epochs'
)

flags.DEFINE_integer(
    name='max_steps',
    default=None,
    help='number of max_steps (have been set to None for max number of steps)'
)

flags.DEFINE_string(
    name="optimizer",
    default='adamw',
    help='which optimizer to use (available Optimizers are lion adamw adafactor )'
)

flags.DEFINE_string(
    name="scheduler",
    default='cosine',
    help='which scheduler to use (available schedulers are cosine linear none warm_up_cosine)'
)

flags.DEFINE_string(
    name="rotary_type",
    default='complex',
    help='what kind of implementation of rotary embedding to be used for model (available are lm2, open, complex) '
)

flags.DEFINE_string(
    name="project_name",
    default='LLama',
    help='name for project and model (be used for model naming and wandb logging)'
)

flags.DEFINE_string(
    name='config_repo',
    default=None,
    help='in case that you want to load configs from an huggingface repo'
)

flags.DEFINE_string(
    name='dtype',
    default='bf16',
    help='dtype for model (bf16,fp16,fp32,fp64)'
)

flags.DEFINE_string(
    name='backend',
    default='tpu',
    help='which backend to use supported backends are (tpu ,gpu ,cpu)'
)

flags.DEFINE_float(
    name='learning_rate',
    default=4e-5,
    help='start of learning_rate'
)

flags.DEFINE_float(
    name='learning_rate_end',
    default=4e-6,
    help='end of learning_rate in case of using scheduler linear'
)

api = HfApi()


def main(argv):
    dataset_train = load_dataset(FLAGS.dataset_name)

    # if FLAGS.config_repo is not None:
    #     conf = None
    #     config = EasyDel.LlamaConfig.from_pretrained(FLAGS.config_repo, trust_remote_code=True)
    #     config.use_flash_attention = FLAGS.use_flash_attention
    #     config.use_sacn_mlp = FLAGS.use_sacn_mlp
    # else:
    #     conf = EasyDel.configs.configs.llama_configs[FLAGS.model_type]
    #     config = EasyDel.LlamaConfig(**conf, rotary_type=FLAGS.rotary_type)
    #     config.use_flash_attention = FLAGS.use_flash_attention
    #     config.use_sacn_mlp = FLAGS.use_sacn_mlp
    #     config.max_sequence_length = FLAGS.max_sequence_length
    #     config.rope_scaling = None

    params, config = llama_from_pretrained(FLAGS.repo_id)

    train_args = TrainArguments(
        model_class=EasyDel.FlaxLlamaForCausalLM,
        configs_to_init_model_class={
            'config': config,
            'dtype': get_float_dtype_by_name(FLAGS.dtype),
            'param_dtype': get_float_dtype_by_name(FLAGS.dtype)
        },
        custom_rule=config.get_partition_rules(True),
        model_name=FLAGS.project_name,
        num_train_epochs=FLAGS.num_train_epochs,
        learning_rate=FLAGS.learning_rate,
        learning_rate_end=FLAGS.learning_rate_end,
        optimizer=FLAGS.optimizer,
        scheduler=FLAGS.scheduler,
        weight_decay=0.01,
        total_batch_size=FLAGS.batch_size,
        max_steps=FLAGS.max_steps,
        do_train=FLAGS.do_train,
        do_eval=FLAGS.do_eval,
        do_test=FLAGS.do_test,
        backend=FLAGS.backend,
        max_length=FLAGS.max_sequence_length,
        gradient_checkpointing='nothing_saveable',
        sharding_array=(1, -1, 1),
        use_pjit_attention_force=False,
        gradient_accumulation_steps=FLAGS.gradient_accumulation_steps,
        remove_ckpt_after_load=FLAGS.remove_ckpt_after_load,

    )

    trainer = CausalLMTrainer(train_args,
                              dataset_train=dataset_train['train'],
                              dataset_eval=dataset_train['eval'] if FLAGS.do_eval else None,
                              ckpt_path=FLAGS.ckpt_path)
    output = trainer.train(
        model_parameters=flax.core.FrozenDict({'params': params})
    )
    # Done You can simply train any llama LLM that you want in less than 50 lines of code


if __name__ == "__main__":
    app.run(main)
