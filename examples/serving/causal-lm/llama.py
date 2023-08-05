from huggingface_hub import HfApi

from transformers import AutoTokenizer
from EasyDel import configs

from EasyDel.serve import JAXServer
import EasyDel
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    name='ckpt_path',
    required=True,
    help='path to model weights for example (ckpt/llama_easydel_format)',
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

flags.DEFINE_integer(
    name='max_sequence_length',
    default=2048,
    help='max sequence length for model to train'
)

flags.DEFINE_string(
    name="rotary_type",
    default='complex',
    help='what kind of implementation of rotary embedding to be used for model (available are lm2, open, complex) '
)


def main(argv):
    conf = EasyDel.configs.configs.llama_configs[FLAGS.model_type]
    config = EasyDel.LlamaConfig(**conf, rotary_type=FLAGS.rotary_type)
    config.use_flash_attention = FLAGS.use_flash_attention
    config.use_sacn_mlp = FLAGS.use_sacn_mlp
    config.rope_scaling = None
    config.max_sequence_length = FLAGS.max_sequence_length
    config.use_pjit_attention_force = False
    model = EasyDel.FlaxLlamaForCausalLM(config, _do_init=False)
    tokenizer = AutoTokenizer.from_pretrained('erfanzar/JaxLLama')

    # This config works fine with all the models supported in EasyDel
    config_server = {"host ": '0.0.0.0',
                     "port ": 2059,
                     "instruct_format ": '### SYSTEM:\n{system}\n### INSTRUCT:\n{instruct}\n### ASSISTANT:\n',
                     "chat_format ": '<|prompter|>{prompt}</s><|assistant|>{assistant}</s>',
                     "batch_size ": 1,
                     "system_prefix ": '',
                     "system ": '',
                     "prompt_prefix_instruct ": '',
                     "prompt_postfix_instruct ": '',
                     "prompt_prefix_chat ": '<|prompter|>',
                     "prompt_postfix_chat ": '</s><|assistant|>',
                     "is_instruct ": False,
                     "chat_prefix ": '',
                     "contains_auto_format ": True,
                     "max_length ": 2048,
                     "max_new_tokens ": 2048,
                     "max_stream_tokens ": 32}
    server = JAXServer.load_from_ckpt(
        ckpt_path=FLAGS.ckpt_path,
        model=model,
        tokenizer=tokenizer,
        partition_rules=config.get_partition_rules(True),
        add_param_field=True,
        config=config_server
    )
