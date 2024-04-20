from typing import List, Union

from absl.app import run
from absl import flags
from EasyDel import JAXServer, JAXServerConfig
import jax
from fjformer import get_dtype
from EasyDel.serve.prompters import GemmaPrompter, Llama2Prompter, OpenChatPrompter, ChatMLPrompter, ZephyrPrompter
from EasyDel.serve.prompters.base_prompter import BasePrompter

FLAGS = flags.FLAGS
flags.DEFINE_enum(
    "prompter_type",
    enum_values=("gemma", "llama", "openchat", "chatml", "zephyr"),
    help="Prompter to be used to prompt the model",
    default="gemma"
)
flags.DEFINE_string(
    "pretrained_model_name_or_path",
    default="google/gemma-7b-it",
    help="The pretrained model path in huggingface.co/models"
)
flags.DEFINE_integer(
    "max_compile_tokens",
    default=256,
    help="Maximum number of compiled tokens"
)

flags.DEFINE_integer(
    "max_new_tokens_ratio",
    default=20,
    help="max new tokens ratio to be multiplied for max_compile_tokens for max_new_tokens"
)

flags.DEFINE_integer(
    "max_sequence_length",
    default=2048,
    help="max sequence length to be used in the model"
)

flags.DEFINE_enum(
    "dtype",
    enum_values=(
        "bf16",
        "fp16",
        "fp32"
    ),
    default="bf16",
    help="The data type of the model"
)

flags.DEFINE_list(
    "sharding_axis_dims",
    default=[1, 1, 1, -1],
    help="Sharding Axis dimensions for the model"
)

flags.DEFINE_bool(
    "use_sharded_kv_caching",
    default=False,
    help="whether to use sharded kv for Large Sequence model up to 1M"
)

flags.DEFINE_bool(
    "scan_ring_attention",
    default=True,
    help="whether to scan ring attention for Large Sequence model up to 1M (works with attn_mechanism='ring')"
)

flags.DEFINE_bool(
    "use_scan_mlp",
    default=True,
    help="whether to scan MLP or FFN Layers for Large Sequence model up to 1M"
)

flags.DEFINE_enum(
    "attn_mechanism",
    enum_values=[
        "vanilla",
        "flash",
        "splash",
        "ring",
        "cudnn",
        "local_ring"
    ],
    default="vanilla",
    help="The attention mechanism to be used in the model"
)

flags.DEFINE_integer(
    "block_k",
    default=128,
    help="the number of chunks for key block in attention (Works with flash, splash, ring Attention mechanism)"
)

flags.DEFINE_integer(
    "block_q",
    default=128,
    help="the number of chunks for query block in attention (Works with flash, splash, ring Attention mechanism)"
)

flags.DEFINE_bool(
    "share_gradio",
    default=True,
    help="whether to share gradio app"
)


def main(argv):
    server_config = JAXServerConfig(
        max_sequence_length=FLAGS.max_sequence_length,
        max_compile_tokens=FLAGS.max_compile_tokens,
        max_new_tokens=FLAGS.max_compile_tokens * FLAGS.max_new_tokens_ratio,
        dtype=FLAGS.dtype
    )
    prompters = {
        "gemma": GemmaPrompter(),
        "llama": Llama2Prompter(),
        "openchat": OpenChatPrompter(),
        "chatml": ChatMLPrompter(),
        "zephyr": ZephyrPrompter()
    }
    prompter: BasePrompter = prompters[FLAGS.prompter_type]

    FLAGS.sharding_axis_dims = tuple([int(s) for s in FLAGS.sharding_axis_dims])

    class JAXServerC(JAXServer):

        def format_chat(self, history: List[List[str]], prompt: str, system: Union[str, None]) -> str:
            return prompter.format_message(
                history=history,
                prompt=prompt,
                system_message=system,
                prefix=None
            )

        def format_instruct(self, system: str, instruction: str) -> str:
            return prompter.format_message(
                prefix=None,
                system_message=system,
                prompt=instruction,
                history=[]
            )

    server = JAXServerC.from_torch_pretrained(
        server_config=server_config,
        pretrained_model_name_or_path=FLAGS.pretrained_model_name_or_path,
        device=jax.devices('cpu')[0],
        dtype=get_dtype(dtype=FLAGS.dtype),
        param_dtype=get_dtype(dtype=FLAGS.dtype),
        precision=jax.lax.Precision("fastest"),
        sharding_axis_dims=FLAGS.sharding_axis_dims,
        sharding_axis_names=("dp", "fsdp", "tp", "sp"),
        input_shape=(1, server_config.max_sequence_length),
        model_config_kwargs=dict(
            fully_sharded_data_parallel=True,
            attn_mechanism=FLAGS.attn_mechanism,
            scan_mlp_chunk_size=FLAGS.max_compile_tokens,
            use_scan_mlp=FLAGS.use_scan_mlp,
            scan_ring_attention=FLAGS.scan_ring_attention,
            block_k=FLAGS.block_k,
            block_q=FLAGS.block_q,
            use_sharded_kv_caching=FLAGS.use_sharded_kv_caching
        )
    )

    server.gradio_inference().launch(
        server_name="0.0.0.0",
        server_port=7680,
        show_api=True,
        share=FLAGS.share_gradio
    )


if __name__ == "__main__":
    run(main)
