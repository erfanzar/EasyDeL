import os
import sys

os.environ["JAX_TRACEBACK_FILTERING"] = "off"
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirname)  # noqa: E402
sys.path.append(
	os.path.join(
		dirname,
		"../..",
	)
)  # noqa: E402
import jax  # noqa: E402

jax.config.update("jax_platform_name", "cpu")  # CPU Test !
from transformers import AutoTokenizer  # noqa: E402

from easydel import (  # noqa: E402
	ApiEngine,
	ChatPipeline,
	FlaxGemma2ForCausalLM,
	Gemma2Config,
	GenerationPipeline,
	GenerationPipelineConfig,
)


def main():
	model = FlaxGemma2ForCausalLM(
		config=Gemma2Config(
			hidden_size=256,
			intermediate_size=512,
			num_attention_heads=8,
			num_key_value_heads=4,
			num_hidden_layers=4,
			vocab_size=32000,
			use_scan_mlp=False,
		),
		_do_init=True,
	)
	tokenizer = AutoTokenizer.from_pretrained(
		"mistralai/Mistral-7B-Instruct-v0.2"
	)  # we use mistral tokenizer wherever we can :)
	tokenizer.pad_token_id = tokenizer.eos_token_id
	pipeline = ChatPipeline(
		pipeline=GenerationPipeline(
			model=model,
			params=model.params,
			tokenizer=tokenizer,
			generation_config=GenerationPipelineConfig(
				max_new_tokens=256,
				temperature=0.4,
			),
		),
		max_prefill_length=128,
	)

	engine = ApiEngine(
		pipeline=pipeline,
	)
	engine.fire()


if __name__ == "__main__":
	main()
