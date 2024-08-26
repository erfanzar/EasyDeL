from transformers import AutoConfig, AutoModelForCausalLM
import os
import sys

os.environ["JAX_TRACEBACK_FILTERING"] = "off"
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirname)
sys.path.append(
	os.path.join(
		dirname,
		"../src",
	)
)
import easydel as ed


def main():
	config = AutoConfig.from_pretrained("tiiuae/falcon-11B", trust_remote_code=True)

	config.num_kv_heads = 8
	config.num_hidden_layers = 4
	config.num_attention_heads = 16
	config.max_position_embeddings = 128
	config.hidden_size = 128
	config.ffn_hidden_size = 256
	config.ff_factor = 2
	print(config)
	model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
	print(model)

	model.save_pretrained("EasyDeL-Checkpoints/PT/Falcon-11BC")
	e_model, params = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
		"EasyDeL-Checkpoints/PT/Falcon-11BC"
	)
	print(e_model)
	print(params)


if __name__ == "__main__":
	main()
