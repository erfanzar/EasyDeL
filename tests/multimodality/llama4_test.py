import torch
import easydel as ed
from jax import numpy as jnp
from transformers import Llama4ForConditionalGeneration, AutoProcessor
from flax import nnx as nn

model_id = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"

header_config = ed.Llama4Config(
	boi_token_index=200080,
	eoi_token_index=200081,
	image_token_index=200092,
)

text_config = ed.Llama4TextConfig(
	hidden_size=512,  # Drastically reduced from 5120
	intermediate_size=2048,  # Typically 4x hidden_size (reduced from 8192/16384)
	intermediate_size_mlp=2048,  # Assuming non-MoE, keep same as intermediate_size
	num_hidden_layers=6,  # Drastically reduced from 48
	num_attention_heads=8,  # Reduced from 40 (512 / 8 = 64 head_dim)
	num_key_value_heads=4,  # Reduced GQA heads from 8 (can be num_attention_heads for MHA)
	head_dim=64,  # Calculated: hidden_size / num_attention_heads
	use_qk_norm=False,
	vocab_size=202048,
	bos_token_id=200000,
	eos_token_id=[200001, 200007, 200008],
	pad_token_id=200018,
	hidden_act="silu",  # Keep common activation
	max_position_embeddings=4096,  # Reduced context window is typical for smaller models
	initializer_range=0.02,
	rms_norm_eps=1e-05,
	use_cache=True,
	attention_bias=False,
	attention_dropout=0.0,
	rope_theta=500000.0,  # Keep original RoPE theta or use a common default like 10000
	rope_scaling=None,  # Simplify: Remove complex rope scaling for smaller model/context
	num_experts_per_tok=1,
	num_local_experts=1,
	output_router_logits=False,
	router_aux_loss_coef=0.0,  # No MoE loss
	router_jitter_noise=0.0,
)

vision_config = ed.Llama4VisionConfig(
	hidden_size=512,  # Reduced from 1408, matching text hidden_size is convenient
	num_hidden_layers=4,  # Reduced from 34 (Often fewer layers than text)
	num_attention_heads=8,  # Reduced from 16 (512 / 8 = 64 head_dim)
	intermediate_size=2048,  # Typically 4x hidden_size (reduced from 5632)
	vision_output_dim=512,  # Set to match vision hidden_size for simplicity here
	projector_input_dim=512,  # Matching vision_output_dim
	projector_output_dim=512,  # Matching text_config.hidden_size
	image_size=336,  # Keep original image size, or reduce if needed
	patch_size=14,  # Keep original patch size
	num_channels=3,
	hidden_act="gelu",  # Common vision activation
	initializer_range=0.02,
	norm_eps=1e-05,
	attention_dropout=0.0,
	rope_theta=10000,  # Keep original vision rope_theta
	pixel_shuffle_ratio=0.5,  # Disable pixel shuffling maybe? Original was 0.5
	projector_dropout=0.0,
	multi_modal_projector_bias=False,
	vision_feature_layer=-1,  # Usually the last layer
	vision_feature_select_strategy="default",
)
header_config.text_config = text_config
header_config.vision_config = vision_config

hf_model = Llama4ForConditionalGeneration(header_config)
ed_model = ed.Llama4ForConditionalGeneration(header_config, rngs=nn.Rngs(0))
ed_model = ed.traversals.merge_model_and_tree(
	ed_model, tree=ed_model.transform_fn(hf_model.state_dict())
)
ed_model.eval()
ed_model = ed_model.shard_model()

print("Sharded")

processor = AutoProcessor.from_pretrained(model_id)


url1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
url2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png"
messages = [
	{
		"role": "user",
		"content": [
			{"type": "image", "url": url1},
			{
				"type": "text",
				"text": "Can you describe how these two images are similar, and how they differ?",
			},
		],
	},
]

inputs = processor.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(torch.float32)

einputs = {k: jnp.asarray(v.detach().cpu().numpy()) for k, v in inputs.items()}
print({k: v.shape for k, v in inputs.items()})
print({k: v.shape for k, v in einputs.items()})

with torch.no_grad():
	output = hf_model(**inputs)

with ed_model.mesh:
	eoutput = ed_model(**einputs)

print(eoutput.logits[-1, -1, -5:])
print(output.logits[-1, -1, -5:])
print(
	jnp.allclose(
		output.logits.detach().cpu().numpy(),
		eoutput.logits,
		atol=0.125,
		rtol=0,
	)
)
