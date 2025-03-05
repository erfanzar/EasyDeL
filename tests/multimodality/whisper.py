# import os

# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from jax import numpy as jnp
from transformers import (
	WhisperProcessor,
	WhisperTokenizer,
)

import easydel as ed


def main():
	REPO_ID = "openai/whisper-large-v3-turbo"
	model = ed.AutoEasyDeLModelForSpeechSeq2Seq.from_pretrained(
		REPO_ID,
		dtype=jnp.bfloat16,
		param_dtype=jnp.bfloat16,
	)
	tokenizer = WhisperTokenizer.from_pretrained(REPO_ID)
	processor = WhisperProcessor.from_pretrained(REPO_ID)
	# model = model.quantize()
	inference = ed.vWhisperInference(
		model=model,
		tokenizer=tokenizer,
		processor=processor,
	)
	result_url = inference(
		"https://www.uclass.psychol.ucl.ac.uk/Release2/Conversation/AudioOnly/wav/F_0126_6y9m_1.wav",
		return_timestamps=True,
	)
	print(result_url)


if __name__ == "__main__":
	main()
