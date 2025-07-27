from jax import lax
from jax import numpy as jnp
from transformers import WhisperProcessor, WhisperTokenizer

import easydel as ed


def main():
    REPO_ID = "openai/whisper-large-v3-turbo"
    model = ed.AutoEasyDeLModelForSpeechSeq2Seq.from_pretrained(
        REPO_ID,
        dtype=jnp.float16,
        param_dtype=jnp.float16,
        precision=lax.Precision.DEFAULT,
        auto_shard_model=True,
        sharding_axis_dims=(1, 1, 1, -1, 1),
        config_kwargs=ed.EasyDeLBaseConfigDict(
            kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
            attn_mechanism=ed.AttentionMechanisms.VANILLA,
            attn_dtype=jnp.float16,
            attn_softmax_dtype=jnp.float16,
            kvdtype=jnp.float16,
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,
        ),
        partition_axis=ed.PartitionAxis(kv_head_axis="tp"),
        quantization_method=ed.EasyDeLQuantizationMethods.NONE,
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
