# Implementation guide

Six walkthroughs that show how to build a Transformer in SpecTrax,
end to end. Files 01-05 are CPU-friendly toy configs; file 06 loads
a real Llama 3.2 3B from HuggingFace and generates text on TPU/GPU.

| File                 | What it shows                                                                                                                                             |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `01_llama3.py`       | Build `examples.models.llama.Llama3` at a tiny config; one forward pass; print output shape and parameter count.                                          |
| `02_qwen.py`         | Same drill for `examples.models.qwen.Qwen` — notable differences from Llama are QKV bias and a larger RoPE base.                                          |
| `03_gpt2.py`         | Minimal GPT-2 from scratch: learned positional embeddings, `LayerNorm`, `CausalSelfAttention`, GELU MLP.                                                  |
| `04_vit.py`          | Minimal Vision Transformer from scratch: patch `Conv2d`, `[CLS]` token + learned positional embeddings, pre-LN encoder stack, classifier head.            |
| `05_custom_block.py` | Author a `spx.Module` transformer block by hand: RMSNorm → GQA → residual → RMSNorm → SwiGLU → residual with column/row-parallel `sharding=` annotations. |
| `06_llama_generation.py` | Load real Llama 3.2 3B from HuggingFace safetensors, convert weights to spectrax, greedy-generate text. Requires `transformers` + HF access token. |

Run any file with:

```bash
python -m examples.02_implementation_guide.01_llama3
```
