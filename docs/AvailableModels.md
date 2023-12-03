# Available Models

| Models          | FP16/FP32/BF16 | DP | FSDP | MP | FlashAttn | Gradient Checkpointing | 8/6/4Bit Interface and Training |
|-----------------|:---------------|:---|------|----|-----------|------------------------|---------------------------------|
| **_Llama_**     | ✅              | ✅  | ✅    | ✅  | ✅         | ✅                      | ✅                               |
| **_Mistral_**   | ✅              | ✅  | ✅    | ✅  | ✅         | ✅                      | ✅                               |
| **_Llama2_**    | ✅              | ✅  | ✅    | ✅  | ❌         | ✅                      | ✅                               |
| **_GPT-J_**     | ✅              | ✅  | ✅    | ✅  | ✅         | ✅                      | ✅                               |
| **_LT_**        | ✅              | ✅  | ✅    | ✅  | ❌         | ✅                      | ❌                               |
| **_MosaicMPT_** | ✅              | ✅  | ✅    | ✅  | ✅         | ✅                      | ✅                               |
| **_GPTNeoX-J_** | ✅              | ✅  | ✅    | ✅  | ✅         | ✅                      | ❌                               |
| **_Falcon_**    | ✅              | ✅  | ✅    | ✅  | ✅         | ✅                      | ✅                               |
| **_Palm_**      | ✅              | ✅  | ✅    | ✅  | ✅         | ✅                      | 🌪️                             |
| **_T5_**        | ✅              | ✅  | ✅    | ✅  | ❌         | ✅                      | 🌪️                             |
| **_OPT_**       | ✅              | ✅  | ✅    | ✅  | ❌         | ✅                      | ❌                               |

you can also tell me the model you want in Flax/Jax version and ill try my best to build it ;)

## Current Update

Some of the models supported by EasyDel will support 8,6,4 bit interface and Train these following models will be
supported

* [X] Llama (Supported via `LlamaConfig(bits=8)` or `LlamaConfig(bits=4)`)
* [X] Falcon (Supported via `FalconConfig(bits=8)` or `FalconConfig(bits=4)`)
* [X] Mistral (Supported via `MistalConfig(bits=8)` or `MistalConfig(bits=4)`)
* [ ] Palm
* [ ] T5
* [X] MosaicGPT / MPT (Supported via `MptConfig(bits=8)` or `MptConfig(bits=4)`)
* [X] GPT-J (Supported via `GPTJConfig(bits=8)` or `GPTJConfig(bits=4)`)

> all the models in future will have the 8,6 and 4 bit Inference and Training