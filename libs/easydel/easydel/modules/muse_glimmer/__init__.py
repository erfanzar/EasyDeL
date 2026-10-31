# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Muse-Glimmer model implementation for EasyDeL.

Muse-Glimmer is a vision-language family (HF ``muse_glimmer``) pairing a packed
windowed ViT tower with a decoder-only language model.

Architecture Overview
---------------------
1. **Vision tower** — patches arrive pre-flattened, a single linear embeds them
   and a learned ``pos_emb_height x pos_emb_width`` grid is bilinearly resampled
   onto each image's patch grid. Blocks alternate window attention and full
   attention over the packed sequence; the tower head pixel-shuffles
   ``merge_size x merge_size`` blocks into the channel axis.

2. **Adapter + projection** — a bias-free two-layer GELU adapter maps
   ``out_hidden_size -> projector_hidden_size``, a linear projects into the text
   hidden size, and a scale-less RMS norm normalizes the result before it is
   scattered into image/video placeholder positions.

3. **Language model** — a decoder with gated attention (``sigmoid(gate_proj(x))``
   on the attention output), scale-less QK-norm plus a ``qk_scale_factor`` query
   scale, a sliding/full attention schedule, per-layer RoPE bases where ``0``
   means NoPE, sandwich norms around each sub-layer, and tanh-soft-capped logits.

Key Components
--------------
- **MuseGlimmerVisionConfig / MuseGlimmerVisionModel**: the standalone vision
  tower (registered under ``muse_glimmer_vision``).
- **MuseGlimmerTextConfig / MuseGlimmerTextModel**: the language-model trunk.
- **MuseGlimmerConfig**: composite configuration wrapping both.
- **MuseGlimmerModel**: multimodal trunk returning hidden states.
- **MuseGlimmerForConditionalGeneration**: adds the LM head and logit capping.

Usage Example
-------------
```python
import easydel as ed
import jax.numpy as jnp
import spectrax as spx

config = ed.MuseGlimmerConfig(
    text_config=ed.MuseGlimmerTextConfig(hidden_size=256, num_hidden_layers=4),
    vision_config=ed.MuseGlimmerVisionConfig(hidden_size=128, num_hidden_layers=4),
)
model = ed.MuseGlimmerForConditionalGeneration(
    config=config,
    dtype=jnp.bfloat16,
    rngs=spx.Rngs(0),
)
```

References
----------
- Model Hub: https://huggingface.co/meta-models/Muse-Glimmer-30B
- Reference implementation: ``transformers/models/muse_glimmer``
"""

from .modeling_muse_glimmer import (
    MuseGlimmerForConditionalGeneration,
    MuseGlimmerModel,
    MuseGlimmerTextModel,
    MuseGlimmerVisionModel,
)
from .muse_glimmer_configuration import (
    MuseGlimmerConfig,
    MuseGlimmerTextConfig,
    MuseGlimmerVisionConfig,
)

__all__ = (
    "MuseGlimmerConfig",
    "MuseGlimmerForConditionalGeneration",
    "MuseGlimmerModel",
    "MuseGlimmerTextConfig",
    "MuseGlimmerTextModel",
    "MuseGlimmerVisionConfig",
    "MuseGlimmerVisionModel",
)
