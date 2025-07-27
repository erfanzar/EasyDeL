import jax
import requests
import torch
from jax import numpy as jnp
from PIL import Image
from transformers import AutoProcessor, SiglipModel

import easydel as ed


def main():
    tmodel = SiglipModel.from_pretrained("google/siglip-base-patch16-224")
    emodel = ed.AutoEasyDeLModel.from_pretrained(
        "google/siglip-base-patch16-224",
        auto_shard_model=True,
        sharding_axis_dims=(1, 1, 1, -1, 1),
    )

    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

    image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

    texts = ["a photo of 2 cats", "a photo of 2 dogs", "a photo of 2 sosks"]
    inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt")

    jinputs = {k: jnp.asarray(v.detach().cpu().numpy()) for k, v in inputs.items()}

    with torch.no_grad():
        tout = tmodel(**inputs)
    with emodel.mesh:
        eout = emodel(**jinputs)

    probs = jax.nn.sigmoid(eout.logits_per_image)
    print(f"E {probs[0][0]:.1%} that image 0 is '{texts[0]}'")

    probs = torch.sigmoid(tout.logits_per_image)
    print(f"T {probs[0][0]:.1%} that image 0 is '{texts[0]}'")


if __name__ == "__main__":
    main()
