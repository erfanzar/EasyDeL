import easydel as ed
import jax
from jax import numpy as jnp
from transformers import CLIPProcessor
from PIL import Image
import requests

REPO = "openai/clip-vit-base-patch32"


def main():
	processor = CLIPProcessor.from_pretrained(REPO)
	model = ed.AutoEasyDeLModelForZeroShotImageClassification.from_pretrained(REPO)
	url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # Cat Photo
	image = Image.open(requests.get(url, stream=True).raw)

	inputs = processor(
		text=[
			"a photo of a cat",
			"a photo of a dog",
			"a photo of a erfan",
			"a photo of a car",
		],
		images=image,
		return_tensors="np",
		padding=True,
	)
	inputs["pixel_values"] = jnp.swapaxes(inputs["pixel_values"], 1, 3)
	outputs = model(**inputs)
	# this is the image-text similarity score
	logits_per_image = outputs.logits_per_image
	# we can take the softmax to get the label probabilities
	probs = jax.nn.softmax(logits_per_image, axis=1)
	print(probs)
	assert probs[0, 0] > probs[0, 1], "test failed!"
	print("loss", model.compute_loss(**inputs)[-1].loss)
	print("test passed.")


if __name__ == "__main__":
	main()
