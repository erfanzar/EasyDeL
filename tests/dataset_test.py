from easydel.trainers.utils import (
	create_constant_length_dataset,
	conversations_formatting_function,
)
from datasets import load_dataset
from transformers import AutoTokenizer


# fmt:off
def create_prompt_creator(tokenizer):
	def to_role_and_content(field):
		return {
			"conversation": [
				{"role": "user", "content": field["conversation"][0]["input"]},
				{"role": "assistant", "content": field["conversation"][0]["output"]},
			]
		}
	def _pc(sample):
		return conversations_formatting_function(tokenizer, messages_field="conversation")(to_role_and_content(sample))
	return _pc
# fmt:on


def main():
	dove = load_dataset(
		"LDJnr/Pure-Dove",
	)
	tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
	print(dove)
	prompt_creator = create_prompt_creator(tokenizer)
	dts = create_constant_length_dataset(
		tokenizer=tokenizer,
		dataset=dove["train"],
		formatting_func=prompt_creator,
	)
	for k in dts():
		print(k)


if __name__ == "__main__":
	main()
