# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

import argparse
import sys

from easydel.cli.train.clm import train_clm as train_clm
from easydel.cli.train.dpo import train_dpo as train_dpo
from easydel.cli.train.sft import train_sft as train_sft
from easydel.cli.utils import get_parser_from_class as get_parser_from_class
from easydel.trainers.training_configurations import (
	TrainingArguments as TrainingArguments,
)


def main():
	parser = argparse.ArgumentParser(
		description="EasyDel CLI for training and serving LLMs"
	)
	subparsers = parser.add_subparsers(dest="command")

	# Train parser
	train_parser = subparsers.add_parser(
		"train",
		help="Train models",
	)
	train_subparsers = train_parser.add_subparsers(
		dest="trainer_type",
	)

	# SFT parser
	train_subparsers.add_parser("sft", help="Supervised Fine-Tuning")
	train_subparsers.add_parser("dpo", help="Direct Preference Optimization")
	train_subparsers.add_parser("clm", help="Causal Language Model")

	train_argument_parser = get_parser_from_class(TrainingArguments)

	args = parser.parse_args()
	if args.command is None:
		parser.print_help()
		sys.exit(1)

	if args.command == "train":
		if args.trainer_type is None:
			train_parser.print_help()
			sys.exit(1)
		elif args.trainer_type == "sft":
			train_sft(
				train_argument_parser=train_argument_parser,
			)
		elif args.trainer_type == "dpo":
			train_dpo(
				train_argument_parser=train_argument_parser,
			)
		elif args.trainer_type == "clm":
			train_clm(
				train_argument_parser=train_argument_parser,
			)
		else:
			print(f"Unknown trainer_type: {args.model_type}")
			sys.exit(1)
	else:
		print(f"Unknown command: {args.command}")
		sys.exit(1)


if __name__ == "__main__":
	main()
