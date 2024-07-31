import argparse
import sys

from easydel.cli.utils import create_parser_from_dataclass
from easydel.trainers.training_configurations import TrainArguments


def train_sft(args):
    config = TrainArguments(**vars(args))
    print("Training SFT model with config:")
    for key, value in vars(config).items():
        print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description="EasyDel CLI for training and serving LLMs"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Train parser
    train_parser = subparsers.add_parser("train", help="Train models")
    train_subparsers = train_parser.add_subparsers(dest="model_type")

    # SFT parser
    sft_parser = train_subparsers.add_parser("sft", help="Supervised Fine-Tuning")
    sft_parser = create_parser_from_dataclass(TrainArguments)
    sft_parser.set_defaults(func=train_sft)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "train":
        if args.model_type is None:
            train_parser.print_help()
            sys.exit(1)
        elif args.model_type == "sft":
            if hasattr(args, "func"):
                args.func(args)
            else:
                print("Error: SFT command not properly configured.")
                sys.exit(1)
        else:
            print(f"Unknown model type: {args.model_type}")
            sys.exit(1)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
