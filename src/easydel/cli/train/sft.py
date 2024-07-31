from easydel.trainers.base_trainer import TrainArguments


def train_sft(train_argument_parser):
    config = TrainArguments(**vars(train_argument_parser))
    print("Training SFT model with config:")
    for key, value in vars(config).items():
        print(f"  {key}: {value}")
