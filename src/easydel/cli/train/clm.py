from easydel.trainers.base_trainer import TrainArguments


def train_clm(train_argument_parser):
	config = TrainArguments(**vars(train_argument_parser))
	print("Training CLM model with config:")
	for key, value in vars(config).items():
		print(f"  {key}: {value}")
