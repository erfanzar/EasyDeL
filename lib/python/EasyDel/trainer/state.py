from flax.training.train_state import TrainState


class EasyState(TrainState):
    """
    EasyState Does the Same as Flax TrainState, but it will be used for training state too.
    """

    def __str__(self) -> str:
        return "EasyState()"

    def __repr__(self) -> str:
        return "EasyState()"
