from EasyDel import EasyDelState
import jax
from jax.sharding import PartitionSpec
from typing import Sequence, Optional


def load_model(
        checkpoint_path: str,
        verbose: bool = True,
        state_shard_fns=None,  # You can pass that
        init_optimizer_state: bool = False
):
    state = EasyDelState.load_state(
        checkpoint_path=checkpoint_path,
        verbose=verbose,
        state_shard_fns=state_shard_fns,  # You can pass that
        init_optimizer_state=init_optimizer_state
    )

    print(state)
