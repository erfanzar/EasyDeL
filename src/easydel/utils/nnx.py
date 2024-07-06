import jax.numpy
import jax.random
import jax  # noqa: E402
from flax import nnx  # noqa: E402


def nnx_init(
    module,
    _add_rngs: bool = True,
    _rng_key: str = "rngs",
    _seed: int = 0,
    _lazy: bool = True,
    **kwargs,
):
    if not _lazy:
        return module(**kwargs, **({_rng_key: nnx.Rngs(_seed)} if _add_rngs else {}))

    return nnx.eval_shape(
        lambda: module(**kwargs, **({_rng_key: nnx.Rngs(_seed)} if _add_rngs else {}))
    )


def create_graphdef(
    module,
    _add_rngs: bool = True,
    _rng_key: str = "rngs",
    _seed: int = 0,
    **kwargs,
):
    return nnx.split(
        nnx_init(
            module=module,
            _rng_key=_rng_key,
            _add_rngs=_add_rngs,
            _seed=_seed,
            _lazy=True,
            **kwargs,
        )
    )[0]


def init_garphstate(
    module,
    _add_rngs: bool = True,
    _rng_key: str = "rngs",
    _seed: int = 0,
    _lazy: bool = True,
    **kwargs,
):
    return nnx.split(
        nnx_init(
            module=module,
            _rng_key=_rng_key,
            _add_rngs=_add_rngs,
            _seed=_seed,
            _lazy=_lazy,
            **kwargs,
        )
    )[1]


def diffrentiate_state(state, init_state: dict):
    missing_attributes = {}
    restored_keys = list(state.keys())
    for key in init_state.keys():
        if key not in restored_keys:
            assert isinstance(
                init_state[key], nnx.VariableState
            ), "only `VariableState` types are restoreable"
            missing_attributes[key] = init_state[key]
    return missing_attributes


def redefine_state(state, missings: dict[str, nnx.VariableState]):
    _miss_count: int = 0
    _state_rngs: jax.random.PRNGKey = jax.random.PRNGKey(42)
    for key, value in missings.items():
        if isinstance(value.type, nnx.Param) or issubclass(value.type, nnx.Param):
            assert (
                value.value is None
            ), "there's missing parameter in state which can't be None."
            state[key] = value
        elif isinstance(value.type, nnx.RngCount) or issubclass(
            value.type, nnx.RngCount
        ):
            state[key] = nnx.VariableState(
                nnx.RngCount,
                jax.numpy.array(_miss_count, dtype=jax.numpy.uint32),
            )
            _miss_count += 1
        elif isinstance(value.type, nnx.RngKey) or issubclass(value.type, nnx.RngKey):
            state[key] = nnx.VariableState(nnx.RngKey, _state_rngs)
            _state_rngs = jax.random.split(_state_rngs)[0]
        else:
            raise AttributeError(
                f"Unexcepted type({value.type}) found which cannot be redefined."
            )
    return state
