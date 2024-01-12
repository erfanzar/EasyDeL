from jax import numpy as jnp
from jax.nn.initializers import lecun_normal, ones
from flax import linen as nn
import jax
from jax.numpy.linalg import eigh, inv, matrix_power
from jax.scipy.signal import convolve


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    def init(key, shape):
        return jax.random.uniform(key, shape) * (
                jnp.log(dt_max) - jnp.log(dt_min)
        ) + jnp.log(dt_min)

    return init


def random_state_space_model(rng, hidden_size):
    a_r, b_r, c_r = jax.random.split(rng, 3)
    a = jax.random.uniform(a_r, (hidden_size, hidden_size))
    b = jax.random.uniform(b_r, (hidden_size, 1))
    c = jax.random.uniform(c_r, (1, hidden_size))
    return a, b, c


def discretize(a, b, c, step):
    i = jnp.eye(a.shape[0])
    bl = inv(i - (step / 2.0) * a)
    ab = bl @ (i + (step / 2.0) * a)
    bb = (bl * step) @ b
    return ab, bb, c


def scan_state_space_model(ab, bb, cb, u, x0):
    def step(x_k_1, u_k):
        x_k = ab @ x_k_1 + bb @ u_k
        y_k = cb @ x_k
        return x_k, y_k

    return jax.lax.scan(step, x0, u)


def k_conv(ab, bb, cb, l):
    return jnp.array(
        [(cb @ matrix_power(ab, times) @ bb).reshape() for times in range(l)]
    )


def causal_convolution(u, k, nofft=False):
    if nofft:
        return convolve(u, k, mode="full")[: u.shape[0]]
    else:
        assert k.shape[0] == u.shape[0], "can not perform this operation (use `nofft=True`)"
        ud = jnp.fft.rfft(jnp.pad(u, (0, k.shape[0])))
        kd = jnp.fft.rfft(jnp.pad(k, (0, u.shape[0])))
        out = ud * kd
        return jnp.fft.irfft(out)[: u.shape[0]]


class StateSpaceModelLayer(nn.Module):
    N: int
    l_max: int
    decode: bool = False

    def setup(self):
        # SSM parameters
        self.A = self.param("A", lecun_normal(), (self.N, self.N))
        self.B = self.param("B", lecun_normal(), (self.N, 1))
        self.C = self.param("C", lecun_normal(), (1, self.N))
        self.D = self.param("D", ones, (1,))

        # Step parameter
        self.log_step = self.param("log_step", log_step_initializer(), (1,))

        step = jnp.exp(self.log_step)
        self.ssm = discretize(self.A, self.B, self.C, step=step)
        self.K = k_conv(*self.ssm, self.l_max)

        self.x_k_1 = self.variable("cache", "cache_x_k", jnp.zeros, (self.N,))

    def __call__(self, u):
        if not self.decode:
            return causal_convolution(u, self.K) + self.D * u
        else:
            x_k, y_s = scan_state_space_model(*self.ssm, u[:, jnp.newaxis], self.x_k_1.value)
            if self.is_mutable_collection("cache"):
                self.x_k_1.value = x_k
            return y_s.reshape(-1).real + self.D * u


def clone_layer(layer):
    return nn.vmap(
        layer,
        in_axes=1,
        out_axes=1,
        variable_axes={"params": 1, "cache": 1, "prime": 1},
        split_rngs={"params": True},
    )


StateSpaceModelLayer = clone_layer(StateSpaceModelLayer)


class SequenceBlock(nn.Module):
    layer_cls: nn.Module
    layer: dict
    dropout: float
    d_model: int
    prenorm: bool = True
    glu: bool = True
    training: bool = True
    decode: bool = False

    def setup(self):
        self.seq = self.layer_cls(
            **self.layer,
            decode=self.decode
        )
        self.norm = nn.LayerNorm()
        self.out = nn.Dense(
            self.d_model
        )
        if self.glu:
            self.out2 = nn.Dense(
                self.d_model
            )
        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0],
            deterministic=not self.training,
        )

    def __call__(self, x):
        skip = x
        if self.prenorm:
            x = self.norm(x)
        x = self.seq(x)
        x = self.drop(nn.gelu(x))
        if self.glu:
            x = self.out(x) * jax.nn.sigmoid(self.out2(x))
        else:
            x = self.out(x)
        x = skip + self.drop(x)
        if not self.prenorm:
            x = self.norm(x)
        return x


class Embedding(nn.Embed):
    num_embeddings: int
    features: int

    @nn.compact
    def __call__(self, x):
        y = nn.Embed(self.num_embeddings, self.features)(x[..., 0])
        return jnp.where(x > 0, y, 0.0)


class StackedModel(nn.Module):
    layer_cls: nn.Module
    layer: dict
    d_output: int
    d_model: int
    n_layers: int
    prenorm: bool = True
    dropout: float = 0.0
    embedding: bool = False
    classification: bool = False
    training: bool = True
    decode: bool = False

    def setup(self):
        if self.embedding:
            self.encoder = Embedding(self.d_output, self.d_model)
        else:
            self.encoder = nn.Dense(self.d_model)
        self.decoder = nn.Dense(self.d_output)
        self.layers = [
            SequenceBlock(
                layer_cls=self.layer_cls,
                layer=self.layer,
                prenorm=self.prenorm,
                d_model=self.d_model,
                dropout=self.dropout,
                training=self.training,
                decode=self.decode,
            )
            for _ in range(self.n_layers)
        ]

    def __call__(self, x):
        if not self.classification:
            if not self.embedding:
                x = x / 255.0  # Normalize
            if not self.decode:
                x = jnp.pad(x[:-1], [(1, 0), (0, 0)])
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x)
        if self.classification:
            x = jnp.mean(x, axis=0)
        x = self.decoder(x)
        return nn.log_softmax(x, axis=-1)


BatchStackedModel = nn.vmap(
    StackedModel,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None, "dropout": None, "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True},
)


def make_hippo(hidden_size):
    p = jnp.sqrt(1 + 2 * jnp.arange(hidden_size))
    a = p[:, jnp.newaxis] * p[jnp.newaxis, :]
    a = jnp.tril(a) - jnp.diag(jnp.arange(hidden_size))
    return -a
