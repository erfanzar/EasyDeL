import typing as tp

import chex as cx
import jax
import jax.numpy as jnp
from eformer import escale as es
from eformer.jaximus import ImplicitArray
from jax.sharding import Mesh, NamedSharding, PartitionSpec

if tp.TYPE_CHECKING:
	from easydel.layers.quantization.quantizers import EasyQuantizer
else:
	EasyQuantizer = object


@cx.dataclass
class PagedAttentionCacheMetaData:
	"""Metadata for Paged Attention KV cache configuration."""

	num_hidden_layers: int
	max_sequences: int
	num_pages: int
	tokens_per_page: int
	max_pages_per_sequence: int
	num_kv_heads: int
	kv_head_dim_size: int

	@classmethod
	def create(
		cls,
		num_hidden_layers: int,
		max_sequences: int,
		num_pages: int,
		tokens_per_page: int,
		max_pages_per_sequence: int,
		num_kv_heads: int,
		kv_head_dim_size: int,
	) -> "PagedAttentionCacheMetaData":
		if num_hidden_layers <= 0:
			raise ValueError("num_hidden_layers must be positive")
		if max_sequences <= 0:
			raise ValueError("max_sequences must be positive")
		if num_pages <= 0:
			raise ValueError("num_pages must be positive")
		if tokens_per_page <= 0:
			raise ValueError("tokens_per_page must be positive")
		if max_pages_per_sequence <= 0:
			raise ValueError("max_pages_per_sequence must be positive")
		if num_kv_heads <= 0:
			raise ValueError("num_kv_heads must be positive")
		if kv_head_dim_size <= 0:
			raise ValueError("kv_head_dim_size must be positive")
		if max_pages_per_sequence > num_pages:
			print(
				f"Warning: max_pages_per_sequence ({max_pages_per_sequence}) > num_pages ({num_pages})."
			)

		return cls(
			num_hidden_layers=num_hidden_layers,
			max_sequences=max_sequences,
			num_pages=num_pages,
			tokens_per_page=tokens_per_page,
			max_pages_per_sequence=max_pages_per_sequence,
			num_kv_heads=num_kv_heads,
			kv_head_dim_size=kv_head_dim_size,
		)


@cx.dataclass
class PagedAttentionCacheView:
	"""Minimal view for a layer within the PagedAttentionCache."""

	metadata: PagedAttentionCacheMetaData
	layer_index: int

	key_pages: tp.Union[cx.Array, ImplicitArray]
	value_pages: tp.Union[cx.Array, ImplicitArray]
	kv_pages_sharding: NamedSharding
	prefill_length: jax.Array
	prefill_pos: jax.Array
	prefill_page_table: jax.Array
	generate_pos: jax.Array
	generate_page_table: jax.Array

	@classmethod
	def init(
		cls,
		metadata: PagedAttentionCacheMetaData,
		layer_index: int,
		mesh: Mesh,
		quantizer: tp.Optional["EasyQuantizer"] = None,
		dtype: tp.Optional[jnp.dtype] = None,
		kv_pages_sharding: tp.Optional[PartitionSpec] = None,
	):
		from easydel.infra.etils import EasyDeLQuantizationMethods
		from easydel.layers.quantization.quantizers import EasyQuantizer

		quantizer = quantizer or EasyQuantizer(EasyDeLQuantizationMethods.NONE)
		kv_pages_sharding = kv_pages_sharding or PartitionSpec(
			None,
			None,
			None,
			None,
		)
		dtype = dtype or jnp.bfloat16
		kv_pages_sharding = NamedSharding(mesh=mesh, spec=kv_pages_sharding)

		kv_pages_shape = (
			metadata.num_kv_heads,
			metadata.num_pages,
			metadata.tokens_per_page,
			metadata.kv_head_dim_size,
		)

		max_len_per_sequence = metadata.max_pages_per_sequence * metadata.tokens_per_page

		with jax.named_scope("easydel-paged-attention-cache-init"):
			key_pages = jnp.zeros(shape=kv_pages_shape, dtype=dtype)
			value_pages = jnp.zeros(shape=kv_pages_shape, dtype=dtype)

			key_pages = quantizer(key_pages)
			value_pages = quantizer(value_pages)
			key_pages = es.with_sharding_constraint(key_pages, kv_pages_sharding)
			value_pages = es.with_sharding_constraint(value_pages, kv_pages_sharding)

			mps = metadata.max_pages_per_sequence
			prefill_length = jnp.array(0, dtype=jnp.int32)
			prefill_pos = jnp.zeros((max_len_per_sequence,), dtype=jnp.int32)
			prefill_page_table = jnp.zeros((mps,), dtype=jnp.int32)
			generate_pos = jnp.zeros((metadata.max_sequences,), dtype=jnp.int32)
			generate_page_table = jnp.zeros((metadata.max_sequences,), dtype=jnp.int32)

			return cls(
				metadata=metadata,
				layer_index=layer_index,
				key_pages=key_pages,
				value_pages=value_pages,
				prefill_length=prefill_length,
				prefill_pos=prefill_pos,
				prefill_page_table=prefill_page_table,
				generate_pos=generate_pos,
				generate_page_table=generate_page_table,
				kv_pages_sharding=kv_pages_sharding,
			)

	def write_prefill_to_cache(self, key: cx.Array, value: cx.Array):
		padded_prefill_len = key.shape[0]
		num_kv_heads_per_device = self.key_pages.shape[0]
		page_size = self.key_pages.shape[2]
		num_pages = padded_prefill_len // page_size
		num_pages = jnp.where(num_pages < 1, 1, num_pages)
		num_active_pages, reminder = jnp.divmod(self.prefill_length, page_size)
		num_active_pages += jnp.where(reminder > 0, 1, 0)

		key = (
			key.transpose((1, 0, 2))
			.reshape((num_kv_heads_per_device, -1, page_size, self.head_dim))
			.astype(self.key_pages.dtype)
		)
		value = (
			value.transpose((1, 0, 2))
			.reshape((num_kv_heads_per_device, -1, page_size, self.head_dim))
			.astype(self.value_pages.dtype)
		)

		def update_cond(carry):
			_, idx = carry
			return idx < num_active_pages

		def per_page_update(carry):
			(kp, vp), idx = carry
			page_k = key[:, idx, :, :][:, None, :, :]
			page_v = value[:, idx, :, :][:, None, :, :]
			mapped_idx = self.prefill_page_table[idx]
			kp = jax.lax.dynamic_update_slice_in_dim(
				kp,
				page_k,
				mapped_idx,
				axis=1,
			)
			vp = jax.lax.dynamic_update_slice_in_dim(
				vp,
				page_v,
				mapped_idx,
				axis=1,
			)
			idx += 1
			return (kp, vp), idx

		idx = 0
		(self.key_pages, self.value_pages), idx = jax.lax.while_loop(
			update_cond,
			per_page_update,
			((self.key_pages, self.value_pages), idx),
		)

	def write_generate_to_cache(self, key: cx.Array, value: cx.Array):
		key = key.transpose((1, 0, 2))
		value = value.transpose((1, 0, 2))

		key = key.astype(self.key_pages.dtype)
		value = value.astype(self.value_pages.dtype)

		num_tokens = key.shape[1]
		num_kv_heads_per_device, num_pages, page_size, head_dim = self.key_pages.shape
		page_idx, offset = jnp.divmod(self.generate_pos, page_size)
		page_to_update = self.generate_page_table[jnp.arange(0, num_tokens), page_idx]

		mapped_page_to_update = page_to_update * page_size + offset
		mapped_page_to_update = jnp.tile(mapped_page_to_update, num_kv_heads_per_device)

		kv_heads_axis_stride = (
			jnp.repeat(jnp.arange(0, num_kv_heads_per_device), num_tokens)
			* num_pages
			* page_size
		)
		mapped_page_to_update = kv_heads_axis_stride + mapped_page_to_update

		key = key.reshape(-1, head_dim)
		value = value.reshape(-1, head_dim)

		self.key_pages = self.key_pages.reshape(-1, head_dim)
		self.value_pages = self.value_pages.reshape(-1, head_dim)

		self.key_pages = self.key_pages.at[mapped_page_to_update, :].set(key)
		self.value_pages = self.value_pages.at[mapped_page_to_update, :].set(value)

		self.key_pages = self.key_pages.reshape(
			num_kv_heads_per_device,
			num_pages,
			page_size,
			head_dim,
		)
		self.value_pages = self.value_pages.reshape(
			num_kv_heads_per_device,
			num_pages,
			page_size,
			head_dim,
		)

	def __repr__(self):
		return f"{self.__class__.__name__}(layer_index={self.layer_index})"

	__str__ = __repr__


@cx.dataclass
class PagedAttentionCache:
	views: tp.List[PagedAttentionCacheView]

	@classmethod
	def init_cache(
		cls,
		metadata: PagedAttentionCacheMetaData,
		mesh: Mesh,
		quantizer: tp.Optional["EasyQuantizer"] = None,
		dtype: tp.Optional[jnp.dtype] = None,
		kv_pages_sharding: tp.Optional[PartitionSpec] = None,
	):
		views = [
			PagedAttentionCacheView.init(
				metadata=metadata,
				layer_index=i,
				mesh=mesh,
				quantizer=quantizer,
				dtype=dtype,
				kv_pages_sharding=kv_pages_sharding,
			)
			for i in range(metadata.num_hidden_layers)
		]
		return cls(views=views)

	def __repr__(self):
		try:
			k_shape = self.key_pages.shape
			v_shape = self.value_pages.shape
		except AttributeError:
			k_shape = "Uninitialized"
			v_shape = "Uninitialized"
		return (
			f"{self.__class__.__name__}(\n"
			f"  metadata={self.metadata},\n"
			f"  key_pages={k_shape},\n"
			f"  value_pages={v_shape},\n"
			f"  num_layers={len(self.views)},\n"
			f"  kv_pages_sharding={self.kv_pages_sharding}\n"
			")"
		)

	__str__ = __repr__
