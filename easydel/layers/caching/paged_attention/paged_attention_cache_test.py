import jax.numpy as jnp
import pytest
from eformer import escale as es
from jax.sharding import NamedSharding, PartitionSpec

from easydel.layers.caching.paged_attention.paged_attention_cache import (
	PagedAttentionCache,
	PagedAttentionCacheMetaData,
	PagedAttentionCacheView,
)
from easydel.layers.quantization import EasyQuantizer


# Test Fixtures
@pytest.fixture(scope="module")
def mesh():
	"""Provides a single-device mesh for testing."""

	return es.create_mesh((1, 1, -1, 1, 1))


@pytest.fixture
def metadata_params():
	"""Provides default parameters for PagedAttentionCacheMetaData."""
	return {
		"num_hidden_layers": 4,
		"max_sequences": 4,
		"num_pages": 16,
		"tokens_per_page": 4,
		"max_pages_per_sequence": 8,  # Max seq len = 8 * 4 = 32
		"num_kv_heads": 2,
		"kv_head_dim_size": 8,
	}


@pytest.fixture
def metadata(metadata_params):
	"""Provides a valid PagedAttentionCacheMetaData instance."""
	return PagedAttentionCacheMetaData.create(**metadata_params)


@pytest.fixture
def quantizer():
	"""Provides a mock quantizer."""
	# return EasyQuantizer(EasyDeLQuantizationMethods.NONE) # If using real one
	return EasyQuantizer()


# --- Test PagedAttentionCacheMetaData ---


def test_metadata_create_valid(metadata_params):
	meta = PagedAttentionCacheMetaData.create(**metadata_params)

	assert meta.num_hidden_layers == metadata_params["num_hidden_layers"]
	assert meta.max_sequences == metadata_params["max_sequences"]
	assert meta.num_pages == metadata_params["num_pages"]
	assert meta.tokens_per_page == metadata_params["tokens_per_page"]
	assert meta.max_pages_per_sequence == metadata_params["max_pages_per_sequence"]
	assert meta.num_kv_heads == metadata_params["num_kv_heads"]
	assert meta.kv_head_dim_size == metadata_params["kv_head_dim_size"]


@pytest.mark.parametrize(
	"invalid_param, value",
	[
		("num_hidden_layers", 0),
		("max_sequences", 0),
		("num_pages", -1),
		("tokens_per_page", 0),
		("max_pages_per_sequence", 0),
		("num_kv_heads", 0),
		("kv_head_dim_size", 0),
	],
)
def test_metadata_create_invalid(metadata_params, invalid_param, value):
	metadata_params[invalid_param] = value
	with pytest.raises(ValueError):
		PagedAttentionCacheMetaData.create(**metadata_params)


def test_metadata_create_warning(metadata_params, capsys):
	metadata_params["max_pages_per_sequence"] = metadata_params["num_pages"] + 1
	PagedAttentionCacheMetaData.create(**metadata_params)
	captured = capsys.readouterr()
	assert "Warning: max_pages_per_sequence" in captured.out


# --- Test PagedAttentionCacheView ---


def test_cache_view_init(metadata, mesh, quantizer):
	view = PagedAttentionCacheView.init(
		metadata=metadata,
		layer_index=5,
		mesh=mesh,
		quantizer=quantizer,
	)
	assert view.metadata == metadata
	assert view.layer_index == 5
	assert repr(view) == "PagedAttentionCacheView(layer_index=5)"


# --- Test PagedAttentionCache ---


def test_cache_init(metadata, mesh, quantizer):
	num_hidden_layers = metadata.num_hidden_layers
	dtype = jnp.float32
	spec = PartitionSpec(None, None, None, None)
	cache = PagedAttentionCache.init_cache(
		metadata=metadata,
		mesh=mesh,
		quantizer=quantizer,
		dtype=dtype,
		kv_pages_sharding=spec,
	)

	assert cache.views[0].metadata == metadata
	assert len(cache.views) == num_hidden_layers
	assert isinstance(cache.views[0], PagedAttentionCacheView)
	assert cache.views[1].layer_index == 1

	expected_shape = (
		metadata.num_kv_heads,
		metadata.num_pages,
		metadata.tokens_per_page,
		metadata.kv_head_dim_size,
	)
	assert cache.key_pages.shape == expected_shape
	assert cache.value_pages.shape == expected_shape
	assert cache.key_pages.dtype == dtype
	assert cache.value_pages.dtype == dtype

	assert isinstance(cache.kv_pages_sharding, NamedSharding)
	assert cache.kv_pages_sharding.mesh == mesh
	assert cache.prefill_length.shape == () or cache.prefill_length.shape == (1,)
	assert cache.generate_pos.shape == (metadata.max_sequences,)


def test_cache_init(metadata, mesh, quantizer):
	num_hidden_layers = metadata.num_hidden_layers
	dtype = jnp.float32
	spec = PartitionSpec(None, None, None, None)  # Fully replicated

	cache = PagedAttentionCache.init_cache(
		metadata=metadata,
		mesh=mesh,
		quantizer=quantizer,
		dtype=dtype,
		kv_pages_sharding=spec,
	)

	assert cache.views[0].metadata == metadata
	assert len(cache.views) == num_hidden_layers
	assert isinstance(cache.views[0], PagedAttentionCacheView)
	assert cache.views[1].layer_index == 1

	expected_kv_shape = (
		metadata.num_kv_heads,
		metadata.num_pages,
		metadata.tokens_per_page,
		metadata.kv_head_dim_size,
	)
	assert cache.views[0].key_pages.shape == expected_kv_shape
	assert cache.views[0].value_pages.shape == expected_kv_shape
	assert cache.views[0].key_pages.dtype == dtype
	assert cache.views[0].value_pages.dtype == dtype

	# Check sharding
	assert isinstance(cache.views[0].kv_pages_sharding, NamedSharding)
	assert cache.views[0].kv_pages_sharding.mesh == mesh

	# --- Check initialized state arrays ---
	max_len_per_seq = metadata.max_pages_per_sequence * metadata.tokens_per_page

	# Prefill state
	assert cache.views[0].prefill_length.shape == ()  # Scalar
	assert cache.views[0].prefill_length.dtype == jnp.int32
	assert cache.views[0].prefill_length == 0
	assert cache.views[0].prefill_pos.shape == (max_len_per_seq,)
	assert cache.views[0].prefill_pos.dtype == jnp.int32
	assert jnp.all(cache.views[0].prefill_pos == 0)
	assert cache.views[0].prefill_page_table.shape == (metadata.max_pages_per_sequence,)
	assert cache.views[0].prefill_page_table.dtype == jnp.int32
	assert jnp.all(cache.views[0].prefill_page_table == 0)
	# Or -1 if initialized differently

	# Generate state
	assert cache.views[0].generate_pos.shape == (metadata.max_sequences,)
	assert cache.views[0].generate_pos.dtype == jnp.int32
	assert jnp.all(cache.views[0].generate_pos == 0)
	assert cache.views[0].generate_page_table.shape == (metadata.max_sequences,)
	assert cache.views[0].generate_page_table.dtype == jnp.int32
	assert jnp.all(cache.views[0].generate_page_table == 0)
	# Or -1 if initialized differently


# Test the replace method
def test_cache_replace(metadata, mesh, quantizer):
	cache = PagedAttentionCache.init_cache(
		metadata=metadata,
		mesh=mesh,
		quantizer=quantizer,
	)
	new_prefill_len = jnp.array(10)
	new_cache = cache.views[0].replace(prefill_length=new_prefill_len)

	assert new_cache is not cache.views[0]  # Should be a new object
	assert new_cache.metadata == cache.views[0].metadata  # Unchanged attribute
	assert new_cache.prefill_length == new_prefill_len  # Changed attribute
	# Check that original cache is unchanged
	assert cache.views[0].prefill_length != new_prefill_len


if __name__ == "__main__":
	pytest.main([__file__])
