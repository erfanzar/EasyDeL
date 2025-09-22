"""Run all caching tests together."""

import pytest


def test_all_imports():
    """Test that all caching modules can be imported."""
    # Abstract classes
    from easydel.layers.caching._abstracts import (  # noqa
        BaseCache,
        BaseCacheMetadata,
        BaseCacheView,
        BaseRunTimeMetadata,
    )

    # Specifications
    from easydel.layers.caching._specs import (  # noqa
        AttentionSpec,
        ChunkedLocalAttentionSpec,
        FullAttentionSpec,
        KVCacheSpec,
        MambaSpec,
        SlidingWindowSpec,
    )

    # Utilities
    from easydel.layers.caching._utils import AttnMaskDetail  # noqa

    # Transformer cache
    from easydel.layers.caching.transformer import (
        TransformerCache,
        TransformerCacheMetaData,
        TransformerCacheView,
        TransformerMetadata,
    )

    # Pages cache
    from easydel.layers.caching.page import (
        PagesCache,
        PagesCacheMetaData,
        PagesCacheView,
        PagesMetadata,
    )

    # Linear attention cache
    from easydel.layers.caching.linear import (
        LinearAttnCache,
        LinearAttnCacheMetaData,
        LinearAttnCacheView,
    )

    # Flex attention cache
    from easydel.layers.caching.flex import FlexAttentionCache

    # Mamba caches
    from easydel.layers.caching.mamba import (
        MambaCache,
        MambaCacheMetaData,
        MambaCacheView,
        MambaMetadata,
    )
    from easydel.layers.caching.mamba2 import (
        Mamba2Cache,
        Mamba2CacheMetaData,
        Mamba2CacheView,
        Mamba2Metadata,
    )

    # Lightning cache
    from easydel.layers.caching.lightning import (
        LightningCache,
        LightningCacheMetaData,
        LightningCacheView,
        LightningMetadata,
    )

    # Check main __init__ exports
    from easydel.layers.caching import (  # noqa
        ChunkedLocalAttentionSpec,
        FlexAttentionCache,
        FullAttentionSpec,
        KVCacheSpec,
        LightningCache,
        LightningCacheMetaData,
        LightningCacheView,
        LightningMetadata,
        LinearAttnCache,
        LinearAttnCacheMetaData,
        LinearAttnCacheView,
        Mamba2Cache,
        Mamba2CacheMetaData,
        Mamba2CacheView,
        Mamba2Metadata,
        MambaCache,
        MambaCacheMetaData,
        MambaCacheView,
        MambaMetadata,
        MambaSpec,
        PagesCache,
        PagesCacheMetaData,
        PagesCacheView,
        PagesMetadata,
        SlidingWindowSpec,
        TransformerCache,
        TransformerCacheMetaData,
        TransformerCacheView,
        TransformerMetadata,
    )

    assert True  # If we got here, all imports worked


def test_module_docstrings():
    """Test that modules have proper docstrings."""
    import easydel.layers.caching as caching
    import easydel.layers.caching._abstracts as abstracts
    import easydel.layers.caching._specs as specs

    assert caching.__doc__ is not None
    assert "Caching systems for efficient inference" in caching.__doc__

    assert abstracts.__doc__ is not None
    assert "Abstract base classes for caching systems" in abstracts.__doc__

    assert specs.__doc__ is not None
    assert "Specification classes for different caching strategies" in specs.__doc__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
