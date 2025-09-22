"""Test Mamba and Mamba2 state-space model caching implementation."""

import pytest
from eformer.escale import PartitionAxis

from easydel.layers.caching.mamba import MambaCacheMetaData
from easydel.layers.caching.mamba2 import Mamba2CacheMetaData


class TestMambaCacheMetaData:
    """Test MambaCacheMetaData class."""

    def test_metadata_creation(self):
        """Test metadata creation for Mamba SSM."""
        metadata = MambaCacheMetaData.create(
            partition_axis=PartitionAxis(),
            batch_size=2,
            num_hidden_layers=8,
            ssm_state_size=16,
            conv_kernel_size=4,
            intermediate_size=1024,
        )

        assert metadata.batch_size == 2
        assert metadata.num_hidden_layers == 8
        assert metadata.ssm_state_size == 16
        assert metadata.conv_kernel_size == 4
        assert metadata.intermediate_size == 1024

    def test_metadata_validation_errors(self):
        """Test metadata validation."""
        # Invalid batch size
        with pytest.raises(ValueError, match="batch_size must be positive"):
            MambaCacheMetaData.create(
                partition_axis=PartitionAxis(),
                batch_size=0,
                num_hidden_layers=4,
                ssm_state_size=16,
                conv_kernel_size=4,
                intermediate_size=512,
            )

        # Invalid SSM state size
        with pytest.raises(ValueError, match="ssm_state_size must be positive"):
            MambaCacheMetaData.create(
                partition_axis=PartitionAxis(),
                batch_size=2,
                num_hidden_layers=4,
                ssm_state_size=0,
                conv_kernel_size=4,
                intermediate_size=512,
            )

        # Invalid conv kernel size
        with pytest.raises(ValueError, match="conv_kernel_size must be positive"):
            MambaCacheMetaData.create(
                partition_axis=PartitionAxis(),
                batch_size=2,
                num_hidden_layers=4,
                ssm_state_size=16,
                conv_kernel_size=0,
                intermediate_size=512,
            )


class TestMamba2CacheMetaData:
    """Test Mamba2CacheMetaData class."""

    def test_metadata_creation(self):
        """Test metadata creation for Mamba2 SSM."""
        metadata = Mamba2CacheMetaData.create(
            partition_axis=PartitionAxis(),  # Note: typo in actual API
            batch_size=2,
            num_hidden_layers=8,
            num_heads=8,
            head_dim=64,
            state_size=128,
            conv_kernel_size=4,
            intermediate_size=512,
            n_groups=2,
        )

        assert metadata.batch_size == 2
        assert metadata.num_hidden_layers == 8
        assert metadata.num_heads == 8
        assert metadata.head_dim == 64
        assert metadata.state_size == 128
        assert metadata.conv_kernel_size == 4

    def test_metadata_validation_errors(self):
        """Test metadata validation."""
        # Invalid num_heads
        with pytest.raises(ValueError, match="num_heads must be positive"):
            Mamba2CacheMetaData.create(
                partition_axis=PartitionAxis(),
                batch_size=2,
                num_hidden_layers=4,
                num_heads=0,
                head_dim=64,
                state_size=128,
                conv_kernel_size=4,
                intermediate_size=512,
                n_groups=2,
            )

        # Invalid head_dim
        with pytest.raises(ValueError, match="head_dim must be positive"):
            Mamba2CacheMetaData.create(
                partition_axis=PartitionAxis(),
                batch_size=2,
                num_hidden_layers=4,
                num_heads=8,
                head_dim=0,
                state_size=128,
                conv_kernel_size=4,
                intermediate_size=512,
                n_groups=2,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
