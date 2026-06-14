GDN kernel tile policy
======================

EasyDeL does not own the GDN kernel tile-policy implementation. The policy
helpers live on the public eJKernel operation surface and are consumed by
eSurge configuration.

.. autodata:: ejkernel.modules.operations.KERNEL_TILE_POLICIES

.. autodata:: ejkernel.modules.operations.KernelTilePolicy

.. autofunction:: ejkernel.modules.operations.normalize_kernel_tile_policy
