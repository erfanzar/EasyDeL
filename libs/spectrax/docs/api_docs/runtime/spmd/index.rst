spectrax.runtime.spmd package
=============================

SPMD-only runtime. These helpers compile one shared HLO through
``shard_map`` and reject MPMD-tagged meshes at the public boundary.
Use ``spectrax.runtime.mpmd`` for true MPMD.

.. toctree::
   :maxdepth: 2

   api
   runtime
   shard_map
