spectrax.runtime.mpmd package
=============================

True MPMD runtime. Public training and gradient entry points route
through schedule-faithful per-rank programs: ``sxjit``, ``sxcall``,
``sxgrad``, and ``sxvalue_and_grad``. Forward-only inference can reuse
``sxjit`` plans through ``MpmdPipelineExecutor``.

.. toctree::
   :maxdepth: 2

   compiler
   markers
   per_rank
   pscan_compiler
   runtime
   treduce
