# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Optional

import jax
from fjformer.checkpoint import CheckpointManager, float_tensor_to_dtype
from flax.traverse_util import flatten_dict
from jax import dlpack


def match_keywords(string, positives, negatives):
	for positive in positives:
		if positive not in string:
			return False
	for negative in negatives:
		if negative in string:
			return False
	return True


def load_and_convert_checkpoint(path):
	import torch

	_, flax_params = CheckpointManager.load_state_checkpoint(path)
	flax_params = flatten_dict(flax_params["params"], sep=".")
	torch_params = {}
	for key, tensor in flax_params.items():
		if match_keywords(key, ["kernel"], ["norm", "ln_f"]):
			tensor = tensor.T
		tensor = float_tensor_to_dtype(tensor, "fp16")
		torch_params[key] = torch.from_numpy(tensor)
	return torch_params


def get_pytorch_model_and_config_by_type(): ...


def jax2pt(x: jax.Array):
	from torch import cuda
	from torch.utils import dlpack as dlpack_pt

	_jax_device = list(x.devices())[0].platform
	cpu_force = not cuda.is_available()
	if (
		_jax_device in ["cpu", "gpu"]
		and not cpu_force
		and not bool(os.environ.get("EASYDEL_FORCE_TORCH_USE_CPU", "false"))
	):
		dl_pack_jax = dlpack.to_dlpack(
			x,
			stream=True if (_jax_device == "gpu" and not cpu_force) else None,
			src_device=list(x.devices())[0],
		)
	else:
		device = os.environ.get("EASYDEL_PERFRED_HOST_COPY", "cpu")
		if device.lower() == "none":
			device = None  # Auto JAX Select
		perfred_host = jax.devices(device)[
			int(os.environ.get("EASYDEL_PERFRED_HOST_COPY_IDEX", "0"))
		]
		x = jax.device_get(x)  # make sure it's local
		x = jax.device_put(x, perfred_host)
		dl_pack_jax = dlpack.to_dlpack(
			x,
			stream=None,
		)
	return dlpack_pt.from_dlpack(dl_pack_jax)


def pt2jax(x, transpose_raw: Optional[tuple] = None):
	# from torch.utils import dlpack as dlpack_pt
	#
	# need_reshape = False
	# try:  # Prevent Major-to-Major BUG.
	#     if x.shape[0] == 1 and len(x.shape) > 1:
	#         need_reshape = True
	#         x = x.view(*x.shape[1:])
	#     device = os.environ.get("EASYDEL_PERFRED_HOST_PUT", "none")
	#     device = None if device.lower() == "none" else device  # Auto JAX Select
	#     array = dlpack.from_dlpack(
	#         dlpack_pt.to_dlpack(x.detach()),
	#         jax.devices(device)[
	#             int(os.environ.get("EASYDEL_PERFRED_HOST_PUT_IDEX", "0"))
	#         ],
	#     )
	#     if need_reshape:
	#         array = jax.numpy.expand_dims(array, 0)
	#     if transpose_raw is not None:
	#         array = array.transpose(*transpose_raw)
	#     return array
	# except Exception as e:
	#     if "minor-to-major" in str(e):
	#         if transpose_raw is not None:
	#             raise OSError("minor-to-major dimensions wont match")
	#         res_tr, excepted_tr = (
	#             str(e).split("minor-to-major dimensions ")[-1].split(", expected ")
	#         )
	#         res_tr, excepted_tr = eval(res_tr), eval(excepted_tr)
	#         row_tr = excepted_tr
	#         if len(row_tr) > 2:
	#             row_tr = tuple(
	#                 [row_tr[s] for s in range(len(row_tr)) if s != row_tr[s]]
	#             )
	#         assert len(row_tr) == 2
	#         return pt2jax(x=x.transpose(*row_tr), transpose_raw=res_tr)
	#     else:
	#         raise OSError(e)
	return jax.numpy.asarray(x.detach().cpu().numpy())
