import logging
import os
import typing

import numpy as np
from jax._src.mesh import Mesh

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

import jax
from fjformer import get_jax_mesh
from EasyDel import FlaxLlamaForCausalLM, LlamaConfig
from typing import Union, Tuple, Sequence, Optional

JaxDevice = jax.Device
TpuMesh = Tuple[int, int, int, int]
OtherMesh = Tuple[int, int]
HardwareMesh = Union[TpuMesh, OtherMesh]
LogicalAxisRules = Sequence[Tuple[str, Optional[str]]]


def get_coords(device: jax.Device) -> HardwareMesh:
    if hasattr(device, 'coords'):
        return *device.coords, device.core_on_chip
    return device.process_index, device.id % jax.local_device_count()


def bounds_from_last_device(last_device: jax.Device) -> HardwareMesh:
    if hasattr(last_device, 'coords'):
        x, y, z = last_device.coords
        return x + 1, y + 1, z + 1, last_device.core_on_chip + 1
    else:
        return jax.process_count(), jax.local_device_count()


def get_mesh(model_parallel_submesh: HardwareMesh,
             input_devices: Sequence[JaxDevice] = (),
             input_local_devices: Sequence[JaxDevice] = (),
             tile_by_host_if_needed: bool = True,
             backend: Optional[str] = None) -> Mesh:
    input_devices = input_devices or jax.devices(backend)
    input_local_devices = input_local_devices or jax.local_devices(0, backend)

    last_device = sorted(input_devices, key=get_coords)[-1]
    last_input_local_devices = sorted(input_local_devices, key=get_coords)[-1]
    logging.info('last device coords : %r\nlast local device coords: %r',
                 get_coords(last_device), get_coords(last_input_local_devices))
    global_hardware_mesh = bounds_from_last_device(last_device)
    mesh_ndim = len(global_hardware_mesh)
    local_hardware_mesh = bounds_from_last_device(last_input_local_devices)
    mesh_err = (
        f'each dimension of the model parallel submesh {model_parallel_submesh} '
        'must be a factor of the corresponding dimension of the global device '
        f'mesh {global_hardware_mesh}')
    assert not any(
        g % m
        for g, m in zip(global_hardware_mesh, model_parallel_submesh)), mesh_err
    assert not any(
        g % l for g, l in zip(global_hardware_mesh, local_hardware_mesh))
    devices = np.empty(global_hardware_mesh, dtype=object)
    for device in input_devices:
        device_coords = get_coords(device)
        devices[device_coords] = device
    tile_by_host = tile_by_host_if_needed
    if len(global_hardware_mesh) == 4:
        # enable contiguous local chunks without host tiling by making Z major
        global_hardware_mesh = typing.cast(Tuple[int, int, int, int],
                                           global_hardware_mesh)
        model_parallel_submesh = typing.cast(Tuple[int, int, int, int],
                                             model_parallel_submesh)
        gx, gy, gz, gc = global_hardware_mesh
        mx, my, mz, mc = model_parallel_submesh
        if (mx == gx > 1 and my == mz == 1) or (mx == 1 and my == gy > 1 and
                                                mz == gz > 1):
            logging.info('ensuring YZ plane has a Z-major device order')
            # YZ should be ZY
            assert mc == gc, (mc, gc)
            global_hardware_mesh = gx, gz, gy, gc
            model_parallel_submesh = mx, mz, my, mc
            devices = devices.swapaxes(1, 2)
            tile_by_host = False
        if (my == gy > 1 and mx == mz == 1) or (my == 1 and mx == gx > 1 and
                                                mz == gz > 1):
            logging.info('ensuring XZ plane has a Z-major device order')
            # XZ should be ZX
            assert mc == gc, (mc, gc)
            global_hardware_mesh = gz, gy, gx, gc
            model_parallel_submesh = mz, my, mx, mc
            devices = devices.swapaxes(0, 2)
            tile_by_host = False
    if tile_by_host:
        logging.warning(
            'Tiling device assignment mesh by hosts, which may lead to '
            'reduced XLA collective performance. To avoid this, modify '
            'the model parallel submesh or run with more tasks per host.')
        tile_err = (
            'to tile the mesh by hosts, each dimension of the model parallel '
            'submesh must be either a factor or a multiple of the corresponding '
            'dimension of the per-host submesh')

        def dh_dd_mh_md(g: int, m: int, l: int) -> Tuple[int, int, int, int]:
            """Split a global mesh dimension into four tiling components.

            Args:
              g: global mesh bounds dimension size
              m: model-parallel submesh bounds dimension size
              l: local submesh bounds dimension size

            Returns:
              The resulting tuple divides the dimension into the hosts component of
              the data-parallel submesh, the devices component of the data-parallel
              submesh, the hosts component of the model-parallel submesh, and the
              devices component of the model-parallel submesh.
            """
            d = g // m
            if m >= l:
                assert not m % l, tile_err
                return (d, 1, m // l, l)
            else:
                assert not l % m, tile_err
                return (d // (l // m), l // m, 1, m)

        dh_dd_mh_md_tups = map(dh_dd_mh_md, global_hardware_mesh,
                               model_parallel_submesh, local_hardware_mesh)
        devices = devices.reshape(*(s for t in dh_dd_mh_md_tups for s in t))
        devices = devices.transpose(*(4 * i for i in range(mesh_ndim)),
                                    *(4 * i + 1 for i in range(mesh_ndim)),
                                    *(4 * i + 2 for i in range(mesh_ndim)),
                                    *(4 * i + 3 for i in range(mesh_ndim)))
    else:
        # e.g. [(x_data, x_model), (y_data, y_model), ...]
        model_data_tups = [
            (g // m, m)
            for g, m in zip(global_hardware_mesh, model_parallel_submesh)
        ]
        # reshape to e.g. (x_data, x_model, y_data, y_model...)
        devices = devices.reshape(*(s for t in model_data_tups for s in t))  # pylint: disable=g-complex-comprehension
        # TODO(jekbradbury): reorder small subgroups for ring locality
        # transpose to e.g. (x_data, y_data, ..., x_model, ...)
        devices = devices.transpose(*(2 * i for i in range(mesh_ndim)),
                                    *(2 * i + 1 for i in range(mesh_ndim)))
    # reshape to (data, model)
    devices = devices.reshape(-1, np.prod(model_parallel_submesh))
    global_mesh = Mesh(devices, ['data', 'model'])
    logging.info('global_mesh axis_names: %s', global_mesh.axis_names)
    logging.info('global_mesh devices: %s', global_mesh.devices)
    logging.info('global_mesh devices shape: %s', global_mesh.devices.shape)
    return global_mesh


def main():
    mesh = get_jax_mesh('1,-1,1', ('dp', "fsdp", 'mp'))
    config = LlamaConfig(
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8
    )

    model = FlaxLlamaForCausalLM(config=config, _do_init=True)


if __name__ == "__main__":
    main()
