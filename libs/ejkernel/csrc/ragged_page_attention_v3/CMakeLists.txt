cmake_minimum_required(VERSION 3.21)
project(ejkernel_ragged_page_attention_v3 LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(EJKERNEL_CUDA_ARCH "" CACHE STRING "CUDA SM architecture (e.g. 80, 90, 100, 110, 120)")
set(EJKERNEL_CUDA_ARCHS "" CACHE STRING "CUDA SM architectures (e.g. 80;90;100;110;120)")
set(EJKERNEL_JAX_FFI_INCLUDE "" CACHE STRING "Path to JAX FFI include dir")

if(NOT EJKERNEL_CUDA_ARCHS AND NOT EJKERNEL_CUDA_ARCH)
  message(FATAL_ERROR "Set EJKERNEL_CUDA_ARCHS or EJKERNEL_CUDA_ARCH (e.g. -DEJKERNEL_CUDA_ARCH=80)")
endif()

if(NOT EJKERNEL_JAX_FFI_INCLUDE)
  message(FATAL_ERROR "EJKERNEL_JAX_FFI_INCLUDE must be set (path to jax/ffi include)")
endif()

set(RPA_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/src")

if(EJKERNEL_CUDA_ARCHS)
  set(_arch_list ${EJKERNEL_CUDA_ARCHS})
else()
  set(_arch_list ${EJKERNEL_CUDA_ARCH})
endif()

foreach(arch IN LISTS _arch_list)
  if(arch LESS 90)
    set(_rpa_stub_arch 80)
  else()
    set(_rpa_stub_arch ${arch})
  endif()

  file(GLOB RPA_FWD "${RPA_SRC_DIR}/rpa_v3_fwd_*_sm${_rpa_stub_arch}.cu")
  set(RPA_SOURCES
    "${RPA_SRC_DIR}/rpa_v3_ffi.cu"
    ${RPA_FWD}
  )

  set(target_name "ejkernel_ragged_page_attention_v3_cuda_sm${arch}")

  add_library(${target_name} SHARED ${RPA_SOURCES})

  set_target_properties(${target_name} PROPERTIES
    CUDA_ARCHITECTURES "${arch}"
    OUTPUT_NAME "ejkernel_ragged_page_attention_v3_cuda_sm${arch}"
  )

  target_compile_options(${target_name} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  )

  target_include_directories(${target_name} PRIVATE
    "${RPA_SRC_DIR}"
    "${EJKERNEL_JAX_FFI_INCLUDE}"
  )

  target_link_libraries(${target_name} PRIVATE CUDA::cudart)
endforeach()
