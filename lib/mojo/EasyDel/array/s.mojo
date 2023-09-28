# from random import rand
# from memory import memset_zero
# from memory.buffer import Buffer
# from utils.index import StaticIntTuple, Index
# from utils.list import Dim, DimList
# from utils.vector import DynamicVector, InlinedFixedVector
# from runtime.llcl import num_cores, Runtime
# from algorithm import vectorize, parallelize, vectorize_unroll
# from algorithm import Static2DTileUnitFunc as Tile2DFunc
# import math

# from .helpers import __check_bounds, __negative_pos_to_positive
# from .tensor_view import TensorView

# alias dims_average_size = 5


# struct TensorG[Type: DType]:
#     var data: DTypePointer[Type]
#     var dims: TensorView
#     var size: Int
#     alias simd_width = simdwidthof[Type]()

#     fn __init__(inout self, random: Bool, *dims: Int):
#         self.dims = TensorView(dims)
#         self.size = 1
#         self.size = self.dims.num_elements()

#         # check if using alligend alloc could be better
#         self.data = DTypePointer[Type].alloc(self.size)
#         if random:
#             rand(self.data, self.size)
#         else:
#             self.zero()

#     fn __init__(inout self, random: Bool, dims: TensorView):
#         self.dims = dims
#         self.size = 1
#         self.size = self.dims.num_elements()

#         self.data = DTypePointer[Type].alloc(self.size)
#         if random:
#             rand(self.data, self.size)
#         else:
#             self.zero()

#     fn __init__(
#         inout self,
#         data: VariadicList[FloatLiteral],
#         dims: TensorView,
#     ):
#         self.dims = dims
#         self.size = 1
#         self.size = self.dims.num_elements()

#         let dims_area_correct = self.size == len(data)
#         debug_assert(
#             dims_area_correct,
#             "Error, the size of the data doesn't match the size of the tensor.",
#         )

#         self.data = DTypePointer[Type].alloc(self.size)
#         for i in range(self.size):
#             self.data.simd_store[1](i, data[i])

#     fn __init__(
#         inout self,
#         data: DynamicVector[FloatLiteral],
#         dims: TensorView,
#     ):
#         self.dims = dims
#         self.size = 1
#         self.size = self.dims.num_elements()

#         let dims_area_correct = self.size == len(data)
#         debug_assert(
#             dims_area_correct,
#             "Error, the size of the data doesn't match the size of the tensor.",
#         )

#         self.data = DTypePointer[Type].alloc(self.size)
#         for i in range(self.size):
#             self.data.simd_store[1](i, data[i])

#     fn __del__(owned self):
#         self.data.free()

#     fn zero(inout self):
#         memset_zero(self.data, self.dims.num_elements())

#     fn __copyinit__(inout self, existing: Self):
#         self.dims = existing.dims
#         self.size = 1
#         self.size = self.dims.num_elements()
#         self.data = DTypePointer[Type].alloc(self.size)

#         for i in range(self.size):
#             self.data.simd_store[1](i, existing.data.simd_load[1](i))

#     fn __moveinit__(inout self, owned existing: Self):
#         self.dims = existing.dims
#         self.data = existing.data
#         self.size = existing.size

#     fn byte_count(self) -> Int:
#         return sizeof[Type]() * self.size

#     @always_inline
#     fn __getitem__[len: Int](self, index: StaticIntTuple[len]) -> SIMD[Type, 1]:
#         return self.load[1](index)

#     @always_inline
#     fn __getitem__[
#         len: Int
#     ](self, index: InlinedFixedVector[len, Int]) -> SIMD[Type, 1]:
#         return self.load[1](index)

#     @always_inline
#     fn __getitem__(self, index: Int) -> SIMD[Type, 1]:
#         """Access the data as a 1D array."""
#         return self.load[1](index)

#     @always_inline
#     fn load[
#         nelts: Int, len: Int
#     ](self, index: StaticIntTuple[len]) -> SIMD[Type, nelts]:
#         let pos = self.dims.get_position(index)
#         __check_bounds(pos, self.size)
#         return self.data.simd_load[nelts](pos)

#     @always_inline
#     fn load[
#         nelts: Int, len: Int
#     ](self, index: InlinedFixedVector[len, Int]) -> SIMD[Type, nelts]:
#         let pos = self.dims.get_position(index)
#         __check_bounds(pos, self.size)
#         return self.data.simd_load[nelts](pos)

#     @always_inline
#     fn load[nelts: Int](self, index: Int) -> SIMD[Type, nelts]:
#         """Access the data as a 1D array."""
#         let pos = __negative_pos_to_positive(index, self.size)
#         __check_bounds(pos, self.size)
#         return self.data.simd_load[nelts](pos)

#     @always_inline
#     fn __setitem__[len: Int](self, index: StaticIntTuple[len], val: SIMD[Type, 1]):
#         return self.store[1](index, val)

#     @always_inline
#     fn __setitem__[
#         len: Int
#     ](self, index: InlinedFixedVector[len, Int], val: SIMD[Type, 1]):
#         return self.store[1](index, val)

#     @always_inline
#     fn __setitem__(self, index: Int, val: SIMD[Type, 1]):
#         return self.store[1](index, val)

#     @always_inline
#     fn store[
#         nelts: Int, len: Int
#     ](self, index: StaticIntTuple[len], val: SIMD[Type, nelts]):
#         let pos = self.dims.get_position(index)
#         __check_bounds(pos, self.size)
#         self.data.simd_store[nelts](pos, val)

#     @always_inline
#     fn store[
#         nelts: Int, len: Int
#     ](self, index: InlinedFixedVector[len, Int], val: SIMD[Type, nelts]):
#         let pos = self.dims.get_position(index)
#         __check_bounds(pos, self.size)
#         self.data.simd_store[nelts](pos, val)

#     @always_inline
#     fn store[nelts: Int](self, index: Int, val: SIMD[Type, nelts]):
#         """Access and store the data as a 1D array."""
#         let pos = __negative_pos_to_positive(index, self.size)
#         __check_bounds(pos, self.size)
#         self.data.simd_store[nelts](pos, val)

#     fn __dim_suffix_product[len: Int](self) -> InlinedFixedVector[len, Int]:
#         var suffix_product = InlinedFixedVector[len, Int](self.dims.rank())
#         suffix_product.append(1)  # the first value has to be 1

#         for index in range(self.dims.rank() - 1):
#             suffix_product.append(
#                 suffix_product[index] * self.dims[self.dims.rank() - 1 - index]
#             )

#         return suffix_product

#     fn __matmul_num_elements(self, other: Self) -> Int:
#         var size = 1
#         for i in range(self.dims.rank() - 2):
#             size *= self.dims[i]

#         size *= self.dims[
#             self.dims.rank() - 2
#         ]  # The different dimension of first tensor
#         size *= other.dims[
#             other.dims.rank() - 1
#         ]  # the different dimension of second tensor
#         size *= self.dims[
#             self.dims.rank() - 1
#         ]  # The dimension that is the same for both tensors

#         return size

#     fn print_all(self):
#         let size = self.dims.num_elements()

#         var suffix_product = InlinedFixedVector[dims_average_size, Int](
#             self.dims.rank() + 1
#         )
#         suffix_product.append(1)
#         for index in range(self.dims.rank()):
#             suffix_product.append(
#                 suffix_product[index] * self.dims[self.dims.rank() - 1 - index]
#             )

#         var count = 0
#         for i in range(size + 1):
#             count = 0
#             for j in range(self.dims.rank()):
#                 if i % suffix_product[j + 1] == 0 and i != 0:
#                     print_no_newline("]")
#                     count += 1

#             if i > 0 and i < size:
#                 print_no_newline(",")
#             if i < size:
#                 for i in range(count):
#                     print()

#             for j in range(self.dims.rank()):
#                 if i % suffix_product[j + 1] == 0 and i != size:
#                     print_no_newline("[")

#             if i < size:
#                 print_no_newline(self[i])

#         print()

#     fn rank(self) -> Int:
#         return self.dims.rank()

#     fn __iterate_binary_op_tensor[
#         nelts: Int,
#         outer_loop_func: fn[func: fn (Int) capturing -> None] (Int) capturing -> None,
#         op_func: fn[T: DType, simd_width: Int] (
#             x: SIMD[T, simd_width], y: SIMD[T, simd_width]
#         ) -> SIMD[T, simd_width],
#     ](self, other: Self) -> Self:
#         let dims_eq = self.dims == other.dims
#         debug_assert(
#             dims_eq, "Error dimension aren't equal can't do operation element wise."
#         )

#         let res = Self(False, self.dims)
#         let size = self.dims.num_elements()

#         let last_dim = self.dims[-1]
#         var dims_rest = size // last_dim

#         @parameter
#         fn outer_loop(i: Int):
#             @parameter
#             fn iterate_vectorize[nelts: Int](j: Int):
#                 let index = i * last_dim + j

#                 res.store[nelts](
#                     index,
#                     op_func[Type, nelts](
#                         self.load[nelts](index), other.load[nelts](index)
#                     ),
#                 )

#             vectorize[nelts, iterate_vectorize](last_dim)

#         outer_loop_func[outer_loop](dims_rest)

#         return res ^

#     @always_inline
#     fn __add__(self, other: Self) -> Self:
#         return self.add[TensorG[Type].simd_width](other)

#     @always_inline
#     fn add[nelts: Int](self, other: Self) -> Self:
#         @parameter
#         fn sum_v[outer_loop: fn (Int) capturing -> None](range_size: Int):
#             for i in range(range_size):
#                 outer_loop(i)

#         let res = self.__iterate_binary_op_tensor[nelts, sum_v, math.add](other)

#         return res ^

#     @always_inline
#     fn add[nelts: Int](self, other: Self, rt: Runtime, n_cores: Int) -> Self:
#         @parameter
#         fn sum_p[outer_loop: fn (Int) capturing -> None](range_size: Int):
#             parallelize[outer_loop](rt, range_size, n_cores)

#         let res = self.__iterate_binary_op_tensor[nelts, sum_p, math.add](other)

#         return res ^

#     fn __mul__(self, other: Self) -> Self:
#         return self.mul[TensorG[Type].simd_width](other)

#     @always_inline
#     fn mul[nelts: Int](self, other: Self) -> Self:
#         @parameter
#         fn mul_v[outer_loop: fn (Int) capturing -> None](range_size: Int):
#             for i in range(range_size):
#                 outer_loop(i)

#         let res = self.__iterate_binary_op_tensor[nelts, mul_v, math.mul](other)

#         return res ^

#     @always_inline
#     fn mul[nelts: Int](self, other: Self, rt: Runtime, n_cores: Int) -> Self:
#         @parameter
#         fn mul_p[outer_loop: fn (Int) capturing -> None](range_size: Int):
#             parallelize[outer_loop](rt, range_size, n_cores)

#         let res = self.__iterate_binary_op_tensor[nelts, mul_p, math.mul](other)

#         return res ^

#     @always_inline
#     fn __eq__(self, other: Self) -> Bool:
#         return self.eq[1](other)

#     @always_inline
#     fn eq[nelts: Int](self, other: Self) -> Bool:
#         let dims_eq = self.dims == other.dims
#         debug_assert(dims_eq, "Error dimension aren't equal can't sum tensors.")

#         var flag = True
#         let size = self.dims.num_elements()

#         @parameter
#         fn iterate_vectorize[nelts: Int](i: Int):
#             if self.load[nelts](i) != other.load[nelts](i):
#                 flag = False

#         vectorize[nelts, iterate_vectorize](size)

#         return flag

#     @always_inline
#     fn eq[nelts: Int](self, other: Self, rt: Runtime, n_cores: Int) -> Bool:
#         let dims_eq = self.dims == other.dims
#         debug_assert(dims_eq, "Error dimension aren't equal can't sum tensors.")

#         var flag = True
#         let size = self.dims.num_elements()

#         let first_dim = self.dims[0]
#         let dims_rest = size // first_dim  # the rest of the dimensions

#         @parameter
#         fn iterate_parallel(i: Int):
#             @parameter
#             fn iterate_vectorize[nelts: Int](j: Int):
#                 let index = i * dims_rest + j

#                 if self.load[nelts](index) != other.load[nelts](index):
#                     flag = False

#             vectorize[nelts, iterate_vectorize](dims_rest)

#         parallelize[iterate_parallel](rt, first_dim, n_cores)

#         return flag

#     @always_inline
#     fn dot[nelts: Int](self, other: Self) -> Self:
#         let dims_1d = self.dims.rank() == 1 and other.dims.rank() == 1
#         debug_assert(dims_1d, "Error dimensions aren't 1D can't dot tensors.")
#         let dims_eq = self.dims == other.dims
#         debug_assert(dims_eq, "Error dimension aren't equal can't sum tensors.")

#         let res = Self(False, 1)
#         let size = self.dims.num_elements()

#         @parameter
#         fn dot_v[nelts: Int](index: Int):
#             res[0] = (
#                 res[0]
#                 + (
#                     self.data.simd_load[nelts](index)
#                     * other.data.simd_load[nelts](index)
#                 ).reduce_add()
#             )

#         vectorize[nelts, dot_v](size)

#         return res ^

#     @always_inline
#     fn __matmul__(self, other: Self) -> Self:
#         return self.matmul[TensorG[Type].simd_width](other)

#     @always_inline
#     fn matmul[nelts: Int](self, other: Self) -> Self:
#         @parameter
#         fn matmul_v[outer_loop: fn (Int) capturing -> None](range_size: Int):
#             for i in range(range_size):
#                 outer_loop(i)

#         let res = self.__matmul[nelts, matmul_v](other)

#         return res ^

#     @always_inline
#     fn matmul[nelts: Int](self, other: Self, rt: Runtime, n_cores: Int) -> Self:
#         @parameter
#         fn matmul_p[outer_loop: fn (Int) capturing -> None](range_size: Int):
#             parallelize[outer_loop](rt, range_size, n_cores)

#         let res = self.__matmul[nelts, matmul_p](other)

#         return res ^

#     @staticmethod
#     # Perform 2D tiling on the iteration space defined by end_x and end_y.
#     fn tile[tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int](end_x: Int, end_y: Int):
#         # Note: this assumes that ends are multiples of the tiles.
#         for y in range(0, end_y, tile_y):
#             for x in range(0, end_x, tile_x):
#                 tiled_fn[tile_x, tile_y](x, y)

#     fn __matmul[
#         nelts: Int,
#         outer_loop_func: fn[func: fn (Int) capturing -> None] (Int) capturing -> None,
#     ](self, other: Self) -> Self:
#         if self.dims.rank() == 1 and other.dims.rank() == 1:
#             return self.dot[nelts](other)

#         let dims_eq = self.dims.eq_matmul(other.dims)
#         debug_assert(dims_eq, "Error dimensions don't conform for a matmul.")

#         var res_dims = InlinedFixedVector[dims_average_size, Int](self.dims.rank())
#         for i in range(self.dims.rank() - 1):
#             res_dims.append(self.dims[i])
#         res_dims.append(other.dims[other.dims.rank() - 1])

#         let res = Self(False, TensorView(res_dims))
#         # let size = self.__matmul_num_elements(other)

#         # The dimension that is different for self and other (other dim)
#         let res_last_dim = res.dims[res.dims.rank() - 1]
#         # the dimension that is the same for self and other
#         let self_last_dim = self.dims[self.dims.rank() - 1]
#         # The other dimension that is different for self and other (self dim)
#         let res_penult_dim = res.dims[res.dims.rank() - 2]

#         let size = self.size * res_last_dim  # size to iterate over

#         # We use the for inside the parallel function to remove data races, so the vectorize function works in the last dimension of the res tensor and the for makes it so the for and vectorize function work on the penultimate dimension of the res tensor (so parallel works on the penultimate dimension of the res tensor)
#         @parameter
#         fn outer_loop(i: Int):
#             @parameter
#             fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
#                 let j_range = math.min(self_last_dim, y + tile_y)
#                 for j in range(y, j_range):

#                     @parameter
#                     fn matmul_v[nelts: Int](k: Int):
#                         let index_res = i * res_last_dim + k + x  # remove data races of parallel function
#                         let index_self = i * self_last_dim + j
#                         let index_other = (
#                             i // res_penult_dim
#                         ) * self_last_dim * res_last_dim + j * res_last_dim + k + x

#                         res.store[nelts](
#                             index_res,
#                             res.load[nelts](index_res)
#                             + self.load[1](index_self) * other.load[nelts](index_other),
#                         )

#                     vectorize_unroll[nelts, tile_x // nelts, matmul_v](
#                         math.min(res_last_dim - x, tile_x)
#                     )

#             alias tile_size = 4
#             self.tile[calc_tile, nelts * tile_size, tile_size](
#                 res_last_dim, self_last_dim
#             )

#         # this function is going to be basically the outer for loop, the function is going to be calling outer_loop using any method it wants until range_size("size // (res_last_dim * self_last_dim)")
#         outer_loop_func[outer_loop](size // (res_last_dim * self_last_dim))

#         return res ^