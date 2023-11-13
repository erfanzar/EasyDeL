# parts of the code are edited from tairov/llama2.mojo

from tensor import Tensor, TensorShape
from algorithm.functional import vectorize, parallelize, unroll
import math
from random import rand


struct TensorSlice[T: DType]:
    # Provides a view into a tensor representing a 1D slice on its first or first 2 dimensions.
    # Same function signatures as Tensor but without owning the data.
    var _data: DTypePointer[T]
    var _shape: TensorShape

    fn __init__(inout self, t: Tensor[T], layer: Int) raises:
        let elements_per_layer = t.num_elements() // t.dim(0)
        self._data = t.data().offset(layer * elements_per_layer)
        if t.rank() == 2:
            self._shape = TensorShape(t.dim(1))
        elif t.rank() == 3:
            self._shape = TensorShape(t.dim(1), t.dim(2))
        else:
            # Compiler complains if _shape not defined
            self._shape = TensorShape(1)
            raise Error("TensorSlice: rank greater than 3 not implemented.")

    fn __init__(inout self, t: Tensor[T], layer: Int, row: Int) raises:
        let elements_per_layer = t.num_elements() // t.dim(0)
        let elements_per_row = elements_per_layer // t.dim(1)
        self._data = t.data().offset(
            layer * elements_per_layer + row * elements_per_row
        )
        if t.rank() == 3:
            self._shape = TensorShape(t.dim(2))
        elif t.rank() == 1:
            # Compiler complains if _shape not defined
            self._shape = TensorShape(1)
            raise Error(
                "Trying to slice a 1D Tensor by layer and row.  This requires a 3D"
                " Tensor."
            )
        else:
            # Compiler complains if _shape not defined
            self._shape = TensorShape(1)
            raise Error("TensorSlice: rank greater than 3 not implemented.")

    fn data(self) -> DTypePointer[T]:
        return self._data

    fn shape(self) -> TensorShape:
        return self._shape

    fn num_elements(self) -> Int:
        return self._shape.num_elements()

    fn dim(self, idx: Int) -> Int:
        return self._shape[idx]

    fn rank(self) -> Int:
        return self._shape.rank()

    fn simd_load[nelts: Int](self, idx: Int) -> SIMD[T, nelts]:
        return self._data.simd_load[nelts](idx)

    fn simd_load[nelts: Int](self, indices: StaticIntTuple[2]) -> SIMD[T, nelts]:
        return self._data.simd_load[nelts](indices[0] * self._shape[1] + indices[1])

    fn __getitem__(self, idx: Int) -> SIMD[T, 1]:
        return self._data.simd_load[1](idx)

    fn simd_store[nelts: Int](self, idx: Int, val: SIMD[T, nelts]):
        return self._data.simd_store[nelts](idx, val)

    fn __setitem__(self, idx: Int, val: SIMD[T, 1]):
        return self.simd_store[1](idx, val)


@always_inline
fn rmsnorm[
    T: DType, nelts: Int
](
    inout o: DTypePointer[T], x: DTypePointer[T], weight: DTypePointer[T], size: Int
) -> None:
    # Calculate sum of squares
    var tmp = SIMD[T, nelts](0)

    @parameter
    fn _sum2[_nelts: Int](j: Int):
        if _nelts < nelts:
            tmp[0] += (x.offset(j).simd_load[_nelts](0) ** 2).reduce_add()
        else:
            tmp += x.offset(j).simd_load[nelts](0) ** 2

    vectorize[nelts, _sum2](size)

    var ss: SIMD[T, 1] = tmp.reduce_add()
    ss = ss / size + 1e-5
    ss = 1.0 / math.sqrt(ss)

    # Normalize and scale
    @parameter
    fn _norm[_nelts: Int](j: Int):
        let val = weight.simd_load[_nelts](j) * ss * x.simd_load[_nelts](j)
        o.offset(j).simd_store[_nelts](0, val)

    vectorize[nelts, _norm](size)


@always_inline
fn rmsnorm[T: DType, nelts: Int](inout o: Tensor[T], x: Tensor[T], weight: Tensor[T]):
    rmsnorm[T, nelts](o._ptr, x.data(), weight.data(), weight.dim(weight.rank() - 1))


@always_inline
fn rmsnorm[
    T: DType, nelts: Int
](inout o: Tensor[T], x: Tensor[T], weight: TensorSlice[T]):
    rmsnorm[T, nelts](o._ptr, x.data(), weight.data(), weight.dim(weight.rank() - 1))


@always_inline
fn softmax[T: DType, nelts: Int](inout x: Tensor[T]) -> None:
    softmax[T, nelts](x, 0, x.dim(0))


@always_inline
fn softmax[T: DType, nelts: Int](inout x: Tensor[T], start: Int, end: Int):
    var max_val: SIMD[T, 1] = -1e9

    @parameter
    fn _max[_nelts: Int](ii: Int):
        let val = x.simd_load[_nelts](start + ii).reduce_max()
        if val > max_val:
            max_val = val

    vectorize[nelts, _max](end - start)

    var ssum: SIMD[T, 1] = 0.0

    @parameter
    fn _exp[_nelts: Int](ii: Int):
        x.simd_store[_nelts](
            start + ii, math.exp(x.simd_load[_nelts](start + ii) - max_val)
        )
        ssum += x.simd_load[_nelts](start + ii).reduce_add()

    vectorize[nelts, _exp](end - start)

    @parameter
    fn _norm[_nelts: Int](ii: Int):
        x.simd_store[_nelts](start + ii, x.simd_load[_nelts](start + ii) / ssum)

    vectorize[nelts, _norm](end - start)


@always_inline
fn batch_matmul[
    T: DType, nelts: Int, cores: Int, n: Int
](
    C: StaticTuple[n, DTypePointer[T]],
    A: DTypePointer[T],
    B: StaticTuple[n, DTypePointer[T]],
    rows: Int,
    cols: Int,
):
    @parameter
    fn compute_row(i: Int):
        var tmp = StaticTuple[n, SIMD[T, nelts]]()

        @parameter
        fn init[k: Int]():
            tmp[k] = SIMD[T, nelts](0)

        unroll[n, init]()
        let row_offset = i * cols

        @parameter
        fn dot[_nelts: Int](j: Int):
            if _nelts < nelts:  # take care of tail array elements with length <  nelts
                let a = A.simd_load[_nelts](j)

                @parameter
                fn _multiply_tail[k: Int]():
                    tmp[k][0] += (
                        a * B[k].simd_load[_nelts](row_offset + j)
                    ).reduce_add()

                unroll[n, _multiply_tail]()
            else:
                let a = A.simd_load[nelts](j)

                @parameter
                fn _multiply[k: Int]():
                    tmp[k] += a * B[k].simd_load[nelts](row_offset + j)

                unroll[n, _multiply]()

        vectorize[nelts, dot](cols)

        @parameter
        fn _reduce[k: Int]():
            C[k].store(i, tmp[k].reduce_add())

        unroll[n, _reduce]()

    parallelize[compute_row](rows, cores)


@always_inline
fn matmul[
    T: DType, nelts: Int, cores: Int
](C: Tensor[T], A: Tensor[T], B: Tensor[T]) raises:
    # B (d,n) @ A (n,) -> C (d,)
    matmul_dimension_checks(A.shape(), B.shape())
    batch_matmul[T, nelts, cores, 1](
        StaticTuple[1, DTypePointer[T]](C.data()),
        A.data(),
        StaticTuple[1, DTypePointer[T]](B.data()),
        B.dim(0),
        B.dim(1),
    )


@always_inline
fn matmul[
    T: DType, nelts: Int, cores: Int
](C: Tensor[T], A: Tensor[T], B: TensorSlice[T]) raises:
    # B (d,n) @ A (n,) -> C (d,)
    matmul_dimension_checks(A.shape(), B.shape())
    batch_matmul[T, nelts, cores, 1](
        StaticTuple[1, DTypePointer[T]](C.data()),
        A.data(),
        StaticTuple[1, DTypePointer[T]](B.data()),
        B.dim(0),
        B.dim(1),
    )


@always_inline
fn matmul[
    T: DType, nelts: Int, cores: Int
](C: TensorSlice[T], A: Tensor[T], B: TensorSlice[T]) raises:
    # B (d,n) @ A (n,) -> C (d,)
    matmul_dimension_checks(A.shape(), B.shape())
    batch_matmul[T, nelts, cores, 1](
        StaticTuple[1, DTypePointer[T]](
            C.data(),
        ),
        A.data(),
        StaticTuple[1, DTypePointer[T]](B.data()),
        B.dim(0),
        B.dim(1),
    )


fn matmul_dimension_checks(a: TensorShape, b: TensorShape) raises:
    if a[0] != b[1]:
        raise Error(
            "matmul dimension mismatch. A rows (dim 0) not equal to B columns (dim 1)"
        )
    if b.rank() != 2:
        raise Error("matmul expects B to be a 2D matrix")


fn argmax[T: DType](v: Tensor[T]) -> Int:
    # return argmax of v
    var max_i: Int = 0
    var max_p: SIMD[T, 1] = v[0]
    for i in range(v.dim(0)):
        if v[i] > max_p:
            max_i = i
            max_p = v[i]
    return max_i


fn sample[T: DType](probabilities: Tensor[T]) -> Int:
    let n = probabilities.dim(0)
    let r = rand[T](1)
    var cdf: SIMD[T, 1] = 0.0
    for i in range(n):
        cdf += probabilities[i]
        if r[0] < cdf:
            return i
    return n - 1


@always_inline
fn tensor_add[T: DType, nelts: Int](inout a: Tensor[T], b: Tensor[T]) -> None:
    let size = a.num_elements()

    @parameter
    fn _acc[_nelts: Int](j: Int):
        a.simd_store[_nelts](j, a.simd_load[_nelts](j) + b.simd_load[_nelts](j))

    vectorize[nelts, _acc](size)


@always_inline
fn tensor_mul[T: DType, nelts: Int](inout a: Tensor[T], b: Tensor[T]) -> None:
    let size = a.num_elements()

    @parameter
    fn _mul[_nelts: Int](j: Int):
        a.simd_store[_nelts](j, a.simd_load[_nelts](j) * b.simd_load[_nelts](j))

    vectorize[nelts, _mul](size)


@always_inline
fn tensor_sub[T: DType, nelts: Int](inout a: Tensor[T], b: Tensor[T]) -> None:
    let size = a.num_elements()

    @parameter
    fn _sub[_nelts: Int](j: Int):
        a.simd_store[_nelts](j, a.simd_load[_nelts](j) - b.simd_load[_nelts](j))

    vectorize[nelts, _sub](size)


@always_inline
fn tensor_div[T: DType, nelts: Int](inout a: Tensor[T], b: Tensor[T]) -> None:
    let size = a.num_elements()

    @parameter
    fn _div[_nelts: Int](j: Int):
        a.simd_store[_nelts](j, a.simd_load[_nelts](j) / b.simd_load[_nelts](j))

    vectorize[nelts, _div](size)


@always_inline
fn tensor_pow[T: DType, nelts: Int](inout a: Tensor[T], b: Tensor[T]) -> None:
    let size = a.num_elements()

    @parameter
    fn _pow[_nelts: Int](j: Int):
        a.simd_store[_nelts](j, a.simd_load[_nelts](j) ** b.simd_load[_nelts](j))

    vectorize[nelts, _pow](size)

@always_inline
fn tensor_pow[T: DType, nelts: Int](inout a: Tensor[T], b: SIMD[T,1]) -> None:
    let size = a.num_elements()

    @parameter
    fn _pow[_nelts: Int](j: Int):
        a.simd_store[_nelts](j, a.simd_load[1](j) ** b)

    vectorize[nelts, _pow](size)