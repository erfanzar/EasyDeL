# parts of the code are edited from tairov/llama2.mojo

from tensor import Tensor, TensorShape
from algorithm.functional import vectorize, parallelize, unroll
import math
from random import rand


fn stride[T: DType](tensor: Tensor[T], axis: Int) -> Int:
    let tensor_rank = tensor.rank() - 1
    if tensor_rank == axis:
        return 1

    var stride = 1
    for i in range(axis - tensor_rank, 1):
        stride *= tensor.shape()[tensor_rank + i]
    return stride


fn a_over_shape[T: DType](depth: Int, A: Tensor[T], B: Tensor[T]) -> Int:
    let _df = max(B.rank() - A.rank(), 0)
    if depth < _df:
        return 1
    return A.shape()[depth - _df]


fn b_over_shape[T: DType](depth: Int, A: Tensor[T], B: Tensor[T]) -> Int:
    let _df = max(A.rank() - B.rank(), 0)
    if depth < _df:
        return 1
    return B.shape()[depth - _df]


fn a_over_strides[T: DType](depth: Int, A: Tensor[T], B: Tensor[T]) -> Int:
    let _df = max(B.rank() - A.rank(), 0)
    if depth < _df:
        return stride[T](A, 0)
    return stride[T](A, depth - _df)


fn b_over_strides[T: DType](depth: Int, A: Tensor[T], B: Tensor[T]) -> Int:
    let _df = max(A.rank() - B.rank(), 0)
    if depth < _df:
        return stride[T](B, 0)
    return stride[T](B, depth - _df)


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
            if _nelts < nelts:  # take care of tail tensor elements with length <  nelts
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
fn tensor_pow[T: DType, nelts: Int](inout a: Tensor[T], b: SIMD[T, 1]) -> None:
    let size = a.num_elements()

    @parameter
    fn _pow[_nelts: Int](j: Int):
        a.simd_store[_nelts](j, a.simd_load[1](j) ** b)

    vectorize[nelts, _pow](size)


@always_inline
fn concatenate[
    T: DType
](A: Tensor[T], B: Tensor[T], inout axis: Int = 0, stragedy: String = "ord") -> Tensor[
    T
]:
    debug_assert(A.rank() == B.rank(), "A,B Shapes Wont Match for concatenate")
    if axis < Int(0):
        axis = A.rank() - axis
    var cat_shape: DynamicVector[Int] = DynamicVector[Int]()
    for i in range(A.rank()):
        if i != axis:
            debug_assert(A.dim(i) == B.dim(i), "A,B Shapes Wont Match for concatenate")
            cat_shape.push_back(A.dim(i))
        else:
            cat_shape.push_back(A.dim(i) + B.dim(i))
    var cat_tensor: Tensor[T] = Tensor[T](cat_shape)
    cat_tensor.alloc(0.0)
    if stragedy == "ord":

        @parameter
        fn _r[nelts: Int](I: Int):
            if I < A.num_elements():
                cat_tensor.store[nelts](I, A.load[nelts](I))
            else:
                cat_tensor.store[nelts](I, B.load[nelts](I))

        vectorize[Tensor[T].nelts, _r](cat_tensor.num_elements())
    return cat_tensor




@always_inline
fn arange[T: DType, cores: Int](start: Int, end: Int) -> Tensor[T]:
    let rng = end - start
    var tensor: Tensor[T] = Tensor[T](rng)
    tensor.alloc()

    @parameter
    fn _row(i: Int):
        tensor.store[1](i, start + i)

    parallelize[_row](rng, cores)
    return tensor


fn recursive_broadcast[
    T: DType,
    nelts: Int,
    cores: Int,
    kernel: fn[T:DType,nelts:Int,cores:Int](inout C: Tensor[T], A: Tensor[T], B: Tensor[T], a_index: Int, b_index: Int,c_index: Int, depth: Int) -> None,
    base_case: fn[T: DType] (depth: Int, A: Tensor[T], B: Tensor[T]) -> Bool,
    parallelized: Bool,
](
    inout C: Tensor[T],
    A: Tensor[T],
    B: Tensor[T],
    a_index: Int = 0,
    b_index: Int = 0,
    c_index: Int = 0,
    depth: Int = 0,
):
    if base_case[T](depth, A, B):
        kernel[T, nelts, cores](C, A, B, a_index, b_index, c_index, depth)
        return

    let a_shape = a_over_shape[T](depth, A, B)
    let b_shape = b_over_shape[T](depth, A, B)
    let c_shape = C.shape()[depth]
    if a_shape != 1 and b_shape == 1:
        if parallelized:

            @parameter
            fn _cols_a(s: Int):
                recursive_broadcast[T, nelts, cores, kernel, base_case, parallelized](
                    C,
                    A,
                    B,
                    a_shape * a_index + s,
                    b_shape * b_index,
                    c_shape * c_index + s,
                    depth + 1,
                )

            parallelize[_cols_a](a_shape, cores)

        for s in range(a_shape):
            recursive_broadcast[T, nelts, cores, kernel, base_case, parallelized](
                C,
                A,
                B,
                a_shape * a_index + s,
                b_shape * b_index,
                c_shape * c_index + s,
                depth + 1,
            )
    elif a_shape == 1 and b_shape != 1:
        if parallelized:

            @parameter
            fn _cols_b(s: Int):
                recursive_broadcast[T, nelts, cores, kernel, base_case, parallelized](
                    C,
                    A,
                    B,
                    a_shape * a_index,
                    b_shape * b_index + s,
                    c_shape * c_index + s,
                    depth + 1,
                )

            parallelize[_cols_b](b_shape, cores)
        else:
            for s in range(b_shape):
                recursive_broadcast[T, nelts, cores, kernel, base_case, parallelized](
                    C,
                    A,
                    B,
                    a_shape * a_index,
                    b_shape * b_index + s,
                    c_shape * c_index + s,
                    depth + 1,
                )
    else:
        if parallelized:

            @parameter
            fn _cols_a_s(s: Int):
                recursive_broadcast[T, nelts, cores, kernel, base_case, parallelized](
                    C,
                    A,
                    B,
                    a_shape * a_index + s,
                    b_shape * b_index + s,
                    c_shape * c_index + s,
                    depth + 1,
                )

            parallelize[_cols_a_s](a_shape, cores)
        else:
            for s in range(a_shape):
                recursive_broadcast[T, nelts, cores, kernel, base_case, parallelized](
                    C,
                    A,
                    B,
                    a_shape * a_index + s,
                    b_shape * b_index + s,
                    c_shape * c_index + s,
                    depth + 1,
                )


@always_inline
fn kernel_matmul[
    T: DType, nelts: Int, cores: Int
](
    inout C: Tensor[T],
    A: Tensor[T],
    B: Tensor[T],
    a_index: Int,
    b_index: Int,
    c_index: Int,
    depth: Int,
) -> None:
    let offset_a = a_index * A.shape()[A.rank() - 2] * A.shape()[A.rank() - 1]
    let offset_b = b_index * B.shape()[B.rank() - 2] * B.shape()[B.rank() - 1]
    let offset_c = c_index * C.shape()[C.rank() - 2] * C.shape()[C.rank() - 1]

    let M = A.shape()[A.rank() - 2]
    let K = B.shape()[B.rank() - 2]
    let N = B.shape()[B.rank() - 1]

    @parameter
    fn calc_row(m: Int):
        for k in range(K):

            @parameter
            fn dot[nelts: Int](n: Int):
                C.data.simd_store[nelts](
                    offset_c + m * N + n,
                    C.simd_load[nelts](offset_c + m * N + n)
                    + A.simd_load[1](offset_a + m * K + k)
                    * B.simd_load[nelts](offset_b + k * N + n),
                )

            vectorize[nelts, dot](N)

    parallelize[calc_row](M, cores if cores > 0 else M)


@always_inline
fn matmul[
    T: DType, nelts: Int = Tensor[T].nelts, cores: Int = 1, parallelized: Bool = True
](inout C: Tensor[T], A: Tensor[T], B: Tensor[T]):
    recursive_broadcast[
        T, nelts, cores, kernel_matmul, base_case_matmul, parallelized
    ](C, A, B)


@always_inline
fn sigmoid[
    T: DType
](inout x: Tensor[T], number_of_cores: Int = 1, size: Int = 0) -> None:
    let num_elements: Int = x.num_elements() if size == 0 else size

    @parameter
    fn _row(size: Int):
        x[size] = 1.0 / (1.0 + math.exp(-x[size]))

    parallelize[_row](num_elements, number_of_cores)


@always_inline
fn silu[T: DType](inout x: Tensor[T], number_of_cores: Int = 1, size: Int = 0) -> None:
    let num_elements: Int = x.num_elements() if size == 0 else size

    @parameter
    fn _row(size: Int):
        let dt: SIMD[T, 1] = x[size]
        x[size] = dt * (1.0 / (1.0 + math.exp(-dt)))

    parallelize[_row](num_elements, number_of_cores)


@always_inline
fn relu[T: DType](inout x: Tensor[T], number_of_cores: Int = 1, size: Int = 0) -> None:
    let num_elements: Int = x.num_elements() if size == 0 else size

    @parameter
    fn _row(size: Int):
        let dt: SIMD[T, 1] = x[size]
        x[size] = dt if dt > 0 else 0.0

    parallelize[_row](num_elements, number_of_cores)


@always_inline
fn leaky_relu[
    T: DType
](
    inout x: Tensor[T], drop: SIMD[T, 1] = 0.1, number_of_cores: Int = 1, size: Int = 0
) -> None:
    let num_elements: Int = x.num_elements() if size == 0 else size

    @parameter
    fn _row(size: Int):
        let dt: SIMD[T, 1] = x[size]
        x[size] = dt if dt > drop else drop

    parallelize[_row](num_elements, number_of_cores)


# POINTER FUNC #


@always_inline
fn softmax[T: DType, nelts: Int](inout x: DTypePointer[T], num_elements: Int) -> None:
    var max_val: SIMD[T, 1] = SIMD[T, 1](-1e9)

    @parameter
    fn _max[_nelts: Int](j: Int):
        let val = x.simd_load[_nelts](j).reduce_max()
        if val > max_val:
            max_val = val

    vectorize[nelts, _max](num_elements)
    var sum_val: SIMD[T, 1] = 0.0

    @parameter
    fn _sum_exp[_nelts: Int](j: Int):
        x.simd_store[_nelts](j, math.exp(x.simd_load[_nelts](j) - max_val))
        sum_val += x.simd_load[_nelts](j).reduce_add()

    vectorize[nelts, _sum_exp](num_elements)

    @parameter
    fn _norm[_nelts: Int](j: Int):
        x.simd_store[_nelts](j, x.simd_load[_nelts](j) / sum_val)

    vectorize[nelts, _norm](num_elements)


@always_inline
fn sigmoid[
    T: DType
](inout x: DTypePointer[T], num_elements: Int, number_of_cores: Int = 1) -> None:
    @parameter
    fn _row(size: Int):
        x.store(size, 1.0 / (1.0 + math.exp(-x.load(size))))

    parallelize[_row](num_elements, number_of_cores)


@always_inline
fn silu[
    T: DType
](inout x: DTypePointer[T], num_elements: Int, number_of_cores: Int = 1) -> None:
    @parameter
    fn _row(size: Int):
        let dt: SIMD[T, 1] = x.load(size)
        x.store(size, dt * (1.0 / (1.0 + math.exp(-dt))))

    parallelize[_row](num_elements, number_of_cores)


@always_inline
fn relu[
    T: DType
](inout x: DTypePointer[T], num_elements: Int, number_of_cores: Int = 1) -> None:
    @parameter
    fn _row(size: Int):
        let dt: SIMD[T, 1] = x.load(size)
        x.store(size, dt if dt > 0 else 0.0)

    parallelize[_row](num_elements, number_of_cores)


@always_inline
fn leaky_relu[
    T: DType
](
    inout x: DTypePointer[T],
    num_elements: Int,
    drop: SIMD[T, 1] = 0.1,
    number_of_cores: Int = 1,
) -> None:
    @parameter
    fn _row(size: Int):
        let dt: SIMD[T, 1] = x.load(size)
        x.store(size, dt if dt > drop else drop)

    parallelize[_row](num_elements, number_of_cores)


@always_inline
fn sample[T: DType](tensor: Tensor[T]) -> Int:
    let number_of_cols = tensor.dim(-1)
    let random_value = DTypePointer[T].alloc(1)
    rand[T](random_value, 1)
    var cdf: SIMD[T, 1] = 0.0
    for i in range(number_of_cols):
        cdf += tensor[i]
        if random_value.load(0) < cdf:
            return i
    return number_of_cols - 1


@always_inline
fn softmax[
    T: DType, nelts: Int, cores: Int
](inout B: Tensor[T], A: Tensor[T], axis: Int = -1):
    let N = A.dim(axis)

    @parameter
    fn rows(s: Int):
        var max_element: SIMD[T, 1] = 0.0

        @parameter
        fn v_max[nelts: Int](i: Int):
            let _x = B.simd_load[nelts](s * N + i).reduce_max()
            max_element = max(max_element, _x)

        vectorize[nelts, v_max](N)
        var sum_loop: SIMD[T, 1] = 0.0

        @parameter
        fn v_exp[nelts: Int](i: Int):
            let _x = math.exp(A.simd_load[nelts](s * N + i) - max_element)
            B.data.simd_store[nelts](s * N + i, _x)
            sum_loop += _x.reduce_add()

        vectorize[nelts, v_exp](N)

        @parameter
        fn v_div[nelts: Int](i: Int):
            B.data.simd_store[nelts](
                s * N + i, B.simd_load[nelts](s * N + i) / sum_loop
            )

        vectorize[nelts, v_div](N)

    parallelize[rows](B.num_elements() // N, cores)


@always_inline
fn softmax[
    T: DType, nelts: Int, cores: Int
](A: Tensor[T], axis: Int = -1) -> Tensor[T]:
    var B: Tensor[T] = Tensor[T](A)
    
    softmax[T, nelts, cores](B, A, axis)
    return B


@always_inline
fn mse[T: DType, nelts: Int](inout C: Tensor[T], A: Tensor[T], B: Tensor[T]):
    @parameter
    fn v_mse[nelts: Int](index: Int):
        let error = (
            A.simd_load[nelts](index) - B.simd_load[nelts](index)
        ) * (A.simd_load[nelts](index) - B.simd_load[nelts](index))
        C.store[1](0, C.simd_load[1](0) + error.reduce_add())

    vectorize[nelts, v_mse](A.num_elements())
    C.store[1](0, C.simd_load[1](0) / SIMD[T, 1](A.num_elements()))


@always_inline
fn ce[T: DType, nelts: Int](inout C: Tensor[T], A: Tensor[T], B: Tensor[T]):
    let N = A.dim(-1)
    let epsilon = SIMD[T, 1](1e-8)

    @parameter
    fn v_ce[nelts: Int](index: Int):
        let error = -A.simd_load[nelts](index) * log(
            B.simd_load[nelts](index) + epsilon
        )
        C.store[1](0, C.simd_load[1](0) + error.reduce_add())

    vectorize[nelts, v_ce](A.num_elements())
    C.store[1](0, C.simd_load[1](0) / (SIMD[T, 1](A.num_elements()) / SIMD[T, 1](N)))


@always_inline
fn mean[T: DType, nelts: Int](inout B: Tensor[T], A: Tensor[T]):
    let dim_len: Int = B.extra_parameters.load(0)

    # Calculate total number of elements in dims
    var total_elements_in_dims: Int = 1
    for d in range(dim_len):
        let dim: Int = B.extra_parameters.load(d + 1)
        total_elements_in_dims *= A.shape()[dim]

    var in_dims = DynamicVector[Bool](B.rank())
    for d in range(B.rank()):
        in_dims[d] = False
    for d in range(dim_len):
        in_dims[B.extra_parameters.load(d + 1)] = True

    # Iterate over all elements in the Tensor[T]
    for i in range(A.num_elements()):
        var indeces = DynamicVector[Int]()
        for dim in range(A.rank()):
            indeces.push_back((i // stride[T](A,dim)) % A.shape()[dim])
        var output_index = 0
        for dim in range(B.rank()):
            if not in_dims[dim]:
                output_index += indeces[dim] * stride[T](V,dim)

        B.data.store(output_index, B.simd_load[1](output_index) + A.simd_load[1](i))

    # Divide each element in output Tensor[T] by total number of elements in dims
    for i in range(B.num_elements()):
        let value: SIMD[T, 1] = B.simd_load[1](i) / SIMD[T, 1](total_elements_in_dims)
        B.data.store(i, value)


@always_inline
fn variance[T: DType, nelts: Int](inout B: Tensor[T], A: Tensor[T]):
    let dim_len: Int = B.extra_parameters.load(0)
    let mean_output = DTypePointer[T].alloc(B.num_elements())
    memset_zero(mean_output, B.num_elements())

    # Calculate total number of elements in dims
    var total_elements_in_dims: Int = 1
    for d in range(dim_len):
        let dim: Int = B.extra_parameters.load(d + 1)
        total_elements_in_dims *= A.shape()[dim]

    var in_dims = DynamicVector[Bool](B.rank())
    for d in range(B.rank()):
        in_dims[d] = False
    for d in range(dim_len):
        in_dims[B.extra_parameters.load(d + 1)] = True

    # Iterate over all elements in the Tensor[T]
    for i in range(A.num_elements()):
        var indeces = DynamicVector[Int]()
        for dim in range(A.rank()):
            indeces.push_back((i // stride[T](A,dim)) % A.shape()[dim])

        var output_index = 0
        for dim in range(B.rank()):
            if not in_dims[dim]:
                output_index += indeces[dim] *stride[T](B,dim)

        mean_output.store(output_index, mean_output.load(output_index) + A.simd_load[1](i))

    # Divide each element in output Tensor[T] by total number of elements in dims
    for i in range(B.num_elements()):
        let value: SIMD[T, 1] = mean_output.load(i) / SIMD[T, 1](
            total_elements_in_dims
        )
        mean_output.store(i, value)

    # Iterate over all elements in the Tensor[T] again to calculate squared _dferences from the mean
    for i in range(A.num_elements()):
        var indeces = DynamicVector[Int]()
        for dim in range(A.rank()):
            indeces.push_back((i // stride[T](A,dim)) % A.shape()[dim])

        var output_index = 0
        for dim in range(B.rank()):
            if not in_dims[dim]:
                output_index += indeces[dim] *stride[T](B,dim)

        let _df = A.simd_load[1](i) - mean_output.load(output_index)
        B.data.store(output_index, B.simd_load[1](output_index) + _df * _df)

    # Divide each element in squared__df_output Tensor[T] by total number of elements in dims to get the variance
    for i in range(B.num_elements()):
        let value: SIMD[T, 1] = B.simd_load[1](i) / SIMD[T, 1](
            total_elements_in_dims - 1
        )
        B.data.store(i, value)

@always_inline
fn kernel_mul[
    T: DType, nelts: Int, cores: Int
](
    inout C: Tensor[T],
    A: Tensor[T],
    B: Tensor[T],
    a_index: Int,
    b_index: Int,
    c_index: Int,
    depth: Int,
) -> None:
    let offset_a = a_index * a_over_shape[T](depth, A, B) * a_over_strides[T](
        depth, A, B
    )
    let offset_b = b_index * b_over_shape[T](depth, A, B) * b_over_strides[T](
        depth, A, B
    )
    let c_rest = C.shape()[depth] * stride[T](C,depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_mul[nelts: Int](i: Int):
        C.data.simd_store[nelts](
            offset_c + i,
            A.simd_load[nelts](offset_a + i)
            * B.simd_load[nelts](offset_b + i),
        )

    vectorize[nelts, v_mul](c_rest)


@always_inline
fn kernel_add[
    T: DType, nelts: Int, cores: Int
](
    inout C: Tensor[T],
    A: Tensor[T],
    B: Tensor[T],
    a_index: Int,
    b_index: Int,
    c_index: Int,
    depth: Int,
) -> None:
    let offset_a = a_index * a_over_shape[T](depth, A, B) * a_over_strides[T](
        depth, A, B
    )
    let offset_b = b_index * b_over_shape[T](depth, A, B) * b_over_strides[T](
        depth, A, B
    )
    let c_rest = C.shape()[depth] * stride[T](C,depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_add[nelts: Int](i: Int):
        C.data.simd_store[nelts](
            offset_c + i,
            A.simd_load[nelts](offset_a + i)
            + B.simd_load[nelts](offset_b + i),
        )

    vectorize[nelts, v_add](c_rest)


@always_inline
fn kernel_sub[
    T: DType, nelts: Int, cores: Int
](
    inout C: Tensor[T],
    A: Tensor[T],
    B: Tensor[T],
    a_index: Int,
    b_index: Int,
    c_index: Int,
    depth: Int,
) -> None:
    let offset_a = a_index * a_over_shape[T](depth, A, B) * a_over_strides[T](
        depth, A, B
    )
    let offset_b = b_index * b_over_shape[T](depth, A, B) * b_over_strides[T](
        depth, A, B
    )
    let c_rest = C.shape()[depth] * stride[T](C,depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_sub[nelts: Int](i: Int):
        C.data.simd_store[nelts](
            offset_c + i,
            A.simd_load[nelts](offset_a + i)
            - B.simd_load[nelts](offset_b + i),
        )

    vectorize[nelts, v_sub](c_rest)


@always_inline
fn kernel_div[
    T: DType, nelts: Int, cores: Int
](
    inout C: Tensor[T],
    A: Tensor[T],
    B: Tensor[T],
    a_index: Int,
    b_index: Int,
    c_index: Int,
    depth: Int,
) -> None:
    let offset_a = a_index * a_over_shape[T](depth, A, B) * a_over_strides[T](
        depth, A, B
    )
    let offset_b = b_index * b_over_shape[T](depth, A, B) * b_over_strides[T](
        depth, A, B
    )
    let c_rest = C.shape()[depth] * stride[T](C,depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_div[nelts: Int](i: Int):
        C.data.simd_store[nelts](
            offset_c + i,
            A.simd_load[nelts](offset_a + i)
            / B.simd_load[nelts](offset_b + i),
        )

    vectorize[nelts, v_div](c_rest)


@always_inline
fn base_case_matmul[T: DType](depth: Int, A: Tensor[T], B: Tensor[T]) -> Bool:
    return depth == max(A.rank(), B.rank()) - Int(2)


@always_inline
fn base_case_mul[T: DType](depth: Int, A: Tensor[T], B: Tensor[T]) -> Bool:
    return a_over_strides(depth, A, B) * a_over_shape[T](
        depth, A, B
    ) == b_over_strides[T](depth, A, B) * b_over_shape[T](depth, A, B)


@always_inline
fn base_case_add[T: DType](depth: Int, A: Tensor[T], B: Tensor[T]) -> Bool:
    return a_over_strides[T](depth, A, B) * a_over_shape[T](
        depth, A, B
    ) == b_over_strides[T](depth, A, B) * b_over_shape[T](depth, A, B)


@always_inline
fn base_case_sub[T: DType](depth: Int, A: Tensor[T], B: Tensor[T]) -> Bool:
    return a_over_strides[T](depth, A, B) * a_over_shape[T](
        depth, A, B
    ) == b_over_strides[T](depth, A, B) * b_over_shape[T](depth, A, B)


@always_inline
fn base_case_div[T: DType](depth: Int, A: Tensor[T], B: Tensor[T]) -> Bool:
    return a_over_strides[T](depth, A, B) * a_over_shape[T](
        depth, A, B
    ) == b_over_strides[T](depth, A, B) * b_over_shape[T](depth, A, B)


@always_inline
fn base_case_pow[T: DType](depth: Int, A: Tensor[T], B: Tensor[T]) -> Bool:
    return a_over_strides[T](depth, A, B) * a_over_shape[T](
        depth, A, B
    ) == b_over_strides[T](depth, A, B) * b_over_shape[T](depth, A, B)


@always_inline
fn kernel_pow[
    T: DType, nelts: Int, cores: Int
](
    inout C: Tensor[T],
    A: Tensor[T],
    B: Tensor[T],
    a_index: Int,
    b_index: Int,
    c_index: Int,
    depth: Int,
) -> None:
    let offset_a = a_index * a_over_shape[T](depth, A, B) * a_over_strides[T](
        depth, A, B
    )
    let offset_b = b_index * b_over_shape[T](depth, A, B) * b_over_strides[T](
        depth, A, B
    )
    let c_rest = C.shape()[depth] * stride[T](C,depth)
    let offset_c = c_index * c_rest

    @parameter
    fn v_pow[nelts: Int](i: Int):
        C.data.simd_store[nelts](
            offset_c + i,
            pow(
                A.simd_load[nelts](offset_a + i),
                B.simd_load[nelts](offset_b + i),
            ),
        )

    vectorize[nelts, v_pow](c_rest)


@always_inline
fn tensor_pow[
    T: DType, nelts: Int, cores: Int, parallelized: Bool = True
](inout C: Tensor[T], A: Tensor[T], B: Tensor[T]):
    recursive_broadcast[T, nelts, cores, kernel_pow, base_case_pow, parallelized](
        C, A, B
    )


fn tensor_pow_all[T: DType, nelts: Int](inout B: Tensor[T], A: Tensor[T]):
    let e = B.extra_parameters.load(0)

    @parameter
    fn v_pow_all[nelts: Int](i: Int):
        let temp = pow(A.simd_load[nelts](i), e)
        B.data.simd_store[nelts](i, temp)

    vectorize[nelts, v_pow_all](A.num_elements())


@always_inline
fn tensor_exp2[T: DType, nelts: Int](inout B: Tensor[T], A: Tensor[T]):
    @parameter
    fn v_exp2[nelts: Int](i: Int):
        let temp = exp2(A.simd_load[nelts](i))
        B.data.simd_store[nelts](i, temp)

    vectorize[nelts, v_exp2](A.num_elements())


@always_inline
fn tensor_exp[T: DType, nelts: Int](inout B: Tensor[T], A: Tensor[T]):
    @parameter
    fn v_exp[nelts: Int](i: Int):
        let temp = exp(A.simd_load[nelts](i))
        B.data.simd_store[nelts](i, temp)

    vectorize[nelts, v_exp](A.num_elements())


@always_inline
fn tensor_log2[T: DType, nelts: Int](inout B: Tensor[T], A: Tensor[T]):
    @parameter
    fn v_log2[nelts: Int](i: Int):
        let temp = log2(A.simd_load[nelts](i))
        B.data.simd_store[nelts](i, temp)

    vectorize[nelts, v_log2](A.num_elements())


@always_inline
fn tensor_log[T: DType, nelts: Int](inout B: Tensor[T], A: Tensor[T]):
    @parameter
    fn v_log[nelts: Int](i: Int):
        let temp = log(A.simd_load[nelts](i))
        B.data.simd_store[nelts](i, temp)

    vectorize[nelts, v_log](A.num_elements())


@always_inline
fn tensor_sin[T: DType, nelts: Int](inout B: Tensor[T], A: Tensor[T]):
    @parameter
    fn v_sin[nelts: Int](i: Int):
        let temp = sin(A.simd_load[nelts](i))
        B.data.simd_store[nelts](i, temp)

    vectorize[nelts, v_sin](A.num_elements())


@always_inline
fn tensor_cos[T: DType, nelts: Int](inout B: Tensor[T], A: Tensor[T]):
    @parameter
    fn v_cos[nelts: Int](i: Int):
        let temp = cos(A.simd_load[nelts](i))
        B.data.simd_store[nelts](i, temp)

    vectorize[nelts, v_cos](A.num_elements())


@always_inline
fn tensor_tan[T: DType, nelts: Int](inout B: Tensor[T], A: Tensor[T]):
    @parameter
    fn v_tan[nelts: Int](i: Int):
        let temp = tan(A.simd_load[nelts](i))
        B.data.simd_store[nelts](i, temp)

    vectorize[nelts, v_tan](A.num_elements())


@always_inline
fn tensor_asin[T: DType, nelts: Int](inout B: Tensor[T], A: Tensor[T]):
    @parameter
    fn v_asin[nelts: Int](i: Int):
        let temp = asin(A.simd_load[nelts](i))
        B.data.simd_store[nelts](i, temp)

    vectorize[nelts, v_asin](A.num_elements())


@always_inline
fn tensor_acos[T: DType, nelts: Int](inout B: Tensor[T], A: Tensor[T]):
    @parameter
    fn v_acos[nelts: Int](i: Int):
        let temp = acos(A.simd_load[nelts](i))
        B.data.simd_store[nelts](i, temp)

    vectorize[nelts, v_acos](A.num_elements())


@always_inline
fn tensor_atan[T: DType, nelts: Int](inout B: Tensor[T], A: Tensor[T]):
    @parameter
    fn v_atan[nelts: Int](i: Int):
        let temp = atan(A.simd_load[nelts](i))
        B.data.simd_store[nelts](i, temp)

    vectorize[nelts, v_atan](A.num_elements())


@always_inline
fn tensor_sinh[T: DType, nelts: Int](inout B: Tensor[T], A: Tensor[T]):
    @parameter
    fn v_sinh[nelts: Int](i: Int):
        let temp = sinh(A.simd_load[nelts](i))
        B.data.simd_store[nelts](i, temp)

    vectorize[nelts, v_sinh](A.num_elements())


@always_inline
fn tensor_cosh[T: DType, nelts: Int](inout B: Tensor[T], A: Tensor[T]):
    @parameter
    fn v_cosh[nelts: Int](i: Int):
        let temp = cosh(A.simd_load[nelts](i))
        B.data.simd_store[nelts](i, temp)

    vectorize[nelts, v_cosh](A.num_elements())


@always_inline
fn tensor_tanh[T: DType, nelts: Int](inout B: Tensor[T], A: Tensor[T]):
    @parameter
    fn v_tanh[nelts: Int](i: Int):
        let temp = tanh(A.simd_load[nelts](i))
        B.data.simd_store[nelts](i, temp)

    vectorize[nelts, v_tanh](A.num_elements())


@always_inline
fn tensor_relu[T: DType, nelts: Int](inout B: Tensor[T], A: Tensor[T]):
    @parameter
    fn v_relu[nelts: Int](i: Int):
        let zeros = SIMD[T, nelts]()
        B.data.simd_store[nelts](
            i,
            (A.simd_load[nelts](i) > zeros).cast[T]()
            * A.simd_load[nelts](i),
        )

    vectorize[nelts, v_relu](B.num_elements())


@always_inline
fn tensor_copy[T: DType, nelts: Int](inout B: Tensor[T], A: Tensor[T]):
    memcpy(B.data, A.data, A.num_elements())


@always_inline
fn tensor_div[
    T: DType, nelts: Int, cores: Int, parallelized: Bool = True
](inout C: Tensor[T], A: Tensor[T], B: Tensor[T]):
    recursive_broadcast[T, nelts, cores, kernel_div, base_case_div, parallelized](
        C, A, B
    )


@always_inline
fn tensor_sqrt[T: DType, nelts: Int](inout B: Tensor[T], A: Tensor[T]):
    @parameter
    fn v_sqrt[nelts: Int](i: Int):
        let temp = sqrt(A.simd_load[nelts](i))
        B.data.simd_store[nelts](i, temp)

    vectorize[nelts, v_sqrt](A.num_elements())


@always_inline
fn tensor_abs[T: DType, nelts: Int](inout B: Tensor[T], A: Tensor[T]):
    @parameter
    fn v_abs[nelts: Int](i: Int):
        let temp = abs(A.simd_load[nelts](i))
        B.data.simd_store[nelts](i, temp)

    vectorize[nelts, v_abs](A.num_elements())


@always_inline
fn tensor_mul[
    T: DType, nelts: Int, cores: Int, parallelized: Bool = True
](inout C: Tensor[T], A: Tensor[T], B: Tensor[T]):
    recursive_broadcast[T, nelts, cores, kernel_mul, base_case_mul, parallelized](
        C, A, B
    )


@always_inline
fn tensor_add[
    T: DType, nelts: Int, cores: Int, parallelized: Bool = True
](inout C: Tensor[T], A: Tensor[T], B: Tensor[T]):
    recursive_broadcast[T, nelts, cores, kernel_add, base_case_add, parallelized](
        C, A, B
    )


@always_inline
fn tensor_sub[
    T: DType, nelts: Int, cores: Int, parallelized: Bool = True
](inout C: Tensor[T], A: Tensor[T], B: Tensor[T]):
    recursive_broadcast[T, nelts, cores, kernel_sub, base_case_sub, parallelized](
        C, A, B
    )


# More Automated Functions


@always_inline
fn tensor_pow[
    T: DType, nelts: Int, cores: Int
](A: Tensor[T], B: Tensor[T]) -> Tensor[T]:
    var C: Tensor[T] = Tensor[T](B.tensor_shape)
    
    tensor_pow[T, nelts, cores](C, A, B)
    return C


fn tensor_pow_all[T: DType, nelts: Int](A: Tensor[T]) -> Tensor[T]:
    var B: Tensor[T] = Tensor[T](A.tensor_shape)
    
    tensor_pow_all[T, nelts](B, A)
    return B


@always_inline
fn tensor_exp2[T: DType, nelts: Int](A: Tensor[T]) -> Tensor[T]:
    var B: Tensor[T] = Tensor[T](A.tensor_shape)
    
    tensor_exp2[T, nelts](B, A)
    return B


@always_inline
fn tensor_exp[T: DType, nelts: Int](A: Tensor[T]) -> Tensor[T]:
    var B: Tensor[T] = Tensor[T](A.tensor_shape)
    
    tensor_exp[T, nelts](B, A)
    return B


@always_inline
fn tensor_log2[T: DType, nelts: Int](A: Tensor[T]) -> Tensor[T]:
    var B: Tensor[T] = Tensor[T](A.tensor_shape)
    
    tensor_log2[T, nelts](B, A)
    return B


@always_inline
fn tensor_log[T: DType, nelts: Int](A: Tensor[T]) -> Tensor[T]:
    var B: Tensor[T] = Tensor[T](A.tensor_shape)
    
    tensor_log[T, nelts](B, A)
    return B


@always_inline
fn tensor_sin[T: DType, nelts: Int](A: Tensor[T]) -> Tensor[T]:
    var B: Tensor[T] = Tensor[T](A.tensor_shape)
    
    tensor_sin[T, nelts](B, A)
    return B


@always_inline
fn tensor_cos[T: DType, nelts: Int](A: Tensor[T]) -> Tensor[T]:
    var B: Tensor[T] = Tensor[T](A.tensor_shape)
    
    tensor_cos[T, nelts](B, A)
    return B


@always_inline
fn tensor_tan[T: DType, nelts: Int](A: Tensor[T]) -> Tensor[T]:
    var B: Tensor[T] = Tensor[T](A.tensor_shape)
    
    tensor_tan[T, nelts](B, A)
    return B


@always_inline
fn tensor_asin[T: DType, nelts: Int](A: Tensor[T]) -> Tensor[T]:
    var B: Tensor[T] = Tensor[T](A.tensor_shape)
    
    tensor_asin[T, nelts](B, A)
    return B


@always_inline
fn tensor_acos[T: DType, nelts: Int](A: Tensor[T]) -> Tensor[T]:
    var B: Tensor[T] = Tensor[T](A.tensor_shape)
    
    tensor_acos[T, nelts](B, A)
    return B


@always_inline
fn tensor_atan[T: DType, nelts: Int](A: Tensor[T]) -> Tensor[T]:
    var B: Tensor[T] = Tensor[T](A.tensor_shape)
    
    tensor_atan[T, nelts](B, A)
    return B


@always_inline
fn tensor_sinh[T: DType, nelts: Int](A: Tensor[T]) -> Tensor[T]:
    var B: Tensor[T] = Tensor[T](A.tensor_shape)
    
    tensor_sinh[T, nelts](B, A)
    return B


@always_inline
fn tensor_cosh[T: DType, nelts: Int](A: Tensor[T]) -> Tensor[T]:
    var B: Tensor[T] = Tensor[T](A.tensor_shape)
    
    tensor_cosh[T, nelts](B, A)
    return B


@always_inline
fn tensor_tanh[T: DType, nelts: Int](A: Tensor[T]) -> Tensor[T]:
    var B: Tensor[T] = Tensor[T](A.tensor_shape)
    
    tensor_tanh[T, nelts](B, A)
    return B


@always_inline
fn tensor_relu[T: DType, nelts: Int](A: Tensor[T]) -> Tensor[T]:
    var B: Tensor[T] = Tensor[T](A.tensor_shape)
    
    tensor_relu[T, nelts](B, A)
    return B


@always_inline
fn tensor_sqrt[T: DType, nelts: Int](A: Tensor[T]) -> Tensor[T]:
    var B: Tensor[T] = Tensor[T](A.tensor_shape)
    
    tensor_sqrt[T, nelts](B, A)
    return B


@always_inline
fn tensor_abs[T: DType, nelts: Int](A: Tensor[T]) -> Tensor[T]:
    var B: Tensor[T] = Tensor[T](A.tensor_shape)
    
    tensor_abs[T, nelts](B, A)
    return B


@always_inline
fn tensor_mul[
    T: DType, nelts: Int, cores: Int
](A: Tensor[T], B: Tensor[T]) -> Tensor[T]:
    var C: Tensor[T] = Tensor[T](B.tensor_shape)
    
    tensor_mul[T, nelts, cores](C, A, B)
    return C


@always_inline
fn tensor_add[
    T: DType, nelts: Int, cores: Int
](A: Tensor[T], B: Tensor[T]) -> Tensor[T]:
    var C: Tensor[T] = Tensor[T](B.tensor_shape)
    
    tensor_add[T, nelts, cores](C, A, B)
    return C


@always_inline
fn tensor_sub[
    T: DType, nelts: Int, cores: Int
](A: Tensor[T], B: Tensor[T]) -> Tensor[T]:
    var C: Tensor[T] = Tensor[T](B.tensor_shape)
    
    tensor_sub[T, nelts, cores](C, A, B)
    return C


@always_inline
fn tensor_div[
    T: DType, nelts: Int, cores: Int
](A: Tensor[T], B: Tensor[T]) -> Tensor[T]:
    var C: Tensor[T] = Tensor[T](B.tensor_shape)
    
    tensor_div[T, nelts, cores](C, A, B)
    return C