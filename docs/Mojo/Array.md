## Array Utils 😇

### convert_numpy_to_easydel_array

```mojo
fn convert_numpy_to_easydel_array[
    T: DType
](np_array: PythonObject, array_spec: ArrayShape) raises -> Array[T]:
```

Converts a Numpy Array To Mojo Array

### matmul_shape 🌪

`fn matmul_shape[T: DType](A: Array[T], B: Array[T]) -> ArrayShape:`

give you the shape of new Array for C in Matmul

### matmul 🪓

`fn matmul[nelts: Int, T: DType](inout C: Array[T], A: Array[T], B: Array[T]) -> None:->`

Apply the Matrix Mul for two Arrays

### Matmul Example with EasyDel

a simple matmul with two arrays

```mojo
import EasyDel as ed

let E1: ed.Array[T] = ed.Array[T](1,2,18)
let E2: ed.Array[T] = ed.Array[T](1,18,64)
let C_Shape: ed.ArrayShape = ed.matmul_shape[T](E1, E2)
var C: ed.Array[T] = ed.Array[T](C_Shape)
C.fill(0.0) # Fill it with zeros
ed.matmul[ed.Array[T].nelts, T](C, E1, E2) # parallelized and accurate
```

in this code we convert 2 numpy array into easydel array and apply matmul on them then do the same in easydel and check
result is same

```mojo
from python import Python as Py
import EasyDel as ed


fn run[T: DType]() raises:
    let np_shape_1 = (2, 20)
    let shape_1 = ed.ArrayShape(2, 20)
    let np_shape_2 = (20, 18)
    let shape_2 = ed.ArrayShape(20, 18)

    let np = Py.import_module("numpy")

    let A1 = np.random.randint(0, 20, np_shape_1)
    let A2 = np.random.randint(0, 20, np_shape_2)
    let E1: ed.Array[T] = ed.convert_numpy_to_easydel_array[T](A1, shape_1)
    let E2: ed.Array[T] = ed.convert_numpy_to_easydel_array[T](A2, shape_2)

    let matmul_np = np.matmul(A1, A2)

    let C_Shape: ed.ArrayShape = ed.matmul_shape[T](E1, E2)
    var C: ed.Array[T] = ed.Array[T](C_Shape) # Prepare Result Array for Matmul
    C.fill(0.0) # Fill it with zeros
    ed.matmul[ed.Array[T].nelts, T](C, E1, E2)
    print(matmul_np)
    C.print_array()


fn main() raises:
    run[DType.float32]()
```

You will get same result

## Mojo ArrayShape

ArrayShape is builder of Arrays In EasyDel Mojo

### Init Methods

`fn __init__(inout self, shape: VariadicList[Int]) -> None:`

`fn __init__(inout self, shape: DynamicVector[Int]) -> None:`

`fn __init__[off: Int](inout self, shape: InlinedFixedVector[off, Int]) -> None:`

`fn __init__(inout self, *elms: Int) -> None:`

### Use Functions

### Shape

`fn shape(self: Self) -> Pointer[Int]:`

- Returns Pointer That have the Shape Of Array

### Dim

`fn dim(self, index: Int) -> Int:`

- Returns shape at given index (supports negative indexing)

### Number Of elements

`fn num_elements(self: Self) -> Int:`

- Return Size of Array or Number Of elements in Array

### Rank

`fn rank(self: Self) -> Int:`

- Returns Length of array shape

### Shape Str

`fn shape_str(self: Self):`
print shape of array

## Mojo Array

Takes DType as dynamic Input like `Array[DType.float32]`

### Init Methods

`fn __init__(inout self: Self, array_shape: ArrayShape):`

`fn __init__(inout self: Self, vl: VariadicList[Int]):`

`fn __init__(inout self: Self, *dim: Int):`

`fn __init__(inout self: Self, value: DynamicVector[FloatLiteral], shape: ArrayShape) -> None:`

`fn __init__(inout self: Self, value: VariadicList[FloatLiteral], shape: ArrayShape) -> None:`

### Set Data from buffer

`fn set_data_from_buffer(inout self: Self, pointer: DTypePointer[T]) -> None:`

sets data from the given buffer

```
fn set_data_from_buffer(
    inout self: Self, pointer: DTypePointer[T], shape: VariadicList[Int]
) -> None:
```

sets data from the given buffer and change shape

### Dim

`fn dim(self, index: Int) -> Int:`

- Returns shape at given index (supports negative indexing)

### Number Of elements

`fn num_elements(self: Self) -> Int:`

- Return Size of Array or Number Of elements in Array

### Rank

`fn rank(self: Self) -> Int:`

- Returns Length of array shape

### Load Functions

```
fn load[
    nelts: Int, off: Int
](self, index: InlinedFixedVector[off, Int]) -> SIMD[T, nelts]:
```

`fn load[nelts: Int, off: Int](self, index: StaticIntTuple[off]) -> SIMD[T, nelts]:`

`fn load[nelts: Int](self, index: Int) -> SIMD[T, nelts]:`

### Store Functions

``` 
fn store[
    nelts: Int, off: Int
](self, index: InlinedFixedVector[off, Int], val: SIMD[T, nelts]) -> None:
```

```
fn store[
    nelts: Int, off: Int
](self, index: StaticIntTuple[off], val: SIMD[T, nelts]) -> None:
```

`fn store[nelts: Int](self, index: Int, val: SIMD[T, nelts]) -> None:`

### `__getitem__` Functions

`fn __getitem__[off: Int](self, index: InlinedFixedVector[off, Int]) -> SIMD[T, 1]:`

`fn __getitem__[off: Int](self, index: StaticIntTuple[off]) -> SIMD[T, 1]:`

`fn __getitem__(self, index: Int) -> SIMD[T, 1]:`

### `__setitem__` Functions

`fn __setitem__(self, index: Int, val: SIMD[T, 1]) -> None:`

`fn __setitem__[off: Int](self, index: InlinedFixedVector[off, Int], val: SIMD[T, 1]):`

`fn __setitem__[off: Int](self, index: StaticIntTuple[off], val: SIMD[T, 1]):`

### Math Functions

```mojo
# cos

fn cos(inout self: Self) -> Self:

fn cos[nelts: Int](inout self: Self) -> Self:

fn cos(inout self: Self, rt: Runtime) -> Self:

# sin

fn sin(inout self: Self) -> Self:

fn sin[nelts: Int](inout self: Self) -> Self:

fn sin(inout self: Self, rt: Runtime) -> Self:

# log

fn log(inout self: Self) -> Self:

fn log[nelts: Int](inout self: Self) -> Self:

fn log(inout self: Self, rt: Runtime) -> Self:

# log2

fn log2(inout self: Self) -> Self:

fn log2[nelts: Int](inout self: Self) -> Self:

fn log2(inout self: Self, rt: Runtime) -> Self:

# tan

fn tan(inout self: Self) -> Self:

fn tan[nelts: Int](inout self: Self) -> Self:

fn tan(inout self: Self, rt: Runtime) -> Self:

# tanh

fn tanh(inout self: Self) -> Self:

fn tanh[nelts: Int](inout self: Self) -> Self:

fn tanh(inout self: Self, rt: Runtime) -> Self:

# sqrt

fn sqrt(inout self: Self) -> Self:

fn sqrt[nelts: Int](inout self: Self) -> Self:

fn sqrt(inout self: Self, rt: Runtime) -> Self:

# atan

fn atan(inout self: Self) -> Self:

fn atan[nelts: Int](inout self: Self) -> Self:

fn atan(inout self: Self, rt: Runtime) -> Self:

# exp

fn exp(inout self: Self) -> Self:

fn exp[nelts: Int](inout self: Self) -> Self:

fn exp(inout self: Self, rt: Runtime) -> Self:

# exp2

fn exp2(inout self: Self) -> Self:

fn exp2[nelts: Int](inout self: Self) -> Self:

fn exp2(inout self: Self, rt: Runtime) -> Self:

# log10

fn log10(inout self: Self) -> Self:

fn log10[nelts: Int](inout self: Self) -> Self:

fn log10(inout self: Self, rt: Runtime) -> Self:

# log1p

fn log1p(inout self: Self) -> Self:

fn log1p[nelts: Int](inout self: Self) -> Self:

fn log1p(inout self: Self, rt: Runtime) -> Self:

# logb

fn logb(inout self: Self) -> Self:

fn logb[nelts: Int](inout self: Self) -> Self:

fn logb(inout self: Self, rt: Runtime) -> Self:


# asin

fn asin(inout self: Self) -> Self:

fn asin[nelts: Int](inout self: Self) -> Self:

fn asin(inout self: Self, rt: Runtime) -> Self:


# acos

fn acos(inout self: Self) -> Self:

fn acos[nelts: Int](inout self: Self) -> Self:

fn acos(inout self: Self, rt: Runtime) -> Self:


# asinh

fn asinh(inout self: Self) -> Self:

fn asinh[nelts: Int](inout self: Self) -> Self:

fn asinh(inout self: Self, rt: Runtime) -> Self:


# acosh

fn acosh(inout self: Self) -> Self:

fn acosh[nelts: Int](inout self: Self) -> Self:

fn acosh(inout self: Self, rt: Runtime) -> Self:

# Fill the whole array with the val 

fn fill(inout self: Self, val: SIMD[T, 1]) -> None:
```