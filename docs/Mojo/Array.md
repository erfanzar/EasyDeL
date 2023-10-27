## Array Utils 😇

### convert_numpy_to_easydel_array

```mojo
fn convert_numpy_to_easydel_array[
    DT: DType
](np_array: PythonObject, array_spec: ArrayShape) raises -> Array[DT]:
```

Converts a Numpy Array To Mojo Array

### matmul_shape 🌪

`fn matmul_shape[DT: DType](A: Array[DT], B: Array[DT]) -> ArrayShape:`

give you the shape of new Array for C in Matmul

### matmul 🪓

`fn matmul[nelts: Int, DT: DType](inout C: Array[DT], A: Array[DT], B: Array[DT]) -> None:->`

Apply the Matrix Mul for two Arrays

### Matmul Example with EasyDel

a simple matmul with two arrays

```mojo
from EasyDel import Array, matmul, matmul_shape


fn run[nelts: Int, DT: DType]() raises:
    # You can change this But Remember Cols of A must match Rows of B
    let A: Array[DT] = Array[DT](True, 1, 3, 1024, 512)  # True Passed to init Array
    let B: Array[DT] = Array[DT](True, 1, 3, 512, 1024)
    var C: Array[DT] = Array[DT](A, B)
    matmul[nelts, DT](C, A, B)  # You Get the same result As Numpy


fn main() raises:
    run[Array[DType.float32].nelts, DType.float32]()
```

in this code we convert 2 numpy array into easydel array and apply matmul on them then do the same in easydel and check
result is same

```mojo
from python import Python as Py
import EasyDel as ed


fn run[DT: DType]() raises:
    let np_shape_1 = (2, 20)
    let shape_1 = ed.ArrayShape(2, 20)
    let np_shape_2 = (20, 18)
    let shape_2 = ed.ArrayShape(20, 18)

    let np = Py.import_module("numpy")

    let A1 = np.random.randint(0, 20, np_shape_1)
    let A2 = np.random.randint(0, 20, np_shape_2)
    let E1: ed.Array[DT] = ed.convert_numpy_to_easydel_array[DT](A1, shape_1)
    let E2: ed.Array[DT] = ed.convert_numpy_to_easydel_array[DT](A2, shape_2)

    let matmul_np = np.matmul(A1, A2)
    var C: ed.Array[DT] = ed.Array[DT](E1, E2) # Prepare Result Array for Matmul
    C.fill(0.0) # Fill it with zeros
    ed.matmul[ed.Array[DT].nelts, DT](C, E1, E2)
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

- Description: Init Array From ArrayShape(Alloc Zero).

`fn __init__(inout self: Self, A: Self, B: Self) -> None:`

- Description: Init Array From Two other Arrays A and B For Matmul(Alloc One).

`fn __init__(inout self: Self, vl: VariadicList[Int]):`

- Description: Init Array from VariadicList[Int](Alloc Zero).

`fn __init__(inout self: Self, init: Bool, *dim: Int) -> None:`

- Description: Init Array from Int Args(Depends on Passed Bool).

`fn __init__(inout self: Self, *dim: Int):`

- Description: Init Array from Int Args(Alloc Zero).

`fn __init__(inout self: Self, value: DynamicVector[FloatLiteral], shape: ArrayShape) -> None:`

- Description: Init Array from ArrayShape and load data from DynamicVector[FloatLiteral](Alloc One).

`fn __init__(inout self: Self, value: VariadicList[FloatLiteral], shape: ArrayShape) -> None:`

- Description: Init Array from ArrayShape and load data from VariadicList[FloatLiteral](Alloc One).

`fn __init__(inout self: Self, pointer: DTypePointer[DT], *dim: Int) -> None:`

- Description: Init Array from IntArgs and load data from DTypePointer[DT](Alloc One).
  
`fn __init__(inout self: Self, pointer: DTypePointer[DT], *dim: Int) -> None:`

- Description: Init Array from given data from DTypePointer[DT](Alloc Zero).

### Alloc

`fn alloc(inout self: Self) -> None:`

- Description: Allocate or Init The Array.

`fn alloc(inout self: Self, fill:SIMD[DT, 1]) -> None:`

- Allocate or Init The Array and fill that with given fill number.

### Random

`fn random(inout self: Self) -> None:`

- Description: Randomize The Data if the Array is Allocated.

### Reshape and View

View:

* INOUT  `fn view(inout self, *dims: Int):`
* View Change Shape totaly and don't care if the new shape fits or doesn't.

Reshape:

* INOUT  `fn reshape(inout self, *dims: Int):`
* Reshape Change Shape totaly and check if the new shape fits or doesn't.

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
](self, index: InlinedFixedVector[off, Int]) -> SIMD[DT, nelts]:
```

`fn load[nelts: Int, off: Int](self, index: StaticIntTuple[off]) -> SIMD[DT, nelts]:`

`fn load[nelts: Int](self, index: Int) -> SIMD[DT, nelts]:`

### Store Functions

```
fn store[
    nelts: Int, off: Int
](self, index: InlinedFixedVector[off, Int], val: SIMD[DT, nelts]) -> None:
```

```
fn store[
    nelts: Int, off: Int
](self, index: StaticIntTuple[off], val: SIMD[DT, nelts]) -> None:
```

`fn store[nelts: Int](self, index: Int, val: SIMD[DT, nelts]) -> None:`

### `__getitem__` Functions

`fn __getitem__[off: Int](self, index: InlinedFixedVector[off, Int]) -> SIMD[DT, 1]:`

`fn __getitem__[off: Int](self, index: StaticIntTuple[off]) -> SIMD[DT, 1]:`

`fn __getitem__(self, index: Int) -> SIMD[DT, 1]:`

`fn __getitem__(self, d1: Int, d2: Int, val:SIMD[DT, 1]) raises->None:`

`fn __getitem__(self, d1: Int, d2: Int, d3: Int, val:SIMD[DT, 1]) raises->None:`

### `__setitem__` Functions

`fn __setitem__(self, index: Int, val: SIMD[DT, 1]) -> None:`

`fn __setitem__[off: Int](self, index: InlinedFixedVector[off, Int], val: SIMD[DT, 1]):`

`fn __setitem__[off: Int](self, index: StaticIntTuple[off], val: SIMD[DT, 1]):`

`fn __setitem__(self, d1: Int, d2: Int, val:SIMD[DT, 1]) raises->None:`

`fn __setitem__(self, d1: Int, d2: Int, d3: Int, val:SIMD[DT, 1]) raises->None:`

### Math Functions

```mojo

fn argmax(self: Self, axis: Int = -1) -> Int:

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

fn fill(inout self: Self, val: SIMD[DT, 1]) -> None:
```
