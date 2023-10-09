# EasyDel Mojo 🔥

EasyDel Mojo differs from EasyDel in Python in significant ways. In Python, you can leverage a vast array of packages to create a mid or high-level API in no time. However, when working with Mojo, it's a different story. Here, you have to build some of the features that other Python libraries provide, such as Jax for arrays and computations. But why not import numpy, Jax, and other similar packages to Mojo and use them?

There are several reasons why building packages in Mojo is more efficient than importing them from Python. Firstly, when you import packages from Python, you incur the overhead of translating and processing the Python code into Mojo code, which takes time. Secondly, the Python code may not be optimized for the Mojo runtime environment, leading to slower performance. Lastly, building packages directly in Mojo allows you to design and optimize them explicitly for the Mojo runtime environment, resulting in faster and more efficient code. With Mojo's built-in array capabilities that are 35000x faster than Python, it's time to take your coding to the next level.

## Docs

* _EasyDel Mojo🔥_ :
    * [README Tree🔥](https://erfanzar.github.io/EasyDeL/lib/mojo)

## Array API

Array API can be used just like Numpy Arrays For example

```mojo
import EasyDel as ed


fn run[T: DType, nelts: Int = 1]() raises:
    # Let Create Two 4D Array
    var A: ed.Array[T] = ed.Array[T](1, 1, 128, 80)
    var B: ed.Array[T] = ed.Array[T](1, 1, 80, 128)

    # What can we do?
    # These are the operations
    A = A.cos()
    print("Gere")
    B = B.sin()
    var copy_A = A

    copy_A = copy_A.sqrt()
    if copy_A != A:
        print(
            "Arrays which be copied or moved have same data but you aplied Sqrt on"
            " CopyA so they are not same anymore"
        )

    # Matmul

    # Method Number One (Use this to get true answers)
    var C_result_matmul_Array: ed.Array[T] = ed.Array[T](ed.matmul_shape[T](A, B))
    # Array by default is randomized so we fill array with zeros
    C_result_matmul_Array.fill(0.0)

    ed.matmul[T ,nelts](C_result_matmul_Array, A, B)
    # Method Number Two (Not recommended at all)
    # Due to Mojo language Bugs this feature apply the same code as above but you get wierd result
    var C = A @ B

    # Works just time np.array.shape[index]
    let last_dim_c: Int = C.dim(-1)
    let second_dim_c: Int = C.dim(2)

    # How to know the Length of shape
    let length: Int = C.rank()

    # How many numbers are there?
    let num_elements: Int = C.num_elements()

    # Want to print Shape?
    C.array_shape.shape_str()

    # Want to print Array itself??
    # C.print_array()  you will se bunch of numbers cause of this large array its not recommended :\
    print(C[0])
    C = C + B
    print(C[0])
    C = C * B
    print(C[0])
    C = C / B
    print(C[0])
    # C = C // 8 # not fully supported yet You know :)
  

fn main() raises:
    run[ed.Array[DType.float32].nelts, DType.float32]()

```

#### Math Supported Operation For Arrays

| Operation | Array[DT.F64]                                 | Array[DT.F32]                                 | Array[DT.F16]                                 | Array[DT.BF16]                                |
| --------- | --------------------------------------------- | --------------------------------------------- | --------------------------------------------- | --------------------------------------------- |
| `Sqrt`    | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) |
| `Sin`     | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) |
| `Cos`     | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) |
| `Tanh`    | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) |
| `Tan`     | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) |
| `Log`     | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) |
| `Log2`    | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) |
| `Atan`    | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) |
| `Exp`     | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) |
| `Exp2`    | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) |
| `Pow`     | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) |
| `Log10`   | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) |
| `Log1p`   | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) |
| `Logb`    | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) |
| `Asin`    | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) |
| `Acos`    | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) |
| `Acosh`   | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) | [✅](https://emojipedia.org/check-mark-button) |

#### Supported Operations Between Arrays

| Operation Sign | Func              | Supported Array TO Array                      | Supported Array TO SIMD |
| -------------- | ----------------- | --------------------------------------------- | ----------------------- |
| `+`            | `__add__()`       | [✅](https://emojipedia.org/check-mark-button) | ❌ Not Yet               |
| `-`            | `__sub__()`       | [✅](https://emojipedia.org/check-mark-button) | ❌ Not Yet               |
| `*`            | `__mul__()`       | [✅](https://emojipedia.org/check-mark-button) | ❌ Not Yet               |
| `/`            | `__truediv__()`   | [✅](https://emojipedia.org/check-mark-button) | ❌ Not Yet               |
| `//`           | `__floordiv__()`  | ❌ Not Yet                                     | ❌ Not Yet               |
| `@`            | `__matmul__()`    | [✅](https://emojipedia.org/check-mark-button) | ❌ Not Yet               |
| `%`            | `__mod__()`       | [✅](https://emojipedia.org/check-mark-button) | ❌ Not Yet               |
| `**`           | `__pow__()`       | [✅](https://emojipedia.org/check-mark-button) | ❌ Not Yet               |
| `+=`           | `__iadd__()`      | [✅](https://emojipedia.org/check-mark-button) | ❌ Not Yet               |
| `-=`           | `__isub__()`      | [✅](https://emojipedia.org/check-mark-button) | ❌ Not Yet               |
| `*=`           | `__imul__()`      | [✅](https://emojipedia.org/check-mark-button) | ❌ Not Yet               |
| `/=`           | `__itruediv__()`  | [✅](https://emojipedia.org/check-mark-button) | ❌ Not Yet               |
| `//=`          | `__ifloordiv__()` | ❌ Not Yet                                     | ❌ Not Yet               |
| `**=`          | `__ipow__()`      | [✅](https://emojipedia.org/check-mark-button) | ❌ Not Yet               |
| `==`           | `__eq__()`        | [✅](https://emojipedia.org/check-mark-button) | ❌ Not Yet               |
| `!=`           | `__ne__()`        | [✅](https://emojipedia.org/check-mark-button) | ❌ Not Yet               |
| `<`            | `__lt__()`        | [✅](https://emojipedia.org/check-mark-button) | ❌ Not Yet               |
| `>`            | `__gt__()`        | [✅](https://emojipedia.org/check-mark-button) | ❌ Not Yet               |
| `[]`           | `__getitem__()`   | [✅](https://emojipedia.org/check-mark-button) | ❌ Not Yet               |
| `[]`           | `__setitem__()`   | [✅](https://emojipedia.org/check-mark-button) | ❌ Not Yet               |

### Road Map

* [X] Build Tokenizer
* [X] Build Utils
* [X] Build StreamReader To Read Data from Buffer
* [X] Build Array API
* [ ] Add Examples to use Library
* [ ] Build Attention Library
* [ ] Build Linen API
* [ ] Gradient Support
