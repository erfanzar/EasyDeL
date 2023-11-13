from ..in_out import File, BufReader


struct FileBuffer:
    var data: DTypePointer[DType.uint8]
    var offset: Int
    var size: Int

    fn __init__(inout self):
        self.data = DTypePointer[DType.uint8].alloc(0)
        self.offset = 0
        self.size = 0

    fn move_offset(inout self: Self, size: Int) raises:
        let dmf: Int = self.offset + size
        if dmf > self.size:
            raise Error(
                "Out of bounderies you can not move offset [Out of Range Data!]"
            )
        else:
            self.offset += size

    fn read_value_float32(
        inout self: Self, size: Int
    ) raises -> DTypePointer[DType.float32]:
        let res = self.data.offset(self.get_offset()).bitcast[DType.float32]()
        self.move_offset(4 * size)
        return res

    fn read_value_int(inout self: Self) raises -> Int:
        let res = self.data.offset(self.get_offset()).bitcast[DType.int32]().load(
            0
        ).to_int()
        self.move_offset(4)
        return res

    fn read_value_float16(
        inout self: Self, size: Int
    ) raises -> DTypePointer[DType.float16]:
        let res = self.data.offset(self.get_offset()).bitcast[DType.float16]()

        self.move_offset(sizeof[DType.float16]() * size)
        return res

    fn read_value_bfloat16(
        inout self: Self, size: Int
    ) raises -> DTypePointer[DType.bfloat16]:
        let res = self.data.offset(self.get_offset()).bitcast[DType.bfloat16]()

        self.move_offset(sizeof[DType.bfloat16]() * size)
        return res

    fn read_value_uint8(
        inout self: Self, size: Int
    ) raises -> DTypePointer[DType.uint8]:
        let res = self.data.offset(self.get_offset()).bitcast[DType.uint8]()

        self.move_offset(sizeof[DType.uint8]() * size)
        return res

    fn read_value_uint16(
        inout self: Self, size: Int
    ) raises -> DTypePointer[DType.uint16]:
        let res = self.data.offset(self.get_offset()).bitcast[DType.uint16]()

        self.move_offset(sizeof[DType.uint16]() * size)
        return res

    fn read_value_uint32(
        inout self: Self, size: Int
    ) raises -> DTypePointer[DType.uint32]:
        let res = self.data.offset(self.get_offset()).bitcast[DType.uint32]()

        self.move_offset(sizeof[DType.uint32]() * size)
        return res

    fn read_numerical_value_dynamic[
        T: DType
    ](inout self: Self, size: Int) raises -> DTypePointer[T]:
        let res = self.data.offset(self.get_offset()).bitcast[T]()

        self.move_offset(sizeof[T]() * size)
        return res

    fn read_numerical_value_dynamic[
        T: DType
    ](inout self: Self, size: Int, fs: Int) raises -> DTypePointer[T]:
        let res = self.data.offset(self.get_offset()).bitcast[T]()

        self.move_offset(fs * size)
        return res

    fn get_offset(self: Self) -> Int:
        return self.offset if self.offset < self.size else self.size


fn read_numerical_value[T: DType](inout buffer: FileBuffer) raises -> SIMD[T, 1]:
    let res: SIMD[T, 1] = buffer.data.offset(buffer.offset).bitcast[T]().load(0)
    buffer.move_offset(sizeof[T]())
    return res


fn read_string_value(
    inout buffer: FileBuffer, string_length: Int
) raises -> Pointer[UInt8]:
    let str = Pointer[UInt8].alloc(string_length + 1)

    for i in range(string_length):
        str.store(i, buffer.data.offset(buffer.offset).load(0))
        buffer.move_offset(1)

    str.store(string_length, 0)
    return str


fn read_file(file_name: String, inout buf: FileBuffer) raises:
    var fd = open(file_name, "r")
    let data = fd.read()
    fd.close()
    let cp_size = data._buffer.size
    let cp_buf: DTypePointer[DType.uint8] = DTypePointer[DType.uint8].alloc(cp_size)
    let data_ptr = data._as_ptr().bitcast[DType.uint8]()
    for i in range(cp_size):
        cp_buf.store(i, data_ptr.load(i))
    _ = data
    buf.data = cp_buf
    buf.size = cp_size
    buf.offset = 0
    return None
