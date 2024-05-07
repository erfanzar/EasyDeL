from utils.list import Dim
from math import min
from math.limit import max_finite
from memory import memcpy
from memory.buffer import Buffer
from memory.unsafe import Pointer, DTypePointer
from sys.info import sizeof
from utils.index import Index
from utils.vector import DynamicVector
import testing


@value
@register_passable("trivial")
struct DIR:
    pass


@value
@register_passable("trivial")
struct dirent:
    var d_ino: UInt64
    var d_off: UInt64
    var d_reclen: UInt16
    var d_type: UInt8
    var d_name: Pointer[UInt8]


# ===--------------------------------------------------------------------===#
# closedir
# ===-----------------------------------------------------------------------===#


fn closedir(arg: Pointer[DIR]) -> Int32:
    return external_call["closedir", Int32, Pointer[DIR]](arg)


# ===--------------------------------------------------------------------===#
# opendir
# ===-----------------------------------------------------------------------===#


fn opendir(arg: Pointer[UInt8]) -> Pointer[DIR]:
    return external_call["opendir", Pointer[DIR], Pointer[UInt8]](arg)


# ===--------------------------------------------------------------------===#
# readdir
# ===-----------------------------------------------------------------------===#


fn readdir(arg: Pointer[DIR]) -> Pointer[dirent]:
    return external_call["readdir", Pointer[dirent], Pointer[DIR]](arg)


# ===--------------------------------------------------------------------===#
# fdopendir
# ===-----------------------------------------------------------------------===#


fn fdopendir(arg: Int32) -> DIR:
    return external_call["fdopendir", DIR](arg)


fn strnlen(pointer: Pointer[UInt8]) -> Int:
    return external_call["strnlen", Int, Pointer[UInt8]](pointer)


@value
@register_passable("trivial")
struct FILE:
    pass


# ===--------------------------------------------------------------------===#
# clearerr
# ===-----------------------------------------------------------------------===#


fn clearerr(arg: Pointer[FILE]) -> UInt8:
    return external_call["clearerr", UInt8, Pointer[FILE]](arg)


# ===--------------------------------------------------------------------===#
# fclose
# ===-----------------------------------------------------------------------===#


fn fclose(arg: Pointer[FILE]) -> Int32:
    return external_call["fclose", Int32](arg)


# ===--------------------------------------------------------------------===#
# feof
# ===-----------------------------------------------------------------------===#


fn feof(arg: Pointer[FILE]) -> Int32:
    return external_call["feof", Int32, Pointer[FILE]](arg)


# ===--------------------------------------------------------------------===#
# ferror
# ===-----------------------------------------------------------------------===#


fn ferror(arg: Pointer[FILE]) -> Int32:
    return external_call["ferror", Int32, Pointer[FILE]](arg)


# ===--------------------------------------------------------------------===#
# fflush
# ===-----------------------------------------------------------------------===#


fn fflush(arg: Pointer[FILE]) -> Int32:
    return external_call["fflush", Int32, Pointer[FILE]](arg)


# ===--------------------------------------------------------------------===#
# fgetc
# ===-----------------------------------------------------------------------===#


fn fgetc(arg: Pointer[FILE]) -> Int32:
    return external_call["fgetc", Int32, Pointer[FILE]](arg)


# ===--------------------------------------------------------------------===#
# fopen
# ===-----------------------------------------------------------------------===#


fn fopen(__filename: Pointer[UInt8], __mode: Pointer[UInt8]) -> Pointer[FILE]:
    return external_call["fopen", Pointer[FILE], Pointer[UInt8], Pointer[UInt8]](
        __filename, __mode
    )


# ===--------------------------------------------------------------------===#
# fread
# ===-----------------------------------------------------------------------===#


fn fread(
    __ptr: Pointer[UInt8], __size: UInt64, __nitems: UInt64, __stream: Pointer[FILE]
) -> UInt64:
    return external_call[
        "fread", UInt64, Pointer[UInt8], UInt64, UInt64, Pointer[FILE]
    ](__ptr, __size, __nitems, __stream)


alias BUF_SIZE = 4096


fn to_char_ptr(s: String) -> Pointer[UInt8]:
    """Only ASCII-based strings."""
    let ptr = Pointer[UInt8]().alloc(len(s) + 1)
    for i in range(len(s)):
        ptr.store(i, ord(s[i]))
    ptr.store(len(s), ord("\0"))
    return ptr


struct File:
    var handle: Pointer[FILE]
    var fname: Pointer[UInt8]
    var mode: Pointer[UInt8]

    fn __init__(inout self, filename: String):
        let fname = to_char_ptr(filename)
        let mode = to_char_ptr("r")
        let handle = fopen(fname, mode)

        self.fname = fname
        self.mode = mode
        self.handle = handle

    fn __bool__(self) -> Bool:
        return self.handle.__bool__()

    fn __del__(owned self) raises:
        if self.handle:
            pass
        if self.fname:
            self.fname.free()
        if self.mode:
            self.mode.free()

    fn __moveinit__(inout self, owned other: Self):
        self.fname = other.fname
        self.mode = other.mode
        self.handle = other.handle
        other.handle = Pointer[FILE]()
        other.fname = Pointer[UInt8]()
        other.mode = Pointer[UInt8]()

    fn do_nothing(self):
        pass

    fn read[D: Dim](self, buffer: Buffer[D, DType.uint8]) raises -> Int:
        return fread(
            buffer.data._as_scalar_pointer(), sizeof[UInt8](), BUF_SIZE, self.handle
        ).to_int()


struct DirEntry:
    var _pointer: Pointer[dirent]
    var name: String

    fn __init__(inout self, pointer: Pointer[dirent]):
        self.name = String()
        if pointer:
            print("hit")
            let name_ptr = pointer.bitcast[UInt8]().offset(
                sizeof[UInt64]() * 2 + sizeof[UInt16]() + sizeof[UInt8]()
            )
            let name_len = strnlen(name_ptr)
            for i in range(name_len):
                self.name += chr(name_ptr.load(i).to_int())
        self._pointer = pointer


@value
@register_passable("trivial")
struct DirIter:
    var handle: Pointer[DIR]
    var data: Pointer[dirent]

    fn __iter__(inout self: Self):
        self.data = readdir(self.handle)

    fn __next__(self: Self) raises -> Pointer[dirent]:
        return self.data

    fn __len__(self: Self) -> Int:
        if self.handle and self.data:
            return 1
        return 0


struct Dir:
    var handle: Pointer[DIR]
    var path: Pointer[UInt8]

    fn __init__(inout self, path: String):
        self.path = to_char_ptr(path)
        self.handle = opendir(self.path)

    fn __bool__(self) -> Bool:
        return self.handle.__bool__()

    fn __iter__(self) -> DirIter:
        return DirIter(self.handle, Pointer[dirent]())

    fn __del__(owned self) raises:
        let c = closedir(self.handle)
        if c != 0:
            raise Error("failed to close dir")
        self.path.free()

    fn do_nothing(self):
        pass


struct BufReader[BUF_SIZE: Int]:
    var unbuffered_reader: File
    var data: DTypePointer[DType.uint8]
    var end: Int
    var start: Int

    fn __init__(inout self, owned reader: File):
        self.unbuffered_reader = reader ^
        self.data = DTypePointer[DType.uint8]().alloc(BUF_SIZE)
        self.end = 0
        self.start = 0

    fn read[D: Dim](inout self, dest: Buffer[D, DType.uint8]) raises -> Int:
        var dest_index = 0
        let buf = Buffer[BUF_SIZE, DType.uint8](self.data)

        while dest_index < len(dest):
            let written = min(len(dest) - dest_index, self.end - self.start)
            memcpy(dest.data.offset(dest_index), self.data.offset(self.start), written)
            if written == 0:
                # buf empty, fill it
                let n = self.unbuffered_reader.read(buf)
                if n == 0:
                    # reading from the unbuffered stream returned nothing
                    # so we have nothing left to read.
                    return dest_index
                self.start = 0
                self.end = n
            self.start += written
            dest_index += written
        return len(dest)

    fn do_nothing(self):
        pass
