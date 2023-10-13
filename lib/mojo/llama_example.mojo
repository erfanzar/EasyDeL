from EasyDel import (
    Array,
    File,
    FileBuffer,
    read_file,
    BufReader,
    softmax,
    Tokenizer,
    byte_pr_tokenizer_encoder,
    print_pointer,
    sample_array,
    loop_sort,
)
from math import math
from python import Python, PythonObject
import time
import random
from random import rand
from EasyDel.models.llama import LlamaConfig, LlamaWeights, llama_forward, LlamaState
from sys import argv


fn run[
    T: DType, nelts: Int = 1
](
    weights_path: StringRef,
    tokenizer_path: StringRef,
    inout next_input_id: Int = -1,
    inout input_id: Int = 1,
    inout position: Int = 0,
    temperature: SIMD[T, 1] = 0.4,
    steps: Int = 512,
    inout start: Int = -1,
    prompt: String = "",
    verbose: Bool = True,
) raises:
    var weights_buffer: FileBuffer = FileBuffer()
    var tokenizer_bufferr: FileBuffer = FileBuffer()

    read_file(weights_buffer, weights_path)
    read_file(tokenizer_bufferr, tokenizer_path)

    var config = LlamaConfig(weights_buffer)
    let sz = weights_buffer.size // 1024 / 1024
    print(
        "\033[1;32mLoaded Model Weights Are ",
        sz if sz < 1000 else weights_buffer.size // 1024 / 1024 / 1024,
        " MB" if sz < 1000 else " GB",
        "\n",
    )

    let is_tied: Bool = True if config.vocab_size > 0 else False
    if not is_tied:
        config.vocab_size = -config.vocab_size
    if verbose:
        config.print_config()
    var tokenizer: Tokenizer = Tokenizer(config.vocab_size, tokenizer_bufferr)

    let llama: LlamaWeights[T] = LlamaWeights[T](config, is_tied, weights_buffer)
    var state: LlamaState[T] = LlamaState[T](config)
    if verbose:
        print("\033[1;32m\nModel Loaded Successfully And Mojo is on 🔥.\033[1;0m\n")

    let now: Int = time.now()
    var input_ids = DynamicVector[Int]()

    if prompt:
        byte_pr_tokenizer_encoder(input_ids, prompt, tokenizer)

    while position < steps:
        llama_forward[T, nelts](input_id, position, llama, state, config, True)
        if position < len(input_ids):
            next_input_id = input_ids[position]
        else:
            if temperature == 0.0:
                next_input_id = state.logits.argmax(-1)
            else:
                for q in range(config.vocab_size):
                    state.logits[q] = state.logits[q] / temperature
                softmax[T, nelts](state.logits, config.vocab_size)
                next_input_id = sample_array[T](state.logits)

            if next_input_id == 1 or next_input_id == 2:
                break

        var token_str: Pointer[UInt8] = tokenizer.vocab[next_input_id]

        if input_id == 1 and token_str[0] == ord(" "):
            token_str = token_str.offset(1)
        print_pointer(token_str)
        input_id = next_input_id
        position += 1
        if start == -1:
            start = time.now() // 1_000_000
    print()
    if verbose:
        let tps = (position - 1) / ((time.now() // 1_000_000) - start) * 1000
        print("\nAchieved Tokens ", tps, " Pre Second")


fn main() raises:
    alias DTYPE = DType.float32
    var next_input_id: Int = -1
    var input_id: Int = 1
    var position: Int = 0
    var temperature: SIMD[DTYPE, 1] = 0.4
    var steps: Int = 512
    var start: Int = -1
    var prompt: String = String("")
    var checkpoint_path: StringRef = StringRef("weights.bin")
    var tokenizer_path: StringRef = StringRef("tokenizer.bin")
    var rng_seed: Int = time.now()
    var verbose: Bool = True

    @parameter
    fn argparse() raises -> Int:
        let args = argv()
        if len(args) < 2:
            return 0
        for i in range(1, len(args), 2):
            if args[i] == "--weights-path" or args[i] == "--wp":
                checkpoint_path = args[i + 1]
            if args[i] == "--steps" or args[i] == "-a":
                steps = atol(args[i + 1])
            if args[i] == "--tokenizer-path":
                tokenizer_path = args[i + 1]
            if args[i] == "-v":
                let eva = atol(args[i + 1])
                verbose = True if eva == 1 else False
            if args[i] == "--seed" or args[i] == "-s":
                rng_seed = atol(args[i + 1])
            if args[i] == "--prompt" or args[i] == "-p":
                prompt = args[i + 1]
            if args[i] == "--temperature" or args[i] == "-t":
                let val = args[i + 1]
                temperature = 0.0
                # hacky parse float, keep only 1 digit
                for c in range(0, len(val)):
                    if val[c] == ".":
                        temperature += atol(val[c + 1]) * 0.1
                        break
                    else:
                        temperature = atol(val[c])
                if temperature < -1e9 or temperature > (1 + 1e9):
                    print("Wrong temperature value", temperature)
                    return 0
        return 1

    let res = argparse()
    if res == 0:
        return
    random.seed(rng_seed)
    run[DTYPE, Array[DTYPE].nelts](
        checkpoint_path,
        tokenizer_path,
        next_input_id,
        input_id,
        position,
        temperature,
        steps,
        start,
        prompt,
        verbose,
    )
