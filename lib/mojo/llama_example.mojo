from EasyDel import (
    Array,
    File,
    FileBuffer,
    read_file,
    BufReader,
    over_all_softmax,
    sample,
    Tokenizer,
    byte_pr_tokenizer_encoder,
    print_pointer,
    loop_sort,
)
from math import math
from python import Python, PythonObject
import time
import random
from random import rand
from EasyDel.models.llama import (
    LlamaConfig,
    LlamaWeights,
    LlamaState,
    llama_forward_call,
)
from sys import argv
from runtime.llcl import num_cores

alias RUNTIME_DTYPE: DType = DType.float32
alias RUNTIME_NELTS: Int = Array[RUNTIME_DTYPE].nelts * 4
alias RUNTIME_CORES: Int = 6


fn run[
    DT: DType, nelts: Int, cores: Int
](
    weights_path: StringRef,
    tokenizer_path: StringRef,
    inout next_input_id: Int = -1,
    inout input_id: Int = 1,
    inout position_id: Int = 0,
    temperature: SIMD[DT, 1] = 0.4,
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

    print("LOADED WEIGHT SIZE [MB] : ", weights_buffer.size // 1024 / 1024)
    print("LOADED TOKENS SIZE [MB] : ", tokenizer_bufferr.size // 1024 / 1024)

    let is_tied: Bool = True if config.vocab_size > 0 else False
    if not is_tied:
        config.vocab_size = -config.vocab_size
    if verbose:
        config.print_config()
    var tokenizer: Tokenizer = Tokenizer(config.vocab_size, tokenizer_bufferr)

    let start_loading: SIMD[DT, 1] = SIMD[DT, 1](time.now())
    let llama: LlamaWeights[DT] = LlamaWeights[DT](config, is_tied, weights_buffer)
    var llama_state: LlamaState[DT] = LlamaState[DT](config)
    let loading_time: SIMD[DT, 1] = (
        SIMD[DT, 1](time.now()) - start_loading
    ) / 1_000_000_000
    var start_time = Float64(0)
    print("MODEL LOADED AND STATE CREATED IN ", loading_time, " SEC/s")

    var input_ids = DynamicVector[Int]()

    if prompt:
        tokenizer.encode(input_ids, prompt)

    for i in range(steps):
        llama_forward_call[DT, nelts, cores, True](
            input_id, position_id, llama_state, llama, config
        )
        if position_id < len(input_ids):
            next_input_id = input_ids[position_id]
        else:
            if temperature == 0.0:
                over_all_softmax[DT, nelts, cores](llama_state.logits)
                next_input_id = llama_state.logits.argmax()
            else:
                for j in range(config.vocab_size):
                    llama_state.logits[j] = llama_state.logits[j] / temperature

                over_all_softmax[DT, nelts, cores](llama_state.logits)
                next_input_id = sample[DT](llama_state.logits)

            if next_input_id == 1 or next_input_id == 2:
                break
        var token_string: Pointer[UInt8] = tokenizer.vocab[next_input_id]
        if input_id == 0 and token_string[0] == ord(" "):
            token_string = token_string.offset(1)
        if start_time == 0:
            start_time = time.now() / 1_000_000
        position_id += 1
        input_id = next_input_id

        print_pointer(token_string)

    print(
        "\n\nWATCHOUT FOR 🔥 , AVG Tokens P/s : ",
        (position_id - 1) / ((time.now() / 1_000_000) - start_time) * 1000,
    )


fn main() raises:
    var next_input_id: Int = -1
    var input_id: Int = 1
    var position_id: Int = 0

    var temperature: SIMD[RUNTIME_DTYPE, 1] = 0.7
    var steps: Int = 1024
    var start: Int = -1

    var prompt: String = String(r"")

    # Prompt Example for TinyLlama <|im_start|>user\nHI<|im_end|>\n<|im_start|>assistant\n

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
    # if res == 0:
    #     return

    print("RUNTIME DTYPE   : ", RUNTIME_DTYPE)
    print("NUMBER OF CORES : ", num_cores())
    print("NUMBER OF NELTS : ", RUNTIME_NELTS)
    random.seed(rng_seed)
    run[RUNTIME_DTYPE, RUNTIME_NELTS, RUNTIME_CORES](
        checkpoint_path,
        tokenizer_path,
        next_input_id,
        input_id,
        position_id,
        temperature,
        steps,
        start,
        prompt,
        verbose,
    )
