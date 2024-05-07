## About Bits in EasyDeL

In easydel bits are totally different from huggingface and in EasyDeL training model with 8 bit is supported too without
needs to change the code just change the bit and that's all you have todo but by the way you still have to pass
the dtype and param_dtype cause unlike the transformers and bitsandbytes which store parameters in int8 and do
operations
in float16, bfloat16, float32 we don't do that like this in Jax we still store parameters as float16,bfloat16 or float32
and
do operations in bits like 8 6 4, and you can still train your model in this way and make it much more accurate than
bitsandbytes or peft fine-tuning

> Right now im looking to make EasyBITs in EasyDeL work on TPU-v3 cause on low amp GPUs and old TPUs it
> might now work as good as it does on TPU-v4/5