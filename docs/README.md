# EasyDeL

EasyDeL (Easy Deep Learning) is an open-source library designed to accelerate and optimize the training process of
machine learning models. This library is primarily focused on Jax/Flax and plans to offer easy and fine solutions to
train Flax/Jax Models on the `TPU/GPU` both for Serving and Training For both Python And Mojo🔥

# EasyDel Mojo 🔥

EasyDel Mojo differs from EasyDel in Python in significant ways. In Python, you can leverage a vast array of packages to
create a mid or high-level API in no time. However, when working with Mojo, it's a different story. Here, you have to
build some of the features that other Python libraries provide, such as Jax for arrays and computations. But why not
import numpy, Jax, and other similar packages to Mojo and use them?

#### Note this Library needs golang to run (for some tracking stuff on TPU/GPU/CPU)

#### Ubuntu GO installation

```shell
sudo apt-get update && apt-get upgrade -y
sudo apt-get install golang -y 
```

#### Manjaro/Arch GO installation

```shell
sudo pacman -Syyuu go
```

_you can install other version too but easydel required at least version of 0.4.10_

```shell
!pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q
```

on GPUs be like

```shell
pip install --upgrade pip
# CUDA 12 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda12_pip]" -f \
  https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

```shell
pip install --upgrade pip
# CUDA 11 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda11_pip]" -f \
  https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Documentations 🧭

* _EasyDel Python_:
    * Configs
    * [Modules](https://erfanzar.github.io/EasyDeL/docs/Python/Models)
    * RLHF
    * [Serve](https://erfanzar.github.io/EasyDeL/docs/Python/Serve)
    * SMI
    * [Trainer](https://erfanzar.github.io/EasyDeL/docs/Python/TrainingExample)
    * Transform
    * Utils

* _EasyDel Mojo🔥_ (Docs are on the way...):
    * [README 🔥](https://erfanzar.github.io/EasyDeL/lib/mojo)
    * [Array 🔥](https://erfanzar.github.io/EasyDeL/docs/Mojo/Array)
    * IO
    * Linen
    * Utilities
    * Tokenizer
    * Models