# EasyDeL

EasyDeL (Easy Deep Learning) is an open-source library designed to accelerate and optimize the training process of
machine learning models. This library is primarily focused on Jax/Flax and plans to offer easy and fine solutions to
train Flax/Jax Models on the `TPU/GPU` both for Serving and Training

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

* _EasyDel_:
    * Configs
    * Modules
    * RLHF
    * [Serve](https://erfanzar.github.io/EasyDeL/docs/Serve)
    * SMI
    * Trainer
    * Transform
    * Utils
