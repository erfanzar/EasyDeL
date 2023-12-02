# EasyDeL 🔮

EasyDeL, an open-source library, is specifically designed to enhance and streamline the training process of machine
learning models. It focuses primarily on Jax/Flax and aims to provide convenient and effective solutions for training
Flax/Jax Models on TPU/GPU for both Serving and Training purposes. Additionally, EasyDeL will support mojo and will be
rewritten for mojo as well.

Some of the key features provided by EasyDeL include:

- Support for 8, 6, and 4 BIT inference and training in JAX
- Integration of flashAttention in JAX for GPUs and TPUs
- Automatic serving of LLMs with mid and high-level APIs in both JAX and PyTorch
- LLM Trainer and fine-tuner in JAX
- RLHF (presumably Reinforcement Learning with Hybrid Functions) in Jax
- And various other features to enhance the training process and optimize performance.

> These features collectively aim to simplify and accelerate the training of machine learning models, making it more
> efficient and accessible for developers working with Jax/Flax.

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
