# Installing EasyDeL

EasyDeL uses FJFormer and JAX as main dependencies in order to run the scripts but there are some packages that needs to
be installed such as GO-lang to JAX specific platform installations, but you can simply install EasyDeL via pip:

```shell
pip install easydel
```

or install from head

```shell
pip install git+https://github.com/erfanzar/EasyDeL.git -U -q
```

### Tensorflow Issues for EasyDeL>=0.0.69
EasyDeL uses tensorflow for dataloaders and streamers so they are not in main process 
and latest tensorflow version and `jax>=0.4.29` which is recommended for easydel, they have package conflicts 
and here's how you should install easydel in case that your facing this issue.

```shell
pip uninstall easydel -y
pip install git+https://github.com/erfanzar/EasyDeL.git -q -U
pip install jax[tpu]==0.4.29 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q -U
python -c 'import easydel;print(True)' 
```

(you can change the TPU jax installation to `pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html` if you want)


## Installing Jax

JAX uses XLA to compile and run your NumPy programs on GPUs and TPUs. Compilation happens under the hood by default,
with library calls getting just-in-time compiled and executed. But JAX also lets you just-in-time compile your own
Python functions into XLA-optimized kernels using a one-function API, jit.
you can install other version too but easydel required at least version of 0.4.16

### TPU

inorder to install jax on TPU Devices use following command

```shell
!pip install jax[tpu]==0.4.29 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q
```

### GPU
inorder to install jax on cuda 12 use following command

```shell
pip install --upgrade pip
# CUDA 12 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

#### Installing GO

Note this Library needs golang to run (for some tracking stuff on TPU/GPU/CPU)

#### Ubuntu GO installation

```shell
sudo apt-get update && apt-get upgrade -y
sudo apt-get install golang -y 
```

#### Manjaro/Arch GO installation

```shell
sudo pacman -Syyuu go
```