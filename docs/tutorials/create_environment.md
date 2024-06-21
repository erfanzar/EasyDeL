# Setting Up the Environment for Easydel

Before you start using Easydel, you need to set up your environment. Easydel relies on JAX and FJFormer as its backbones. Here's a step-by-step guide on how to set up your environment and install the necessary dependencies.

#### NOTE

if you want to install JAX Version higher than 0.4.28 make sure to first install EasyDeL and then after that install JAX like this 

```sh
pip uninstall easydel -y
pip install git+https://github.com/erfanzar/EasyDeL.git -q -U
pip install jax[tpu]==0.4.29 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q -U
python -c 'import easydel;print(True)' 
```

### Installing JAX

JAX is a key dependency for Easydel, and the installation instructions vary depending on your hardware. 

#### CPU
If you are using a CPU, you can install JAX using the following command:

```sh
pip install -U jax
```

#### NVIDIA GPU
If you have an NVIDIA GPU, you should install JAX with CUDA support. Use the following command:

```sh
pip install -U "jax[cuda12]"
```

#### Google TPU
For Google TPU, you need to use a specific installation command that includes a link to the TPU releases:

```sh
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### Installing FJFormer

FJFormer is also a crucial component for Easydel. To install FJFormer, use the following command:

```sh
pip install fjformer
```

### Installing Easydel

Once JAX and FJFormer are installed, you can proceed to install Easydel. Easydel requires JAX version 0.4.28 or higher. Ensure you have this version installed to avoid compatibility issues.

```sh
pip install easydel
```

### Verifying the Installation

After installing the dependencies, it's a good idea to verify that everything is set up correctly. You can do this by running a simple script to check the versions of JAX and FJFormer.

```python
import jax
import fjformer
import easydel

print(f"JAX version: {jax.__version__}")
print(f"FJFormer version: {fjformer.__version__}")
print(f"Easydel version: {easydel.__version__}")
```

If the versions match the requirements (JAX >= 0.4.28), you're ready to start using Easydel.

### Example: Setting Up the Environment on an NVIDIA GPU

Here’s a complete example of setting up the environment on an NVIDIA GPU:

1. **Create a virtual environment** (optional but recommended):

   ```sh
   python -m venv easydel-env
   source easydel-env/bin/activate
   ```

2. **Install JAX with CUDA support**:

   ```sh
   pip install -U "jax[cuda12]"
   ```

3. **Install FJFormer**:

   ```sh
   pip install fjformer
   ```

4. **Install Easydel**:

   ```sh
   pip install easydel
   ```

5. **Verify the installation**:

   ```python
   import jax
   import fjformer
   import easydel

   print(f"JAX version: {jax.__version__}")
   print(f"FJFormer version: {fjformer.__version__}")
   print(f"Easydel version: {easydel.__version__}")
   ```

With these steps, you should have a fully configured environment ready to take advantage of Easydel's capabilities for training and serving machine learning models.

---

This guide should help users set up their environment correctly to ensure smooth operation with Easydel. Feel free to customize or expand upon these instructions as necessary.