FROM ubuntu:latest
LABEL authors="Erfan Zare Chavoshi"
FROM python:3.11
WORKDIR /app

RUN apt-get update && apt-get upgrade -y -q
RUN apt-get install golang -y -q

ARG device
RUN if [ "$device" = "tpu" ]; then \
  pip install torch torchvision torchaudio --index-url ... ... https://download.pytorch.org/whl/cpu  \
  && pip install jax[tpu]==0.4.10 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; \
else \
  pip install torch torchvision torchaudio && pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html; \
fi


# Install dependencies
RUN pip install  \
    chex \
    typing \
    jax>=0.4.10  \
    jaxlib>=0.4.10 \
    flax \
    fjformer>=0.0.7  \
    transformers>=4.33.0 \
    einops \
    optax \
    msgpack  \
    ipython \
    tqdm  \
    pydantic==2.4.2 \
    datasets==2.14.3  \
    setuptools \
    gradio  \
    distrax  \
    rlax  \
    wandb>=0.15.9 \
    tensorboard \
    pydantic_core==2.11.0

# Copy your application code
COPY . /app
ENTRYPOINT ["top", "-b"]

