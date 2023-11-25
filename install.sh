#!/bin/bash

read -p "What type of device are you using? (tpu/gpu/cpu): " device
read -p "Do you have go Lang Already Installed? (y/n): " have_lang


if [[ "$device" == "tpu" ]]; then
  # Install the dependencies for TPU
  pip install jax[tpu]==0.4.10 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q
  pip install datasets flax optax --upgrade -q
  pip install einops ml_collections gradio sentencepiece fjformer>=0.0.0 protobuf==3.20.0 -q
  pip install transformers --upgrade
  echo "Done."
elif [[ "$device" == "gpu" ]]; then
  pip install numpy==1.25.2 -q
  pip install transformers einops ml_collections gradio sentencepiece fjformer>=0.0.0 protobuf -q
  pip install datasets flax optax --upgrade -q
elif [[ "$device" == "cpu" ]]; then
  pip install numpy==1.25.2 -q
  pip install transformers einops ml_collections gradio sentencepiece fjformer>=0.0.0 protobuf -q
  pip install datasets flax optax --upgrade -q
else
  # Print an error message
  echo "Invalid device type."
fi

if [[ "$have_lang" == "n" ]]; then
  read -p "Linux Distro? (arch/ubuntu): " distro

  if [[ "$distro" == "ubuntu" ]]; then
    apt-get update && apt-get upgrade -y
    apt-get install golang -y
  elif [[ "$distro" == "arch" ]]; then
    sudo pacman -Syyuu go
  else
    echo 'The Given Distro is not valid you can install Go Lang Manually'
  fi
fi