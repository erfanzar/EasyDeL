# Tutorial: Model Distillation with Streaming Datasets on TPUs using EasyDeL

This tutorial explores **model distillation**, an advanced technique for creating smaller, faster, and more efficient language models by transferring knowledge from a large "teacher" model to a smaller "student" model. We will use EasyDeL to distill knowledge from the powerful **Qwen3-14B** model into a custom, smaller student model.

A key feature of this tutorial is demonstrating how to train on **massive, streaming datasets** that are too large to download locally, a common scenario in large-scale model training.

**What is Model Distillation?**
Imagine a wise, experienced professor (the **teacher model**) who knows a vast amount about a subject. Now, imagine a bright, eager undergraduate (the **student model**) who needs to learn the subject quickly and efficiently. Distillation is the process of the professor teaching the student.

Instead of just giving the student the right answers (which is like standard SFT), the professor also explains *why* some wrong answers are "less wrong" than others. In LLM terms, the teacher model provides its full probability distribution over the vocabulary—its "soft labels." The student model then learns to mimic this rich distribution, not just the single most likely token. This helps the student learn the teacher's "reasoning process," leading to a much more capable small model.

**Why Use Distillation?**

* **Create Compact Models:** Produce smaller models that are faster and cheaper to run for inference.
* **Specialize Models:** Distill the capabilities of a general-purpose model into a smaller one for a specific task.
* **Knowledge Transfer:** Efficiently transfer the complex patterns learned by a large model to a smaller one.

**Key Technologies Used:**

* **EasyDeL:** A JAX-based library for efficient training of LLMs, with a dedicated `DistillationTrainer`.
* **Ray:** An open-source framework for distributed applications, managing our TPU resources.
* **JAX:** A high-performance numerical computing library, ideal for TPUs.
* **Hugging Face Datasets & Transformers:** For model and dataset access.
* **Streaming Datasets:** Using EasyDeL's `DataManager` to handle datasets too large to fit in memory.

---

## Prerequisites

1. **Google Cloud TPU:** Access to a TPU environment (e.g., a TPU VM). This script is configured for a `v4-64` slice.
2. **Google Cloud Account & Project:** Properly configured for TPU usage and ideally, Google Cloud Storage (GCS) for saving checkpoints.
3. **Basic Python & ML Knowledge:** Familiarity with Python and machine learning concepts.

---

## Step 1: Setting up your TPU Environment

The setup process is identical to previous tutorials.

1. **SSH into your TPU VM.**
2. **Run the EasyDeL setup script:**

    ```bash
    curl -Ls https://raw.githubusercontent.com/erfanzar/EasyDeL/refs/heads/main/tpu_setup.sh | bash
    ```

3. **Set Environment Variables:**
    Ensure your Hugging Face and (optional) Weights & Biases environment variables are set.

    ```bash
    export HF_TOKEN_FOR_EASYDEL="hf_YOUR_HUGGINGFACE_TOKEN_HERE"
    export WANDB_API_KEY_FOR_EASYDEL="YOUR_WANDB_API_KEY_HERE"
    export WANDB_ENTITY="your_wandb_username"
    # Add to ~/.bashrc for persistence
    ```

---

## Step 2: Understanding the Distillation Script

The polished script above is our guide. Let's break down its most important sections.

### The Student/Teacher Dynamic

The entire process revolves around two models:

* **`teacher_model`**: A large, powerful, pre-trained model (`Qwen/Qwen3-14B`) that is loaded in inference mode. Its weights are frozen.
* **`student_model`**: A smaller model that we define from scratch using `ed.Qwen3Config`. Its weights are randomly initialized and are the only ones being trained.

The goal is to make the `student_model`'s predictions on the training data as close as possible to the `teacher_model`'s predictions.

### Streaming Large Datasets with `DataManager`

For datasets like `tiiuae/falcon-refinedweb` (which is over a terabyte), downloading is not feasible. The script handles this elegantly:

```python
# Describes the data source, its location, and the field containing text.
informs = [ed.TextDatasetInform(content_field="content", path="tiiuae/falcon-refinedweb", split="train")]
# Combines one or more sources.
mixture = ed.DatasetMixture(batch_size=1, informs=informs)
# Creates a live, iterable dataset that fetches data on-the-fly.
train_dataset = ed.DataManager.create_dataset_from_mixture(mixture)
```

This setup allows you to train on virtually any size dataset without worrying about local storage.

### Distillation Hyperparameters

In the `ed.DistillationConfig`, two parameters are critical for the distillation process:

* **`temperature=2.0`**: This "softens" the teacher's output probabilities. A higher temperature makes the probability distribution less "spiky," giving the student a richer, more nuanced signal to learn from. It prevents the student from only paying attention to the single most likely token.
* **`alpha=0.9`**: This is a weighting factor that balances two different losses:
    1. **Distillation Loss (weighted by `alpha`)**: Measures how well the student's soft predictions match the teacher's soft predictions.
    2. **SFT Loss (weighted by `1 - alpha`)**: The standard cross-entropy loss, measuring how well the student predicts the correct next token from the data.

    An `alpha` of `0.9` means the training is heavily focused on mimicking the teacher's "thought process."

### `per_epoch_training_steps` for Streaming

Since a streaming dataset has no defined "end," you must tell the trainer how many steps constitute one epoch.

```python
per_epoch_training_steps=98_000_000,
```

This value should be approximately the total number of samples in the dataset. This allows the learning rate scheduler and other epoch-based logic to function correctly. **This argument is mandatory for streaming datasets.**

---

## Step 3: Running the Script

1. **Save the Code:** Ensure the polished Python script is saved as `distill_finetune.py` on your TPU VM.
2. **Execute the Script:**
    From your TPU VM's terminal:

    ```bash
    python distill_finetune.py
    ```

    The script will begin streaming data from Hugging Face, processing it on the fly, and training the student model by comparing its outputs to the teacher's. This is a long-running job, so using a terminal multiplexer like `tmux` or `screen` is highly recommended.

---

## Key Points and Customization

* **Design Your Student:** The true power of this script is the ability to define any student architecture you want via `ed.Qwen3Config` (or any other EasyDeL config). You can experiment with fewer layers, smaller hidden dimensions, or different attention mechanisms to create a model perfectly sized for your deployment needs.
* **Tune Distillation Hyperparameters:** The `temperature` and `alpha` are the most important knobs to turn. Finding the right balance is key to successful knowledge transfer.
* **Use Google Cloud Storage (GCS):** For any serious, long-running training job, saving checkpoints to a GCS bucket is essential for reliability. Make sure to change `save_directory="gs://your-bucket/distillation"` to your own bucket.
* **Dataset Choice:** While this script uses a general web text dataset, you can perform distillation on more specialized, in-domain data to create a powerful, domain-specific small model. Just update the `path` in `TextDatasetInform`.

This tutorial provides a powerful template for creating high-quality, compact models through distillation, leveraging EasyDeL's advanced features for handling massive datasets and distributed training on TPUs.
