# Tutorial: Advanced Model Distillation with Streaming Datasets on TPUs

This tutorial dives into an advanced and powerful technique: **model distillation**. We will demonstrate how to create a smaller, more efficient language model by transferring knowledge from a massive "teacher" model (`Qwen/Qwen3-14B`) to a custom-designed "student" model.

What makes this tutorial particularly advanced is its focus on a real-world, large-scale scenario: training on a **massive, web-scale dataset** (`tiiuae/falcon-refinedweb`) by **streaming** it directly from its source. This approach, powered by EasyDeL's `DataManager`, is essential for working with datasets that are too large to download.

**What is Model Distillation?**
Model distillation is like an apprenticeship. A large, knowledgeable "teacher" model guides a smaller "student" model. Instead of just teaching the student the correct answers (like in SFT), the teacher also imparts its "intuition" by providing its full probability distribution over all possible next words (its "soft labels"). The student learns to mimic this nuanced distribution, effectively learning the teacher's reasoning process. This results in a much more capable and robust small model.

**Why is this important?**

* **Create Production-Ready Models:** Build smaller, faster models that are cheaper to deploy and run for inference.
* **Train on Massive Datasets:** Learn from web-scale data without needing terabytes of local storage.
* **Build Custom Architectures:** Gain full control over your model's size and structure.

**Key Technologies Used:**

* **EasyDeL:** A JAX-based library with advanced features like the `DistillationTrainer` and `DataManager`.
* **Ray:** An open-source framework for distributed computing on our TPU cluster.
* **JAX:** A high-performance numerical computing library, ideal for TPUs.
* **Streaming Datasets:** A technique to process data on the fly from sources like the Hugging Face Hub or Google Cloud Storage.

---

## Prerequisites

1. **Google Cloud TPU:** Access to a TPU environment (e.g., a TPU VM). This script is configured for a `v4-64` slice.
2. **Google Cloud Account & Project:** Properly configured for TPU usage and, importantly, Google Cloud Storage (GCS) for saving checkpoints from long-running jobs.
3. **Basic Python & ML Knowledge:** Familiarity with Python and machine learning concepts.

---

## Step 1: Environment Setup

The setup process is identical to previous tutorials.

1. **SSH into your TPU VM.**
2. **Run the EasyDeL setup script:**

    ```bash
    bash <(curl -sL https://raw.githubusercontent.com/erfanzar/EasyDeL/refs/heads/main/tpu_setup.sh)
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

The provided script is a powerful template for large-scale distillation. Let's break down its most important sections.

### The Student/Teacher Dynamic

The core of this process is the relationship between two models:

* **`teacher_model`**: A large, powerful, pre-trained model (`Qwen/Qwen3-14B`). It is loaded in inference mode, and its weights are kept frozen throughout training. It acts as the source of knowledge.
* **`student_model`**: A smaller model that we define from scratch using EasyDeL's `Qwen3Config` object. Its weights are randomly initialized, and it is the only model whose parameters are updated during training.

The goal is to train the `student_model` so that its predictions on the training data become as close as possible to the `teacher_model`'s rich, nuanced predictions.

### Streaming Large Datasets with `DataManager`

For datasets like `tiiuae/falcon-refinedweb`, which is over a terabyte in size, downloading is not an option. The script handles this using EasyDeL's `DataManager`, a powerful tool for streaming and mixing datasets from various sources.

```python
# 1. Define the dataset source(s). Each `TextDatasetInform` points to a location.
#    You can stream from the Hugging Face Hub, a GCS bucket, or local files.
informs = [
    ed.TextDatasetInform(content_field="content", path="tiiuae/falcon-refinedweb", split="train"),
    # The commented-out lines show examples of streaming from GCS buckets.
    # You can mix and match multiple sources to create a custom data blend.
]
# 2. Combine the sources into a mixture.
mixture = ed.DatasetMixture(batch_size=1, informs=informs)
# 3. Create a live, iterable dataset that fetches data on-the-fly.
train_dataset = ed.DataManager.create_dataset_from_mixture(mixture)
```

This setup allows you to train on virtually any size dataset without worrying about local storage limitations.

### Designing the Student Model

A key feature of this script is the flexibility to define a custom student architecture.

```python
student_model = ed.Qwen3ForCausalLM(
    config=ed.Qwen3Config(
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=4096 * 2,
        num_hidden_layers=16,  # We've reduced the layers, making it much smaller.
        num_attention_heads=32,
        # ... other architecture parameters ...
    ),
    # ... other model settings ...
).shard_model()
```

You have complete control to experiment with different numbers of layers, hidden sizes, and attention heads to create a model that perfectly balances performance and computational cost for your specific needs.

### Distillation Hyperparameters

In the `ed.DistillationConfig`, two parameters are critical for the distillation process:

* **`temperature=2.0`**: This hyperparameter "softens" the teacher's output probability distribution. A higher temperature makes the distribution less "spiky" (less focused on the single best token), providing a richer, more nuanced signal for the student to learn from.
* **`alpha=0.9`**: This is a weighting factor that balances two different loss components:
    1. **Distillation Loss (weighted by `alpha`)**: Measures how well the student's soft predictions match the teacher's soft predictions.
    2. **SFT Loss (weighted by `1 - alpha`)**: The standard cross-entropy loss, which measures how well the student predicts the single correct next token from the data.

An `alpha` of `0.9` means the training is heavily focused (90% of the loss) on mimicking the teacher's "thought process."

### `per_epoch_training_steps`: A Must for Streaming

Since a streaming dataset has no predefined length, you **must** tell the trainer how many steps constitute one "epoch."

```python
per_epoch_training_steps=98_000_000,
```

This value allows the learning rate scheduler and other epoch-based logic to function correctly. You should set it to roughly the total number of samples in your dataset(s).

---

## Step 3: Running the Script

1. **Save the Code:** Ensure the Python script is saved as `distill_finetune.py` on your TPU VM.
2. **Update GCS Bucket:** **Crucially**, change the `save_directory` in `DistillationConfig` to your own Google Cloud Storage bucket path (e.g., `save_directory="gs://my-awesome-bucket/distillation-checkpoints"`). Saving to GCS is essential for long-running jobs.
3. **Execute the Script:**
    From your TPU VM's terminal:

    ```bash
    python distill_finetune.py
    ```

    The script will begin streaming data, processing it on the fly, and training the student model. This is a large-scale job that will run for a long time. It's highly recommended to use a terminal multiplexer like `tmux` or `screen` to keep the process running even if your SSH connection drops.

---

## Key Points and Customization

* **The Power of `DataManager`:** Don't underestimate the flexibility of `DataManager`. The commented-out examples in the `informs` list show how you can easily create custom data mixtures by streaming from different GCS buckets containing Parquet or JSON files. This is how large, proprietary datasets are often trained on.
* **Tune Distillation Hyperparameters:** `temperature` and `alpha` are the most important knobs to turn for successful distillation. Experiment with different values to find the best results for your specific student/teacher pair.
* **Start Small, Scale Up:** When designing a new student model, it can be useful to start with a very small architecture (e.g., 4-6 layers) and train for a short time to ensure the pipeline works. Then, you can scale up to your target architecture.
* **Cost and Time:** Be aware that training on a web-scale dataset is a significant undertaking in terms of both time and cost. Monitor your cloud billing and plan your experiments accordingly.

This tutorial provides a powerful and realistic template for creating high-quality, compact models through distillation, leveraging EasyDeL's advanced features for handling massive datasets and distributed training on TPUs.
