# **EasyDeLState: A Snapshot of Your EasyDeL Model**

The `EasyDeLState` class acts like a comprehensive container that holds all the essential information about your EasyDeL
model at a given point in time. Think of it as a snapshot of your model. It includes:

* **Training Progress:**
    * `step`: Tracks the current training step.
* **Model Itself:**
    * `module`:  Holds the actual instance of your EasyDeL model.
    * `module_config`: Stores the model's configuration settings.
    * `module_config_args`:  Keeps track of arguments used to create the configuration (useful for reloading).
    * `apply_fn`:  References the core function that applies your model to data.
* **Learned Parameters:**
    * `params`: Contains the trained weights and biases of your model.
* **Optimizer Information:**
    * `tx`: Stores the optimizer you're using to update the model's parameters (e.g., AdamW).
    * `opt_state`: Keeps track of the optimizer's internal state (this is important for things like momentum in
      optimizers).
    * `tx_init`: Remembers the initial settings used to create the optimizer (again, for reloading purposes).
* **Additional Settings:**
    * `hyperparameters`:  Provides a flexible place to store other hyperparameters related to your model or training
      process.

**Key Capabilities of EasyDeLState:**

* **Initialization (`create`)**: Lets you create a brand new `EasyDeLState` to start training.
* **Loading (`load`, `load_state`, `from_pretrained`)**: Enables you to reload a saved model from a checkpoint file or
  even a pre-trained model from a repository like Hugging Face Hub.
* **Saving (`save_state`)**: Allows you to save your model's current state, including its parameters and optimizer
  state.
* **Optimizer Management (`apply_gradients`, `free_opt_state`, `init_opt_state`)**: Provides methods for updating the
  model's parameters using gradients, releasing optimizer memory, and re-initializing the optimizer if needed.
* **Sharding (`shard_model`)**:  Helps you distribute your model's parameters efficiently across multiple devices (
  important for training large models).
* **PyTorch Conversion (`to_pytorch`)**:  Gives you a way to convert your EasyDeL model to its PyTorch equivalent.

**In Essence:**

`EasyDeLState` streamlines the process of managing, saving, loading, and even converting your EasyDeL models. It ensures
that you can easily work with your models and maintain consistency throughout your machine learning workflow. 
