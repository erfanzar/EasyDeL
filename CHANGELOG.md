# EasyDeL ChangeLOG 

## 0.0.69 TO Beta Version 0.0.70

- `TrainingArguments` Structure is changed and now it's a dataclass.
- `BaseTrainer`: This improved version of the `BaseTrainer`
> [!TIP]
>
>    This improved version of the `BaseTrainer` class includes the following enhancements:
>
>    1. Better code organization: Methods are grouped logically and broken down into smaller, more focused functions.
>    2. Improved readability: Complex operations are split into separate methods with descriptive names.
>    3. Type hinting: Added more type hints to improve code clarity and catch potential errors.
>    4. Error handling: Added more explicit error messages and checks.
>    5. Reduced code duplication: Extracted common operations into separate methods.
>    6. Improved configurability: Made it easier to extend and customize the trainer for different use cases.
>
>    Some potential further improvements could include:
>
>    1. Adding more extensive logging throughout the training process.
>    2. Implementing a progress bar for long-running operations.
>    3. Implementing early stopping based on validation metrics.
>    4. Adding support for mixed precision training.
>    5. Adding support for custom metrics and loss functions.
>    6. Implementing model checkpointing based on best validation scores.
> 
>    These improvements make the `BaseTrainer` class more robust, easier to understand, and more maintainable. >    The abstract methods `train` and `eval` still need to be implemented in subclasses to provide the specific >    training and evaluation logic for different model types.

- Overall improvement in `OrpoTrainer`, `DPOTrainer`, `CLMTrainer`, `SFTTrainer`.
- `ApiEngine` and `engine_client` are added.