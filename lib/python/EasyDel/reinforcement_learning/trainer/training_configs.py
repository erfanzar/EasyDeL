from dataclasses import dataclass
from typing import Optional


@dataclass
class RewardConfig:
    max_length: Optional[int] = None
    """
    The maximum length of the sequences in the batch. This argument is 
    required if you want to use the default data collator.
    """
    gradient_checkpointing: Optional[bool] = True
    """If True, use gradient checkpointing to save memory at the expense of slower backward pass."""
    gradient_checkpointing_kwargs: Optional[dict] = None
    """Keyword arguments to pass to the gradient checkpointing function."""
