"""Using This Feature is not recommended since it's not fully completed
"""
from .models import AutoRLModelForCasualLMWithValueHead
from .trainer import (
    DPOTrainer as DPOTrainer,
    create_dpo_eval_function as create_dpo_eval_function,
    create_dpo_train_function as create_dpo_train_function,
    create_concatenated_forward as create_concatenated_forward
)

__all__ = (
    "create_concatenated_forward",
    "create_dpo_train_function",
    "create_dpo_eval_function",
    "DPOTrainer",
    "AutoRLModelForCasualLMWithValueHead"
)
