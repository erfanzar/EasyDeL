import dataclasses
from dataclasses import field
from typing import Optional, List, Union, Tuple

from .. import is_torch_available

if is_torch_available():
    import torch
    from torch import nn as nn
    from torch.nn import functional as fu
    from torch.utils.data import Dataset, DataLoader


@dataclasses.dataclass
class Tasks:
    TEXT_GENERATION = 'text_generation'
    TEXT_CLASSIFICATION = 'text_classification'
    QUESTION_ANSWERING = 'question_answering'


@dataclasses.dataclass
class TrainConfig:
    epochs: int = 10,
    task: str = Tasks.TEXT_GENERATION
    betas: Optional[Tuple[float, flaot]] = (0.90, 0.999)
    lr: float = 1e-4


class Trainer(nn.Module):
    def __init__(self,
                 model: Optional[Union[nn.Module, torch.jit.ScriptModule]],
                 args: TrainConfig = TrainConfig(),
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 accelerate: Optional[bool] = True,
                 train_dataset: Optional[Dataset] = None,

                 ):
        super(Trainer, self).__init__()

    def text_generation_forward(self):
        ...

    def text_classification_forward(self):
        ...

    def question_answering_forward(self):
        ...
