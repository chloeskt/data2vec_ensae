from abc import ABC
from dataclasses import dataclass, field

import torch
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader


@dataclass
class TrainerArguments:
    """
    Arguments needed to initiate a Trainer
    """
    # TODO: complete field
    device: str = field()
    criterion: CrossEntropyLoss = field()
    optimizer: torch.optim.Optimizer = field()
    learning_rate: float = field()
    lr_scheduler: LambdaLR = field()
    train_loader: DataLoader = field()
    val_loader: DataLoader = field()
    epochs: int = field()
    output_dir: str = field()


class Trainer(ABC):
    """General Trainer signature"""

    def __init__(self):
        pass

    def train(self):
        raise NotImplementedError
