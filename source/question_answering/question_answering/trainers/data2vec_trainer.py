from .trainer import Trainer


class Data2VecTrainer(Trainer):
    def __init__(self):
        super().__init__()

    def train(self):
        raise NotImplementedError
