from .trainer import TrainerArguments, DataArguments
from .token_based_model_trainer import TokenBasedModelTrainer


class MBertTrainer(TokenBasedModelTrainer):
    def __init__(
        self,
        trainer_args: TrainerArguments,
        data_args: DataArguments,
        model_name: str = "mbert",
    ) -> None:
        super().__init__(trainer_args, data_args, model_name)
