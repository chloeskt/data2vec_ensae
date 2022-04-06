from .character_based_model_trainer import CharacterBasedModelTrainer
from .trainer import TrainerArguments, DataArguments


class CanineCTrainer(CharacterBasedModelTrainer):
    def __init__(
        self,
        trainer_args: TrainerArguments,
        data_args: DataArguments,
        model_name: str = "CANINE-C",
    ) -> None:
        super().__init__(trainer_args, data_args, model_name)


class CanineSTrainer(CharacterBasedModelTrainer):
    def __init__(
        self,
        trainer_args: TrainerArguments,
        data_args: DataArguments,
        model_name: str = "CANINE-S",
    ) -> None:
        super().__init__(trainer_args, data_args, model_name)
