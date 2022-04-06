import os
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, List, OrderedDict, Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from datasets import DatasetDict, Dataset
from transformers import (
    PreTrainedTokenizer,
    IntervalStrategy,
    SchedulerType,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.data.data_collator import InputDataClass
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers.trainer_utils import PredictionOutput

from ..utils.utils import remove_answer_end

QA_METRICS = Tuple[float, float]


@dataclass
class TrainerArguments:
    """
    Arguments needed to initiate a Trainer
    """

    # TODO: complete field
    model: nn.Module = field()
    learning_rate: float = field()
    lr_scheduler: SchedulerType = field()
    warmup_ratio: float = field()
    save_strategy: IntervalStrategy = field()
    save_steps: int = field()
    epochs: int = field()
    output_dir: str = field()
    metric: Any = field()
    evaluation_strategy: IntervalStrategy = field()
    weight_decay: float = field()
    data_collator: Callable[[List[InputDataClass]], Dict[str, Any]] = field()
    model_save_path: str = field()
    device: str = field()
    early_stopping_patience: int = field()


@dataclass
class DataArguments:
    """
    Data arguments needed to initiate a Trainer
    """

    datasets: DatasetDict = field()
    validation_features: Dataset = field()
    batch_size: int = field()
    tokenizer: PreTrainedTokenizer = field()
    n_best_size: int = field()
    max_answer_length: int = field()
    tokenized_datasets: DatasetDict = field()
    squad_v2: bool = field()


class CustomTrainer(ABC):
    """General Trainer signature"""

    def __init__(
        self, trainer_args: TrainerArguments, data_args: DataArguments, model_name: str
    ) -> None:
        self.trainer_args = trainer_args
        self.data_args = data_args
        self.model_name = model_name

        # Trainer
        self.trainer = None
        self.model = None

    def train(self) -> None:
        # Define training arguments
        args = TrainingArguments(
            output_dir=os.path.join(
                self.trainer_args.output_dir, self.model_name + "-finetuned"
            ),
            evaluation_strategy=self.trainer_args.evaluation_strategy,
            learning_rate=self.trainer_args.learning_rate,
            weight_decay=self.trainer_args.weight_decay,
            num_train_epochs=self.trainer_args.epochs,
            lr_scheduler_type=self.trainer_args.lr_scheduler,
            warmup_ratio=self.trainer_args.warmup_ratio,
            per_device_train_batch_size=self.data_args.batch_size,
            per_device_eval_batch_size=self.data_args.batch_size,
            save_strategy=self.trainer_args.save_strategy,
            save_steps=self.trainer_args.save_steps,
            push_to_hub=False,
            metric_for_best_model="eval_loss",
            load_best_model_at_end=True,
            logging_steps=self.trainer_args.save_steps,
            no_cuda=False if self.trainer_args.device == "cuda" else True,
        )

        # Initiate Hugging Face Trainer
        self.trainer = Trainer(
            self.trainer_args.model,
            args,
            train_dataset=self.data_args.tokenized_datasets["train"],
            eval_dataset=self.data_args.tokenized_datasets["validation"],
            data_collator=self.trainer_args.data_collator,
            tokenizer=self.data_args.tokenizer,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.trainer_args.early_stopping_patience
                )
            ],
        )

        self.trainer.train()
        # Load best model at the end of training
        self.model = self.trainer.model

    def evaluate(self, mode: str, features: Optional[Dataset] = None) -> QA_METRICS:
        if mode == "val":
            _features = self.data_args.validation_features
        elif mode == "test":
            _features = features
        else:
            raise ValueError(
                "Mode should either be val or test. If val, the model will be evaluated on validation features"
                "defined in the DataArguments. If test, one must provide a Dataset of features in the correct"
                "format."
            )
        raw_predictions = self.trainer.predict(_features)
        self.data_args.validation_features.set_format(
            type=self.data_args.validation_features.format["type"],
            columns=list(self.data_args.validation_features.features.keys()),
        )
        final_predictions = self._postprocess_qa_predictions(
            self.data_args.datasets["validation"],
            self.data_args.validation_features,
            raw_predictions.predictions,
        )
        results = self._compute_metrics(
            self.trainer_args.metric,
            self.data_args.datasets["validation"],
            final_predictions,
            self.data_args.squad_v2,
        )
        if self.data_args.squad_v2:
            return results["f1"], results["exact"]
        else:
            return results["f1"], results["exact_match"]

    def _postprocess_qa_predictions(
        self,
        data: Dataset,
        features: Dataset,
        raw_predictions: Union[PredictionOutput, QuestionAnsweringModelOutput],
    ) -> OrderedDict:
        raise NotImplementedError

    def save_model(self) -> None:
        print(f"Saving best trained model at {self.trainer_args.model_save_path}")
        torch.save(self.model.state_dict(), self.trainer_args.model_save_path)

    @staticmethod
    def _compute_metrics(
        metric: Any,
        data: Dataset,
        predictions: OrderedDict,
        squad_v2: bool = True,
    ) -> Dict[str, float]:
        data = data.map(remove_answer_end, batched=True)
        if squad_v2:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
                for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [
                {"id": k, "prediction_text": v} for k, v in predictions.items()
            ]

        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in data]

        return metric.compute(predictions=formatted_predictions, references=references)
