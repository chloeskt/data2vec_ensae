import argparse
import os

from tqdm import tqdm

tqdm.pandas()

from datasets import load_dataset, Dataset, load_metric
from transformers import (
    CanineForQuestionAnswering,
    CanineTokenizer,
    RobertaTokenizerFast,
    Data2VecTextForQuestionAnswering,
    default_data_collator,
    IntervalStrategy,
    SchedulerType,
)

from question_answering import (
    set_seed,
    Preprocessor,
    CanineDatasetTokenizer,
    Data2VecDatasetTokenizer,
    CanineCTrainer,
    CanineSTrainer,
    Data2VecTrainer,
    TrainerArguments,
    DataArguments,
    to_pandas,
    remove_examples_longer_than_threshold,
)

seed = 0
set_seed(seed)


def train_model(
    model_name: str,
    learning_rate: float,
    weight_decay: float,
    type_lr_scheduler: SchedulerType,
    warmup_ratio: float,
    save_strategy: IntervalStrategy,
    save_steps: int,
    num_epochs: int,
    early_stopping_patience: int,
    output_dir: str,
    device: str,
    batch_size: int,
    max_length: int,
    doc_stride: int,
    n_best_size: int,
    max_answer_length: int,
    squad_v2: bool,
) -> None:
    datasets = load_dataset("squad_v2" if squad_v2 else "squad")

    preprocessor = Preprocessor(datasets)
    datasets = preprocessor.preprocess()

    if model_name == "canine-s" or model_name == "canine-c":
        df_train = to_pandas(datasets["train"])
        df_validation = to_pandas(datasets["validation"])

        df_train = remove_examples_longer_than_threshold(
            df_train, max_length=max_length * 2, doc_stride=doc_stride
        )
        df_validation = remove_examples_longer_than_threshold(
            df_validation, max_length=max_length * 2, doc_stride=doc_stride
        )

        datasets["train"] = Dataset.from_pandas(df_train)
        datasets["validation"] = Dataset.from_pandas(df_validation)

        del df_train, df_validation

        pretrained_model_name = f"google/{model_name}"
        tokenizer = CanineTokenizer.from_pretrained(pretrained_model_name)

        model = CanineForQuestionAnswering.from_pretrained(pretrained_model_name)

        tokenizer_dataset_train = CanineDatasetTokenizer(
            tokenizer,
            max_length,
            doc_stride,
            train=True,
            squad_v2=squad_v2,
            language="en",
        )
        tokenizer_dataset_val = CanineDatasetTokenizer(
            tokenizer,
            max_length,
            doc_stride,
            train=True,
            squad_v2=squad_v2,
            language="en",
        )

    elif model_name == "data2vec":
        pretrained_model_name = "facebook/data2vec-text-base"

        tokenizer = RobertaTokenizerFast.from_pretrained(pretrained_model_name)

        model = Data2VecTextForQuestionAnswering.from_pretrained(pretrained_model_name)

        tokenizer_dataset_train = Data2VecDatasetTokenizer(
            tokenizer, max_length, doc_stride, train=True
        )
        tokenizer_dataset_val = Data2VecDatasetTokenizer(
            tokenizer, max_length, doc_stride, train=False
        )

    elif model_name == "bert":
        pretrained_model_name = "bert-base-uncased"

        tokenizer = RobertaTokenizerFast.from_pretrained(pretrained_model_name)

        model = Data2VecTextForQuestionAnswering.from_pretrained(pretrained_model_name)

        tokenizer_dataset_train = Data2VecDatasetTokenizer(
            tokenizer, max_length, doc_stride, train=True
        )
        tokenizer_dataset_val = Data2VecDatasetTokenizer(
            tokenizer, max_length, doc_stride, train=False
        )
        pass

    elif model_name == "mbert":
        pretrained_model_name = "facebook/data2vec-text-base"

        tokenizer = RobertaTokenizerFast.from_pretrained(pretrained_model_name)

        model = Data2VecTextForQuestionAnswering.from_pretrained(pretrained_model_name)

        tokenizer_dataset_train = Data2VecDatasetTokenizer(
            tokenizer, max_length, doc_stride, train=True
        )
        tokenizer_dataset_val = Data2VecDatasetTokenizer(
            tokenizer, max_length, doc_stride, train=False
        )
        pass

    elif model_name == "xlm-roberta":
        pretrained_model_name = "facebook/data2vec-text-base"

        tokenizer = RobertaTokenizerFast.from_pretrained(pretrained_model_name)

        model = Data2VecTextForQuestionAnswering.from_pretrained(pretrained_model_name)

        tokenizer_dataset_train = Data2VecDatasetTokenizer(
            tokenizer, max_length, doc_stride, train=True
        )
        tokenizer_dataset_val = Data2VecDatasetTokenizer(
            tokenizer, max_length, doc_stride, train=False
        )
        pass

    else:
        raise NotImplementedError

    tokenized_datasets = datasets.map(
        tokenizer_dataset_train.tokenize,
        batched=True,
        remove_columns=datasets["train"].column_names,
    )

    validation_features = datasets["validation"].map(
        tokenizer_dataset_val.tokenize,
        batched=True,
        remove_columns=datasets["validation"].column_names,
    )

    data_collator = default_data_collator
    metric = load_metric("squad_v2" if squad_v2 else "squad")

    trainer_args = TrainerArguments(
        model=model,
        learning_rate=learning_rate,
        lr_scheduler=type_lr_scheduler,
        warmup_ratio=warmup_ratio,
        save_strategy=save_strategy,
        save_steps=save_steps,
        epochs=num_epochs,
        output_dir=output_dir,
        metric=metric,
        evaluation_strategy=save_strategy,
        weight_decay=weight_decay,
        data_collator=data_collator,
        model_save_path=os.path.join(
            output_dir, f"{model_name}-finetuned", "best_model.pt"
        ),
        device=device,
        early_stopping_patience=early_stopping_patience,
    )
    data_args = DataArguments(
        datasets=datasets,
        validation_features=validation_features,
        batch_size=batch_size,
        tokenizer=tokenizer,
        n_best_size=n_best_size,
        max_answer_length=max_answer_length,
        tokenized_datasets=tokenized_datasets,
        squad_v2=squad_v2,
    )

    if model_name == "canine-c":
        trainer = CanineCTrainer(trainer_args, data_args)
    elif model_name == "canine-s":
        trainer = CanineSTrainer(trainer_args, data_args)
    elif model_name == "data2vec":
        trainer = Data2VecTrainer(trainer_args, data_args)
    else:
        raise NotImplementedError

    trainer.train()

    f1, exact_match = trainer.evaluate(mode="val")
    print("Obtained F1-score: ", f1, "Obtained Exact Match: ", exact_match)

    # Save best model
    trainer.save_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parser for training and data arguments"
    )

    # TODO: add help field
    parser.add_argument("--model_name", type=str, help="Name of the model")
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--type_lr_scheduler", type=str)
    parser.add_argument("--warmup_ratio", type=float)
    parser.add_argument("--save_strategy", type=str)
    parser.add_argument("--save_steps", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--early_stopping_patience", type=int)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--doc_stride", type=int)
    parser.add_argument("--n_best_size", type=int)
    parser.add_argument("--max_answer_length", type=int)
    parser.add_argument("--squad_v2", type=bool)

    args = parser.parse_args()

    train_model(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        type_lr_scheduler=args.type_lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping_patience,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        doc_stride=args.doc_stride,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
        squad_v2=args.squad_v2,
    )
