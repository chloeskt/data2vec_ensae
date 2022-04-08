import argparse
import logging
import os

from datasets import Dataset, load_dataset, load_metric
from tqdm import tqdm

tqdm.pandas()

from transformers import (
    CanineForQuestionAnswering,
    CanineTokenizer,
    Data2VecTextForQuestionAnswering,
    IntervalStrategy,
    RobertaTokenizerFast,
    SchedulerType,
    default_data_collator,
    BertTokenizerFast,
    BertForQuestionAnswering,
    RobertaForQuestionAnswering,
)

from question_answering import (
    DataArguments,
    DatasetCharacterBasedTokenizer,
    DatasetTokenBasedTokenizer,
    Preprocessor,
    TrainerArguments,
    CharacterBasedModelTrainer,
    TokenBasedModelTrainer,
    remove_examples_longer_than_threshold,
    set_seed,
    to_pandas,
)

SEED = 0
set_seed(SEED)

CANINE_S_MODEL = "canine-s"
CANINE_C_MODEL = "canine-c"
BERT_MODEL = "bert"
MBERT_MODEL = "mbert"
XLM_ROBERTA_MODEL = "xlm_roberta"
DATA2VEC_MODEL = "data2vec"
ROBERTA_MODEL = "roberta"

logger = logging.getLogger(__name__)


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
    logger.info("Loading dataset")
    datasets = load_dataset("squad_v2" if squad_v2 else "squad")

    logger.info("Adding end_answers to each question")
    preprocessor = Preprocessor(datasets)
    datasets = preprocessor.preprocess()

    logger.info(f"Preparing for model {model_name}")
    if model_name in [CANINE_C_MODEL, CANINE_S_MODEL]:
        df_train = to_pandas(datasets["train"])
        df_validation = to_pandas(datasets["validation"])

        logger.info(f"Removing examples longer than threshold for model {model_name}")
        df_train = remove_examples_longer_than_threshold(
            df_train, max_length=max_length * 2, doc_stride=doc_stride
        )
        df_validation = remove_examples_longer_than_threshold(
            df_validation, max_length=max_length * 2, doc_stride=doc_stride
        )
        logger.info("Done removing examples longer than threshold")

        datasets["train"] = Dataset.from_pandas(df_train)
        datasets["validation"] = Dataset.from_pandas(df_validation)

        del df_train, df_validation

        pretrained_model_name = f"google/{model_name}"
        tokenizer = CanineTokenizer.from_pretrained(pretrained_model_name)
        model = CanineForQuestionAnswering.from_pretrained(pretrained_model_name)

        tokenizer_dataset_train = DatasetCharacterBasedTokenizer(
            tokenizer,
            max_length,
            doc_stride,
            train=True,
            squad_v2=squad_v2,
            language="en",
        )
        tokenizer_dataset_val = DatasetCharacterBasedTokenizer(
            tokenizer,
            max_length,
            doc_stride,
            train=False,
            squad_v2=squad_v2,
            language="en",
        )
    else:
        if model_name == DATA2VEC_MODEL:
            pretrained_model_name = "facebook/data2vec-text-base"
            tokenizer = RobertaTokenizerFast.from_pretrained(pretrained_model_name)
            model = Data2VecTextForQuestionAnswering.from_pretrained(
                pretrained_model_name
            )

        elif model_name == BERT_MODEL:
            pretrained_model_name = "bert-base-uncased"
            tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name)
            model = BertForQuestionAnswering.from_pretrained(pretrained_model_name)

        elif model_name == MBERT_MODEL:
            pretrained_model_name = "bert-base-multilingual-cased"
            tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name)
            model = BertForQuestionAnswering.from_pretrained(pretrained_model_name)

        elif model_name == XLM_ROBERTA_MODEL:
            pretrained_model_name = "xlm-roberta-base"
            tokenizer = RobertaTokenizerFast.from_pretrained(pretrained_model_name)
            model = RobertaForQuestionAnswering.from_pretrained(pretrained_model_name)

        elif model_name == ROBERTA_MODEL:
            pretrained_model_name = "roberta-base"
            tokenizer = RobertaTokenizerFast.from_pretrained(pretrained_model_name)
            model = RobertaForQuestionAnswering.from_pretrained(pretrained_model_name)

        else:
            raise NotImplementedError

        tokenizer_dataset_train = DatasetTokenBasedTokenizer(
            tokenizer, max_length, doc_stride, train=True
        )
        tokenizer_dataset_val = DatasetTokenBasedTokenizer(
            tokenizer, max_length, doc_stride, train=False
        )

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

    if model_name in [CANINE_C_MODEL, CANINE_S_MODEL]:
        trainer = CharacterBasedModelTrainer(trainer_args, data_args, model_name)
    elif model_name in [DATA2VEC_MODEL, BERT_MODEL, MBERT_MODEL, XLM_ROBERTA_MODEL]:
        trainer = TokenBasedModelTrainer(trainer_args, data_args, model_name)
    else:
        raise NotImplementedError

    trainer.train()

    logger.info("START FINAL EVALUATION")
    f1, exact_match = trainer.evaluate(mode="val")
    print("Obtained F1-score: ", f1, "Obtained Exact Match: ", exact_match)

    # Save best model
    trainer.save_model()


if __name__ == "__main__":
    debug = False
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("datasets.arrow_dataset").setLevel(logging.WARNING)
    if debug:
        logger.getChild("question_answering.DatasetCharacterBasedTokenizer").setLevel(
            logging.DEBUG
        )

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
