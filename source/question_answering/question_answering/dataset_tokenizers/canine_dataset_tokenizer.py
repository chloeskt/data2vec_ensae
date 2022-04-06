from transformers import PreTrainedTokenizer

from .dataset_character_based_tokenizer import DatasetCharacterBasedTokenizer


class CanineDatasetTokenizer(DatasetCharacterBasedTokenizer):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        doc_stride: int,
        train: bool,
        squad_v2: bool,
        language: str,
    ):
        super().__init__(tokenizer, max_length, doc_stride, train, squad_v2, language)
