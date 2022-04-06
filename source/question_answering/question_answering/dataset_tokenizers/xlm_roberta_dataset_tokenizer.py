from transformers import PreTrainedTokenizer

from .dataset_token_based_tokenizer import DatasetTokenBasedTokenizer


class XlmRobertaDatasetTokenizer(DatasetTokenBasedTokenizer):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        doc_stride: int,
        train: bool,
    ) -> None:
        super().__init__(tokenizer, max_length, doc_stride, train)
