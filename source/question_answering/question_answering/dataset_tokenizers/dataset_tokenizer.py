from abc import ABC
from typing import Union, Dict, List

from datasets import Dataset
from transformers import BatchEncoding

CANINE_TOKENIZED_EXAMPLES = Dict[str, Union[List[List[int]], List[int]]]


class DatasetTokenizer(ABC):
    def __init__(self) -> None:
        pass

    def tokenize(
        self, data: Dataset
    ) -> Union[BatchEncoding, CANINE_TOKENIZED_EXAMPLES]:
        raise NotImplementedError
