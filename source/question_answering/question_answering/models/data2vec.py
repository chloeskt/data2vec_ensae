from transformers import Data2VecTextModel, Data2VecTextConfig

from .model import Model


class Data2VecQA(Model):
    """Data2Vec model for Question Answering Tasks"""

    def __init__(
            self, pretrained_model_name: str = "facebook/data2vec-text-base"
    ) -> None:
        config = Data2VecTextConfig()
        data2vec = Data2VecTextModel.from_pretrained(pretrained_model_name, add_pooling_layer=False)
        Model.__init__(self, data2vec, config)
