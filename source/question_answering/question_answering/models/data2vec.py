import torch.nn as nn
from transformers import Data2VecTextModel, Data2VecTextConfig


class Data2VecQA(nn.Module):
    """Data2Vec model for Question Answering Tasks"""

    def __init__(
        self, pretrained_model_name: str = "facebook/data2vec-text-base"
    ) -> None:
        super().__init__()

        self.pretrained_model_name = pretrained_model_name
        self.config = Data2VecTextConfig()
        self.data2vec = Data2VecTextModel.from_pretrained(self.pretrained_model_name, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(self.config.hidden_size, self.config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.data2vec(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        return self.qa_outputs(sequence_output)
