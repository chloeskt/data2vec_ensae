import torch
from transformers import RobertaTokenizerFast

from question_answering.models.data2vec import Data2VecQA

pretrained_model_name = "facebook/data2vec-text-base"
tokenizer = RobertaTokenizerFast.from_pretrained(pretrained_model_name)
model = Data2VecQA(pretrained_model_name)

question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
inputs = tokenizer(question, text, return_tensors="pt")

start_positions = torch.tensor([1])
end_positions = torch.tensor([3])

logits = model(**inputs)

start_logits, end_logits = logits.split(1, dim=-1)
start_logits = start_logits.squeeze(-1).contiguous()
end_logits = end_logits.squeeze(-1).contiguous()

print(start_logits, end_logits)
