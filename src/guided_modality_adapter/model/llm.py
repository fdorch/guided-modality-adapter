import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMModel(nn.Module):
    def __init__(self, model_name: str):
        super(LLMModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs