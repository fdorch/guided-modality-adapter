import torch
from torch import nn, Tensor
from transformers import WhisperModel, WhisperConfig


class WhisperEncoder(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.whisper = WhisperModel(config)
        self.encoder = self.whisper.encoder

    def forward(self, input_features: Tensor, attention_mask: Tensor = None) -> Tensor:
        encoder_outputs = self.encoder(
            input_features,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return encoder_outputs.last_hidden_state
