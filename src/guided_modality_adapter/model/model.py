from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from embedding import SpeakerEmbeddingModel
from llm import LLMModel
from projectors import SAASRAdapter
from torch import Tensor
from typing_extensions import Self
from utils.interpolate import interpolate_speaker_embeddings_in_time as interpolate
from whisper import WhisperEncoderWrapper


class SAASRModel(nn.Module):
    """
    Unified SA-ASR model that composes:
      SpeakerEmbeddingModel + WhisperEncoderWrapper + SAASRAdapter + LLMModel
    """

    def __init__(self, cfg: Dict[str, Any], device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        spk_cfg = cfg.get("speaker", {})
        self.speaker_model = SpeakerEmbeddingModel(spk_cfg)
        self.speaker_model.to(self.device)

        whisper_cfg = cfg.get("whisper", {})
        whisper_model_name = whisper_cfg.get("model_name", "openai/whisper-medium")
        self.whisper = WhisperEncoderWrapper(
            whisper_model_name, device=str(self.device)
        )
        self.whisper.to(self.device)

        proj_cfg = cfg.get("projectors", {})
        self.adapter = SAASRAdapter(proj_cfg)
        self.adapter.to(self.device)

        llm_cfg = cfg.get("llm", {})
        # ensure adapter_dim is in llm_cfg if needed
        self.llm = LLMModel(llm_cfg)
        self.llm.to(self.device)

        # sample_rate expected by speaker model
        self.sample_rate = int(
            spk_cfg.get(
                "sample_rate", getattr(self.speaker_model, "sample_rate", 16000)
            )
        )
        # window size for speaker model (used in interpolation)
        self.window_sec = float(spk_cfg.get("window_sec", 1.0))

    def _extract_dense(self, waveform: Tensor) -> Tuple[Tensor, Tensor]:
        waveform = waveform.to(next(self.speaker_model.parameters()).device)

        return self.speaker_model.extract_dense(waveform)

    def forward(
        self,
        waveform: Tensor,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through the SA-ASR model.

        Args:
            waveform (Tensor): Audio waveform tensor of shape (B, T).
            input_ids (Tensor): Input token IDs for the LLM of shape (B, S).
            attention_mask (Optional[Tensor]): Attention mask for the LLM of shape (B, S).

        Returns:
            Tensor: Output logits from the LLM.
        """
        # Step 1: Extract dense speaker embeddings
        dense_embeddings, dense_times = self._extract_dense(waveform)

        # Step 2: Pass audio through Whisper encoder
        encoder_outputs = self.whisper(waveform)

        # Step 3: Upsample speaker embeddings to match Whisper encoder time steps
        dense_embeddings = interpolate(
            dense_embeddings,
            dense_times,
            self.sample_rate,
            self.window_sec,
            encoder_outputs["frame_times_seconds"],
            device=self.device,
        )

        # Step 4: Adapt Whisper outputs with speaker embeddings
        adapted_outputs = self.adapter(encoder_outputs["hidden_states"], dense_embeddings)

        # Step 5: Pass adapted outputs to LLM
        llm_outputs = self.llm(
            adapted_outputs, input_ids, attention_mask=attention_mask
        )

        return llm_outputs

    @torch.no_grad()
    def generate_from_audio(
        self,
        waveform: Tensor,
        max_length: int = 4096,
        num_beams: int = 5,
        **generate_kwargs: Any,
    ) -> Tensor:
        """
        Generate text from audio waveform.

        Args:
            waveform (Tensor): Audio waveform tensor of shape (B, T).
            max_length (int): Maximum length of generated sequences.
            num_beams (int): Number of beams for beam search.
            **generate_kwargs: Additional keyword arguments for generation.
        Returns:
            Tensor: Generated token IDs.
        """
        # Step 1: Extract dense speaker embeddings
        dense_embeddings, dense_times = self._extract_dense(waveform)

        # Step 2: Pass audio through Whisper encoder
        encoder_outputs = self.whisper(waveform)

        # Step 3: Adapt Whisper outputs with speaker embeddings
        adapted_outputs = self.adapter(encoder_outputs, dense_embeddings, dense_times)

        # Step 4: Generate text using LLM
        generated_ids = self.llm.generate(
            adapted_outputs,
            max_length=max_length,
            num_beams=num_beams,
            **generate_kwargs,
        )

        return generated_ids
