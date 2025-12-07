import torch
from torch import nn, Tensor
from typing import Dict, Optional, Tuple

from transformers import (
    WhisperModel,
    WhisperConfig,
    AutoFeatureExtractor,
)


class WhisperEncoderWrapper(nn.Module):
    """
    Wrapper around Whisper encoder providing:

    - Batch log-mel→encoder outputs
    - Frame-level encoder hidden states
    - Frame timestamps in seconds
    - Optional raw-waveform → feature conversion
    """

    def __init__(
        self,
        model_name_or_config: "str | WhisperConfig",
        device: str = "cuda",
    ):
        super().__init__()

        if isinstance(model_name_or_config, WhisperConfig):
            self.model = WhisperModel(model_name_or_config)
        else:
            self.model = WhisperModel.from_pretrained(model_name_or_config)

        self.encoder = self.model.encoder
        self.config: WhisperConfig = self.model.config
        self.device = torch.device(device)
        self.to(self.device)

        # lazy load feature extractor
        self._feature_extractor = None

    @property
    def feature_extractor(self) -> AutoFeatureExtractor:
        if self._feature_extractor is None:
            self._feature_extractor = AutoFeatureExtractor.from_pretrained(
                getattr(self.config, "_name_or_path", "openai/whisper-small")
            )
        return self._feature_extractor

    @torch.no_grad()
    def waveform_to_features(
        self,
        waveform: Tensor,
        sample_rate: Optional[int] = 16000,
    ) -> Tensor:
        """
        Convert raw waveform → Whisper log-mel features.

        Args:
            waveform: (B, samples) or (samples,)
            sample_rate: native sampling rate of waveform

        Returns:
            input_features: (B, n_mels, T_whisper)
        """
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # HF expects list of (N,)
        inputs = self.feature_extractor(
            [w.cpu().numpy() for w in waveform],
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        return inputs.input_features.to(self.device)

    #@torch.no_grad()
    def forward(
        self,
        input_features: Tensor,
        *,
        attention_mask: Optional[Tensor] = None,
        audio_length_sec: Optional[Tensor] = None,
        return_timestamps: bool = True,
    ) -> Dict[str, Tensor]:
        """
        Args:
            input_features: (B, n_mels, T_whisper)
            attention_mask: (B, T_whisper)
            audio_length_sec: (B,) duration in seconds
            return_timestamps: return frame_times_seconds

        Returns:
            {   
              "hidden_states": (B, T, D)
              "frame_times_seconds": (B, T)
            }
        """
        # convert tensor to acceptable format
        input_features = self.waveform_to_features(input_features)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        enc = self.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            return_dict=True,
        )

        hidden = enc.last_hidden_state  # (B, T, D)
        out = {"hidden_states": hidden}

        if return_timestamps and audio_length_sec is not None:
            B, T, _ = hidden.shape

            if audio_length_sec.ndim == 0:
                audio_length_sec = audio_length_sec.view(1)

            if audio_length_sec.shape[0] == 1 and B > 1:
                audio_length_sec = audio_length_sec.expand(B)

            out["frame_times_seconds"] = frames_to_time_indices_batch(
                T, audio_length_sec.to(self.device)
            )

        return out


def frames_to_time_indices(
    T_whisper: int,
    audio_length_sec: float,
    device="cpu",
) -> Tensor:
    """Compute center time (sec) of each Whisper encoder frame."""
    if T_whisper == 0:
        return torch.empty(0, device=device)
    frame = torch.arange(T_whisper, device=device) + 0.5
    return frame * (audio_length_sec / T_whisper)


def frames_to_time_indices_batch(
    T: int,
    audio_length_sec: Tensor,
) -> Tensor:
    """Vectorized version for batch audio."""
    if T == 0:
        return audio_length_sec.new_zeros(audio_length_sec.size(0), 0)

    B = audio_length_sec.size(0)
    base = (torch.arange(T, device=audio_length_sec.device) + 0.5).unsqueeze(0)
    scale = (audio_length_sec / float(T)).unsqueeze(1)
    return base * scale
