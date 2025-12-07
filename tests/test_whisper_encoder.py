import torch
from transformers import WhisperConfig

from guided_modality_adapter.model.whisper import WhisperEncoderWrapper, frames_to_time_indices


def test_frames_to_time_indices_shape():
    T = 100
    dur = 10.0
    times = frames_to_time_indices(T, dur)
    assert times.shape[0] == T
    assert torch.all(times > 0)


def test_whisper_encoder_wrapper_shapes():
    # Tiny config to keep this a light unit test
    config = WhisperConfig(
        d_model=32,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
    )
    model = WhisperEncoderWrapper(config, device="cpu")

    B = 2
    n_mels = 128
    T_whisper = 50
    input_features = torch.randn(B, n_mels, T_whisper)
    audio_length_sec = torch.tensor([5.0, 7.5])

    out = model(
        input_features=input_features,
        attention_mask=None,
        audio_length_sec=audio_length_sec,
        return_timestamps=True,
    )

    hidden = out["last_hidden_state"]
    times = out["frame_times_seconds"]

    assert hidden.shape == (B, T_whisper, config.d_model)
    assert times.shape == (B, T_whisper)