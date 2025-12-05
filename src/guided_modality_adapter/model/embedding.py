import os
import torch
import torch.nn as nn
from torch import Tensor
from typing import Union, Optional, Dict, Any, Tuple

from pyannote.audio.models.embedding.xvector import XVectorSincNet

# --- Configuration ---
LOCAL_WEIGHTS_PATH = "/gpfs/mariana/home/artfed/projects/pyannote_embed.bin"
DEFAULT_EMBEDDING_DIM = 512
DEFAULT_SAMPLE_RATE = 16000
# ---------------------


class SpeakerEmbeddingModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ):
        super().__init__()

        sincnet_params = {"stride": 10}

        self.xvector = XVectorSincNet(
            sample_rate=sample_rate,
            num_channels=1,
            dimension=embedding_dim,
            sincnet=sincnet_params,
        )

        self.sample_rate = sample_rate
        print(
            f"Initialized XVectorSincNet. Ready to load weights from {LOCAL_WEIGHTS_PATH}."
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)  # (1, samples)
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (batch, 1, samples)
        return self.xvector(x)

    def extract_embedding_from_waveform(self, waveform: Tensor) -> Tensor:
        device = next(self.parameters()).device
        waveform = waveform.to(device)
        self.eval()
        with torch.no_grad():
            return self.forward(waveform)

    def extract_dense_embeddings(
        self,
        waveform: Tensor,
        window_sec: float = 1.0,
        hop_sec: float = 0.5,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a long waveform into dense per-window speaker embeddings.

        Args:
            waveform: Tensor of shape (samples,) or (1, samples)
            window_sec: size of sliding window in seconds
            hop_sec: hop between consecutive windows in seconds
            device: device to run model on

        Returns:
            embeddings: Tensor of shape (num_windows, embedding_dim)
            window_starts: Tensor of shape (num_windows,) with start sample indices
        """
        self.eval()
        self.to(device)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(device)

        sample_rate = self.sample_rate
        window_size = int(window_sec * sample_rate)
        hop_size = int(hop_sec * sample_rate)
        total_samples = waveform.shape[1]

        embeddings = []
        window_starts = []

        for start in range(0, total_samples - window_size + 1, hop_size):
            end = start + window_size
            win_wave = waveform[:, start:end]
            with torch.no_grad():
                emb = self.extract_embedding_from_waveform(win_wave)  # (1, emb_dim)
            embeddings.append(emb)
            window_starts.append(start)

        if len(embeddings) == 0:
            with torch.no_grad():
                emb = self.extract_embedding_from_waveform(waveform)
            embeddings = [emb]
            window_starts = [0]

        embeddings = torch.vstack(embeddings)  # (num_windows, embedding_dim)
        window_starts = torch.tensor(window_starts, dtype=torch.int)

        return embeddings, window_starts


# --- Checkpoint loading and prefix mapping functions ---
def _strip_prefix(k: str, prefixes=("model.", "module.", "xvector.")) -> str:
    for p in prefixes:
        if k.startswith(p):
            return k[len(p) :]
    return k


def build_key_mapping(
    checkpoint_state_dict: Dict[str, Any], model_state_dict_keys: set
) -> Dict[str, Any]:
    mapped: Dict[str, Any] = {}
    used_model_keys = set()

    for ck_k, ck_v in checkpoint_state_dict.items():
        candidates = [ck_k, _strip_prefix(ck_k), ck_k.removeprefix("xvector.")]
        matched = None
        for c in candidates:
            if c in model_state_dict_keys and c not in used_model_keys:
                matched = c
                break

        if matched is None:
            alt_candidates = [
                "xvector." + ck_k,
                "model." + ck_k,
                "module." + ck_k,
                ck_k.replace("xvector.", ""),
            ]
            for c in alt_candidates:
                if c in model_state_dict_keys and c not in used_model_keys:
                    matched = c
                    break

        if matched:
            mapped[matched] = ck_v
            used_model_keys.add(matched)

    return mapped


def load_checkpoint_into_model(model: SpeakerEmbeddingModel, ckpt_path: str):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    model_keys = set(model.xvector.state_dict().keys())
    mapped_state = build_key_mapping(raw_state, model_keys)

    model_sd = model.xvector.state_dict()
    to_load = model_sd.copy()
    to_load.update(mapped_state)

    try:
        if set(mapped_state.keys()) == model_keys:
            model.xvector.load_state_dict(to_load, strict=True)
        else:
            model.xvector.load_state_dict(to_load, strict=False)
    except Exception:
        cleaned = {k: v for k, v in raw_state.items()}
        model.xvector.load_state_dict(cleaned, strict=False)


# --- Example usage ---
if __name__ == "__main__":
    model = SpeakerEmbeddingModel()
    load_checkpoint_into_model(model, LOCAL_WEIGHTS_PATH)

    sr = model.sample_rate
    duration_sec = 30.0
    n_samples = int(sr * duration_sec)
    dummy_waveform = torch.randn(1, n_samples)

    embeddings, starts = model.extract_dense_embeddings(
        dummy_waveform, window_sec=1.0, hop_sec=0.5
    )
    print("Dense embeddings shape:", embeddings.shape)
    print("Window start indices:", starts)
