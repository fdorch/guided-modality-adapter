import logging
import os
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from pyannote.audio.models.embedding.xvector import XVectorSincNet
from rich.logging import RichHandler
from torch import Tensor

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("rich")


class SpeakerEmbeddingModel(nn.Module):
    """
    Project-compliant wrapper around XVectorSincNet.

    Responsibilities:
        - forward(): process a single window → (B, D)
        - extract_dense(): sliding-window extraction for one waveform
        - load_checkpoint(): handle arbitrary .bin checkpoints
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        self.cfg = cfg
        self.sample_rate = cfg["sample_rate"]
        self.window_sec = cfg.get("window_sec", 1.0)
        self.hop_sec = cfg.get("hop_sec", 0.5)
        self.batch_size_windows = cfg.get("batch_size_windows", 32)
        embedding_dim = cfg.get("embedding_dim", 512)

        sincnet_params = {"stride": cfg.get("stride", 10)}

        self.xvector = XVectorSincNet(
            sample_rate=self.sample_rate,
            num_channels=1,
            dimension=embedding_dim,
            sincnet=sincnet_params,
        )

        self.load_checkpoint(cfg["checkpoint_path"])
        logger.info(
            f"Loaded speaker embedding model from {cfg['checkpoint_path']}"
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, samples) or (B, 1, samples)
        Returns:
            embeddings: (B, D)
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.xvector(x)

    def load_checkpoint(self, path: str):
        """
        Load weights from a pyannote .bin checkpoint or raw state_dict.
        Handles prefix stripping and mismatched keys.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError as e:
            logger.warning(f"Failed to load checkpoint with weights_only=False: {e}")
            return

        state = ckpt.get("state_dict", ckpt)

        model_keys = set(self.xvector.state_dict().keys())
        mapped = _map_state_dict_with_prefix_stripping(state, model_keys)

        final_state = self.xvector.state_dict()
        final_state.update(mapped)

        self.xvector.load_state_dict(final_state, strict=False)

    @torch.no_grad()
    def extract_dense(
        self,
        waveform: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Extract dense embeddings for a single waveform.

        Returns:
            embeddings:    (N, D)
            window_starts: (N,)
        """
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 2 and waveform.size(0) != 1:
            raise ValueError("extract_dense expects a single waveform: (T,) or (1,T)")

        embs, starts = _dense_batch_extract(
            self,
            waveform,
            window_sec=self.window_sec,
            hop_sec=self.hop_sec,
            device=waveform.device,
            batch_size_windows=self.batch_size_windows,
        )

        return embs[0], starts[0]


def _dense_batch_extract(
    model: SpeakerEmbeddingModel,
    waveforms: Tensor,
    window_sec: float,
    hop_sec: float,
    device: torch.device,
    batch_size_windows: int,
) -> Tuple[Tensor, Tensor]:
    """
    Internal dense window extraction for batched waveforms.
    Produces right-padded outputs for consistent shape.
    """
    model.eval()
    model.to(device)
    waveforms = waveforms.to(device)

    B, _ = waveforms.shape
    sr = model.sample_rate
    window_size = int(window_sec * sr)
    hop_size = int(hop_sec * sr)

    all_windows: List[Tensor] = []
    all_indices: List[Tuple[int, int]] = []
    counts: List[int] = []

    for b in range(B):
        wav = waveforms[b]
        T = wav.shape[-1]
        starts = []

        if T >= window_size:
            for start in range(0, T - window_size + 1, hop_size):
                all_windows.append(wav[start : start + window_size].unsqueeze(0))
                all_indices.append((b, start))
                starts.append(start)
        else:
            # fallback: whole waveform as one window
            all_windows.append(wav.unsqueeze(0))
            all_indices.append((b, 0))
            starts.append(0)

        counts.append(len(starts))

    if not all_windows:
        D = model.cfg.get("embedding_dim", 512)
        return (
            waveforms.new_zeros(B, 0, D),
            waveforms.new_zeros(B, 0, dtype=torch.long),
        )

    # Stack windows
    M = len(all_windows)
    windows_tensor = torch.cat(all_windows, dim=0).unsqueeze(1)  # (M,1,L)

    # Forward in chunks
    embs_list = []
    for i in range(0, M, batch_size_windows):
        chunk = windows_tensor[i : i + batch_size_windows]
        emb = model.forward(chunk)  # (m, D)
        embs_list.append(emb)

    embs = torch.cat(embs_list, dim=0)  # (M, D)
    D = embs.shape[-1]

    max_N = max(counts)
    batch_embs = waveforms.new_zeros(B, max_N, D)
    batch_starts = waveforms.new_zeros(B, max_N, dtype=torch.long)

    cursor = 0
    for b in range(B):
        n_w = counts[b]
        if n_w > 0:
            batch_embs[b, :n_w] = embs[cursor : cursor + n_w]
            starts_for_b = [
                idx for (bb, idx) in all_indices[cursor : cursor + n_w] if bb == b
            ]
            batch_starts[b, :n_w] = torch.tensor(
                starts_for_b, dtype=torch.long, device=device
            )
            cursor += n_w

    return batch_embs, batch_starts


def _map_state_dict_with_prefix_stripping(
    raw_state: Dict[str, Any],
    model_keys: set,
) -> Dict[str, Any]:
    """
    Map arbitrary checkpoint keys → model keys,
    stripping prefixes like:
        - module.
        - model.
        - xvector.
    """
    mapped = {}
    for k, v in raw_state.items():
        candidates = [
            k,
            k.removeprefix("module."),
            k.removeprefix("model."),
            k.removeprefix("xvector."),
        ]
        for c in candidates:
            if c in model_keys:
                mapped[c] = v
                break
    return mapped
