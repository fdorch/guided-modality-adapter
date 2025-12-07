import numpy as np
import torch
from torch import Tensor, device


def interpolate_speaker_embeddings_in_time(
    speaker_embeddings: Tensor,
    window_starts: Tensor,
    sample_rate: int,
    window_size_sec: float,
    frame_times: Tensor,
    device: device,
) -> Tensor:
    """
    Linearly interpolate speaker embeddings from window centers -> frame times.

    Args:
    spk_emb:       (B, N, D_spk)  dtype=float32
    window_starts: (B, N)         start sample index for each window (int)
    sample_rate:   int (e.g. 16000)
    frame_times:   (B, T)         frame center times in seconds (float)
    device:        torch.device or None (output device). If None uses spk_emb.device

    Returns:
    upsampled:     (B, T, D_spk) on `device`
    """

    if device is None:
        device = speaker_embeddings.device

    B, N, D = speaker_embeddings.shape
    T = frame_times.shape[1]
    window_size_samples = int(round(sample_rate * window_size_sec))
    centers = (window_starts.to(torch.float32) + window_size_samples / 2.0) / float(
        sample_rate
    )

    out = np.zeros((B, T, D), dtype=np.float32)
    spk_np = speaker_embeddings.detach().cpu().numpy()
    centers_np = centers.detach().cpu().numpy()
    frames_np = frame_times.detach().cpu().numpy()

    for b in range(B):
        if centers_np[b].size == 0:
            continue

        cs = centers_np[b]

        if np.any(np.diff(cs) < 0):
            cs = np.maximum.accumulate(cs + 1e-8 * np.arange(cs.shape[0]))

        for d in range(D):
            y = spk_np[b, :, d]
            out[b, :, d] = np.interp(frames_np[b], cs, y)

    out_tensor = torch.from_numpy(out).to(device)  # (B, T, D)
    return out_tensor
