import math
import torch

from guided_modality_adapter.model.embedding import SpeakerEmbeddingModel


def test_batch_extract_dense_embeddings_shape():
    sr = 16000
    duration_sec = 30.0
    n_samples = int(sr * duration_sec)

    # One example, (B, samples)
    waveforms = torch.randn(1, n_samples)

    model = SpeakerEmbeddingModel(sample_rate=sr)

    with torch.no_grad():
        embs, starts = model.batch_extract_dense_embeddings(
            waveforms,
            window_sec=1.0,
            hop_sec=0.5,
            device="cpu",
            batch_size_windows=16,
        )

    # For 30 s, window=1s, hop=0.5s
    # N_windows = floor((30 - 1)/0.5) + 1 = 59
    expected_n = math.floor((duration_sec - 1.0) / 0.5) + 1

    assert embs.shape[0] == 1
    assert embs.shape[1] == expected_n
    assert embs.shape[2] == 512  # DEFAULT_EMBEDDING_DIM

    assert starts.shape[0] == 1
    assert starts.shape[1] == expected_n

    # No NaNs
    assert not torch.isnan(embs).any()
