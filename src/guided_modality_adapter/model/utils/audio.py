import torchaudio


def w2t(path, target_sample_rate=16000):
    waveform, sr = torchaudio.load(path)
    if sr != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, target_sample_rate)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform
