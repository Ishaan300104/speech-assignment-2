"""Shared audio I/O and preprocessing utilities used across all parts."""

import numpy as np
import torch
import torchaudio
import soundfile as sf
from pathlib import Path


SAMPLE_RATE = 16000


def load_audio(path, target_sr=SAMPLE_RATE, mono=True):
    """Load audio file, resample if needed, return (numpy float32, sr)."""
    sig, sr = sf.read(str(path))
    if sig.ndim == 2 and mono:
        sig = sig.mean(axis=1)
    sig = sig.astype(np.float32)

    if sr != target_sr:
        t = torch.from_numpy(sig).unsqueeze(0)
        t = torchaudio.functional.resample(t, sr, target_sr)
        sig = t.squeeze(0).numpy()
        sr  = target_sr

    return sig, sr


def save_audio(sig, path, sr=22050):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), sig, sr)


def split_audio_segments(sig, sr, segment_ms=10000, overlap_ms=500):
    """Split long audio into overlapping chunks. Returns list of numpy arrays."""
    seg_samples = int(segment_ms * sr / 1000)
    hop_samples = seg_samples - int(overlap_ms * sr / 1000)
    segments = []
    for start in range(0, len(sig), hop_samples):
        chunk = sig[start: start + seg_samples]
        if len(chunk) > sr * 0.1:  # skip chunks under 100ms
            segments.append(chunk)
    return segments


def frame_to_time_ms(frame_idx, hop_length=320, sr=16000):
    return frame_idx * hop_length / sr * 1000


def time_ms_to_frame(time_ms, hop_length=320, sr=16000):
    return int(time_ms * sr / (1000 * hop_length))
