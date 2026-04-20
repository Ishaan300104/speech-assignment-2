"""
Task 3.1 – Speaker Embedding Extraction (d-vector)

Records or loads a 60s reference audio and extracts a high-dimensional
speaker embedding using a d-vector encoder (3-layer LSTM).

The d-vector approach (Wan et al., 2018) trains a speaker verification network
on a large corpus and uses the final hidden state as the speaker representation.
Here I use a pretrained SpeechBrain ECAPA-TDNN model for x-vector extraction,
with a fallback LSTM-based d-vector encoder trained from scratch.

Output: 256-dimensional speaker embedding (normalized L2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import soundfile as sf
from pathlib import Path


SAMPLE_RATE     = 16000
N_MFCC          = 40
N_MELS          = 80
EMBEDDING_DIM   = 256
WINDOW_SIZE_S   = 1.6   # segment length for d-vector extraction
HOP_SIZE_S      = 0.5   # overlap between segments


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_mfcc(waveform, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    """Extract MFCC features: (n_mfcc, time_frames)."""
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).float()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    transform = torchaudio.transforms.MFCC(
        sample_rate=sr,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft":    512,
            "hop_length": 160,
            "n_mels":   80,
            "f_min":    20,
            "f_max":    8000,
        },
    )
    mfcc = transform(waveform)  # (1, n_mfcc, T)
    # delta and delta-delta
    delta1 = torchaudio.functional.compute_deltas(mfcc)
    delta2 = torchaudio.functional.compute_deltas(delta1)
    features = torch.cat([mfcc, delta1, delta2], dim=1)  # (1, 3*n_mfcc, T)
    return features.squeeze(0)  # (3*n_mfcc, T)


# ──────────────────────────────────────────────────────────────────────────────
# D-vector LSTM encoder
# ──────────────────────────────────────────────────────────────────────────────

class DVectorEncoder(nn.Module):
    """
    3-layer LSTM d-vector encoder.
    Input: MFCC features (batch, n_mfcc*3, time)
    Output: L2-normalized speaker embedding (batch, embedding_dim)

    Architecture follows Wan et al. (2018) GE2E training setup.
    """

    def __init__(self, input_dim=N_MFCC * 3, hidden_dim=768,
                 num_layers=3, embedding_dim=EMBEDDING_DIM, dropout=0.0):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.projection = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        """
        x: (batch, feat_dim, time)  – MFCC layout
        returns: (batch, embedding_dim) L2-normalized
        """
        # LSTM wants (batch, time, feat)
        x = x.transpose(1, 2)  # (batch, time, feat)
        lstm_out, (h_n, _) = self.lstm(x)
        # use last hidden state of top layer
        last_hidden = h_n[-1]  # (batch, hidden_dim)
        embedding = self.projection(last_hidden)
        return F.normalize(embedding, p=2, dim=1)

    def embed_segment(self, segment_feats):
        """
        Embed a single segment. segment_feats: (feat_dim, time) tensor.
        """
        x = segment_feats.unsqueeze(0)  # add batch dim
        with torch.no_grad():
            return self(x).squeeze(0)


# ──────────────────────────────────────────────────────────────────────────────
# X-vector extraction via SpeechBrain (preferred if available)
# ──────────────────────────────────────────────────────────────────────────────

def try_load_speechbrain_xvector(device="cpu"):
    """Try to load SpeechBrain ECAPA-TDNN for x-vector extraction."""
    try:
        from speechbrain.pretrained import EncoderClassifier
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
        print("SpeechBrain ECAPA-TDNN loaded for x-vector extraction.")
        return classifier
    except Exception as e:
        print(f"SpeechBrain not available ({e}). Using LSTM d-vector instead.")
        return None


def extract_xvector_speechbrain(waveform, sr, classifier):
    """Extract x-vector using SpeechBrain ECAPA-TDNN."""
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).float()
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform.unsqueeze(0), sr, 16000).squeeze(0)
    embeddings = classifier.encode_batch(waveform.unsqueeze(0))
    return embeddings.squeeze().detach()


# ──────────────────────────────────────────────────────────────────────────────
# Main embedding extractor
# ──────────────────────────────────────────────────────────────────────────────

class SpeakerEmbeddingExtractor:
    """
    Extracts a robust speaker embedding from a 60s reference recording.

    Process:
      1. Load audio → resample to 16kHz
      2. Split into overlapping 1.6s windows
      3. Extract d-vector or x-vector per window
      4. Mean-pool across windows → final embedding
    """

    def __init__(self, device="cpu", model_path=None):
        self.device = device

        # prefer SpeechBrain x-vector
        self.xvec_model = try_load_speechbrain_xvector(device)

        if self.xvec_model is None:
            # fallback to LSTM d-vector
            self.dvec_model = DVectorEncoder().to(device)
            if model_path and Path(model_path).exists():
                self.dvec_model.load_state_dict(
                    torch.load(model_path, map_location=device)
                )
                print(f"Loaded d-vector weights from {model_path}")
            else:
                print("Warning: using untrained d-vector encoder (random weights).")
                print("For proper speaker cloning, train on VoxCeleb first.")
            self.dvec_model.eval()

    def extract_from_file(self, audio_path):
        """
        Load audio and extract speaker embedding.
        Returns (embedding, sample_rate) as (torch.Tensor, int).
        """
        sig, sr = sf.read(audio_path)
        if sig.ndim == 2:
            sig = sig.mean(axis=1)
        sig = sig.astype(np.float32)

        waveform = torch.from_numpy(sig)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform.unsqueeze(0), sr, SAMPLE_RATE
            ).squeeze(0)
            sr = SAMPLE_RATE

        embedding = self.extract_from_waveform(waveform, sr)
        return embedding, sr

    def extract_from_waveform(self, waveform, sr=SAMPLE_RATE):
        """
        waveform: 1D torch.Tensor at 16kHz
        Returns: (EMBEDDING_DIM,) normalized speaker embedding
        """
        if self.xvec_model is not None:
            return extract_xvector_speechbrain(waveform, sr, self.xvec_model)

        # LSTM d-vector path
        win_samples = int(WINDOW_SIZE_S * sr)
        hop_samples = int(HOP_SIZE_S * sr)

        segments = []
        start = 0
        while start + win_samples <= len(waveform):
            seg = waveform[start: start + win_samples]
            feats = extract_mfcc(seg, sr).to(self.device)
            emb = self.dvec_model.embed_segment(feats)
            segments.append(emb.cpu())
            start += hop_samples

        if not segments:
            # audio shorter than one window – process it directly
            feats = extract_mfcc(waveform, sr).to(self.device)
            return self.dvec_model.embed_segment(feats).cpu()

        # mean-pool and renormalize
        stacked = torch.stack(segments, dim=0)
        mean_emb = stacked.mean(dim=0)
        return F.normalize(mean_emb, p=2, dim=0)

    def save_embedding(self, embedding, save_path):
        torch.save(embedding, save_path)
        print(f"Embedding saved to {save_path} (shape: {embedding.shape})")

    def load_embedding(self, path):
        return torch.load(path, map_location="cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Cosine similarity between two embeddings
# ──────────────────────────────────────────────────────────────────────────────

def speaker_similarity(emb1, emb2):
    """Cosine similarity, returns float in [-1, 1]."""
    return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python voice_embedding.py <reference_audio.wav> [output_emb.pt]")
        print("Running architecture test instead...")
        model = DVectorEncoder()
        dummy = torch.randn(2, N_MFCC * 3, 100)
        out = model(dummy)
        print(f"D-vector shape: {out.shape}")  # should be (2, 256)
        print(f"L2 norms: {out.norm(dim=1).tolist()}")  # should be ~[1.0, 1.0]
        sys.exit(0)

    ref_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "models/speaker_embedding.pt"

    extractor = SpeakerEmbeddingExtractor()
    emb, sr = extractor.extract_from_file(ref_path)
    print(f"Embedding shape: {emb.shape}, norm: {emb.norm().item():.4f}")
    extractor.save_embedding(emb, out_path)
