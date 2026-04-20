"""
Task 4.1 – Anti-Spoofing Countermeasure (CM) System

Architecture:
  - Feature: 60-dim LFCC (Linear Frequency Cepstral Coefficients)
             + delta + delta-delta = 180 features per frame
  - Model: LCNN (Light CNN) binary classifier
           Bona Fide (0) vs. Spoof (1)
  - Evaluation: Equal Error Rate (EER)

LFCC vs MFCC for anti-spoofing:
  LFCC uses a linear filter bank instead of mel-spaced, which makes it
  more sensitive to fine spectral artifacts in synthesized/replayed audio.
  This is why ASVspoof evaluations use LFCC as a standard feature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import soundfile as sf
from scipy.signal import lfilter
from sklearn.metrics import roc_curve
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path


SAMPLE_RATE  = 16000
N_LFCC       = 60
N_LINEAR     = 512   # linear filterbank bins
FRAME_LEN_MS = 25
FRAME_SH_MS  = 10
LABEL_BONA   = 0
LABEL_SPOOF  = 1


# ──────────────────────────────────────────────────────────────────────────────
# LFCC feature extraction
# ──────────────────────────────────────────────────────────────────────────────

def linear_filterbank(n_filters, n_fft, sr):
    """
    Build a linear-spaced filterbank matrix.
    Unlike mel, the filters are evenly spaced in Hz (not log-Hz).
    Returns: (n_filters, n_fft//2 + 1) matrix
    """
    freq_bins = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2, freq_bins)

    # filter centers: linear spacing from 0 to sr/2
    centers = np.linspace(0, sr / 2, n_filters + 2)

    fb = np.zeros((n_filters, freq_bins))
    for i in range(1, n_filters + 1):
        left   = centers[i - 1]
        center = centers[i]
        right  = centers[i + 1]
        # triangular filter
        for j, f in enumerate(freqs):
            if left <= f < center:
                fb[i - 1, j] = (f - left) / (center - left + 1e-9)
            elif center <= f <= right:
                fb[i - 1, j] = (right - f) / (right - center + 1e-9)

    return fb.astype(np.float32)


def extract_lfcc(waveform, sr=SAMPLE_RATE, n_lfcc=N_LFCC, n_linear=N_LINEAR):
    """
    Extract LFCC features.
    Returns: (3 * n_lfcc, T) array  [LFCC + delta + delta-delta]
    """
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=0)

    sr_int    = int(sr)
    n_fft     = 512
    hop_len   = int(FRAME_SH_MS * sr / 1000)
    win_len   = int(FRAME_LEN_MS * sr / 1000)

    # STFT magnitude
    wt = torch.from_numpy(waveform).float()
    window = torch.hann_window(win_len)
    S = torch.stft(wt, n_fft=n_fft, hop_length=hop_len,
                   win_length=win_len, window=window,
                   return_complex=True)
    power_spec = S.abs() ** 2  # (freq_bins, T)

    # apply linear filterbank
    fb = torch.from_numpy(linear_filterbank(n_linear, n_fft, sr_int))  # (n_linear, freq_bins)
    linear_spec = torch.matmul(fb, power_spec)  # (n_linear, T)
    log_spec    = torch.log(linear_spec + 1e-9)

    # DCT to get LFCC (same as MFCC but on linear filterbank)
    # Use torch DCT-II approximation
    n_lin = log_spec.shape[0]
    T     = log_spec.shape[1]

    # Build DCT matrix
    dct_mat = torch.zeros(n_lfcc, n_lin)
    for k in range(n_lfcc):
        for n in range(n_lin):
            dct_mat[k, n] = np.cos(np.pi * k * (2 * n + 1) / (2 * n_lin))
    dct_mat /= n_lin

    lfcc = torch.matmul(dct_mat, log_spec)  # (n_lfcc, T)

    # compute deltas
    delta1 = torchaudio.functional.compute_deltas(lfcc.unsqueeze(0)).squeeze(0)
    delta2 = torchaudio.functional.compute_deltas(delta1.unsqueeze(0)).squeeze(0)

    features = torch.cat([lfcc, delta1, delta2], dim=0)  # (3*n_lfcc, T)
    return features.numpy()


# ──────────────────────────────────────────────────────────────────────────────
# LCNN (Light CNN) classifier
# ──────────────────────────────────────────────────────────────────────────────

class MaxFeatureMap(nn.Module):
    """Max Feature Map (MFM) activation – key component of LCNN."""
    def forward(self, x):
        # split channels in half and take element-wise max
        half = x.shape[1] // 2
        return torch.max(x[:, :half], x[:, half:])


class LCNN(nn.Module):
    """
    Light CNN for anti-spoofing, following Wu et al. (2020).
    Input: (batch, 1, n_lfcc*3, T) spectrogram-like feature map
    Output: (batch, 2) logits [bona_fide, spoof]
    """

    def __init__(self, input_channels=1, feat_dim=N_LFCC * 3):
        super().__init__()

        self.features = nn.Sequential(
            # block 1
            nn.Conv2d(input_channels, 64, kernel_size=5, padding=2),
            MaxFeatureMap(),                        # → 32 channels
            nn.MaxPool2d(kernel_size=2, stride=2),

            # block 2
            nn.Conv2d(32, 64, kernel_size=1),
            MaxFeatureMap(),                        # → 32
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 128, kernel_size=3, padding=1),
            MaxFeatureMap(),                        # → 64
            nn.MaxPool2d(kernel_size=2, stride=2),

            # block 3
            nn.Conv2d(64, 128, kernel_size=1),
            MaxFeatureMap(),                        # → 64
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            MaxFeatureMap(),                        # → 32
            nn.MaxPool2d(kernel_size=2, stride=2),

            # block 4
            nn.Conv2d(32, 64, kernel_size=1),
            MaxFeatureMap(),                        # → 32
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            MaxFeatureMap(),                        # → 32
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.75),
            nn.Linear(32 * self._calc_flat_dim(feat_dim), 160),
            MaxFeatureMap(),  # → 80
            nn.Linear(80, 2),
        )

    def _calc_flat_dim(self, h):
        # after 4 MaxPool2d(2,2) with input height h
        # (assuming T dimension gets pooled too, we average over T)
        # returns spatial height after 4 halvings
        for _ in range(4):
            h = h // 2
        return h

    def forward(self, x):
        """x: (B, 1, feat_dim, T)"""
        h = self.features(x)
        # average pool over time dimension to handle variable-length input
        h = h.mean(dim=-1, keepdim=True)  # (B, C, H, 1)
        out = self.classifier(h)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class AntiSpoofDataset(Dataset):
    """
    Expects list of {"audio_path": str, "label": 0 or 1} dicts.
    LABEL_BONA = 0, LABEL_SPOOF = 1
    """

    def __init__(self, samples, max_frames=300):
        self.samples    = samples
        self.max_frames = max_frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        sig, sr = sf.read(item["audio_path"])
        if sig.ndim == 2:
            sig = sig.mean(axis=1)
        sig = sig.astype(np.float32)

        feats = extract_lfcc(sig, sr)  # (180, T)

        # pad/trim time dimension
        T = feats.shape[1]
        if T < self.max_frames:
            feats = np.pad(feats, ((0, 0), (0, self.max_frames - T)))
        else:
            feats = feats[:, :self.max_frames]

        # add channel dim → (1, 180, max_frames)
        feats_t = torch.from_numpy(feats).float().unsqueeze(0)
        label   = torch.tensor(item["label"], dtype=torch.long)
        return feats_t, label


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_anti_spoofing(train_samples, val_samples, epochs=20, lr=1e-3,
                        batch_size=16, device="cuda",
                        save_path="models/anti_spoof_weights.pt"):
    model = LCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_loader = DataLoader(AntiSpoofDataset(train_samples),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(AntiSpoofDataset(val_samples),
                              batch_size=batch_size)

    best_eer = 1.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for feats, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            feats, labels = feats.to(device), labels.to(device)
            logits = model(feats)
            loss   = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # EER on validation set
        eer = evaluate_eer(model, val_loader, device)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}  val_EER={eer*100:.2f}%")

        if eer < best_eer:
            best_eer = eer
            torch.save(model.state_dict(), save_path)
            print(f"  -> saved (best EER: {best_eer*100:.2f}%)")

    print(f"\nBest val EER: {best_eer*100:.2f}%")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# EER computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_eer(scores, labels):
    """
    Compute Equal Error Rate.
    scores: numpy array of spoof scores (higher = more likely spoof)
    labels: numpy array of ground truth (0=bona, 1=spoof)
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    # EER is where FPR ≈ FNR
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    threshold_at_eer = thresholds[eer_idx]
    return eer, threshold_at_eer


def evaluate_eer(model, data_loader, device="cpu"):
    """Run model on data_loader and compute EER."""
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for feats, labels in data_loader:
            feats = feats.to(device)
            logits = model(feats)
            # spoof score = softmax probability of class 1 (spoof)
            scores = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend(labels.numpy())

    eer, _ = compute_eer(np.array(all_scores), np.array(all_labels))
    return eer


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

def load_anti_spoof_model(weights_path, device="cpu"):
    model = LCNN()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def classify_audio(model, audio_path, device="cpu"):
    """
    Classify a single audio file as bona fide or spoof.
    Returns {"label": "bona_fide"/"spoof", "score": float, "eer_threshold": float}
    """
    sig, sr = sf.read(audio_path)
    if sig.ndim == 2:
        sig = sig.mean(axis=1)
    sig = sig.astype(np.float32)

    feats = extract_lfcc(sig, sr)  # (180, T)
    # pad to fixed length
    max_frames = 300
    T = feats.shape[1]
    if T < max_frames:
        feats = np.pad(feats, ((0, 0), (0, max_frames - T)))
    else:
        feats = feats[:, :max_frames]

    feats_t = torch.from_numpy(feats).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(feats_t)
        probs  = F.softmax(logits, dim=1).squeeze()

    spoof_score = probs[1].item()
    label = "spoof" if spoof_score > 0.5 else "bona_fide"

    return {"label": label, "spoof_score": spoof_score, "bona_score": probs[0].item()}


if __name__ == "__main__":
    # Test LFCC extraction
    print("Testing LFCC extraction...")
    dummy_wav = np.random.randn(16000 * 3).astype(np.float32)
    feats = extract_lfcc(dummy_wav)
    print(f"LFCC features shape: {feats.shape}")  # should be (180, T)

    # Test LCNN forward pass
    print("Testing LCNN forward pass...")
    model = LCNN()
    dummy_input = torch.randn(2, 1, N_LFCC * 3, 300)
    out = model(dummy_input)
    print(f"LCNN output shape: {out.shape}")  # should be (2, 2)

    # Test EER calculation
    np.random.seed(42)
    scores  = np.concatenate([np.random.normal(0.3, 0.15, 500),
                               np.random.normal(0.7, 0.15, 500)])
    labels  = np.concatenate([np.zeros(500), np.ones(500)]).astype(int)
    eer, thr = compute_eer(scores, labels)
    print(f"Dummy EER: {eer*100:.1f}% at threshold {thr:.3f}")
