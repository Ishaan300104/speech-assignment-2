"""
Task 1.1 – Multi-Head Frame-Level Language Identification

Architecture:
  - Backbone: wav2vec2-base (pretrained), upper 4 layers fine-tuned
  - Head A: phoneme-discriminative features (point-wise MLP)
  - Head B: prosody/rhythm features (Conv1D with wider receptive field)
  - Fusion: learned weighted combination → frame-level softmax

Output: per-frame language label  (0 = English, 1 = Hindi)
        at ~20ms resolution (wav2vec2 outputs ~50 frames/sec)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm


SAMPLE_RATE = 16000
LANG_ENGLISH = 0
LANG_HINDI = 1


class MultiHeadLID(nn.Module):
    def __init__(self, num_languages=2, hidden_dim=256, dropout=0.15):
        super().__init__()

        self.backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

        # freeze everything except the top 4 transformer encoder layers
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        for i in range(8, 12):
            for param in self.backbone.encoder.layers[i].parameters():
                param.requires_grad = True
        # also unfreeze the projection layer
        for param in self.backbone.feature_projection.parameters():
            param.requires_grad = True

        feat_dim = 768  # wav2vec2-base hidden size

        # Head A: phoneme-focused
        # narrow context, just a 2-layer MLP per frame
        self.head_phoneme = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_languages),
        )

        # Head B: prosody/rhythm-focused
        # Conv1d over the time axis captures ~300ms context (kernel=15 @ 50fps)
        self.prosody_conv = nn.Sequential(
            nn.Conv1d(feat_dim, hidden_dim, kernel_size=15, padding=7),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=7, padding=3),
            nn.GELU(),
        )
        self.head_prosody = nn.Linear(hidden_dim // 2, num_languages)

        # Learned fusion weights (scalar per head)
        self.fusion_weight = nn.Parameter(torch.tensor([0.5, 0.5]))

    def forward(self, waveform, attention_mask=None):
        """
        waveform: (B, T) float32, 16kHz
        returns:  (B, F, num_languages) logits  where F = num frames
        """
        out = self.backbone(waveform, attention_mask=attention_mask)
        h = out.last_hidden_state  # (B, F, 768)

        # Head A
        logits_a = self.head_phoneme(h)  # (B, F, 2)

        # Head B – need (B, 768, F) for Conv1d
        h_t = h.transpose(1, 2)
        prosody_feat = self.prosody_conv(h_t).transpose(1, 2)  # (B, F, 128)
        logits_b = self.head_prosody(prosody_feat)  # (B, F, 2)

        # weighted fusion
        w = F.softmax(self.fusion_weight, dim=0)
        logits = w[0] * logits_a + w[1] * logits_b  # (B, F, 2)

        return logits

    def predict_frames(self, waveform, device="cpu"):
        """Convenience: returns per-frame language labels as numpy array."""
        self.eval()
        with torch.no_grad():
            wav = waveform.to(device).unsqueeze(0) if waveform.dim() == 1 else waveform.to(device)
            logits = self(wav)
            preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy()
        return preds


# ──────────────────────────────────────────────────────────────────────────────
# Dataset for frame-level LID training
# ──────────────────────────────────────────────────────────────────────────────

class LIDDataset(Dataset):
    """
    Expects a list of dicts:
        {"audio": np.ndarray (16kHz), "frame_labels": np.ndarray (int, 0/1)}
    frame_labels length should match the number of wav2vec2 output frames.
    wav2vec2-base outputs 1 frame per 320 samples = 20ms at 16kHz.
    """

    def __init__(self, samples, max_duration_sec=10.0):
        self.samples = samples
        self.max_len = int(max_duration_sec * SAMPLE_RATE)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        audio = item["audio"].astype(np.float32)
        labels = item["frame_labels"].astype(np.int64)

        # truncate / pad audio
        if len(audio) > self.max_len:
            audio = audio[: self.max_len]
        else:
            audio = np.pad(audio, (0, self.max_len - len(audio)))

        # expected frames for wav2vec2-base (320-sample stride)
        n_frames = (self.max_len - 400) // 320 + 1  # approximate
        if len(labels) < n_frames:
            labels = np.pad(labels, (0, n_frames - len(labels)), constant_values=0)
        else:
            labels = labels[:n_frames]

        return torch.tensor(audio), torch.tensor(labels)


def train_lid_model(train_samples, val_samples, epochs=10, lr=2e-4,
                    batch_size=8, device="cuda", save_path="models/lid_weights.pt"):
    model = MultiHeadLID().to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-2
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(LIDDataset(train_samples), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(LIDDataset(val_samples),   batch_size=batch_size)

    best_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for wavs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            wavs, labels = wavs.to(device), labels.to(device)

            logits = model(wavs)  # (B, F, 2)

            # flatten for cross-entropy
            B, F, C = logits.shape
            loss = F.cross_entropy(logits.reshape(B * F, C), labels.reshape(B * F))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # ── validation ──
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for wavs, labels in val_loader:
                wavs = wavs.to(device)
                logits = model(wavs)
                preds = logits.argmax(dim=-1).cpu().numpy().flatten()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy().flatten())

        f1 = f1_score(all_labels, all_preds, average="macro")
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}  val_F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), save_path)
            print(f"  -> saved (best F1 so far: {best_f1:.4f})")

    print(f"\nTraining done. Best val F1: {best_f1:.4f}")
    return model


def load_lid_model(weights_path, device="cpu"):
    model = MultiHeadLID()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Post-processing: smooth frame predictions and extract switch timestamps
# ──────────────────────────────────────────────────────────────────────────────

def smooth_predictions(frame_preds, window=5):
    """Majority vote in a sliding window to reduce spurious switches."""
    smoothed = np.array(frame_preds, dtype=int)
    half = window // 2
    for i in range(half, len(frame_preds) - half):
        neighborhood = frame_preds[i - half: i + half + 1]
        smoothed[i] = int(np.bincount(neighborhood).argmax())
    return smoothed


def get_switch_timestamps(frame_preds, frame_duration_ms=20.0):
    """
    Returns list of (timestamp_ms, from_lang, to_lang) tuples
    where a code-switch was detected.
    """
    switches = []
    preds = smooth_predictions(frame_preds)
    for i in range(1, len(preds)):
        if preds[i] != preds[i - 1]:
            t_ms = i * frame_duration_ms
            switches.append((t_ms, preds[i - 1], preds[i]))
    return switches


def frames_to_segments(frame_preds, frame_duration_ms=20.0):
    """
    Converts frame-level predictions to language segments.
    Returns list of (start_ms, end_ms, lang_id).
    """
    if len(frame_preds) == 0:
        return []

    segments = []
    current_lang = frame_preds[0]
    seg_start = 0

    for i in range(1, len(frame_preds)):
        if frame_preds[i] != current_lang:
            segments.append((
                seg_start * frame_duration_ms,
                i * frame_duration_ms,
                current_lang
            ))
            current_lang = frame_preds[i]
            seg_start = i

    segments.append((
        seg_start * frame_duration_ms,
        len(frame_preds) * frame_duration_ms,
        current_lang
    ))
    return segments


if __name__ == "__main__":
    # Quick sanity check
    import soundfile as sf

    model = MultiHeadLID()
    print("Model parameters (trainable):",
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    # dummy forward pass
    dummy_wav = torch.randn(1, 16000 * 5)  # 5 seconds
    with torch.no_grad():
        logits = model(dummy_wav)
    print(f"Output shape: {logits.shape}")  # should be (1, ~250, 2)
