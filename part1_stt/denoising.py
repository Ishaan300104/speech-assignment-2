"""
Task 1.3 – Audio Denoising & Normalization

Two methods implemented:
  1. Spectral Subtraction (primary) – classic but effective for stationary noise
  2. Wiener Filtering (secondary) – smoother output, used as optional second pass

Noise profile estimation:
  - First 0.5s if silence detected (energy below threshold)
  - Otherwise, minimum statistics over the whole signal (Martin 2001 approach)
"""

import numpy as np
import torch
import torchaudio
import soundfile as sf
from scipy.signal import butter, filtfilt


SAMPLE_RATE = 16000
FRAME_LEN   = 0.025   # 25ms
FRAME_SHIFT = 0.010   # 10ms
N_FFT       = 512


def frames_to_samples(duration_s, sr=SAMPLE_RATE):
    return int(duration_s * sr)


def stft(signal, n_fft=N_FFT, hop_length=None, win_length=None):
    """Short-time Fourier transform using torch."""
    hop = hop_length or n_fft // 4
    win = win_length or n_fft
    window = torch.hann_window(win)
    sig_t = torch.from_numpy(signal).float()
    S = torch.stft(sig_t, n_fft=n_fft, hop_length=hop,
                   win_length=win, window=window,
                   return_complex=True)
    return S  # (freq_bins, time_frames)


def istft(S, n_fft=N_FFT, hop_length=None, win_length=None, length=None):
    hop = hop_length or n_fft // 4
    win = win_length or n_fft
    window = torch.hann_window(win)
    return torch.istft(S, n_fft=n_fft, hop_length=hop,
                       win_length=win, window=window,
                       length=length).numpy()


# ──────────────────────────────────────────────────────────────────────────────
# Noise profile estimation
# ──────────────────────────────────────────────────────────────────────────────

def estimate_noise_profile(signal, sr=SAMPLE_RATE, noise_duration=0.5,
                            energy_threshold_db=-35.0):
    """
    Estimate noise power spectrum.
    First tries to find a silent segment at the beginning.
    Falls back to minimum statistics across all frames.
    """
    n_fft   = N_FFT
    hop     = n_fft // 4
    n_noise = frames_to_samples(noise_duration, sr)

    # check if the first 0.5s is actually quiet
    leading_chunk = signal[:n_noise]
    rms_db = 20 * np.log10(np.sqrt(np.mean(leading_chunk**2)) + 1e-9)

    if rms_db < energy_threshold_db:
        # use leading silence as noise reference
        noise_ref = leading_chunk
        S_noise   = stft(noise_ref, n_fft=n_fft, hop_length=hop)
        noise_psd = (S_noise.abs() ** 2).mean(dim=1, keepdim=True)
    else:
        # minimum statistics: take the minimum magnitude in each freq bin
        # over all frames (rough but works well for classroom noise)
        S_full    = stft(signal, n_fft=n_fft, hop_length=hop)
        mag_sq    = S_full.abs() ** 2  # (freq, frames)
        # minimum over time (biased estimator of noise floor)
        noise_psd = mag_sq.min(dim=1, keepdim=True).values
        # scale up slightly since min is an underestimate
        noise_psd = noise_psd * 1.5

    return noise_psd  # (freq_bins, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Spectral Subtraction
# ──────────────────────────────────────────────────────────────────────────────

def spectral_subtraction(signal, sr=SAMPLE_RATE, alpha=1.2, beta=0.002,
                          noise_estimation="auto"):
    """
    Classic spectral subtraction (Boll, 1979).

    alpha: over-subtraction factor (>1 subtracts more noise, but risks musical noise)
    beta:  spectral floor (prevents negative values; set between 0 and 0.1)

    musical noise suppression: half-wave rectification + spectral floor
    """
    n_fft = N_FFT
    hop   = n_fft // 4
    sig_len = len(signal)

    S = stft(signal, n_fft=n_fft, hop_length=hop)  # complex (freq, frames)
    magnitude = S.abs()  # (freq, frames)
    phase     = S / (magnitude + 1e-9)  # unit phasor

    noise_psd = estimate_noise_profile(signal, sr)
    noise_mag = noise_psd.sqrt()  # (freq, 1)

    # subtract: |S_clean| = max(|S| - alpha * |N|, beta * |S|)
    magnitude_clean = torch.clamp(
        magnitude - alpha * noise_mag,
        min=beta * magnitude
    )

    S_clean = magnitude_clean * phase

    signal_clean = istft(S_clean, n_fft=n_fft, hop_length=hop, length=sig_len)
    return signal_clean


# ──────────────────────────────────────────────────────────────────────────────
# Wiener Filter (optional second pass)
# ──────────────────────────────────────────────────────────────────────────────

def wiener_filter(signal, sr=SAMPLE_RATE, noise_estimation_frames=10):
    """
    Frequency-domain Wiener filter.
    H(f) = SNR(f) / (SNR(f) + 1)   where SNR is estimated per bin.
    """
    n_fft = N_FFT
    hop   = n_fft // 4
    sig_len = len(signal)

    S = stft(signal, n_fft=n_fft, hop_length=hop)
    psd_signal = S.abs() ** 2  # (freq, frames)

    noise_psd  = estimate_noise_profile(signal, sr)  # (freq, 1)
    speech_psd = torch.clamp(psd_signal - noise_psd, min=0)

    snr = speech_psd / (noise_psd + 1e-9)
    H   = snr / (snr + 1.0)  # Wiener gain, between 0 and 1

    S_filtered = H * S
    return istft(S_filtered, n_fft=n_fft, hop_length=hop, length=sig_len)


# ──────────────────────────────────────────────────────────────────────────────
# Pre-emphasis and normalization
# ──────────────────────────────────────────────────────────────────────────────

def pre_emphasis(signal, coef=0.97):
    """Boost high frequencies to compensate for the spectral tilt of speech."""
    return np.append(signal[0], signal[1:] - coef * signal[:-1])


def normalize_loudness(signal, target_rms=0.1):
    """RMS normalization."""
    rms = np.sqrt(np.mean(signal**2)) + 1e-9
    return signal * (target_rms / rms)


def highpass_filter(signal, sr=SAMPLE_RATE, cutoff_hz=80.0):
    """Remove low-frequency hum (HVAC noise, power line interference)."""
    nyq = sr / 2.0
    b, a = butter(4, cutoff_hz / nyq, btype="high")
    return filtfilt(b, a, signal)


# ──────────────────────────────────────────────────────────────────────────────
# Full denoising pipeline
# ──────────────────────────────────────────────────────────────────────────────

def denoise_audio(audio_path, output_path=None, use_wiener=True):
    """
    Full preprocessing pipeline:
      1. Load & resample to 16kHz
      2. High-pass filter (remove HVAC hum)
      3. Spectral subtraction
      4. Optional Wiener filter pass
      5. Loudness normalization
    """
    sig, sr = sf.read(audio_path)

    # convert stereo to mono
    if sig.ndim == 2:
        sig = sig.mean(axis=1)
    sig = sig.astype(np.float32)

    # resample if needed
    if sr != SAMPLE_RATE:
        sig_t = torch.from_numpy(sig).unsqueeze(0)
        sig_t = torchaudio.functional.resample(sig_t, sr, SAMPLE_RATE)
        sig = sig_t.squeeze(0).numpy()
        sr = SAMPLE_RATE

    # step 1: high-pass to kill hum
    sig = highpass_filter(sig, sr)

    # step 2: pre-emphasis
    sig = pre_emphasis(sig)

    # step 3: spectral subtraction
    sig = spectral_subtraction(sig, sr, alpha=1.3, beta=0.005)

    # step 4: optional Wiener pass
    if use_wiener:
        sig = wiener_filter(sig, sr)

    # step 5: normalize
    sig = normalize_loudness(sig, target_rms=0.08)

    if output_path:
        sf.write(output_path, sig, SAMPLE_RATE)
        print(f"Denoised audio saved to {output_path}")

    return sig, SAMPLE_RATE


def compute_snr(clean, noisy):
    """Estimate SNR improvement (reference-free heuristic)."""
    signal_power = np.mean(clean**2)
    diff         = clean - noisy[:len(clean)]
    noise_power  = np.mean(diff**2)
    snr_db       = 10 * np.log10(signal_power / (noise_power + 1e-9))
    return snr_db


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python denoising.py <input.wav> [output.wav]")
        sys.exit(0)

    inp  = sys.argv[1]
    outp = sys.argv[2] if len(sys.argv) > 2 else inp.replace(".wav", "_denoised.wav")
    clean_sig, _ = denoise_audio(inp, outp)
    print(f"Done. Output written to {outp}")
