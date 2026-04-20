"""
Task 3.2 – Prosody Warping via Dynamic Time Warping

Goal: transfer the "teaching style" prosody from the professor's original lecture
      onto the synthesized Santhali speech.

What gets transferred:
  - F0 (fundamental frequency / pitch contour)
  - Energy (RMS amplitude contour)
  - Duration scaling (speaking rate)

Pipeline:
  1. Extract F0 + energy from reference (professor's lecture)
  2. Extract F0 + energy from synthesized output (flat/neutral TTS)
  3. DTW to align the two time axes
  4. Warp the synthesized speech to match the reference prosody
     (pitch shifting + amplitude scaling frame-by-frame)
"""

import numpy as np
import torch
import torchaudio
import soundfile as sf
from scipy.signal import medfilt
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d


SAMPLE_RATE    = 22050   # VITS outputs at 22.05kHz
HOP_LENGTH     = 256     # ~11.6ms per frame at 22050Hz
WIN_LENGTH     = 1024
N_FFT          = 1024

# F0 extraction range (Hz)
F0_MIN = 60.0
F0_MAX = 400.0


# ──────────────────────────────────────────────────────────────────────────────
# F0 extraction using PYIN (more robust than autocorrelation)
# ──────────────────────────────────────────────────────────────────────────────

def extract_f0_pyin(waveform, sr=SAMPLE_RATE):
    """
    Extract F0 using PYIN algorithm via torchaudio.
    Returns (f0_hz, voiced_flags) both as numpy arrays of shape (T,).
    Unvoiced frames have f0 = 0.0.
    """
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).float()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # torchaudio PYIN
    f0, voiced_flag, voiced_prob = torchaudio.functional.detect_pitch_frequency(
        waveform,
        sample_rate=sr,
        frame_time=HOP_LENGTH / sr,
        win_length=WIN_LENGTH // HOP_LENGTH,
        freq_low=F0_MIN,
        freq_high=F0_MAX,
    )

    f0_np = f0.squeeze().numpy()
    voiced_np = voiced_flag.squeeze().numpy().astype(bool)
    f0_np[~voiced_np] = 0.0
    return f0_np, voiced_np


def extract_f0_autocorr(waveform, sr=SAMPLE_RATE):
    """
    Fallback F0 extraction using autocorrelation (simpler but noisier).
    Used if PYIN isn't available.
    """
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()

    f0_list = []
    voiced_list = []
    hop = HOP_LENGTH
    win = WIN_LENGTH

    min_period = int(sr / F0_MAX)
    max_period = int(sr / F0_MIN)

    for start in range(0, len(waveform) - win, hop):
        frame = waveform[start: start + win]
        frame = frame - frame.mean()  # remove DC

        # normalized autocorrelation
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2:]
        corr /= (corr[0] + 1e-9)

        # find peak in valid range
        search = corr[min_period: max_period]
        if len(search) == 0 or search.max() < 0.3:
            f0_list.append(0.0)
            voiced_list.append(False)
        else:
            peak_idx = search.argmax() + min_period
            f0_list.append(sr / peak_idx)
            voiced_list.append(True)

    return np.array(f0_list), np.array(voiced_list, dtype=bool)


# ──────────────────────────────────────────────────────────────────────────────
# Energy extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_energy(waveform, hop=HOP_LENGTH, win=WIN_LENGTH):
    """RMS energy per frame. Returns numpy array of shape (T,)."""
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()

    energy = []
    for start in range(0, len(waveform) - win, hop):
        frame = waveform[start: start + win]
        rms = np.sqrt(np.mean(frame ** 2))
        energy.append(rms)
    return np.array(energy)


# ──────────────────────────────────────────────────────────────────────────────
# DTW alignment
# ──────────────────────────────────────────────────────────────────────────────

def dtw_align(ref_seq, synth_seq):
    """
    DTW alignment between two 1D sequences.
    Returns (ref_path, synth_path) index arrays.

    Uses Euclidean distance between scalar values.
    This is the standard O(NM) DP implementation.
    """
    N = len(ref_seq)
    M = len(synth_seq)

    # cost matrix
    D = np.full((N + 1, M + 1), np.inf)
    D[0, 0] = 0.0

    ref_n   = (ref_seq - ref_seq.mean()) / (ref_seq.std() + 1e-9)
    synth_n = (synth_seq - synth_seq.mean()) / (synth_seq.std() + 1e-9)

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            cost = abs(ref_n[i - 1] - synth_n[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    # backtrack
    i, j = N, M
    path_ref, path_synth = [], []
    while i > 0 and j > 0:
        path_ref.append(i - 1)
        path_synth.append(j - 1)
        choices = [D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]]
        move = np.argmin(choices)
        if move == 0:
            i -= 1
        elif move == 1:
            j -= 1
        else:
            i -= 1
            j -= 1

    return np.array(path_ref[::-1]), np.array(path_synth[::-1])


def dtw_align_fast(ref_seq, synth_seq):
    """
    Faster DTW using scipy (if available), otherwise falls back to manual.
    """
    try:
        from scipy.spatial.distance import cdist
        # vectorized approach
        ref_n   = (ref_seq - ref_seq.mean()) / (ref_seq.std() + 1e-9)
        synth_n = (synth_seq - synth_seq.mean()) / (synth_seq.std() + 1e-9)

        # For long sequences, use a Sakoe-Chiba band
        N, M = len(ref_n), len(synth_n)
        band  = max(int(0.1 * max(N, M)), 50)  # 10% band or minimum 50 frames

        D = np.full((N + 1, M + 1), np.inf)
        D[0, 0] = 0.0
        for i in range(1, N + 1):
            j_start = max(1, i - band)
            j_end   = min(M + 1, i + band + 1)
            for j in range(j_start, j_end):
                cost = abs(ref_n[i - 1] - synth_n[j - 1])
                D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

        # backtrack
        i, j = N, M
        path_ref, path_synth = [], []
        while i > 0 and j > 0:
            path_ref.append(i - 1)
            path_synth.append(j - 1)
            choices = [D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]]
            move = np.argmin(choices)
            if move == 0:
                i -= 1
            elif move == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
        return np.array(path_ref[::-1]), np.array(path_synth[::-1])

    except Exception:
        return dtw_align(ref_seq, synth_seq)


# ──────────────────────────────────────────────────────────────────────────────
# Prosody transfer
# ──────────────────────────────────────────────────────────────────────────────

def interpolate_f0(f0, voiced):
    """Linear interpolation over unvoiced regions for smooth prosody transfer."""
    f0_interp = f0.copy()
    voiced_idx = np.where(voiced)[0]
    if len(voiced_idx) < 2:
        return f0_interp

    interp_fn = interp1d(
        voiced_idx, f0[voiced_idx],
        kind="linear", bounds_error=False,
        fill_value=(f0[voiced_idx[0]], f0[voiced_idx[-1]])
    )
    unvoiced_idx = np.where(~voiced)[0]
    f0_interp[unvoiced_idx] = interp_fn(unvoiced_idx)
    return f0_interp


def warp_prosody(synth_waveform, ref_f0, ref_energy, synth_f0, synth_energy,
                 synth_voiced, sr=SAMPLE_RATE):
    """
    Apply prosody warping to the synthesized waveform.

    Steps:
      1. Align ref and synth prosody via DTW on F0 contours
      2. For each synth frame, compute the target F0 from the aligned ref
      3. Shift pitch using PSOLA (approximated via phase vocoder)
      4. Scale energy to match reference

    Returns warped waveform as numpy array.
    """
    # smooth f0 contours first
    ref_f0_s   = medfilt(ref_f0,   kernel_size=5)
    synth_f0_s = medfilt(synth_f0, kernel_size=5)

    # interpolate over unvoiced regions for DTW
    ref_voiced   = ref_f0_s > 0
    synth_voiced_mask = synth_f0_s > 0
    ref_f0_interp   = interpolate_f0(ref_f0_s, ref_voiced)
    synth_f0_interp = interpolate_f0(synth_f0_s, synth_voiced_mask)

    # DTW alignment
    path_ref, path_synth = dtw_align_fast(ref_f0_interp, synth_f0_interp)

    # build target F0 for each synth frame
    target_f0 = np.zeros(len(synth_f0))
    for pr, ps in zip(path_ref, path_synth):
        if ps < len(target_f0):
            target_f0[ps] = ref_f0_interp[pr]

    # fill any zeros by interpolation
    nonzero = np.where(target_f0 > 0)[0]
    if len(nonzero) >= 2:
        fn = interp1d(nonzero, target_f0[nonzero], bounds_error=False,
                      fill_value=(target_f0[nonzero[0]], target_f0[nonzero[-1]]))
        zero_idx = np.where(target_f0 == 0)[0]
        target_f0[zero_idx] = fn(zero_idx)

    # pitch shift via STFT phase vocoder
    warped = pitch_shift_stft(synth_waveform, synth_f0_s, target_f0, sr)

    # energy warping: scale amplitude frame-by-frame
    warped = energy_warp(warped, ref_energy, synth_energy, sr)

    return warped


def pitch_shift_stft(waveform, source_f0, target_f0, sr=SAMPLE_RATE):
    """
    Pitch shift using STFT magnitude manipulation.
    For each voiced frame, shift the pitch by the ratio target_f0 / source_f0.
    """
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()

    hop  = HOP_LENGTH
    nfft = N_FFT

    # STFT
    window = np.hanning(nfft)
    spec   = []
    for start in range(0, len(waveform) - nfft, hop):
        frame = waveform[start: start + nfft] * window
        spec.append(np.fft.rfft(frame))
    spec = np.array(spec)  # (T, freq_bins)

    # shift each frame's spectrum
    freq_bins = spec.shape[1]
    shifted_spec = spec.copy()

    for t in range(min(len(source_f0), len(spec))):
        src = source_f0[t] if t < len(source_f0) else 0
        tgt = target_f0[t] if t < len(target_f0) else 0
        if src > F0_MIN and tgt > F0_MIN:
            ratio = tgt / src
            # shift spectrum by interpolating magnitudes
            mag   = np.abs(spec[t])
            phase = np.angle(spec[t])
            new_mag = np.interp(
                np.arange(freq_bins) / ratio,
                np.arange(freq_bins),
                mag,
                left=0, right=0
            )
            shifted_spec[t] = new_mag * np.exp(1j * phase)

    # iSTFT via overlap-add
    output = np.zeros(len(waveform))
    for t, frame_spec in enumerate(shifted_spec):
        start = t * hop
        frame = np.fft.irfft(frame_spec)[:nfft] * window
        end = min(start + nfft, len(output))
        output[start:end] += frame[:end - start]

    return output


def energy_warp(waveform, ref_energy, synth_energy, sr=SAMPLE_RATE):
    """Scale amplitude of each frame to match reference energy."""
    if isinstance(waveform, np.ndarray):
        waveform = waveform.copy()
    else:
        waveform = waveform.numpy()

    hop = HOP_LENGTH
    win = WIN_LENGTH

    # interpolate ref_energy to match synth length
    T_synth = (len(waveform) - win) // hop
    if len(ref_energy) != T_synth:
        x_old = np.linspace(0, 1, len(ref_energy))
        x_new = np.linspace(0, 1, T_synth)
        ref_e_interp = np.interp(x_new, x_old, ref_energy)
    else:
        ref_e_interp = ref_energy

    for t in range(min(T_synth, len(synth_energy))):
        start = t * hop
        end   = start + win
        if end > len(waveform):
            break
        synth_rms = synth_energy[t] + 1e-9
        ref_rms   = ref_e_interp[t] + 1e-9
        gain = ref_rms / synth_rms
        # clip gain to avoid extreme scaling
        gain = np.clip(gain, 0.1, 10.0)
        waveform[start:end] *= gain

    return waveform


# ──────────────────────────────────────────────────────────────────────────────
# High-level API
# ──────────────────────────────────────────────────────────────────────────────

def apply_prosody_warping(ref_audio_path, synth_audio_path, output_path=None):
    """
    Full prosody warping pipeline.
    ref_audio_path:   professor's original lecture segment
    synth_audio_path: flat TTS synthesis of the Santhali translation
    output_path:      where to save the warped audio (optional)
    """
    ref_sig,   ref_sr   = sf.read(ref_audio_path)
    synth_sig, synth_sr = sf.read(synth_audio_path)

    if ref_sig.ndim == 2:   ref_sig   = ref_sig.mean(axis=1)
    if synth_sig.ndim == 2: synth_sig = synth_sig.mean(axis=1)

    ref_sig   = ref_sig.astype(np.float32)
    synth_sig = synth_sig.astype(np.float32)

    # resample to 22050 if needed
    def maybe_resample(sig, sr_in, sr_out=SAMPLE_RATE):
        if sr_in == sr_out:
            return sig
        t = torch.from_numpy(sig).unsqueeze(0)
        t = torchaudio.functional.resample(t, sr_in, sr_out)
        return t.squeeze(0).numpy()

    ref_sig   = maybe_resample(ref_sig,   ref_sr)
    synth_sig = maybe_resample(synth_sig, synth_sr)

    print("Extracting reference prosody...")
    try:
        ref_f0, ref_voiced = extract_f0_pyin(ref_sig)
    except Exception:
        ref_f0, ref_voiced = extract_f0_autocorr(ref_sig)
    ref_energy = extract_energy(ref_sig)

    print("Extracting synthesis prosody...")
    try:
        synth_f0, synth_voiced = extract_f0_pyin(synth_sig)
    except Exception:
        synth_f0, synth_voiced = extract_f0_autocorr(synth_sig)
    synth_energy = extract_energy(synth_sig)

    print("Warping prosody...")
    warped = warp_prosody(
        synth_sig, ref_f0, ref_energy, synth_f0, synth_energy, synth_voiced
    )

    # normalize output
    warped = warped / (np.abs(warped).max() + 1e-9) * 0.9

    if output_path:
        sf.write(output_path, warped, SAMPLE_RATE)
        print(f"Prosody-warped audio saved to {output_path}")

    return warped


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        ref_path   = sys.argv[1]
        synth_path = sys.argv[2]
        out_path   = sys.argv[3] if len(sys.argv) > 3 else "output_warped.wav"
        apply_prosody_warping(ref_path, synth_path, out_path)
    else:
        print("Usage: python prosody_warping.py <ref.wav> <synth.wav> [output.wav]")
        print("Testing DTW with dummy data...")
        ref   = np.sin(np.linspace(0, 4 * np.pi, 200)) * 100 + 150
        synth = np.sin(np.linspace(0, 4 * np.pi, 180)) * 80  + 140
        pr, ps = dtw_align_fast(ref, synth)
        print(f"DTW path length: {len(pr)}")
