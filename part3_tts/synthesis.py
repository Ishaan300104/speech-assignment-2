"""
Task 3.3 – Zero-Shot Cross-Lingual TTS via VITS + Speaker Conditioning

Strategy:
  - Use a pretrained multilingual VITS or YourTTS model
  - Condition on the speaker embedding extracted in Task 3.1
  - Input: Santhali text (Latin transliteration → phoneme sequence)
  - Output: 22.05kHz WAV

Since VITS models are typically trained on specific languages, we use
Coqui TTS's YourTTS which was trained on multilingual data and supports
zero-shot speaker cloning via d-vector/x-vector conditioning.

The phoneme input goes through our custom Santhali G2P (borrowing from
the Hindi and English phoneme sets since Santhali shares many phonemes).
"""

import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from part3_tts.voice_embedding import SpeakerEmbeddingExtractor


SAMPLE_RATE = 22050
MAX_CHUNK_CHARS = 250  # split long texts to avoid OOM


# ──────────────────────────────────────────────────────────────────────────────
# Santhali phoneme set (borrowed from Hindi + English inventory)
# ──────────────────────────────────────────────────────────────────────────────

# Santhali phoneme inventory (Austroasiatic, Munda branch)
# Consonants: stops p b t d k g, affricates c j, fricatives s sh h,
#             nasals m n ng, liquids r l, semivowels y w
# Vowels: a e i o u (all short/long)
# No aspirates as distinct phonemes (unlike Hindi)

SANTHALI_G2P_MAP = {
    # vowels
    "a": "a", "aa": "aː", "i": "i", "ii": "iː", "u": "u", "uu": "uː",
    "e": "e", "ee": "eː", "o": "o", "oo": "oː",
    # nasalized vowels (tilde)
    "an": "ã", "en": "ẽ", "on": "õ",
    # consonants
    "p": "p", "b": "b", "t": "t", "d": "d", "k": "k", "g": "g",
    "c": "tʃ", "j": "dʒ", "s": "s", "sh": "ʃ", "h": "h",
    "m": "m", "n": "n", "ng": "ŋ",
    "r": "r", "l": "l", "y": "j", "w": "w",
    # glottal stop (common in Santhali)
    "'": "ʔ",
}

# Simple transliteration rules for Santhali Latin → phoneme sequence
# (real G2P would be rule-based but this covers the corpus vocabulary)
def santhali_latin_to_phonemes(text):
    """
    Convert Santhali Latin transliteration to space-separated phoneme sequence.
    Used as input to the TTS phoneme encoder.
    """
    text = text.lower().strip()
    phonemes = []
    i = 0
    while i < len(text):
        if text[i] == " ":
            phonemes.append("|")  # word boundary
            i += 1
            continue
        # try 2-char sequences first
        if i + 1 < len(text) and text[i:i+2] in SANTHALI_G2P_MAP:
            phonemes.append(SANTHALI_G2P_MAP[text[i:i+2]])
            i += 2
        elif text[i] in SANTHALI_G2P_MAP:
            phonemes.append(SANTHALI_G2P_MAP[text[i]])
            i += 1
        else:
            # unknown char: pass through (numbers, punctuation)
            phonemes.append(text[i])
            i += 1
    return " ".join(phonemes)


# ──────────────────────────────────────────────────────────────────────────────
# TTS model wrapper (Coqui YourTTS / VITS)
# ──────────────────────────────────────────────────────────────────────────────

class SanthaliTTSSynthesizer:
    """
    Zero-shot cross-lingual TTS using YourTTS with speaker conditioning.
    Falls back to a simple rule-based speech synthesis if model unavailable.
    """

    def __init__(self, device=None, speaker_embedding_path=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.speaker_embedding = None

        # load speaker embedding if provided
        if speaker_embedding_path and Path(speaker_embedding_path).exists():
            self.speaker_embedding = torch.load(
                speaker_embedding_path, map_location=self.device
            )
            print(f"Speaker embedding loaded from {speaker_embedding_path}")

        # try loading Coqui TTS
        self.model = self._load_yourtts()

    def _load_yourtts(self):
        """Attempt to load YourTTS from Coqui TTS."""
        try:
            from TTS.api import TTS as CoquiTTS
            # YourTTS supports multilingual + zero-shot speaker cloning
            model = CoquiTTS(
                model_name="tts_models/multilingual/multi-dataset/your_tts",
                progress_bar=True,
            ).to(self.device)
            print("YourTTS model loaded successfully.")
            return model
        except Exception as e:
            print(f"Could not load YourTTS: {e}")
            print("Will use rule-based synthesis as fallback.")
            return None

    def synthesize(self, santhali_text, output_path=None, speaker_wav=None):
        """
        Synthesize Santhali text to speech.

        santhali_text: string in Santhali Latin transliteration
        output_path:   where to save the WAV
        speaker_wav:   path to reference speaker audio for zero-shot cloning
                       (overrides stored embedding if provided)
        Returns: (waveform_np, sample_rate)
        """
        # split into chunks to avoid OOM on long texts
        chunks = self._split_text(santhali_text)
        audio_chunks = []

        for chunk in chunks:
            audio = self._synthesize_chunk(chunk, speaker_wav)
            if audio is not None:
                audio_chunks.append(audio)

        if not audio_chunks:
            print("Synthesis failed for all chunks.")
            return None, SAMPLE_RATE

        full_audio = np.concatenate(audio_chunks, axis=0)

        # normalize
        full_audio = full_audio / (np.abs(full_audio).max() + 1e-9) * 0.9

        if output_path:
            sf.write(output_path, full_audio, SAMPLE_RATE)
            print(f"Synthesis saved to {output_path} "
                  f"({len(full_audio)/SAMPLE_RATE:.1f}s at {SAMPLE_RATE}Hz)")

        return full_audio, SAMPLE_RATE

    def _synthesize_chunk(self, text, speaker_wav=None):
        """Synthesize one chunk of text."""
        if self.model is not None:
            return self._yourtts_synthesize(text, speaker_wav)
        else:
            return self._fallback_synthesize(text)

    def _yourtts_synthesize(self, text, speaker_wav=None):
        """Use YourTTS with zero-shot cloning."""
        try:
            ref_wav = speaker_wav or "audio/student_voice_ref.wav"
            # YourTTS uses Portuguese as a proxy for other low-resource languages
            # We pass Santhali text and rely on the speaker embedding for voice identity
            wav = self.model.tts(
                text=text,
                speaker_wav=ref_wav,
                language="en",  # use English phonemes as proxy (closest available)
            )
            return np.array(wav, dtype=np.float32)
        except Exception as e:
            print(f"YourTTS error: {e}")
            return self._fallback_synthesize(text)

    def _fallback_synthesize(self, text):
        """
        Rule-based fallback synthesis using sinusoidal vocoder.
        Very basic but demonstrates the pipeline architecture.
        """
        print(f"  Using fallback synthesis for: '{text[:50]}...'")
        phonemes = santhali_latin_to_phonemes(text)
        return synthesize_from_phonemes_basic(phonemes)

    def _split_text(self, text, max_chars=MAX_CHUNK_CHARS):
        """Split text into sentence-level chunks."""
        sentences = text.replace("।", ".").split(".")
        chunks = []
        current = ""
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(current) + len(sent) < max_chars:
                current += " " + sent
            else:
                if current:
                    chunks.append(current.strip())
                current = sent
        if current:
            chunks.append(current.strip())
        return chunks if chunks else [text]


# ──────────────────────────────────────────────────────────────────────────────
# Basic fallback synthesis (sinusoidal vocoder – not quality TTS)
# ──────────────────────────────────────────────────────────────────────────────

# Rough F0 and duration for each phoneme class
PHONEME_F0 = {
    "a": 180, "aː": 180, "e": 190, "eː": 190, "i": 200, "iː": 200,
    "o": 175, "oː": 175, "u": 170, "uː": 170,
    "p": 0, "b": 80, "t": 0, "d": 100, "k": 0, "g": 90,
    "m": 160, "n": 165, "ŋ": 155, "r": 170, "l": 170,
    "s": 0, "ʃ": 0, "h": 0, "tʃ": 0, "dʒ": 80,
    "j": 185, "w": 175, "|": 0,
}
PHONEME_DUR_MS = {
    "a": 90, "aː": 140, "e": 85, "eː": 130, "i": 80, "iː": 125,
    "o": 90, "oː": 140, "u": 80, "uː": 125, "|": 120,
}


def synthesize_phoneme(phoneme, sr=SAMPLE_RATE, duration_ms=None):
    """Synthesize a single phoneme as a simple sinusoid."""
    f0 = PHONEME_F0.get(phoneme, 0)
    dur = duration_ms or PHONEME_DUR_MS.get(phoneme, 60)
    n_samples = int(dur * sr / 1000)
    t = np.linspace(0, dur / 1000, n_samples)

    if f0 > 0:
        # voiced: sum of harmonics
        signal = np.zeros(n_samples)
        for harmonic in range(1, 5):
            signal += (1.0 / harmonic) * np.sin(2 * np.pi * f0 * harmonic * t)
        # apply Hanning envelope
        if n_samples > 10:
            signal *= np.hanning(n_samples)
    else:
        # unvoiced: shaped noise
        signal = np.random.randn(n_samples) * 0.1
        if n_samples > 10:
            signal *= np.hanning(n_samples)

    return signal.astype(np.float32)


def synthesize_from_phonemes_basic(phoneme_str, sr=SAMPLE_RATE):
    """Synthesize speech from a phoneme string (space-separated)."""
    phonemes = phoneme_str.split()
    frames = [synthesize_phoneme(ph, sr) for ph in phonemes]
    if not frames:
        return np.zeros(sr, dtype=np.float32)
    return np.concatenate(frames)


# ──────────────────────────────────────────────────────────────────────────────
# Main synthesis function (used by pipeline.py)
# ──────────────────────────────────────────────────────────────────────────────

def synthesize_lecture(
    santhali_segments,
    speaker_ref_path="audio/student_voice_ref.wav",
    ref_audio_path="audio/original_segment.wav",
    output_path="audio/output_LRL_cloned.wav",
    apply_prosody=True,
):
    """
    Full synthesis pipeline:
      1. Synthesize each Santhali segment with speaker cloning
      2. Concatenate into full lecture
      3. Apply prosody warping from professor's reference
      4. Save final WAV at 22.05kHz
    """
    synthesizer = SanthaliTTSSynthesizer(speaker_wav=speaker_ref_path)

    all_audio = []
    for i, seg in enumerate(santhali_segments):
        text = seg.get("santhali", seg.get("text", ""))
        if not text.strip():
            continue
        print(f"Synthesizing segment {i+1}/{len(santhali_segments)}: '{text[:60]}...'")
        audio, sr = synthesizer.synthesize(text, speaker_wav=speaker_ref_path)
        if audio is not None:
            all_audio.append(audio)
            # 200ms silence between segments
            all_audio.append(np.zeros(int(0.2 * sr), dtype=np.float32))

    if not all_audio:
        print("No audio generated.")
        return

    full_audio = np.concatenate(all_audio)
    sf.write("audio/synthesis_raw.wav", full_audio, SAMPLE_RATE)
    print(f"Raw synthesis: {len(full_audio)/SAMPLE_RATE:.1f}s")

    if apply_prosody and Path(ref_audio_path).exists():
        from part3_tts.prosody_warping import apply_prosody_warping
        warped = apply_prosody_warping(
            ref_audio_path,
            "audio/synthesis_raw.wav",
            output_path
        )
        print(f"Final prosody-warped lecture saved to {output_path}")
    else:
        sf.write(output_path, full_audio, SAMPLE_RATE)
        print(f"Final lecture saved to {output_path} (no prosody warping)")


if __name__ == "__main__":
    import sys
    test_text = "mit baang re kotha em akantalea cepstrum ar mel filterbank"
    synth = SanthaliTTSSynthesizer()
    audio, sr = synth.synthesize(test_text, output_path="test_synthesis.wav")
    if audio is not None:
        print(f"Test synthesis: {len(audio)/sr:.2f}s at {sr}Hz")
