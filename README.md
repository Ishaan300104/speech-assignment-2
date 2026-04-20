# Speech Understanding – Programming Assignment 2
## Code-Switched Transcription + LRL Voice Cloning Pipeline

This repo contains my implementation for PA2. The pipeline does four things:
1. Transcribes Hinglish (code-switched Hindi-English) lecture audio
2. Converts the transcript to IPA and translates it into Santhali (my chosen LRL)
3. Clones the professor's teaching style onto my own voice and synthesizes the Santhali lecture
4. Tests robustness against spoofing and adversarial noise

Chosen Low-Resource Language: **Santhali** (Santali, ISO 639-3: `sat`) – an Austroasiatic language spoken primarily in Jharkhand, Odisha, and West Bengal. I picked it because it has very limited digital resources, making it a realistic LRL challenge.

---

## Directory Structure

```
SpeechAssignment2/
├── pipeline.py                  # End-to-end pipeline runner
├── requirements.txt
├── environment.yml
│
├── part1_stt/
│   ├── lid_model.py             # Task 1.1 – Multi-head frame-level LID
│   ├── constrained_decoding.py  # Task 1.2 – Whisper + N-gram logit bias
│   └── denoising.py             # Task 1.3 – Spectral subtraction denoiser
│
├── part2_phonetic/
│   ├── g2p_hinglish.py          # Task 2.1 – Hinglish → IPA conversion
│   └── translation.py           # Task 2.2 – IPA/text → Santhali
│
├── part3_tts/
│   ├── voice_embedding.py       # Task 3.1 – d-vector / x-vector extraction
│   ├── prosody_warping.py       # Task 3.2 – F0 + energy DTW warping
│   └── synthesis.py             # Task 3.3 – VITS-based synthesis
│
├── part4_adversarial/
│   ├── anti_spoofing.py         # Task 4.1 – LFCC-based CM classifier
│   └── adversarial_attack.py    # Task 4.2 – FGSM perturbation on LID
│
├── utils/
│   ├── audio_utils.py           # Shared audio I/O helpers
│   └── metrics.py               # WER, MCD, EER computations
│
├── data/
│   ├── ngram_lm/
│   │   └── syllabus_terms.txt   # Technical terms for N-gram LM
│   └── parallel_corpus/
│       └── santhali_corpus.json # 500-word Hinglish↔Santhali corpus
│
├── audio/
│   ├── original_segment.wav     # Source lecture snippet (10 min)
│   ├── student_voice_ref.wav    # My 60s reference recording
│   └── output_LRL_cloned.wav    # Final synthesized Santhali lecture
│
└── models/                      # Saved model checkpoints (not tracked in git)
    ├── lid_weights.pt
    └── anti_spoof_weights.pt
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate speech_pa2

# or with pip:
pip install -r requirements.txt
```

---

## Running the Pipeline

```bash
# Full end-to-end
python pipeline.py --input audio/original_segment.wav \
                   --voice_ref audio/student_voice_ref.wav \
                   --output audio/output_LRL_cloned.wav

# Individual parts
python pipeline.py --part 1 --input audio/original_segment.wav   # STT only
python pipeline.py --part 2 --transcript transcript.txt           # Phonetic mapping
python pipeline.py --part 3 --voice_ref audio/student_voice_ref.wav  # TTS
python pipeline.py --part 4 --eval                                 # Adversarial eval
```

---

## Task-wise Notes

### Task 1.1 – Language ID
- Backbone: `facebook/wav2vec2-base` (pretrained, upper layers fine-tuned)
- Two attention heads: one for phoneme-discriminative features (narrow temporal window), one for prosody/rhythm (wider context via Conv1D)
- Frame-level output at ~20ms resolution (50 frames/sec)
- Trained on MUCS dataset + synthetic Hinglish mixtures

### Task 1.2 – Constrained Decoding
- Model: `openai/whisper-large-v3`
- Custom trigram LM trained on the Speech Understanding course syllabus
- Logit bias applied at each decoding step – technical terms (cepstrum, stochastic, mel-filterbank etc.) get a +2.0 log-prob boost
- Beam search width = 5

### Task 1.3 – Denoising
- Spectral subtraction with adaptive noise floor estimation
- Noise profile estimated from first 0.5s of silence / low-energy frames
- Wiener filter as optional second stage

### Task 2.1 – Hinglish G2P
- Hindi (Devanagari) → IPA: rule-based mapping table covering all Hindi phonemes
- English → IPA: CMU Pronouncing Dictionary lookup with fallback rules
- Code-switch detection: Unicode range check (Devanagari = U+0900-U+097F)

### Task 2.2 – Santhali Translation
- Created a 500-term parallel corpus covering speech/NLP technical vocabulary
- Rule-based morphological adaptation for untranslatable technical terms

### Task 3.1 – Voice Embedding
- d-vector extraction using a 3-layer LSTM trained on speaker verification
- Input: 40-dim MFCC, output: 256-dim speaker embedding
- 60s reference audio → mean-pooled embedding

### Task 3.2 – Prosody Warping
- F0 extraction: PYIN algorithm (more robust than autocorrelation for speech)
- Energy: frame-level RMS
- DTW (scipy) to align professor's prosody contour onto synthesis duration

### Task 3.3 – Synthesis
- VITS model with speaker embedding conditioning
- Output: 22050 Hz mono WAV
- Used Coqui TTS as the base framework

### Task 4.1 – Anti-Spoofing
- 60-dim LFCC features (40 LFCC + delta + delta-delta)
- LCNN (Light CNN) binary classifier: Bona Fide vs. Spoof
- EER computed on held-out test set

### Task 4.2 – FGSM Attack
- Targeted FGSM on raw waveform features
- Target: make LID misclassify Hindi frames as English
- Reports minimum epsilon where misclassification occurs (SNR > 40dB constraint)

---

## Results Summary

| Metric                  | Target    | Achieved  |
|-------------------------|-----------|-----------|
| WER (English)           | < 15%     | 12.3%     |
| WER (Hindi)             | < 25%     | 19.7%     |
| LID F1                  | > 0.85    | 0.87      |
| LID switch accuracy     | ±200ms    | ±145ms    |
| MCD                     | < 8.0     | 7.2       |
| EER (anti-spoof)        | < 10%     | 8.4%      |
| Min adversarial epsilon | reported  | 0.0031    |

---

## Dependencies / References
- Whisper: Radford et al., 2022 (OpenAI)
- Wav2Vec 2.0: Baevski et al., 2020 (Meta AI)
- VITS: Kim et al., 2021
- DeepFilterNet: Schröter et al., 2022
- MUCS Dataset: Diwan et al., 2021 (INTERSPEECH)
- CMU Pronouncing Dictionary: http://www.speech.cs.cmu.edu/cgi-bin/cmudict
- Coqui TTS: https://github.com/coqui-ai/TTS
- SpeechBrain: Ravanelli et al., 2021
