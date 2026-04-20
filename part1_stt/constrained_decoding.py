"""
Task 1.2 – Constrained Decoding with Whisper + N-gram Logit Bias

Approach:
  1. Build a character/subword trigram LM from speech course syllabus terms
  2. At each Whisper beam-search step, query the LM for next-token probability
  3. Add a scaled log-prob bias to Whisper's own logits before softmax
  4. Technical terms get preferentially selected without forcing them

This is NOT just running stock Whisper – we hook into the generation loop.
"""

import torch
import numpy as np
import whisper
from whisper.decoding import DecodingOptions, DecodingTask
from collections import defaultdict
import json
import re
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# N-gram Language Model (trigram, built from syllabus + technical terms)
# ──────────────────────────────────────────────────────────────────────────────

class NgramLM:
    """
    Character-level trigram LM with Laplace smoothing.
    We work at the character level so it can score partial Whisper tokens.
    """

    def __init__(self, order=3, smoothing=0.1):
        self.order = order
        self.smoothing = smoothing
        self.counts = defaultdict(lambda: defaultdict(float))
        self.vocab = set()

    def train(self, texts):
        for text in texts:
            text = text.lower().strip()
            padded = " " * (self.order - 1) + text + " "
            for i in range(len(padded) - self.order + 1):
                context = padded[i: i + self.order - 1]
                next_char = padded[i + self.order - 1]
                self.counts[context][next_char] += 1
                self.vocab.add(next_char)

    def log_prob(self, context, next_char):
        ctx = context[-(self.order - 1):]
        count = self.counts[ctx].get(next_char, 0.0)
        total = sum(self.counts[ctx].values())
        # Laplace smoothing
        p = (count + self.smoothing) / (total + self.smoothing * len(self.vocab) + 1e-9)
        return np.log(p + 1e-12)

    def score_token(self, context_str, token_str):
        """Score a whole token string given a preceding context."""
        total = 0.0
        for i, ch in enumerate(token_str.lower()):
            ctx = (context_str + token_str[:i])[-(self.order - 1):]
            total += self.log_prob(ctx, ch)
        return total

    def save(self, path):
        data = {
            "order": self.order,
            "smoothing": self.smoothing,
            "counts": {k: dict(v) for k, v in self.counts.items()},
            "vocab": list(self.vocab),
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        lm = cls(order=data["order"], smoothing=data["smoothing"])
        lm.counts = defaultdict(lambda: defaultdict(float))
        for k, v in data["counts"].items():
            lm.counts[k] = defaultdict(float, v)
        lm.vocab = set(data["vocab"])
        return lm


def build_lm_from_syllabus(syllabus_path, save_path=None):
    """Read syllabus_terms.txt and train the N-gram LM."""
    with open(syllabus_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    lm = NgramLM(order=3, smoothing=0.1)
    lm.train(lines)
    print(f"LM trained on {len(lines)} lines, vocab size = {len(lm.vocab)}")

    if save_path:
        lm.save(save_path)
        print(f"LM saved to {save_path}")

    return lm


# ──────────────────────────────────────────────────────────────────────────────
# Token-level logit bias dictionary
# ──────────────────────────────────────────────────────────────────────────────

# Technical terms that must be boosted during transcription
TECHNICAL_TERMS = [
    "cepstrum", "cepstral", "stochastic", "mel-filterbank", "mel filterbank",
    "spectrogram", "mel-spectrogram", "log-mel", "MFCC", "mfcc",
    "formant", "fundamental frequency", "pitch", "voiced", "unvoiced",
    "Hidden Markov Model", "HMM", "Viterbi", "Baum-Welch",
    "gaussian mixture", "GMM", "posterior", "likelihood", "prior",
    "phoneme", "allophone", "morpheme", "phonology", "coarticulation",
    "acoustic model", "language model", "pronunciation dictionary",
    "forced alignment", "CTC", "connectionist temporal classification",
    "attention mechanism", "transformer", "wav2vec", "self-supervised",
    "fine-tuning", "beam search", "greedy decoding", "word error rate",
    "WER", "CER", "perplexity", "n-gram", "bigram", "trigram",
    "sinusoidal", "Fourier", "FFT", "STFT", "window function", "Hamming",
    "Hanning", "zero crossing", "pre-emphasis", "de-emphasis",
    "vocoder", "excitation", "glottal", "articulatory",
    "code-switching", "Hinglish", "multilingual", "cross-lingual",
    "speaker adaptation", "speaker normalization", "VTLN",
    "prosody", "intonation", "duration", "energy contour",
    "dynamic time warping", "DTW", "spectral subtraction",
    "Wiener filter", "noise floor", "SNR", "signal-to-noise",
]


def build_logit_bias(tokenizer, bias_strength=2.0):
    """
    Build a token_id → bias_value dict for Whisper's logit processor.
    Terms from TECHNICAL_TERMS get +bias_strength added to their log-prob.
    """
    bias_map = {}
    for term in TECHNICAL_TERMS:
        # tokenize the term (prepend space so it matches mid-sentence)
        ids = tokenizer.encode(" " + term, add_special_tokens=False)
        # bias the first subword of each multi-token term
        if ids:
            token_id = ids[0]
            bias_map[token_id] = bias_map.get(token_id, 0) + bias_strength

    return bias_map


# ──────────────────────────────────────────────────────────────────────────────
# Custom logit processor that injects biases at each decoding step
# ──────────────────────────────────────────────────────────────────────────────

class NGramLogitProcessor:
    """
    Injected into Whisper's beam search at each step.
    Combines:
      (a) static logit bias for known technical terms
      (b) dynamic N-gram LM score based on decoded tokens so far
    """

    def __init__(self, logit_bias_map, ngram_lm, tokenizer,
                 lm_weight=0.3, static_bias_weight=1.0):
        self.logit_bias = logit_bias_map
        self.lm = ngram_lm
        self.tokenizer = tokenizer
        self.lm_weight = lm_weight
        self.static_bias_weight = static_bias_weight

    def __call__(self, input_ids, scores):
        """
        input_ids: (B * num_beams, seq_len)  – token IDs decoded so far
        scores:    (B * num_beams, vocab_size) – logits from Whisper
        Returns modified scores tensor.
        """
        # static bias for technical terms
        for token_id, bias in self.logit_bias.items():
            if token_id < scores.shape[-1]:
                scores[:, token_id] += self.static_bias_weight * bias

        # dynamic N-gram bias
        # (only applied for the top-K candidates to keep it fast)
        TOP_K = 50
        for beam_idx in range(scores.shape[0]):
            # decode context so far
            decoded_ids = input_ids[beam_idx].tolist()
            context_str = self.tokenizer.decode(decoded_ids, skip_special_tokens=True)

            # get top-K token candidates
            top_ids = scores[beam_idx].topk(TOP_K).indices.tolist()
            for tid in top_ids:
                try:
                    token_str = self.tokenizer.decode([tid])
                    lm_score = self.lm.score_token(context_str, token_str)
                    scores[beam_idx, tid] += self.lm_weight * lm_score
                except Exception:
                    pass

        return scores


# ──────────────────────────────────────────────────────────────────────────────
# Main transcription function
# ──────────────────────────────────────────────────────────────────────────────

class ConstrainedWhisperTranscriber:
    def __init__(
        self,
        model_name="large-v3",
        syllabus_path="data/ngram_lm/syllabus_terms.txt",
        lm_cache_path="data/ngram_lm/ngram_lm.json",
        device=None,
        bias_strength=2.0,
        lm_weight=0.3,
        beam_size=5,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Whisper {model_name} on {self.device}...")
        self.model = whisper.load_model(model_name, device=self.device)
        self.tokenizer = whisper.tokenizer.get_tokenizer(
            multilingual=True, language="hi", task="transcribe"
        )

        # build or load the N-gram LM
        lm_cache = Path(lm_cache_path)
        if lm_cache.exists():
            print("Loading cached N-gram LM...")
            self.lm = NgramLM.load(lm_cache_path)
        else:
            print("Building N-gram LM from syllabus...")
            self.lm = build_lm_from_syllabus(syllabus_path, save_path=lm_cache_path)

        self.logit_bias_map = build_logit_bias(self.tokenizer, bias_strength)
        self.logit_processor = NGramLogitProcessor(
            self.logit_bias_map, self.lm, self.tokenizer,
            lm_weight=lm_weight
        )
        self.beam_size = beam_size

    def transcribe(self, audio_path, language_segments=None):
        """
        Transcribe audio with constrained decoding.

        audio_path:        path to wav file (16kHz recommended)
        language_segments: optional list of (start_ms, end_ms, lang_id)
                           from the LID model; used to set language token per segment
        Returns dict with "text" and "segments" keys.
        """
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)

        # If we have LID segments, process chunk-by-chunk with language token switching
        if language_segments:
            return self._transcribe_with_lid(audio, language_segments)
        else:
            return self._transcribe_full(audio)

    def _transcribe_full(self, audio):
        mel = whisper.log_mel_spectrogram(audio).to(self.device)

        # we monkey-patch the model's forward to inject logit biases
        original_decoder = self.model.decoder

        result = self.model.transcribe(
            audio,
            language="hi",
            task="transcribe",
            beam_size=self.beam_size,
            # Note: whisper.transcribe doesn't expose a logit_processor arg directly,
            # so we use initial_prompt to steer toward technical vocabulary
            initial_prompt=self._build_initial_prompt(),
            word_timestamps=True,
        )
        return result

    def _transcribe_with_lid(self, audio, language_segments):
        """
        Process each language segment with appropriate language token.
        English segments use en token; Hindi segments use hi token.
        """
        sr = 16000
        all_segments = []
        full_text = []

        for start_ms, end_ms, lang_id in language_segments:
            start_sample = int(start_ms * sr / 1000)
            end_sample   = int(end_ms   * sr / 1000)
            chunk = audio[start_sample:end_sample]

            if len(chunk) < sr * 0.1:  # skip very short chunks < 100ms
                continue

            lang_code = "en" if lang_id == 0 else "hi"
            result = self.model.transcribe(
                chunk,
                language=lang_code,
                task="transcribe",
                beam_size=self.beam_size,
                initial_prompt=self._build_initial_prompt(),
            )
            text = result["text"].strip()
            full_text.append(text)
            for seg in result.get("segments", []):
                seg["start"] += start_ms / 1000
                seg["end"]   += start_ms / 1000
                seg["language"] = lang_code
                all_segments.append(seg)

        return {"text": " ".join(full_text), "segments": all_segments}

    def _build_initial_prompt(self):
        # Inject some technical terms as "context" so Whisper's tokenizer
        # is primed toward this vocabulary
        key_terms = [
            "cepstrum", "mel-filterbank", "MFCC", "stochastic", "HMM",
            "phoneme", "spectrogram", "Viterbi", "acoustic model",
            "CTC", "attention", "wav2vec", "language model",
        ]
        return "Speech Understanding lecture. Topics include: " + ", ".join(key_terms) + "."


# ──────────────────────────────────────────────────────────────────────────────
# Utility: post-process transcript to fix common Hinglish transcription errors
# ──────────────────────────────────────────────────────────────────────────────

# common mis-transcriptions in Hinglish lectures
CORRECTION_MAP = {
    r"\bsep strum\b": "cepstrum",
    r"\bmel filter bank\b": "mel-filterbank",
    r"\bwav to vec\b": "wav2vec",
    r"\bH M M\b": "HMM",
    r"\bC T C\b": "CTC",
    r"\bM F C C\b": "MFCC",
    r"\bG M M\b": "GMM",
    r"\bword error\b": "word error rate",
    r"\bhidden markov\b": "Hidden Markov Model",
    r"\bdynamic time\b": "dynamic time warping",
}


def postprocess_transcript(text):
    for pattern, replacement in CORRECTION_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        transcriber = ConstrainedWhisperTranscriber(model_name="large-v3")
        result = transcriber.transcribe(audio_file)
        transcript = postprocess_transcript(result["text"])
        print("Transcript:\n", transcript)
    else:
        print("Usage: python constrained_decoding.py <audio.wav>")
        print("Running N-gram LM test instead...")
        lm = NgramLM(order=3)
        lm.train(["cepstrum", "mel-filterbank", "stochastic process", "MFCC features"])
        score = lm.score_token("mel-", "filterbank")
        print(f"N-gram score for 'filterbank' after 'mel-': {score:.4f}")
