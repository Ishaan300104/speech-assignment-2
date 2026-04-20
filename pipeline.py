"""
Speech Understanding PA2 – End-to-End Pipeline

Usage:
    python pipeline.py --input audio/original_segment.wav \
                       --voice_ref audio/student_voice_ref.wav \
                       --output audio/output_LRL_cloned.wav

    python pipeline.py --part 1 --input audio/original_segment.wav
    python pipeline.py --part 2 --transcript transcript.json
    python pipeline.py --part 3 --voice_ref audio/student_voice_ref.wav
    python pipeline.py --part 4 --eval
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
from pathlib import Path

from utils.audio_utils import load_audio, save_audio
from utils.metrics import (
    compute_wer_on_segments, lid_switching_accuracy,
    print_evaluation_report
)


def run_part1(args):
    """Part I: Robust Code-Switched Transcription."""
    from part1_stt.denoising import denoise_audio
    from part1_stt.lid_model import load_lid_model, get_switch_timestamps, frames_to_segments
    from part1_stt.constrained_decoding import ConstrainedWhisperTranscriber, postprocess_transcript

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Part 1] Running STT pipeline on {args.input}")
    print(f"         Device: {device}")

    # Task 1.3: Denoise
    print("\n[1.3] Denoising audio...")
    denoised_path = args.input.replace(".wav", "_denoised.wav")
    clean_sig, sr = denoise_audio(args.input, denoised_path, use_wiener=True)
    print(f"      Denoised audio saved: {denoised_path}")

    # Task 1.1: Language ID
    print("\n[1.1] Running frame-level Language ID...")
    lid_weights = "models/lid_weights.pt"
    if not Path(lid_weights).exists():
        print(f"      WARNING: LID weights not found at {lid_weights}")
        print("      Using untrained model – run training first for proper results.")
        from part1_stt.lid_model import MultiHeadLID
        lid_model = MultiHeadLID().to(device)
    else:
        lid_model = load_lid_model(lid_weights, device=device)

    sig_t = torch.from_numpy(clean_sig)
    frame_preds = lid_model.predict_frames(sig_t, device=device)
    segments    = frames_to_segments(frame_preds)
    switches    = get_switch_timestamps(frame_preds)

    print(f"      Detected {len(segments)} language segments")
    print(f"      Code switches: {len(switches)}")
    for ts, from_l, to_l in switches[:5]:
        lang_names = {0: "English", 1: "Hindi"}
        print(f"        {ts:.0f}ms: {lang_names.get(from_l, '?')} → {lang_names.get(to_l, '?')}")

    # Task 1.2: Constrained transcription
    print("\n[1.2] Constrained Whisper transcription...")
    transcriber = ConstrainedWhisperTranscriber(
        model_name=args.whisper_model,
        device=device,
    )
    result = transcriber.transcribe(denoised_path, language_segments=segments)
    transcript = postprocess_transcript(result["text"])

    print(f"\n      Transcript preview (first 300 chars):")
    print(f"      {transcript[:300]}...")

    # save outputs
    transcript_path = args.input.replace(".wav", "_transcript.json")
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump({
            "text":     transcript,
            "segments": result.get("segments", []),
            "lid_segments": [
                {"start_ms": s, "end_ms": e, "lang_id": int(l)}
                for s, e, l in segments
            ],
            "code_switches": [
                {"timestamp_ms": t, "from_lang": int(f), "to_lang": int(t2)}
                for t, f, t2 in switches
            ],
        }, f, ensure_ascii=False, indent=2)

    print(f"\n      Transcript saved to {transcript_path}")
    return transcript_path


def run_part2(args):
    """Part II: Phonetic Mapping & Translation."""
    from part2_phonetic.g2p_hinglish import hinglish_to_ipa, process_transcript_to_ipa
    from part2_phonetic.translation import HinglishToSanthaliTranslator

    print("[Part 2] Phonetic mapping & translation...")

    # load transcript
    transcript_path = getattr(args, "transcript", None)
    if transcript_path and Path(transcript_path).exists():
        with open(transcript_path, encoding="utf-8") as f:
            data = json.load(f)
        segments = data.get("segments", [{"text": data.get("text", "")}])
    else:
        print("No transcript file provided. Using sample text.")
        segments = [
            {"text": "ab hum cepstrum ke baare mein baat karte hain", "language": "hi"},
            {"text": "the mel filterbank extracts spectral features", "language": "en"},
        ]

    # Task 2.1: IPA conversion
    print("\n[2.1] Converting to IPA...")
    segments_with_ipa = process_transcript_to_ipa(segments)
    for seg in segments_with_ipa[:3]:
        print(f"      Text: {seg['text'][:60]}")
        print(f"      IPA:  {seg.get('ipa', '')[:60]}\n")

    # Task 2.2: Santhali translation
    print("[2.2] Translating to Santhali...")
    translator = HinglishToSanthaliTranslator()
    translated_segments = translator.translate_segments(segments_with_ipa)

    for seg in translated_segments[:3]:
        print(f"      Source:   {seg['text'][:60]}")
        print(f"      Santhali: {seg.get('santhali', '')[:60]}\n")

    # save
    out_path = "santhali_translation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(translated_segments, f, ensure_ascii=False, indent=2)
    print(f"Translation saved to {out_path}")

    return out_path


def run_part3(args):
    """Part III: Zero-Shot Voice Cloning & Synthesis."""
    from part3_tts.voice_embedding import SpeakerEmbeddingExtractor
    from part3_tts.synthesis import synthesize_lecture

    print("[Part 3] Voice cloning & synthesis...")

    # Task 3.1: Extract speaker embedding
    print("\n[3.1] Extracting speaker embedding from reference audio...")
    extractor = SpeakerEmbeddingExtractor()
    emb_path  = "models/speaker_embedding.pt"
    emb, sr   = extractor.extract_from_file(args.voice_ref)
    extractor.save_embedding(emb, emb_path)
    print(f"      Embedding shape: {emb.shape}")

    # load translation
    translation_path = getattr(args, "translation", "santhali_translation.json")
    if Path(translation_path).exists():
        with open(translation_path, encoding="utf-8") as f:
            santhali_segs = json.load(f)
    else:
        print("Translation file not found. Using dummy segments.")
        santhali_segs = [
            {"santhali": "mit baang re cepstrum ar mel filterbank kotha akantalea"}
        ]

    # Task 3.2 + 3.3: Prosody warping + synthesis
    print("\n[3.2/3.3] Synthesizing Santhali lecture with prosody warping...")
    ref_audio = getattr(args, "input", "audio/original_segment.wav")
    output    = getattr(args, "output", "audio/output_LRL_cloned.wav")

    synthesize_lecture(
        santhali_segs,
        speaker_ref_path=args.voice_ref,
        ref_audio_path=ref_audio,
        output_path=output,
        apply_prosody=Path(ref_audio).exists(),
    )


def run_part4(args):
    """Part IV: Adversarial Robustness & Spoofing Detection."""
    from part4_adversarial.anti_spoofing import (
        load_anti_spoof_model, classify_audio, evaluate_eer, compute_eer
    )
    from part4_adversarial.adversarial_attack import run_adversarial_evaluation

    print("[Part 4] Adversarial robustness & anti-spoofing evaluation...")

    # Task 4.1: Anti-spoofing evaluation
    print("\n[4.1] Anti-spoofing classifier evaluation...")
    spoof_weights = "models/anti_spoof_weights.pt"

    if Path(spoof_weights).exists():
        spoof_model = load_anti_spoof_model(spoof_weights)
        # classify the real voice and cloned voice
        real_path   = "audio/student_voice_ref.wav"
        cloned_path = "audio/output_LRL_cloned.wav"

        results = {}
        for path, expected in [(real_path, "bona_fide"), (cloned_path, "spoof")]:
            if Path(path).exists():
                r = classify_audio(spoof_model, path)
                results[path] = r
                status = "CORRECT" if r["label"] == expected else "WRONG"
                print(f"      {Path(path).name}: {r['label']} (score={r['spoof_score']:.3f}) [{status}]")
    else:
        print(f"      Anti-spoof model not found at {spoof_weights}")
        print("      Train the model first with part4_adversarial/anti_spoofing.py")

    # Task 4.2: Adversarial attack
    print("\n[4.2] Adversarial attack on LID model...")
    lid_weights = "models/lid_weights.pt"
    input_audio = getattr(args, "input", "audio/original_segment.wav")

    if Path(lid_weights).exists() and Path(input_audio).exists():
        adv_results = run_adversarial_evaluation(
            input_audio, lid_weights, device="cpu"
        )
        if adv_results:
            print(f"\n      Min epsilon: {adv_results.get('min_epsilon')}")
    else:
        print(f"      Missing: LID weights ({lid_weights}) or audio ({input_audio})")


def main():
    parser = argparse.ArgumentParser(description="Speech PA2 Pipeline")
    parser.add_argument("--part", type=int, choices=[1, 2, 3, 4],
                        help="Run only a specific part (1-4). Omit to run all.")
    parser.add_argument("--input",       default="audio/original_segment.wav",
                        help="Input lecture audio (WAV)")
    parser.add_argument("--voice_ref",   default="audio/student_voice_ref.wav",
                        help="60s speaker reference audio")
    parser.add_argument("--output",      default="audio/output_LRL_cloned.wav",
                        help="Output synthesized lecture path")
    parser.add_argument("--transcript",  default=None,
                        help="Path to existing transcript JSON (skips Part 1)")
    parser.add_argument("--translation", default=None,
                        help="Path to existing translation JSON (skips Part 2)")
    parser.add_argument("--whisper_model", default="large-v3",
                        choices=["tiny", "base", "small", "medium", "large-v3"],
                        help="Whisper model size")
    parser.add_argument("--eval", action="store_true",
                        help="Run evaluation metrics at end")

    args = parser.parse_args()

    # create output directories
    os.makedirs("audio",  exist_ok=True)
    os.makedirs("models", exist_ok=True)

    if args.part is None or args.part == 1:
        if not Path(args.input).exists():
            print(f"ERROR: Input file not found: {args.input}")
            print("Please place your lecture audio at audio/original_segment.wav")
            if args.part == 1:
                sys.exit(1)
        else:
            transcript_path = run_part1(args)
            if args.transcript is None:
                args.transcript = transcript_path

    if args.part is None or args.part == 2:
        translation_path = run_part2(args)
        if args.translation is None:
            args.translation = translation_path

    if args.part is None or args.part == 3:
        if not Path(args.voice_ref).exists():
            print(f"ERROR: Voice reference not found: {args.voice_ref}")
            print("Please record 60s of your voice and save to audio/student_voice_ref.wav")
            if args.part == 3:
                sys.exit(1)
        else:
            run_part3(args)

    if args.part is None or args.part == 4:
        run_part4(args)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
