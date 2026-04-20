"""
Task 4.2 – Adversarial Noise Injection (FGSM on LID model)

Goal: find the minimum perturbation epsilon that causes our LID model
      (Task 1.1) to misclassify Hindi frames as English.

Method: Fast Gradient Sign Method (FGSM)
  x_adv = x + epsilon * sign(∇_x L(f(x), y_target))

Constraint: perturbation must be perceptually inaudible
  SNR of the perturbed signal vs original must be > 40dB

We test on a 5-second segment and report:
  - Minimum epsilon where misclassification occurs
  - Actual SNR at that epsilon
  - Confusion matrix before/after attack
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path

from part1_stt.lid_model import load_lid_model, MultiHeadLID, LANG_ENGLISH, LANG_HINDI


SAMPLE_RATE    = 16000
TARGET_LANG    = LANG_ENGLISH   # we want Hindi to be classified as English
MAX_EPSILON    = 0.1
SNR_THRESHOLD  = 40.0           # minimum acceptable SNR (dB)


# ──────────────────────────────────────────────────────────────────────────────
# SNR computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_snr_db(original, perturbed):
    """
    SNR = 10 * log10( power(signal) / power(noise) )
    where noise = perturbed - original
    """
    sig_power   = torch.mean(original ** 2)
    noise       = perturbed - original
    noise_power = torch.mean(noise ** 2)
    snr = 10.0 * torch.log10(sig_power / (noise_power + 1e-12))
    return snr.item()


# ──────────────────────────────────────────────────────────────────────────────
# FGSM attack (single step)
# ──────────────────────────────────────────────────────────────────────────────

def fgsm_attack(model, waveform, target_label, epsilon, device="cpu"):
    """
    Single-step FGSM.

    model:        MultiHeadLID model
    waveform:     (1, T) float32 tensor
    target_label: int, desired (wrong) prediction for all frames
    epsilon:      perturbation magnitude

    Returns: (adversarial_waveform, grad_sign)
    """
    model.eval()
    x = waveform.clone().detach().to(device).requires_grad_(True)

    logits = model(x)  # (1, F, 2)
    B, F, C = logits.shape

    # we want all frames to predict target_label (English)
    target = torch.full((B, F), target_label, dtype=torch.long, device=device)

    loss = F.cross_entropy(logits.reshape(B * F, C), target.reshape(B * F))
    loss.backward()

    grad_sign = x.grad.sign()
    x_adv = x.detach() + epsilon * grad_sign

    # clip to [-1, 1] (valid audio range)
    x_adv = torch.clamp(x_adv, -1.0, 1.0)

    return x_adv.detach(), grad_sign.detach()


def pgd_attack(model, waveform, target_label, epsilon, alpha=0.001,
               n_steps=20, device="cpu"):
    """
    Projected Gradient Descent (multi-step FGSM) for a stronger attack.
    More reliable for finding tight epsilon bounds.
    """
    model.eval()
    x_orig = waveform.clone().detach().to(device)
    x_adv  = x_orig.clone()

    for step in range(n_steps):
        x_adv = x_adv.clone().requires_grad_(True)

        logits = model(x_adv)
        B, F, C = logits.shape
        target = torch.full((B, F), target_label, dtype=torch.long, device=device)
        loss = F.cross_entropy(logits.reshape(B * F, C), target.reshape(B * F))
        loss.backward()

        grad_sign = x_adv.grad.sign()
        x_adv = x_adv.detach() + alpha * grad_sign

        # project back into epsilon-ball around original
        delta = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x_orig + delta, -1.0, 1.0).detach()

    return x_adv


# ──────────────────────────────────────────────────────────────────────────────
# Epsilon sweep: find minimum epsilon for misclassification
# ──────────────────────────────────────────────────────────────────────────────

def find_minimum_epsilon(model, waveform, true_labels_per_frame,
                         device="cpu", use_pgd=True):
    """
    Binary search for the minimum epsilon that causes misclassification
    while maintaining SNR > 40dB.

    true_labels_per_frame: numpy array of true per-frame language labels
    Returns: {
        "min_epsilon": float,
        "snr_at_min_eps": float,
        "flip_fraction": float,  (fraction of Hindi frames misclassified)
        "epsilon_sweep": dict,   (results at each tested epsilon)
    }
    """
    waveform = waveform.to(device)
    model    = model.to(device)

    # frames that are genuinely Hindi
    hindi_frame_mask = (true_labels_per_frame == LANG_HINDI)
    n_hindi = hindi_frame_mask.sum()
    if n_hindi == 0:
        print("No Hindi frames found in input.")
        return None

    epsilon_values = np.logspace(-4, -1, 30)  # from 1e-4 to 0.1
    sweep_results  = {}
    min_epsilon_found = None

    print(f"Sweeping epsilon from {epsilon_values[0]:.5f} to {epsilon_values[-1]:.3f}")
    print(f"Hindi frames: {n_hindi}, SNR threshold: {SNR_THRESHOLD}dB\n")

    for eps in epsilon_values:
        if use_pgd:
            x_adv = pgd_attack(model, waveform.unsqueeze(0), TARGET_LANG,
                                epsilon=eps, device=device)
        else:
            x_adv, _ = fgsm_attack(model, waveform.unsqueeze(0), TARGET_LANG,
                                    epsilon=eps, device=device)

        snr = compute_snr_db(waveform.unsqueeze(0), x_adv)

        # evaluate LID on adversarial audio
        with torch.no_grad():
            logits = model(x_adv)
            preds  = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

        # how many Hindi frames got flipped to English?
        n_frames = min(len(preds), len(hindi_frame_mask))
        flipped  = np.sum(
            (preds[:n_frames] == LANG_ENGLISH) & (hindi_frame_mask[:n_frames])
        )
        flip_rate = flipped / max(n_hindi, 1)

        sweep_results[float(eps)] = {
            "snr_db":     snr,
            "flip_rate":  float(flip_rate),
            "snr_ok":     snr > SNR_THRESHOLD,
        }

        print(f"  eps={eps:.5f}: flip_rate={flip_rate:.2%}  SNR={snr:.1f}dB  "
              f"{'[SNR OK]' if snr > SNR_THRESHOLD else '[SNR FAIL]'}")

        if flip_rate > 0.5 and snr > SNR_THRESHOLD and min_epsilon_found is None:
            min_epsilon_found = float(eps)
            print(f"  *** Minimum epsilon found: {min_epsilon_found:.5f} ***")

    if min_epsilon_found is None:
        print("\nCould not achieve >50% flip rate within SNR constraint.")
        # find the epsilon that came closest
        best_snr_ok = {k: v for k, v in sweep_results.items() if v["snr_ok"]}
        if best_snr_ok:
            min_epsilon_found = max(best_snr_ok.keys(),
                                    key=lambda e: best_snr_ok[e]["flip_rate"])

    result = {
        "min_epsilon":    min_epsilon_found,
        "snr_at_min_eps": sweep_results.get(min_epsilon_found, {}).get("snr_db", None),
        "flip_fraction":  sweep_results.get(min_epsilon_found, {}).get("flip_rate", None),
        "epsilon_sweep":  sweep_results,
    }
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Confusion matrix for LID before/after attack
# ──────────────────────────────────────────────────────────────────────────────

def build_confusion_matrix(true_labels, pred_labels, n_classes=2):
    """Returns (n_classes, n_classes) confusion matrix."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def print_confusion_matrix(cm, class_names=None):
    names = class_names or [str(i) for i in range(cm.shape[0])]
    header = "         " + "  ".join(f"Pred {n:<8}" for n in names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>12}" for v in row)
        print(f"True {names[i]:<4}: {row_str}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot epsilon sweep results
# ──────────────────────────────────────────────────────────────────────────────

def plot_epsilon_sweep(sweep_results, save_path="adversarial_sweep.png"):
    eps_vals  = sorted(sweep_results.keys())
    flip_rates = [sweep_results[e]["flip_rate"] * 100 for e in eps_vals]
    snr_vals   = [sweep_results[e]["snr_db"] for e in eps_vals]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.semilogx(eps_vals, flip_rates, "b-o", label="Hindi→English flip rate (%)")
    ax2.semilogx(eps_vals, snr_vals,   "r--s", label="SNR (dB)")

    ax1.axhline(50, color="blue",   linestyle=":", alpha=0.5, label="50% flip threshold")
    ax2.axhline(SNR_THRESHOLD, color="red", linestyle=":", alpha=0.5,
                label=f"{SNR_THRESHOLD}dB SNR threshold")

    ax1.set_xlabel("Epsilon (log scale)")
    ax1.set_ylabel("Flip Rate (%)", color="blue")
    ax2.set_ylabel("SNR (dB)",       color="red")
    ax1.set_title("FGSM Adversarial Attack: Epsilon vs. LID Flip Rate & SNR")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Sweep plot saved to {save_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation function
# ──────────────────────────────────────────────────────────────────────────────

def run_adversarial_evaluation(
    audio_path,
    lid_weights_path,
    frame_labels=None,
    device=None,
    segment_duration_s=5.0,
):
    """
    Full adversarial robustness evaluation.

    audio_path:       path to a Hinglish audio segment
    lid_weights_path: path to trained LID model weights
    frame_labels:     optional numpy array of true per-frame labels
                      if None, we assume all frames are Hindi (worst case)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # load audio
    sig, sr = sf.read(audio_path)
    if sig.ndim == 2:
        sig = sig.mean(axis=1)
    sig = sig.astype(np.float32)

    # use only 5 seconds
    n_samples = int(segment_duration_s * SAMPLE_RATE)
    sig = sig[:n_samples]
    if sr != SAMPLE_RATE:
        import torchaudio
        t = torch.from_numpy(sig).unsqueeze(0)
        t = torchaudio.functional.resample(t, sr, SAMPLE_RATE)
        sig = t.squeeze(0).numpy()

    waveform = torch.from_numpy(sig)

    # load LID model
    model = load_lid_model(lid_weights_path, device=device)

    # get clean predictions
    with torch.no_grad():
        clean_logits = model(waveform.unsqueeze(0).to(device))
        clean_preds  = clean_logits.argmax(dim=-1).squeeze(0).cpu().numpy()

    n_frames = len(clean_preds)

    # if no ground truth labels, assume all frames are Hindi
    if frame_labels is None:
        true_labels = np.ones(n_frames, dtype=int)  # all Hindi
        print("No ground truth labels provided – assuming all frames are Hindi.")
    else:
        true_labels = frame_labels[:n_frames]

    # confusion matrix before attack
    print("=== Clean LID Confusion Matrix ===")
    cm_clean = build_confusion_matrix(true_labels, clean_preds)
    print_confusion_matrix(cm_clean, class_names=["English", "Hindi"])
    print()

    # run epsilon sweep
    results = find_minimum_epsilon(model, waveform, true_labels, device=device)

    print("\n=== Adversarial Robustness Summary ===")
    if results and results["min_epsilon"]:
        print(f"Minimum epsilon: {results['min_epsilon']:.5f}")
        print(f"SNR at min eps:  {results['snr_at_min_eps']:.1f} dB")
        print(f"Flip fraction:   {results['flip_fraction']*100:.1f}%")
    else:
        print("Could not find epsilon satisfying constraints.")

    # generate adversarial audio at min epsilon and show confusion matrix
    if results and results["min_epsilon"]:
        min_eps = results["min_epsilon"]
        x_adv   = pgd_attack(model, waveform.unsqueeze(0), TARGET_LANG,
                              epsilon=min_eps, device=device)
        with torch.no_grad():
            adv_logits = model(x_adv)
            adv_preds  = adv_logits.argmax(dim=-1).squeeze(0).cpu().numpy()

        print("\n=== Adversarial LID Confusion Matrix (at min epsilon) ===")
        cm_adv = build_confusion_matrix(true_labels, adv_preds[:len(true_labels)])
        print_confusion_matrix(cm_adv, class_names=["English", "Hindi"])

        # save adversarial audio
        adv_wav_path = audio_path.replace(".wav", "_adversarial.wav")
        import soundfile as sf
        sf.write(adv_wav_path, x_adv.squeeze(0).cpu().numpy(), SAMPLE_RATE)
        print(f"\nAdversarial audio saved to {adv_wav_path}")

        # plot
        plot_epsilon_sweep(results["epsilon_sweep"])

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        audio  = sys.argv[1]
        weights = sys.argv[2]
        run_adversarial_evaluation(audio, weights)
    else:
        print("Usage: python adversarial_attack.py <audio.wav> <lid_weights.pt>")
        print("\nTesting FGSM with dummy model + audio...")

        model   = MultiHeadLID()
        dummy_w = torch.randn(1, 16000 * 5)
        dummy_w = dummy_w / dummy_w.abs().max()

        x_adv, grad = fgsm_attack(model, dummy_w, TARGET_LANG, epsilon=0.01)
        snr = compute_snr_db(dummy_w, x_adv)
        print(f"FGSM test: SNR={snr:.1f}dB, adv shape={x_adv.shape}")

        # check flip
        with torch.no_grad():
            orig_preds = model(dummy_w).argmax(dim=-1).squeeze()
            adv_preds  = model(x_adv).argmax(dim=-1).squeeze()
        flip_rate = (orig_preds != adv_preds).float().mean().item()
        print(f"Flip rate with eps=0.01: {flip_rate:.2%}")
