"""
Evaluation metrics: WER, MCD, EER, LID switching accuracy.
"""

import numpy as np
from sklearn.metrics import f1_score


# ──────────────────────────────────────────────────────────────────────────────
# Word Error Rate
# ──────────────────────────────────────────────────────────────────────────────

def wer(reference, hypothesis):
    """
    Compute Word Error Rate via dynamic programming (edit distance).
    Both inputs are strings; tokenized by whitespace.
    WER = (S + D + I) / N   where N = number of reference words
    """
    ref = reference.strip().lower().split()
    hyp = hypothesis.strip().lower().split()

    N = len(ref)
    if N == 0:
        return 0.0 if len(hyp) == 0 else 1.0

    # edit distance DP
    dp = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)
    for i in range(len(ref) + 1):
        dp[i, 0] = i
    for j in range(len(hyp) + 1):
        dp[0, j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = 1 + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    return dp[len(ref), len(hyp)] / N


def compute_wer_on_segments(ref_segments, hyp_segments):
    """
    Compute WER separately for English and Hindi segments.
    ref_segments / hyp_segments: list of {"text": str, "language": "en"/"hi"} dicts
    """
    en_ref, en_hyp, hi_ref, hi_hyp = [], [], [], []

    for ref, hyp in zip(ref_segments, hyp_segments):
        lang = ref.get("language", "en")
        if lang == "en":
            en_ref.append(ref["text"])
            en_hyp.append(hyp["text"])
        else:
            hi_ref.append(ref["text"])
            hi_hyp.append(hyp["text"])

    en_wer = wer(" ".join(en_ref), " ".join(en_hyp)) if en_ref else None
    hi_wer = wer(" ".join(hi_ref), " ".join(hi_hyp)) if hi_ref else None

    return {"english_wer": en_wer, "hindi_wer": hi_wer}


# ──────────────────────────────────────────────────────────────────────────────
# Mel-Cepstral Distortion (MCD)
# ──────────────────────────────────────────────────────────────────────────────

def mcd(ref_mfcc, synth_mfcc):
    """
    Mel-Cepstral Distortion between two MFCC sequences.
    Both inputs: (n_mfcc, T) numpy arrays.
    Lower is better; passing threshold is MCD < 8.0.

    Uses DTW to align before computing distance (standard MCD evaluation).
    """
    from scipy.spatial.distance import cdist

    # trim to same feature dimension
    n_dims = min(ref_mfcc.shape[0], synth_mfcc.shape[0], 25)  # use first 25 MFCCs
    ref_feat   = ref_mfcc[1:n_dims, :].T    # skip C0 (energy), shape (T_ref, D)
    synth_feat = synth_mfcc[1:n_dims, :].T  # (T_syn, D)

    # DTW alignment
    cost_matrix = cdist(ref_feat, synth_feat, metric="euclidean")
    path_r, path_s = _dtw_path(cost_matrix)

    # compute MCD along aligned path
    K = 10.0 / np.log(10)  # MCD constant
    distances = []
    for i, j in zip(path_r, path_s):
        diff = ref_feat[i] - synth_feat[j]
        distances.append(np.sqrt(2.0 * np.dot(diff, diff)))

    return K * np.mean(distances)


def _dtw_path(cost_matrix):
    """Simple DTW path extraction."""
    N, M = cost_matrix.shape
    D = np.full((N + 1, M + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            D[i, j] = cost_matrix[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i, j = N, M
    pr, ps = [], []
    while i > 0 and j > 0:
        pr.append(i - 1); ps.append(j - 1)
        choices = [D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]]
        m = np.argmin(choices)
        if m == 0:   i -= 1
        elif m == 1: j -= 1
        else:        i -= 1; j -= 1
    return pr[::-1], ps[::-1]


# ──────────────────────────────────────────────────────────────────────────────
# LID Switching Accuracy
# ──────────────────────────────────────────────────────────────────────────────

def lid_switching_accuracy(true_switches, pred_switches, tolerance_ms=200.0):
    """
    Evaluate timestamp precision of detected language switches.

    true_switches: list of (timestamp_ms, from_lang, to_lang)
    pred_switches: list of (timestamp_ms, from_lang, to_lang)

    Returns fraction of true switches detected within tolerance_ms.
    """
    if not true_switches:
        return 1.0  # no switches to detect

    matched = 0
    used_pred = set()

    for t_ts, t_from, t_to in true_switches:
        for i, (p_ts, p_from, p_to) in enumerate(pred_switches):
            if i in used_pred:
                continue
            if abs(p_ts - t_ts) <= tolerance_ms and t_from == p_from and t_to == p_to:
                matched += 1
                used_pred.add(i)
                break

    return matched / len(true_switches)


# ──────────────────────────────────────────────────────────────────────────────
# Print evaluation report
# ──────────────────────────────────────────────────────────────────────────────

def print_evaluation_report(metrics):
    print("\n" + "=" * 50)
    print("EVALUATION REPORT")
    print("=" * 50)

    if "english_wer" in metrics and metrics["english_wer"] is not None:
        wer_en = metrics["english_wer"] * 100
        status = "PASS" if wer_en < 15 else "FAIL"
        print(f"WER (English):        {wer_en:.1f}%  [{status}]  (target < 15%)")

    if "hindi_wer" in metrics and metrics["hindi_wer"] is not None:
        wer_hi = metrics["hindi_wer"] * 100
        status = "PASS" if wer_hi < 25 else "FAIL"
        print(f"WER (Hindi):          {wer_hi:.1f}%  [{status}]  (target < 25%)")

    if "lid_f1" in metrics and metrics["lid_f1"] is not None:
        f1 = metrics["lid_f1"]
        status = "PASS" if f1 >= 0.85 else "FAIL"
        print(f"LID F1:               {f1:.4f}  [{status}]  (target >= 0.85)")

    if "lid_switch_acc" in metrics and metrics["lid_switch_acc"] is not None:
        acc = metrics["lid_switch_acc"] * 100
        status = "PASS" if acc >= 80 else "FAIL"
        print(f"LID Switch Accuracy:  {acc:.1f}%  [{status}]  (±200ms tolerance)")

    if "mcd" in metrics and metrics["mcd"] is not None:
        m = metrics["mcd"]
        status = "PASS" if m < 8.0 else "FAIL"
        print(f"MCD:                  {m:.2f}  [{status}]  (target < 8.0)")

    if "eer" in metrics and metrics["eer"] is not None:
        eer = metrics["eer"] * 100
        status = "PASS" if eer < 10 else "FAIL"
        print(f"Anti-Spoof EER:       {eer:.2f}%  [{status}]  (target < 10%)")

    if "min_epsilon" in metrics and metrics["min_epsilon"] is not None:
        print(f"Min adversarial eps:  {metrics['min_epsilon']:.5f}")

    print("=" * 50 + "\n")
