"""
Task 2.1 – IPA Unified Representation for Hinglish

Standard G2P tools (phonemizer, espeak) fail on code-switched text because:
  - They can't handle mixed Devanagari + Latin scripts in one string
  - Hindi phonemes like ट ड ण ष have no direct English IPA equivalent
  - Common Hinglish borrowings like "yaar", "accha" have non-standard pronunciation

This module does:
  1. Detect language of each word/token (Devanagari vs Latin)
  2. Route to Hindi G2P or English G2P accordingly
  3. Return a single unified IPA string with a delimiter between words
"""

import re
import unicodedata


# ──────────────────────────────────────────────────────────────────────────────
# Hindi (Devanagari) → IPA mapping table
# ──────────────────────────────────────────────────────────────────────────────

# Vowels (matras)
HINDI_VOWELS = {
    "अ": "ə", "आ": "aː", "इ": "ɪ", "ई": "iː",
    "उ": "ʊ", "ऊ": "uː", "ए": "eː", "ऐ": "æː",
    "ओ": "oː", "औ": "ɔː", "ऋ": "rɪ", "अं": "ən",
    "ा": "aː", "ि": "ɪ",  "ी": "iː", "ु": "ʊ",
    "ू": "uː",  "े": "eː", "ै": "æː", "ो": "oː",
    "ौ": "ɔː",  "ं": "n",  "ः": "h",  "ँ": "̃",
    "ृ": "rɪ",
}

# Consonants
HINDI_CONSONANTS = {
    # Velars
    "क": "k",   "ख": "kʰ",  "ग": "ɡ",   "घ": "ɡʰ",  "ङ": "ŋ",
    # Palatals
    "च": "tʃ",  "छ": "tʃʰ", "ज": "dʒ",  "झ": "dʒʰ", "ञ": "ɲ",
    # Retroflex
    "ट": "ʈ",   "ठ": "ʈʰ",  "ड": "ɖ",   "ढ": "ɖʰ",  "ण": "ɳ",
    # Dentals
    "त": "t̪",   "थ": "t̪ʰ",  "द": "d̪",   "ध": "d̪ʰ",  "न": "n",
    # Labials
    "प": "p",   "फ": "pʰ",  "ब": "b",   "भ": "bʰ",  "म": "m",
    # Approximants & liquids
    "य": "j",   "र": "r",   "ल": "l",   "व": "ʋ",
    # Sibilants & aspirates
    "श": "ʃ",   "ष": "ʂ",   "स": "s",   "ह": "ɦ",
    # Conjuncts / special
    "क्ष": "kʂ", "त्र": "t̪r", "ज्ञ": "ɡj",
    # Nuqta consonants (borrowed sounds)
    "क़": "q",  "ख़": "x",   "ग़": "ɣ",  "ज़": "z",
    "ड़": "ɽ",  "ढ़": "ɽʰ",  "फ़": "f",
    # Halant (removes inherent vowel)
    "्": "",
    # Anusvara and visarga handled via vowels above
}

# Inherent vowel 'a' (schwa) in Hindi – added after consonants unless halant follows
INHERENT_VOWEL = "ə"

# Combine for full lookup
DEVANAGARI_TO_IPA = {**HINDI_VOWELS, **HINDI_CONSONANTS}


def devanagari_to_ipa(text):
    """
    Convert a Hindi word in Devanagari to IPA.
    Handles the inherent vowel (schwa) insertion rule.
    """
    ipa = []
    chars = list(text)
    i = 0
    while i < len(chars):
        ch = chars[i]

        # check two-character sequences first (e.g., क्ष)
        if i + 1 < len(chars):
            pair = chars[i] + chars[i + 1]
            if pair in DEVANAGARI_TO_IPA:
                ipa.append(DEVANAGARI_TO_IPA[pair])
                i += 2
                continue

        if ch in HINDI_CONSONANTS:
            ipa.append(HINDI_CONSONANTS[ch])
            # check next char – if it's not a matra/halant, add inherent vowel
            if i + 1 < len(chars):
                next_ch = chars[i + 1]
                if next_ch not in HINDI_VOWELS and next_ch not in ("्",):
                    ipa.append(INHERENT_VOWEL)
            else:
                # word-final consonant in Hindi usually retains inherent vowel
                # EXCEPT in common patterns – simplified: add schwa
                ipa.append(INHERENT_VOWEL)
        elif ch in HINDI_VOWELS:
            ipa.append(HINDI_VOWELS[ch])
        elif ch == "्":
            # halant: remove the inherent vowel we just added
            if ipa and ipa[-1] == INHERENT_VOWEL:
                ipa.pop()
        else:
            # Unknown char – pass through (handles numbers, punctuation)
            ipa.append(ch)

        i += 1

    return "".join(ipa)


# ──────────────────────────────────────────────────────────────────────────────
# English → IPA (rule-based + CMU dict fallback)
# ──────────────────────────────────────────────────────────────────────────────

# Common English words / technical terms used in Hinglish lectures
# Manually verified pronunciations for domain-specific words
ENGLISH_IPA_DICT = {
    # common function words
    "the": "ðə", "a": "ə", "an": "æn", "is": "ɪz", "are": "ɑːr",
    "and": "ænd", "or": "ɔːr", "of": "əv", "in": "ɪn", "on": "ɒn",
    "to": "tuː", "for": "fɔːr", "with": "wɪð", "this": "ðɪs", "that": "ðæt",
    "we": "wiː", "so": "soʊ", "it": "ɪt", "as": "æz", "at": "æt",
    # lecture common words
    "ok": "oʊˈkeɪ", "right": "raɪt", "yes": "jɛs", "no": "noʊ",
    "now": "naʊ", "then": "ðɛn", "here": "hɪər", "there": "ðɛər",
    # speech processing technical terms
    "cepstrum": "ˈsɛpstrəm", "cepstral": "ˈsɛpstrəl",
    "mel": "mɛl", "mfcc": "ɛm.ɛf.siː.siː",
    "spectrogram": "ˈspɛktrəɡræm",
    "filterbank": "ˈfɪltərˌbæŋk",
    "phoneme": "ˈfoʊniːm", "phonemes": "ˈfoʊniːmz",
    "formant": "ˈfɔːrmænt",
    "stochastic": "stəˈkæstɪk",
    "gaussian": "ˈɡaʊsiən",
    "viterbi": "vɪˈtɜːrbiː",
    "hmm": "eɪtʃ.ɛm.ɛm",
    "ctc": "siː.tiː.siː",
    "transformer": "trænsˈfɔːrmər",
    "attention": "əˈtɛnʃən",
    "softmax": "ˈsɒftmæks",
    "embedding": "ɛmˈbɛdɪŋ",
    "acoustic": "əˈkuːstɪk",
    "frequency": "ˈfriːkwənsiː",
    "amplitude": "ˈæmplɪtjuːd",
    "sinusoidal": "ˌsɪnjʊˈsɔɪdəl",
    "fourier": "ˈfʊərieɪ",
    "waveform": "ˈweɪvfɔːrm",
    "sampling": "ˈsæmplɪŋ",
    "quantization": "ˌkwɒntɪˈzeɪʃən",
    "prosody": "ˈprɒsədiː",
    "intonation": "ˌɪntəˈneɪʃən",
    "pitch": "pɪtʃ",
    "voiced": "vɔɪst",
    "unvoiced": "ʌnˈvɔɪst",
    "diphthong": "ˈdɪfθɒŋ",
    "articulatory": "ɑːrˈtɪkjʊlətɔːriː",
    "coarticulation": "koʊˌɑːrtɪkjʊˈleɪʃən",
    "spectral": "ˈspɛktrəl",
    "subtraction": "səbˈtrækʃən",
    "wiener": "ˈwiːnər",
    "concatenation": "kənˌkætɪˈneɪʃən",
    "perplexity": "pərˈplɛksɪtiː",
    "trigram": "ˈtraɪɡræm",
    "bigram": "ˈbaɪɡræm",
    "wav2vec": "wɛv.tuː.vɛk",
    "whisper": "ˈwɪspər",
}

# Fallback English G2P rules (very simplified – handles ~80% of cases)
ENGLISH_RULES = [
    # multi-character patterns first
    (r"tion",  "ʃən"),
    (r"sion",  "ʒən"),
    (r"ck",    "k"),
    (r"ph",    "f"),
    (r"th",    "ð"),
    (r"sh",    "ʃ"),
    (r"ch",    "tʃ"),
    (r"wh",    "w"),
    (r"ng",    "ŋ"),
    (r"qu",    "kw"),
    (r"ee",    "iː"),
    (r"ea",    "iː"),
    (r"oo",    "uː"),
    (r"ou",    "aʊ"),
    (r"ow",    "oʊ"),
    (r"ai",    "eɪ"),
    (r"ay",    "eɪ"),
    (r"ie",    "iː"),
    (r"igh",   "aɪ"),
    # silent e rule (very rough)
    (r"a(?=.*e$)", "eɪ"),
    # single chars
    (r"a",    "æ"),
    (r"e",    "ɛ"),
    (r"i",    "ɪ"),
    (r"o",    "ɒ"),
    (r"u",    "ʌ"),
    (r"y",    "j"),
    (r"b",    "b"), (r"c",  "k"), (r"d",  "d"),
    (r"f",    "f"), (r"g",  "ɡ"), (r"h",  "h"),
    (r"j",    "dʒ"), (r"k", "k"), (r"l",  "l"),
    (r"m",    "m"), (r"n",  "n"), (r"p",  "p"),
    (r"r",    "r"), (r"s",  "s"), (r"t",  "t"),
    (r"v",    "v"), (r"w",  "w"), (r"x",  "ks"),
    (r"z",    "z"),
]


def english_to_ipa_rules(word):
    """Rule-based English → IPA (fallback when word not in dict)."""
    word = word.lower()
    result = word
    for pattern, replacement in ENGLISH_RULES:
        result = re.sub(pattern, replacement, result)
    return result


def english_to_ipa(word):
    """Dictionary lookup → rule-based fallback."""
    word_lower = word.lower().strip(".,!?;:")
    if word_lower in ENGLISH_IPA_DICT:
        return ENGLISH_IPA_DICT[word_lower]
    return english_to_ipa_rules(word_lower)


# ──────────────────────────────────────────────────────────────────────────────
# Language detection at word level
# ──────────────────────────────────────────────────────────────────────────────

DEVANAGARI_RANGE = (0x0900, 0x097F)


def is_devanagari(word):
    """Check if the word contains Devanagari characters."""
    for ch in word:
        cp = ord(ch)
        if DEVANAGARI_RANGE[0] <= cp <= DEVANAGARI_RANGE[1]:
            return True
    return False


def is_latin(word):
    """Check if the word is predominantly ASCII Latin."""
    latin_count = sum(1 for ch in word if ch.isascii() and ch.isalpha())
    return latin_count > 0.5 * max(len(word), 1)


# ──────────────────────────────────────────────────────────────────────────────
# Unified Hinglish → IPA converter
# ──────────────────────────────────────────────────────────────────────────────

# Common Hinglish colloquial words that don't transcribe well with pure rules
HINGLISH_OVERRIDES = {
    "yaar":    "jaːr",
    "accha":   "ətʃʰaː",
    "acha":    "ətʃʰaː",
    "matlab":  "mət̪ləb",
    "matlab":  "mət̪ləb",
    "nahi":    "nəɦiː",
    "nahin":   "nəɦɪ̃",
    "haan":    "ɦãː",
    "hona":    "ɦoːnaː",
    "karna":   "kərnaː",
    "dekho":   "d̪ɛkʰoː",
    "samajh":  "səmədʒ",
    "wala":    "ʋaːlaː",
    "wali":    "ʋaːliː",
    "kya":     "kjæː",
    "kyunki":  "kjʊ̃kiː",
    "agar":    "əɡər",
    "toh":     "t̪oː",
    "bhi":     "bʰiː",
    "ho":      "ɦoː",
    "hai":     "ɦæː",
    "hain":    "ɦɛ̃",
    "mein":    "mɛ̃",
    "matlab":  "mət̪ləb",
    "basically": "beɪsɪkliː",
    "like":    "laɪk",
    "actually": "æktʃuəliː",
    "means":   "miːnz",
}


def tokenize_hinglish(text):
    """
    Tokenize a Hinglish string into words, preserving script info.
    Returns list of (word, script) where script is 'hindi', 'english', or 'other'.
    """
    tokens = []
    # split on whitespace and punctuation, keeping punctuation as separate tokens
    raw_tokens = re.split(r"(\s+|[.,!?;:()\[\]\"'])", text)
    for tok in raw_tokens:
        if not tok or tok.isspace():
            tokens.append((" ", "space"))
        elif re.match(r"[.,!?;:()\[\]\"']", tok):
            tokens.append((tok, "punct"))
        elif is_devanagari(tok):
            tokens.append((tok, "hindi"))
        elif is_latin(tok):
            tokens.append((tok, "english"))
        else:
            tokens.append((tok, "other"))
    return tokens


def hinglish_to_ipa(text, word_boundary=" "):
    """
    Convert a Hinglish (Hindi-English code-switched) string to a unified IPA string.

    word_boundary: separator inserted between words in IPA output
                   Use "" for continuous IPA, " " for readable word-separated output
    """
    tokens = tokenize_hinglish(text)
    ipa_parts = []

    for word, script in tokens:
        if script == "space":
            ipa_parts.append(word_boundary)
        elif script == "punct":
            ipa_parts.append(word)
        elif script == "hindi":
            ipa_parts.append(devanagari_to_ipa(word))
        elif script == "english":
            # check Hinglish override first
            override = HINGLISH_OVERRIDES.get(word.lower())
            if override:
                ipa_parts.append(override)
            else:
                ipa_parts.append(english_to_ipa(word))
        else:
            ipa_parts.append(word)

    return "".join(ipa_parts)


def process_transcript_to_ipa(transcript_segments):
    """
    Process a list of transcript segments (dicts with 'text' and 'language' keys)
    and return a list of dicts with an added 'ipa' field.
    """
    result = []
    for seg in transcript_segments:
        ipa = hinglish_to_ipa(seg["text"])
        result.append({**seg, "ipa": ipa})
    return result


if __name__ == "__main__":
    test_cases = [
        "अब हम cepstrum के बारे में बात करते हैं",
        "basically यह एक spectral feature है",
        "the mel filterbank is applied after STFT",
        "yaar, इसको समझना बहुत important है",
        "HMM matlab Hidden Markov Model होता है",
    ]

    print("Hinglish → IPA conversion tests:\n")
    for sentence in test_cases:
        ipa = hinglish_to_ipa(sentence)
        print(f"  Input: {sentence}")
        print(f"  IPA:   {ipa}\n")
