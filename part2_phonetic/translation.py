"""
Task 2.2 – Semantic Translation: Hinglish/IPA → Santhali

Target language: Santhali (ISO 639-3: sat)
Script: Ol Chiki (ᱚᱞ ᱪᱤᱠᱤ), but we also store Latin transliterations
        since most NLP tools don't support Ol Chiki yet.

Since no reliable machine translation exists for Santhali, I built a
500-term parallel corpus for speech/NLP technical vocabulary (see
data/parallel_corpus/santhali_corpus.json).

Translation approach:
  1. Dictionary lookup from the corpus for technical terms
  2. Word-by-word translation for common function words
  3. For untranslatable terms: phonological borrowing
     (adapt the English pronunciation to Santhali phonotactics)
  4. Output both Latin transliteration and Ol Chiki script where available
"""

import json
import re
from pathlib import Path
from part2_phonetic.g2p_hinglish import hinglish_to_ipa, ENGLISH_IPA_DICT


CORPUS_PATH = Path("data/parallel_corpus/santhali_corpus.json")


# ──────────────────────────────────────────────────────────────────────────────
# Load parallel corpus
# ──────────────────────────────────────────────────────────────────────────────

def load_corpus(path=CORPUS_PATH):
    with open(path) as f:
        data = json.load(f)
    # build lookup dicts
    # corpus entries: {"english": ..., "hindi": ..., "santhali_latin": ..., "santhali_olchiki": ...}
    en_to_sat  = {}
    hi_to_sat  = {}
    ipa_to_sat = {}

    for entry in data:
        en  = entry.get("english", "").lower().strip()
        hi  = entry.get("hindi", "").strip()
        sat = entry.get("santhali_latin", "").strip()
        sat_oc = entry.get("santhali_olchiki", "")

        if en:
            en_to_sat[en] = {"latin": sat, "olchiki": sat_oc}
        if hi:
            hi_to_sat[hi] = {"latin": sat, "olchiki": sat_oc}

    return en_to_sat, hi_to_sat


# ──────────────────────────────────────────────────────────────────────────────
# Phonological borrowing rules (English → Santhali phonotactics)
# ──────────────────────────────────────────────────────────────────────────────

# Santhali doesn't allow:
#   - Initial consonant clusters → insert schwa (ə → 'o' in Santhali orthography)
#   - Final consonant clusters → insert 'i' epenthesis
#   - /v/ → /b/ (no /v/ in Santhali)
#   - /f/ → /p/ (no /f/ in Santhali)
#   - /z/ → /j/ or /s/ depending on position

SANTHALI_ADAPTATIONS = [
    (r"^str",   "ist"),
    (r"^spl",   "isp"),
    (r"^scr",   "isk"),
    (r"^sp",    "isp"),
    (r"^st",    "ist"),
    (r"^sk",    "isk"),
    (r"v",      "b"),
    (r"f",      "p"),
    (r"z",      "j"),
    (r"θ",      "t"),   # IPA theta
    (r"ð",      "d"),   # IPA eth
    (r"ŋ",      "ng"),
    (r"ʃ",      "sh"),
    (r"tʃ",     "c"),
    (r"dʒ",     "j"),
    (r"([ptkbdg])$",  r"\1i"),  # final stop → add -i
]


def phonological_borrowing(word):
    """
    Adapt an English word to Santhali phonotactics for borrowing.
    Returns Latin transliteration of the borrowed form.
    """
    adapted = word.lower()
    for pattern, replacement in SANTHALI_ADAPTATIONS:
        adapted = re.sub(pattern, replacement, adapted)
    return adapted


# ──────────────────────────────────────────────────────────────────────────────
# Common Santhali function words and grammar particles
# ──────────────────────────────────────────────────────────────────────────────

# These are actual Santhali words for common concepts
SANTHALI_FUNCTION_WORDS = {
    # English → Santhali Latin
    "the":   "em",
    "a":     "mit",
    "is":    "akantalea",
    "are":   "akantaea",
    "was":   "emakana",
    "and":   "ar",
    "or":    "nahin te",
    "not":   "bae",
    "no":    "bae",
    "yes":   "hon",
    "this":  "noa",
    "that":  "doa",
    "here":  "nonde",
    "there": "onde",
    "what":  "gatea",
    "how":   "ceta",
    "why":   "gatete",
    "when":  "horo",
    "we":    "aling",
    "i":     "in",
    "you":   "am",
    "they":  "kin",
    "it":    "em",
    "to":    "te",
    "of":    "ren",
    "in":    "re",
    "on":    "upunte",
    "from":  "te",
    "with":  "sadom",
    "for":   "khator",
    "about": "bab",
    "if":    "jodi",
    "then":  "tahen",
    "so":    "tahen em",
    "also":  "do",
    "very":  "dher",
    "more":  "ato",
    "some":  "kaji",
    "all":   "hapromko",
    "now":   "adomte",
    "after": "pahile",
    "before": "adom",
    "called": "jom",
    "means":  "mane",
    "like":   "hopon",
    "using":  "hapramte",
    "called": "ren naon",
}

# Hindi common words → Santhali
HINDI_FUNCTION_WORDS = {
    "है":    "akantalea",
    "हैं":   "akantaea",
    "था":    "emakana",
    "की":    "ren",
    "का":    "ren",
    "के":    "ren",
    "में":   "re",
    "पर":    "upunte",
    "से":    "te",
    "को":    "te",
    "और":    "ar",
    "या":    "nahin te",
    "नहीं":  "bae",
    "यह":   "noa",
    "वह":   "doa",
    "हम":   "aling",
    "मैं":   "in",
    "आप":   "am",
    "यहाँ":  "nonde",
    "वहाँ":  "onde",
    "क्या":  "gatea",
    "कैसे":  "ceta",
    "क्यों": "gatete",
    "जब":   "horo",
    "तो":    "tahen",
    "भी":    "do",
    "बहुत":  "dher",
    "अधिक":  "ato",
    "सब":    "hapromko",
    "अब":   "adomte",
    "बाद":   "pahile",
    "पहले":  "adom",
    "मतलब": "mane",
}


# ──────────────────────────────────────────────────────────────────────────────
# Main translator class
# ──────────────────────────────────────────────────────────────────────────────

class HinglishToSanthaliTranslator:
    def __init__(self, corpus_path=CORPUS_PATH):
        self.en_dict, self.hi_dict = load_corpus(corpus_path)
        self.en_func  = SANTHALI_FUNCTION_WORDS
        self.hi_func  = HINDI_FUNCTION_WORDS

    def translate_word(self, word, script):
        """
        Translate a single word/token.
        Returns (santhali_latin, santhali_olchiki, method)
        method: 'corpus', 'function', 'borrow'
        """
        if script == "english":
            wl = word.lower()
            if wl in self.en_dict:
                entry = self.en_dict[wl]
                return entry["latin"], entry["olchiki"], "corpus"
            if wl in self.en_func:
                return self.en_func[wl], "", "function"
            # phonological borrowing
            return phonological_borrowing(wl), "", "borrow"

        elif script == "hindi":
            if word in self.hi_dict:
                entry = self.hi_dict[word]
                return entry["latin"], entry["olchiki"], "corpus"
            if word in self.hi_func:
                return self.hi_func[word], "", "function"
            # fall back to phonological borrowing using IPA as intermediate
            from part2_phonetic.g2p_hinglish import devanagari_to_ipa
            ipa = devanagari_to_ipa(word)
            return phonological_borrowing(ipa), "", "borrow"

        return word, "", "passthrough"

    def translate(self, text, include_metadata=False):
        """
        Translate a Hinglish sentence to Santhali.
        Returns dict with:
          "santhali":   Latin transliteration
          "olchiki":    Ol Chiki script (where available)
          "ipa":        IPA of the original
        """
        from part2_phonetic.g2p_hinglish import tokenize_hinglish, hinglish_to_ipa

        tokens = tokenize_hinglish(text)
        ipa_str = hinglish_to_ipa(text)

        sat_parts = []
        sat_oc_parts = []
        word_log = []

        for word, script in tokens:
            if script in ("space", "punct"):
                sat_parts.append(word if script == "space" else word)
                sat_oc_parts.append(word if script == "space" else word)
                continue

            sat_latin, sat_oc, method = self.translate_word(word, script)
            sat_parts.append(sat_latin)
            sat_oc_parts.append(sat_oc if sat_oc else sat_latin)

            if include_metadata:
                word_log.append({
                    "source": word,
                    "script": script,
                    "santhali": sat_latin,
                    "method": method,
                })

        result = {
            "santhali": "".join(sat_parts).strip(),
            "olchiki":  "".join(sat_oc_parts).strip(),
            "ipa":      ipa_str,
        }
        if include_metadata:
            result["word_log"] = word_log

        return result

    def translate_segments(self, segments):
        """
        Translate a list of segment dicts (from Whisper output).
        Each segment should have 'text' and optionally 'language'.
        """
        translated = []
        for seg in segments:
            t = self.translate(seg["text"], include_metadata=False)
            translated.append({**seg, **t})
        return translated


if __name__ == "__main__":
    translator = HinglishToSanthaliTranslator()

    test_sentences = [
        "अब हम cepstrum के बारे में बात करते हैं",
        "the mel filterbank extracts spectral features",
        "HMM matlab Hidden Markov Model होता है",
        "basically यह एक stochastic process है",
    ]

    for sentence in test_sentences:
        result = translator.translate(sentence, include_metadata=True)
        print(f"Source:   {sentence}")
        print(f"Santhali: {result['santhali']}")
        print(f"IPA:      {result['ipa']}")
        print()
