import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# Devanagari Unicode ranges and common punctuation marks seen in transcripts.
DEVANAGARI_RE = re.compile(r"^[\u0900-\u097F]+$")
DEVANAGARI_OR_PUNC_RE = re.compile(r"^[\u0900-\u097F\u0964\u0965'’\-]+$")

# Standalone signs that cannot start/end words in standard orthography.
VOWEL_SIGNS = set("ािीुूृॄॅेैॉोौ")
NASAL_SIGNS = set("ंँः")
MATRAS = VOWEL_SIGNS | NASAL_SIGNS
HALANT = "्"
NUKTA = "़"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Question-3: classify unique Hindi words into correct vs incorrect spelling."
    )
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to unique words CSV (expected column: word)",
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Output sheet with columns: word, spelling_label",
    )
    parser.add_argument(
        "--summary_json",
        required=True,
        help="Summary JSON with counts and method details",
    )
    parser.add_argument(
        "--methodology_md",
        default=None,
        help="Optional short methodology markdown output",
    )
    return parser.parse_args()


def normalize_word(word: str) -> str:
    w = unicodedata.normalize("NFC", str(word or "")).strip()
    # Normalize dash variants and punctuation spacing.
    w = w.replace("–", "-").replace("—", "-").replace("―", "-")
    w = re.sub(r"\s+", "", w)
    return w


def has_non_devanagari_content(word: str) -> bool:
    if not word:
        return True
    # Allow danda, apostrophe, hyphen. If other punctuation/script appears, mark as incorrect.
    return DEVANAGARI_OR_PUNC_RE.fullmatch(word) is None


def invalid_orthography(word: str) -> Tuple[bool, str]:
    # Remove hyphen/apostrophe/danda for structural checks.
    core = re.sub(r"[\u0964\u0965'’\-]", "", word)
    if not core:
        return True, "empty_after_cleanup"

    # Must be Devanagari after cleanup.
    if DEVANAGARI_RE.fullmatch(core) is None:
        return True, "non_devanagari_chars"

    # Word should not start with matra/halant/nukta.
    if core[0] in MATRAS or core[0] in {HALANT, NUKTA}:
        return True, "invalid_start"
    # Word should not end with halant/nukta (ending with vowel signs is valid in Hindi).
    if core[-1] in {HALANT, NUKTA}:
        return True, "invalid_end"

    # Invalid local sequences.
    if "््" in core:
        return True, "double_halant"
    if "़़" in core:
        return True, "double_nukta"

    # Matra immediately after halant is generally invalid in Hindi orthography.
    for i in range(1, len(core)):
        prev_c, c = core[i - 1], core[i]
        if prev_c == HALANT and c in MATRAS:
            return True, "matra_after_halant"
        if prev_c in VOWEL_SIGNS and c in VOWEL_SIGNS:
            return True, "consecutive_matras"
        if prev_c == NUKTA and c in MATRAS:
            return True, "matra_after_nukta"

    # Very long repeated-char runs are usually transcription noise/typos.
    if re.search(r"(.)\1{3,}", core):
        return True, "long_repetition_noise"

    return False, "valid"


def classify_word(word: str) -> Tuple[str, str]:
    w = normalize_word(word)
    if not w:
        return "incorrect spelling", "empty"

    # English words in this project should appear in Devanagari transcription.
    if has_non_devanagari_content(w):
        return "incorrect spelling", "contains_non_devanagari"

    invalid, reason = invalid_orthography(w)
    if invalid:
        return "incorrect spelling", reason

    return "correct spelling", "valid_orthography"


def write_methodology(path: Path, summary: Dict):
    text = f"""# Question-3 Methodology Summary

## How words were classified
Used a rule-based Hindi orthography validation pipeline. A word is marked **incorrect spelling** if it has non-Devanagari characters (English/other scripts/symbol noise) or invalid Devanagari spelling structure (invalid start/end, illegal sign sequences, noisy repeated characters). Otherwise it is marked **correct spelling**.

## Why this approach
The task asks to catch spelling mistakes at scale in ~1.75 lakh words. A deterministic validator is efficient, reproducible, and targets obvious spelling errors without re-transcribing the full dataset.

## Final counts
- Total unique words processed: {summary["total_unique_words"]}
- Correct spelling words: {summary["correct_spelling_words"]}
- Incorrect spelling words: {summary["incorrect_spelling_words"]}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    summary_json = Path(args.summary_json)

    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    if "word" not in df.columns:
        raise ValueError(f"Expected 'word' column in {input_csv}")

    # Preserve first-seen order, remove null duplicates.
    words: List[str] = df["word"].dropna().astype(str).tolist()
    seen = set()
    unique_words = []
    for w in words:
        n = normalize_word(w)
        if n in seen:
            continue
        seen.add(n)
        unique_words.append(w)

    out_rows = []
    reason_counts: Dict[str, int] = {}
    for w in unique_words:
        label, reason = classify_word(w)
        out_rows.append({"word": normalize_word(w), "spelling_label": label})
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    out_df = pd.DataFrame(out_rows, columns=["word", "spelling_label"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    correct_count = int((out_df["spelling_label"] == "correct spelling").sum())
    incorrect_count = int((out_df["spelling_label"] == "incorrect spelling").sum())

    summary = {
        "input_csv": str(input_csv.resolve()),
        "output_csv": str(output_csv.resolve()),
        "total_unique_words": int(len(out_df)),
        "correct_spelling_words": correct_count,
        "incorrect_spelling_words": incorrect_count,
        "reason_breakdown": reason_counts,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.methodology_md:
        write_methodology(Path(args.methodology_md), summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
