import argparse
import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import soundfile as sf
from tqdm import tqdm


OUTPUT_COLUMNS = [
    "occurrence_id",
    "recording_id",
    "segment_idx",
    "disfluency_type",
    "matched_text",
    "match_rule",
    "segment_start_sec",
    "segment_end_sec",
    "clip_duration_sec",
    "clip_path",
    "clip_rel_path",
    "transcription_snippet",
    "transcription_url",
    "audio_url",
    "user_id",
    "language",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Question-2 pipeline: detect Hindi disfluencies and export segmented clips + sheet."
    )
    parser.add_argument("--metadata_csv", required=True, help="Path to metadata CSV (e.g. FT_Data - data.csv)")
    parser.add_argument("--disfluency_csv", required=True, help="Path to disfluency lexicon CSV")
    parser.add_argument("--raw_audio_dir", required=True, help="Directory with full .wav recordings by recording_id")
    parser.add_argument("--output_csv", required=True, help="Output CSV (one row per disfluency occurrence)")
    parser.add_argument(
        "--output_clips_dir",
        required=True,
        help="Directory to store segmented disfluency clips",
    )
    parser.add_argument("--summary_json", required=True, help="Output summary JSON")
    parser.add_argument(
        "--methodology_md",
        default=None,
        help="Optional path for short methodology summary markdown",
    )
    parser.add_argument("--language", default="hi", help="Language filter from metadata CSV")
    parser.add_argument("--num_workers", type=int, default=6, help="Parallel workers by recording")
    parser.add_argument("--max_rows", type=int, default=None, help="Optional debug limit")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    t = unicodedata.normalize("NFC", str(text)).strip().lower()
    t = t.replace("\n", " ").replace("\r", " ")
    t = t.replace("–", "-").replace("—", "-").replace("―", "-").replace("‑", "-")
    t = re.sub(r"\s+", " ", t)
    return t


def compile_phrase_pattern(phrase_norm: str) -> re.Pattern:
    escaped = re.escape(phrase_norm)
    # allow variable spacing and dash variants between phrase tokens
    escaped = escaped.replace(r"\ ", r"\s+").replace(r"\-", r"[\-\s]*")
    return re.compile(rf"(?<![\w\u0900-\u097f]){escaped}(?![\w\u0900-\u097f])")


def load_lexicon(disfluency_csv: Path) -> Dict[str, List[Tuple[str, re.Pattern]]]:
    df = pd.read_csv(disfluency_csv, encoding="utf-8-sig")
    lexicon: Dict[str, List[Tuple[str, re.Pattern]]] = {}
    for col in df.columns:
        col_name = str(col).strip()
        entries: List[Tuple[str, re.Pattern]] = []
        for item in df[col].dropna().tolist():
            phrase = normalize_text(item)
            if not phrase:
                continue
            entries.append((phrase, compile_phrase_pattern(phrase)))
        if entries:
            # dedupe while preserving order
            seen = set()
            uniq = []
            for p, pat in entries:
                if p in seen:
                    continue
                seen.add(p)
                uniq.append((p, pat))
            lexicon[col_name] = uniq
    if not lexicon:
        raise ValueError(f"Lexicon load failed: no usable entries in {disfluency_csv}")
    return lexicon


def regex_hits(text_norm: str) -> List[Tuple[str, str, str]]:
    hits: List[Tuple[str, str, str]] = []

    # repetition: "मैं मैं", "मैं-मैं", "जी जी जी"
    repetition = re.compile(r"\b([\w\u0900-\u097f]+)(?:[\s\-]+\1){1,}\b")
    for m in repetition.finditer(text_norm):
        hits.append(("Repetition", m.group(0), "regex_repetition"))

    # prolongation: same char repeated >= 3
    prolong = re.compile(r"([\u0900-\u097fa-z])\1{2,}")
    for m in prolong.finditer(text_norm):
        hits.append(("Prolongation", m.group(0), "regex_prolongation"))

    # false start marker: token with trailing hyphen
    false_start = re.compile(r"\b([\w\u0900-\u097f]+-)\b")
    for m in false_start.finditer(text_norm):
        hits.append(("False Start", m.group(1), "regex_false_start"))

    return hits


def detect_disfluencies(
    text: str, lexicon: Dict[str, List[Tuple[str, re.Pattern]]]
) -> Tuple[str, str, List[Tuple[str, str, str]]]:
    raw_text = str(text or "")
    text_norm = normalize_text(raw_text)
    hits: List[Tuple[str, str, str]] = []

    for disfluency_type, phrases in lexicon.items():
        for phrase, pattern in phrases:
            if pattern.search(text_norm):
                hits.append((disfluency_type, phrase, "lexicon"))

    hits.extend(regex_hits(text_norm))

    uniq: List[Tuple[str, str, str]] = []
    seen = set()
    for disfluency_type, matched_text, rule in hits:
        key = (disfluency_type, matched_text, rule)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((disfluency_type, matched_text, rule))
    return raw_text, text_norm, uniq


def infer_upload_folder_id(row: dict) -> str:
    for col in ["rec_url_gcp", "transcription_url_gcp", "metadata_url_gcp", "audio_url", "transcription_url"]:
        url = str(row.get(col, "")).strip()
        m = re.search(r"/hi/([^/]+)/", url)
        if m:
            return m.group(1)
        m = re.search(r"upload_goai/([^/]+)/", url)
        if m:
            return m.group(1)
    return ""


def resolve_urls(row: dict) -> Tuple[str, str]:
    recording_id = str(row["recording_id"])
    folder_id = infer_upload_folder_id(row)
    if not folder_id:
        raise ValueError(f"Unable to infer folder_id for recording_id={recording_id}")

    transcript_candidates = [
        str(row.get("transcription_url", "")).strip(),
        str(row.get("transcription_url_gcp", "")).strip(),
        f"https://storage.googleapis.com/upload_goai/{folder_id}/{recording_id}_transcription.json",
    ]
    audio_candidates = [
        str(row.get("audio_url", "")).strip(),
        str(row.get("rec_url_gcp", "")).strip(),
        f"https://storage.googleapis.com/upload_goai/{folder_id}/{recording_id}_audio.wav",
    ]

    transcript_url = ""
    last_error = None
    for url in transcript_candidates:
        if not url:
            continue
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                _ = r.json()
                transcript_url = url
                break
        except Exception as exc:
            last_error = exc

    if not transcript_url:
        raise ValueError(f"Cannot resolve transcription URL for {recording_id}. Last error: {last_error}")

    audio_url = next((u for u in audio_candidates if u), "")
    if "upload_goai" in transcript_url:
        audio_url = f"https://storage.googleapis.com/upload_goai/{folder_id}/{recording_id}_audio.wav"
    return audio_url, transcript_url


def fetch_segments(transcript_url: str) -> List[dict]:
    resp = requests.get(transcript_url, timeout=45)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected transcript payload type: {type(payload)} for {transcript_url}")
    return payload


def clip_audio(audio: List[float], sr: int, start_sec: float, end_sec: float) -> List[float]:
    start_idx = max(0, int(round(start_sec * sr)))
    end_idx = min(len(audio), int(round(end_sec * sr)))
    if end_idx <= start_idx:
        return []
    return audio[start_idx:end_idx]


def process_recording(
    row: dict,
    raw_audio_dir: Path,
    output_clips_dir: Path,
    lexicon: Dict[str, List[Tuple[str, re.Pattern]]],
) -> Tuple[List[dict], Optional[str]]:
    recording_id = str(row["recording_id"])
    user_id = str(row.get("user_id", ""))
    language = str(row.get("language", ""))
    audio_path = raw_audio_dir / f"{recording_id}.wav"
    if not audio_path.exists():
        return [], f"Missing raw audio: {audio_path}"

    try:
        audio_url, transcript_url = resolve_urls(row)
        segments = fetch_segments(transcript_url)
        audio, sr = sf.read(str(audio_path), always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)

        rec_out_dir = output_clips_dir / recording_id
        rec_out_dir.mkdir(parents=True, exist_ok=True)

        out_rows: List[dict] = []
        for seg_idx, seg in enumerate(segments):
            start_sec = float(seg.get("start", 0.0))
            end_sec = float(seg.get("end", 0.0))
            if end_sec <= start_sec:
                continue

            raw_text, _, hits = detect_disfluencies(seg.get("text", ""), lexicon)
            if not hits:
                continue

            # task asks segment-level clipping by segment timestamps
            seg_audio = clip_audio(audio, sr, start_sec, end_sec)
            if len(seg_audio) == 0:
                continue

            for hit_idx, (dtype, matched_text, match_rule) in enumerate(hits):
                clip_name = f"{recording_id}_{seg_idx:04d}_{hit_idx:02d}.wav"
                clip_path = rec_out_dir / clip_name
                sf.write(str(clip_path), seg_audio, sr)

                out_rows.append(
                    {
                        "occurrence_id": f"{recording_id}_{seg_idx:04d}_{hit_idx:02d}",
                        "recording_id": recording_id,
                        "segment_idx": seg_idx,
                        "disfluency_type": dtype,
                        "matched_text": matched_text,
                        "match_rule": match_rule,
                        "segment_start_sec": round(start_sec, 3),
                        "segment_end_sec": round(end_sec, 3),
                        "clip_duration_sec": round(end_sec - start_sec, 3),
                        "clip_path": str(clip_path.resolve()),
                        "clip_rel_path": str(clip_path.relative_to(output_clips_dir.parent)),
                        "transcription_snippet": raw_text,
                        "transcription_url": transcript_url,
                        "audio_url": audio_url,
                        "user_id": user_id,
                        "language": language,
                    }
                )
        return out_rows, None
    except Exception as exc:
        return [], f"recording_id={recording_id}: {exc}"


def write_methodology(path: Path, summary: dict):
    text = f"""# Question-2 Methodology Summary

## 1) How disfluencies were detected (Short answer)
Used a hybrid rule-based method: lexicon phrase matching from the provided disfluency list plus regex rules for repetitions, prolongations, and false-start hyphen patterns on normalized Hindi text.

## 2) How clips were cut from complete recordings (Short answer)
Each disfluency-positive transcript segment was cut from the full `.wav` recording using segment `start` and `end` timestamps from the transcription JSON, then saved as a standalone clip per detected occurrence.

## 3) Preprocessing / normalization applied (Short answer)
Applied Unicode NFC normalization, lowercasing, dash unification (`–/—` -> `-`), and whitespace normalization before matching.

## Run Summary
- Input Hindi recordings: {summary["input_rows_hi"]}
- Output disfluency rows: {summary["detected_disfluency_rows"]}
- Recordings with hits: {summary["unique_recordings_with_hits"]}
- Clips directory: `{summary["clips_dir"]}`
- Output sheet: `{summary["output_csv"]}`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main():
    args = parse_args()

    metadata_path = Path(args.metadata_csv)
    disfluency_csv = Path(args.disfluency_csv)
    raw_audio_dir = Path(args.raw_audio_dir)
    output_csv = Path(args.output_csv)
    output_clips_dir = Path(args.output_clips_dir)
    summary_json = Path(args.summary_json)

    df = pd.read_csv(metadata_path)
    required = {"recording_id", "language"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required metadata columns: {sorted(missing)}")

    df = df[df["language"].astype(str).str.lower() == args.language.lower()].copy()
    if args.max_rows:
        df = df.head(args.max_rows)
    rows = df.to_dict(orient="records")

    lexicon = load_lexicon(disfluency_csv)
    output_clips_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    all_rows: List[dict] = []
    failures: List[str] = []

    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = [ex.submit(process_recording, row, raw_audio_dir, output_clips_dir, lexicon) for row in rows]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing recordings"):
            rows_out, err = fut.result()
            if err:
                failures.append(err)
            if rows_out:
                all_rows.extend(rows_out)

    out_df = pd.DataFrame(all_rows, columns=OUTPUT_COLUMNS)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    summary = {
        "input_rows_hi": len(rows),
        "detected_disfluency_rows": int(len(out_df)),
        "unique_recordings_with_hits": int(out_df["recording_id"].nunique()) if len(out_df) else 0,
        "disfluency_type_counts": out_df["disfluency_type"].value_counts().to_dict() if len(out_df) else {},
        "lexicon_source": str(disfluency_csv.resolve()),
        "clips_dir": str(output_clips_dir.resolve()),
        "output_csv": str(output_csv.resolve()),
        "failures": len(failures),
        "failure_examples": failures[:20],
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.methodology_md:
        write_methodology(Path(args.methodology_md), summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
